mod engine;

use std::{
    path::PathBuf,
    sync::{Arc, OnceLock, RwLock},
};

use anyhow::{Context, Result};
use deepseek_ocr_core::{
    CancellationToken,
    inference::{DecodeParameters, ModelKind, VisionSettings},
};
use engine::{AndroidOcrEngine, EngineArgs, EngineModelConfig, EngineSettings};
use image::DynamicImage;
use thiserror::Error;

#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum AndroidModelKind {
    Deepseek,
    PaddleOcrVl,
}

impl From<AndroidModelKind> for ModelKind {
    fn from(value: AndroidModelKind) -> Self {
        match value {
            AndroidModelKind::Deepseek => ModelKind::Deepseek,
            AndroidModelKind::PaddleOcrVl => ModelKind::PaddleOcrVl,
        }
    }
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct AndroidModelPaths {
    pub kind: AndroidModelKind,
    pub config_path: String,
    pub tokenizer_path: String,
    pub weights_path: String,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct AndroidInferenceOptions {
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
    pub max_new_tokens: u32,
    pub use_cache: bool,
    pub do_sample: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<u32>,
    pub repetition_penalty: f64,
    pub no_repeat_ngram_size: Option<u32>,
    pub seed: Option<u64>,
    pub template: String,
    pub system_prompt: Option<String>,
}

impl Default for AndroidInferenceOptions {
    fn default() -> Self {
        Self {
            base_size: 1024,
            image_size: 640,
            crop_mode: true,
            max_new_tokens: 512,
            use_cache: true,
            do_sample: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: Some(20),
            seed: None,
            template: "plain".to_string(),
            system_prompt: None,
        }
    }
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct AndroidRunConfig {
    pub model: AndroidModelPaths,
    pub inference: AndroidInferenceOptions,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct AndroidImageInput {
    pub data: Vec<u8>,
    pub mime_type: Option<String>,
}

#[derive(Clone, Copy, Debug, uniffi::Enum)]
pub enum AndroidLogLevel {
    Debug,
    Info,
    Warn,
}

#[uniffi::export(callback_interface)]
pub trait AndroidLogCallback: Send + Sync {
    fn on_log(&self, level: AndroidLogLevel, message: String);
}

#[derive(uniffi::Object, Debug)]
pub struct AndroidStopHandle {
    token: CancellationToken,
}

#[uniffi::export]
impl AndroidStopHandle {
    #[uniffi::constructor]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            token: CancellationToken::new(),
        })
    }

    pub fn cancel(&self) {
        self.token.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }
}

impl AndroidStopHandle {
    pub(crate) fn token(&self) -> CancellationToken {
        self.token.clone()
    }
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct AndroidProgressEvent {
    pub token_count: u32,
    pub delta_text: String,
    pub is_final: bool,
}

#[uniffi::export(callback_interface)]
pub trait AndroidProgressCallback: Send + Sync {
    fn on_progress(&self, event: AndroidProgressEvent);
}

#[derive(Debug, Error, uniffi::Error)]
pub enum AndroidOcrError {
    #[error("{0}")]
    Failure(String),
}

impl From<anyhow::Error> for AndroidOcrError {
    fn from(err: anyhow::Error) -> Self {
        AndroidOcrError::Failure(err.to_string())
    }
}

#[uniffi::export]
pub fn android_run_ocr(
    config: AndroidRunConfig,
    prompt: String,
    images: Vec<AndroidImageInput>,
    log_callback: Option<Box<dyn AndroidLogCallback>>,
    progress_callback: Option<Box<dyn AndroidProgressCallback>>,
    stop_handle: Option<Arc<AndroidStopHandle>>,
) -> Result<String, AndroidOcrError> {
    let _scoped_logger = ScopedLogCallback::install(log_callback);
    let decoded_images = decode_images(images).map_err(AndroidOcrError::from)?;
    let args = EngineArgs::try_from(config).map_err(AndroidOcrError::from)?;
    let engine = AndroidOcrEngine::new(args).map_err(AndroidOcrError::from)?;
    let progress_callback = progress_callback.map(|cb| Arc::from(cb));
    let cancel_token = stop_handle.as_ref().map(|handle| handle.token());
    let outcome = engine
        .infer(
            &prompt,
            &decoded_images,
            progress_callback.clone(),
            cancel_token,
        )
        .map_err(AndroidOcrError::from)?;
    Ok(outcome.text)
}

fn decode_images(inputs: Vec<AndroidImageInput>) -> Result<Vec<DynamicImage>> {
    inputs
        .into_iter()
        .enumerate()
        .map(|(idx, input)| {
            image::load_from_memory(&input.data)
                .with_context(|| format!("failed to decode image #{idx}"))
        })
        .collect()
}

impl TryFrom<AndroidRunConfig> for EngineArgs {
    type Error = anyhow::Error;

    fn try_from(config: AndroidRunConfig) -> Result<Self> {
        let AndroidRunConfig { model, inference } = config;
        let model_paths = EngineModelConfig {
            kind: model.kind.into(),
            config_path: PathBuf::from(model.config_path),
            tokenizer_path: PathBuf::from(model.tokenizer_path),
            weights_path: PathBuf::from(model.weights_path),
        };

        let AndroidInferenceOptions {
            base_size,
            image_size,
            crop_mode,
            max_new_tokens,
            use_cache,
            do_sample,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            no_repeat_ngram_size,
            seed,
            template,
            system_prompt,
        } = inference;

        let template_value = if template.is_empty() {
            "plain".to_string()
        } else {
            template
        };
        let system_prompt_value = system_prompt.and_then(|value| match value.trim() {
            "" => None,
            other => Some(other.to_string()),
        });

        let vision = VisionSettings {
            base_size,
            image_size,
            crop_mode,
        };
        let decode = DecodeParameters {
            max_new_tokens: max_new_tokens as usize,
            do_sample,
            temperature,
            top_p: if top_p < 1.0 { Some(top_p) } else { None },
            top_k: top_k.map(|value| value as usize),
            repetition_penalty: repetition_penalty as f32,
            no_repeat_ngram_size: no_repeat_ngram_size.map(|value| value as usize),
            seed,
            use_cache,
        };

        Ok(EngineArgs {
            model: model_paths,
            settings: EngineSettings {
                template: template_value,
                system_prompt: system_prompt_value,
                vision,
                decode,
            },
        })
    }
}

uniffi::setup_scaffolding!();

static LOG_CALLBACK: OnceLock<RwLock<Option<Arc<dyn AndroidLogCallback>>>> = OnceLock::new();

fn log_callback_slot() -> &'static RwLock<Option<Arc<dyn AndroidLogCallback>>> {
    LOG_CALLBACK.get_or_init(|| RwLock::new(None))
}

struct ScopedLogCallback {
    previous: Option<Arc<dyn AndroidLogCallback>>,
}

impl ScopedLogCallback {
    fn install(callback: Option<Box<dyn AndroidLogCallback>>) -> Self {
        let mut slot = log_callback_slot()
            .write()
            .expect("log callback lock poisoned");
        let previous = slot.clone();
        *slot = callback.map(|cb| cb.into());
        Self { previous }
    }
}

impl Drop for ScopedLogCallback {
    fn drop(&mut self) {
        let mut slot = log_callback_slot()
            .write()
            .expect("log callback lock poisoned");
        *slot = self.previous.clone();
    }
}

pub(crate) fn dispatch_log(level: AndroidLogLevel, message: &str) -> bool {
    let maybe_callback = {
        let guard = log_callback_slot()
            .read()
            .expect("log callback lock poisoned");
        guard.clone()
    };
    if let Some(callback) = maybe_callback {
        callback.on_log(level, message.to_string());
        true
    } else {
        false
    }
}
