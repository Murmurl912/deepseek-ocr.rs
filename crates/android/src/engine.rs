use std::{
    cell::RefCell,
    convert::TryFrom,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
};

use crate::{AndroidLogLevel, AndroidProgressCallback, AndroidProgressEvent, dispatch_log};
use anyhow::{Context, Result, anyhow, ensure};
use candle_core::{DType, Device};
use deepseek_ocr_core::{
    CancellationToken, ModelKind, ModelLoadArgs,
    inference::{DecodeOutcome, DecodeParameters, OcrEngine, VisionSettings, render_prompt},
    streaming::DeltaTracker,
};
use deepseek_ocr_infer_deepseek::load_model as load_deepseek_model;
use deepseek_ocr_infer_paddleocr::load_model as load_paddle_model;
use image::DynamicImage;
#[derive(Clone, Copy, Debug)]
enum LogPriority {
    Debug,
    Info,
    Warn,
}

impl From<LogPriority> for AndroidLogLevel {
    fn from(value: LogPriority) -> Self {
        match value {
            LogPriority::Debug => AndroidLogLevel::Debug,
            LogPriority::Info => AndroidLogLevel::Info,
            LogPriority::Warn => AndroidLogLevel::Warn,
        }
    }
}
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct EngineModelConfig {
    pub kind: ModelKind,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weights_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct EngineSettings {
    pub template: String,
    pub system_prompt: Option<String>,
    pub vision: VisionSettings,
    pub decode: DecodeParameters,
}

pub struct EngineArgs {
    pub model: EngineModelConfig,
    pub settings: EngineSettings,
}

pub struct AndroidOcrEngine {
    backend: Box<dyn OcrEngine>,
    tokenizer: Tokenizer,
    settings: EngineSettings,
}

impl AndroidOcrEngine {
    pub fn new(args: EngineArgs) -> Result<Self> {
        ensure_exists(&args.model.config_path, "model config")?;
        ensure_exists(&args.model.tokenizer_path, "tokenizer")?;
        ensure_exists(&args.model.weights_path, "weights")?;

        log(
            LogPriority::Info,
            format!(
                "initialising engine (kind={:?}, config={}, tokenizer={}, weights={})",
                args.model.kind,
                args.model.config_path.display(),
                args.model.tokenizer_path.display(),
                args.model.weights_path.display()
            ),
        );

        let tokenizer = Tokenizer::from_file(&args.model.tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer from {}: {err}",
                args.model.tokenizer_path.display()
            )
        })?;

        let backend = load_backend(&args.model)?;
        log(
            LogPriority::Info,
            format!("model weights loaded (kind={:?})", backend.kind()),
        );
        Ok(Self {
            backend,
            tokenizer,
            settings: args.settings,
        })
    }

    pub fn infer(
        &self,
        raw_prompt: &str,
        images: &[DynamicImage],
        progress: Option<Arc<dyn AndroidProgressCallback>>,
        cancel: Option<CancellationToken>,
    ) -> Result<DecodeOutcome> {
        let system_prompt = self.settings.system_prompt.as_deref().unwrap_or("");
        let prompt = render_prompt(&self.settings.template, system_prompt, raw_prompt)
            .context("failed to render prompt")?;
        let slots = prompt.matches("<image>").count();
        ensure!(
            slots == images.len(),
            "prompt includes {slots} <image> tokens but {} image(s) provided",
            images.len()
        );

        log(
            LogPriority::Info,
            format!(
                "starting decode (kind={:?}, prompt_chars={}, images={}, base_size={}, image_size={}, crop_mode={}, max_new_tokens={}, do_sample={}, temperature={})",
                self.backend.kind(),
                prompt.chars().count(),
                images.len(),
                self.settings.vision.base_size,
                self.settings.vision.image_size,
                self.settings.vision.crop_mode,
                self.settings.decode.max_new_tokens,
                self.settings.decode.do_sample,
                self.settings.decode.temperature
            ),
        );

        if prompt.len() > 512 {
            log(
                LogPriority::Debug,
                format!(
                    "prompt preview (truncated): {}",
                    prompt.chars().take(256).collect::<String>()
                ),
            );
        } else {
            log(LogPriority::Debug, format!("prompt preview: {}", prompt));
        }

        if images.is_empty() {
            log(
                LogPriority::Warn,
                "prompt contains no <image> tokens; OCR result may be empty",
            );
        }

        let progress_state = progress.as_ref().map(|callback| {
            Rc::new(RefCell::new(ProgressDispatcher::new(
                self.tokenizer.clone(),
                Arc::clone(callback),
            )))
        });
        let mut callback_holder: Option<Box<dyn Fn(usize, &[i64])>> = None;
        if let Some(state) = progress_state.as_ref() {
            let state = Rc::clone(state);
            callback_holder = Some(Box::new(move |count, ids| {
                state.borrow_mut().handle(count, ids, false);
            }));
        }

        let outcome = self.backend.decode(
            &self.tokenizer,
            &prompt,
            images,
            self.settings.vision,
            &self.settings.decode,
            callback_holder.as_deref(),
            cancel.as_ref(),
        )?;
        if let Some(state) = progress_state.as_ref() {
            state.borrow_mut().finalize(&outcome.generated_tokens);
        }
        log(
            LogPriority::Info,
            format!(
                "decode finished (prompt_tokens={}, response_tokens={})",
                outcome.prompt_tokens, outcome.response_tokens
            ),
        );
        Ok(outcome)
    }
}

fn load_backend(config: &EngineModelConfig) -> Result<Box<dyn OcrEngine>> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let load_args = ModelLoadArgs {
        kind: config.kind,
        config_path: Some(config.config_path.as_path()),
        weights_path: Some(config.weights_path.as_path()),
        device,
        dtype,
    };
    match config.kind {
        ModelKind::Deepseek => load_deepseek_model(load_args),
        ModelKind::PaddleOcrVl => load_paddle_model(load_args),
    }
}

fn ensure_exists(path: &Path, label: &str) -> Result<()> {
    ensure!(path.exists(), "{label} not found at {}", path.display());
    Ok(())
}

fn log(priority: LogPriority, message: impl Into<String>) {
    let msg_string = message.into();
    if dispatch_log(priority.into(), &msg_string) {
        return;
    }
    println!("[AndroidOCR][{:?}] {}", priority, msg_string);
}

struct ProgressDispatcher {
    tokenizer: Tokenizer,
    tracker: DeltaTracker,
    last_count: usize,
    callback: Arc<dyn AndroidProgressCallback>,
}

impl ProgressDispatcher {
    fn new(tokenizer: Tokenizer, callback: Arc<dyn AndroidProgressCallback>) -> Self {
        Self {
            tokenizer,
            tracker: DeltaTracker::default(),
            last_count: 0,
            callback,
        }
    }

    fn handle(&mut self, count: usize, ids: &[i64], is_final: bool) {
        if (!is_final && count <= self.last_count) || count == 0 {
            if count == 0 {
                self.last_count = 0;
            }
            return;
        }

        let upto = count.min(ids.len());
        if upto == 0 {
            self.last_count = count;
            return;
        }

        let token_slice: Vec<u32> = ids[..upto]
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();
        if token_slice.is_empty() {
            self.last_count = count;
            return;
        }

        if let Ok(full_text) = self.tokenizer.decode(&token_slice, true) {
            let delta = self.tracker.advance(&full_text, is_final);
            if !delta.is_empty() {
                self.callback.on_progress(AndroidProgressEvent {
                    token_count: count.min(u32::MAX as usize) as u32,
                    delta_text: delta,
                    is_final,
                });
            }
        }

        self.last_count = count;
    }

    fn finalize(&mut self, tokens: &[i64]) {
        self.handle(tokens.len(), tokens, true);
    }
}
