use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use deepseek_ocr_android::{
    AndroidImageInput, AndroidInferenceOptions, AndroidLogCallback, AndroidLogLevel,
    AndroidModelKind, AndroidModelPaths, AndroidRunConfig, android_run_ocr,
};

const DEFAULT_BASE_SIZE: u32 = 1024;
const DEFAULT_IMAGE_SIZE: u32 = 640;
const DEFAULT_CROP_MODE: bool = true;
const DEFAULT_MAX_NEW_TOKENS: u32 = 512;
const DEFAULT_USE_CACHE: bool = true;
const DEFAULT_DO_SAMPLE: bool = false;
const DEFAULT_TEMPERATURE: f64 = 0.0;
const DEFAULT_TOP_P: f64 = 1.0;
const DEFAULT_REPETITION_PENALTY: f64 = 1.0;
const DEFAULT_TEMPLATE: &str = "plain";

fn main() -> Result<()> {
    let args = CliArgs::parse();
    let prompt = args.prompt_text()?;
    let images = load_images(&args.images)?;
    let run_config = AndroidRunConfig {
        model: AndroidModelPaths {
            kind: args.model_kind.into(),
            config_path: path_to_string(&args.config_path),
            tokenizer_path: path_to_string(&args.tokenizer_path),
            weights_path: path_to_string(&args.weights_path),
        },
        inference: AndroidInferenceOptions {
            base_size: args.base_size,
            image_size: args.image_size,
            crop_mode: args.crop_mode,
            max_new_tokens: args.max_new_tokens,
            use_cache: args.use_cache,
            do_sample: args.do_sample,
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: args.top_k,
            repetition_penalty: args.repetition_penalty,
            no_repeat_ngram_size: args.no_repeat_ngram_size,
            seed: args.seed,
            template: args.template,
            system_prompt: args.system_prompt,
        },
    };

    let logger = StdoutLogCallback;
    let response = android_run_ocr(
        run_config,
        prompt,
        images,
        Some(Box::new(logger)),
        None,
        None,
    )
    .context("android_run_ocr failed")?;
    println!("{response}");
    Ok(())
}

#[derive(Parser, Debug)]
#[command(
    name = "android-cli",
    about = "Run the DeepSeek OCR Android engine on desktop hosts",
    version
)]
struct CliArgs {
    /// Model family to load
    #[arg(long, value_enum, default_value_t = ModelKindArg::Deepseek)]
    model_kind: ModelKindArg,

    /// Path to the model config json
    #[arg(long)]
    config_path: PathBuf,

    /// Path to the tokenizer json
    #[arg(long)]
    tokenizer_path: PathBuf,

    /// Path to the model weights (safetensors)
    #[arg(long)]
    weights_path: PathBuf,

    /// Prompt text (use --prompt-file to read from disk)
    #[arg(long, value_name = "TEXT")]
    prompt: Option<String>,

    /// Prompt file
    #[arg(long, value_name = "FILE")]
    prompt_file: Option<PathBuf>,

    /// Image inputs (repeatable)
    #[arg(long = "image", value_name = "PATH", required = true)]
    images: Vec<PathBuf>,

    #[arg(long, default_value_t = DEFAULT_BASE_SIZE)]
    base_size: u32,

    #[arg(long, default_value_t = DEFAULT_IMAGE_SIZE)]
    image_size: u32,

    #[arg(long, default_value_t = DEFAULT_CROP_MODE)]
    crop_mode: bool,

    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_new_tokens: u32,

    #[arg(long, default_value_t = DEFAULT_USE_CACHE)]
    use_cache: bool,

    #[arg(long, default_value_t = DEFAULT_DO_SAMPLE)]
    do_sample: bool,

    #[arg(long, default_value_t = DEFAULT_TEMPERATURE)]
    temperature: f64,

    #[arg(long, default_value_t = DEFAULT_TOP_P)]
    top_p: f64,

    #[arg(long)]
    top_k: Option<u32>,

    #[arg(long, default_value_t = DEFAULT_REPETITION_PENALTY)]
    repetition_penalty: f64,

    #[arg(long)]
    no_repeat_ngram_size: Option<u32>,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long, default_value = DEFAULT_TEMPLATE)]
    template: String,

    #[arg(long)]
    system_prompt: Option<String>,
}

impl CliArgs {
    fn prompt_text(&self) -> Result<String> {
        if let Some(text) = &self.prompt {
            return Ok(text.clone());
        }
        if let Some(path) = &self.prompt_file {
            let data = fs::read_to_string(path)
                .with_context(|| format!("failed to load prompt file at {}", path.display()))?;
            return Ok(data);
        }
        Err(anyhow!("either --prompt or --prompt-file must be provided"))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ModelKindArg {
    Deepseek,
    PaddleOcrVl,
}

impl From<ModelKindArg> for AndroidModelKind {
    fn from(value: ModelKindArg) -> Self {
        match value {
            ModelKindArg::Deepseek => AndroidModelKind::Deepseek,
            ModelKindArg::PaddleOcrVl => AndroidModelKind::PaddleOcrVl,
        }
    }
}

fn load_images(paths: &[PathBuf]) -> Result<Vec<AndroidImageInput>> {
    paths
        .iter()
        .enumerate()
        .map(|(idx, path)| {
            let data = fs::read(path)
                .with_context(|| format!("failed to read image #{idx} at {}", path.display()))?;
            Ok(AndroidImageInput {
                data,
                mime_type: mime_from_path(path),
            })
        })
        .collect()
}

fn mime_from_path(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .and_then(|ext| match ext.as_str() {
            "png" => Some("image/png".to_string()),
            "jpg" | "jpeg" => Some("image/jpeg".to_string()),
            "webp" => Some("image/webp".to_string()),
            "bmp" => Some("image/bmp".to_string()),
            "gif" => Some("image/gif".to_string()),
            _ => None,
        })
}

fn path_to_string(path: &PathBuf) -> String {
    path.display().to_string()
}

struct StdoutLogCallback;

impl AndroidLogCallback for StdoutLogCallback {
    fn on_log(&self, level: AndroidLogLevel, message: String) {
        println!("[AndroidCLI][{level:?}] {message}");
    }
}
