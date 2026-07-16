//! MLX generation engine backed by a dedicated OS worker thread.
//!
//! MLX arrays, streams, and the loaded model wrap raw C pointers and are neither
//! `Send` nor `Sync`. Rather than fight that, we confine every MLX object to a
//! single worker thread that owns them for its whole life, and talk to it only
//! with `Send` messages:
//!
//! - a `Send + Sync` job channel (tokio unbounded) carries generation requests;
//! - each job carries a per-request token channel the worker streams results on.
//!
//! This also naturally serializes GPU access (one generation at a time), which
//! matches how goose drives safemlx today.

use std::path::PathBuf;
use std::thread;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokio::sync::mpsc;

use safemlx::transforms::async_eval;
use safemlx::{Device, DeviceType, Stream};
use safemlx_lm::models::input::{InputPart, ModelInput};
use safemlx_lm::models::{LoadedModel, ModelLoadOptions};
use safemlx_lm::quantization::AffineQuantization;
use safemlx_lm::sampler::DefaultSampler;

/// How the worker should load and run a model.
#[derive(Clone, Debug)]
pub struct MlxEngineConfig {
    pub model_dir: PathBuf,
    pub model_id: String,
    /// JIT-quantize eligible dense weights to this bit width on load (Metal only).
    pub quantize_bits: Option<i32>,
    pub quant_group_size: i32,
    pub default_max_tokens: usize,
    pub max_tokens_cap: usize,
}

/// One chat turn, in `Send` form (no MLX types).
#[derive(Clone, Debug)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// A generation request handed to the worker.
#[derive(Debug)]
pub struct GenerateRequest {
    pub messages: Vec<ChatTurn>,
    /// If set, skip the chat template and feed this text verbatim.
    pub raw_prompt: Option<String>,
    pub max_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// Streamed output from the worker for one request.
#[derive(Debug)]
pub enum TokenMsg {
    Delta(String),
    Done {
        finish_reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Error(String),
}

struct Job {
    req: GenerateRequest,
    reply: mpsc::UnboundedSender<TokenMsg>,
}

/// Handle to the MLX worker thread. `Send + Sync`, safe to share in an `Arc`.
pub struct MlxEngine {
    job_tx: mpsc::UnboundedSender<Job>,
    config: MlxEngineConfig,
}

impl MlxEngine {
    /// Spawns the worker and blocks until the model has finished loading.
    /// Call from a blocking context (e.g. `tokio::task::spawn_blocking`).
    pub fn spawn(config: MlxEngineConfig) -> Result<Self> {
        let (job_tx, job_rx) = mpsc::unbounded_channel::<Job>();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

        let worker_config = config.clone();
        thread::Builder::new()
            .name("mlx-engine".into())
            .spawn(move || run_worker(worker_config, job_rx, ready_tx))?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { job_tx, config }),
            Ok(Err(e)) => Err(anyhow!("MLX model load failed: {e}")),
            Err(_) => Err(anyhow!("MLX worker exited before signalling readiness")),
        }
    }

    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    pub fn clamp_max_tokens(&self, requested: Option<usize>) -> usize {
        requested
            .unwrap_or(self.config.default_max_tokens)
            .clamp(1, self.config.max_tokens_cap)
    }

    /// Submits a request and returns the channel its tokens will stream on.
    pub fn submit(&self, req: GenerateRequest) -> mpsc::UnboundedReceiver<TokenMsg> {
        let (tx, rx) = mpsc::unbounded_channel();
        if self
            .job_tx
            .send(Job {
                req,
                reply: tx.clone(),
            })
            .is_err()
        {
            let _ = tx.send(TokenMsg::Error("MLX worker is not running".into()));
        }
        rx
    }
}

struct LoadedEngine {
    model: LoadedModel,
    stream: Stream,
    tokenizer: tokenizers::Tokenizer,
    eos: Vec<u32>,
}

fn load_engine(config: &MlxEngineConfig) -> Result<LoadedEngine> {
    // Metal GPU stream for compute, CPU stream for weight staging (goose's split).
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));

    let options = match config.quantize_bits {
        Some(bits) => ModelLoadOptions::with_quantization(AffineQuantization::new(
            config.quant_group_size,
            bits,
        )?),
        None => ModelLoadOptions::default(),
    };

    let started = Instant::now();
    let model =
        LoadedModel::load_with_options(&config.model_dir, options, &stream, &weights_stream)
            .map_err(|e| anyhow!("load {}: {e}", config.model_dir.display()))?;
    stream.synchronize().map_err(|e| anyhow!("sync: {e}"))?;

    let tokenizer = tokenizers::Tokenizer::from_file(config.model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer.json: {e}"))?;
    let eos = model.eos_token_ids().to_vec();

    tracing::info!(
        model = %config.model_id,
        kind = model.model_type(),
        load_secs = started.elapsed().as_secs_f64(),
        "MLX model loaded"
    );
    Ok(LoadedEngine {
        model,
        stream,
        tokenizer,
        eos,
    })
}

fn run_worker(
    config: MlxEngineConfig,
    mut job_rx: mpsc::UnboundedReceiver<Job>,
    ready_tx: std::sync::mpsc::Sender<Result<(), String>>,
) {
    let mut engine = match load_engine(&config) {
        Ok(engine) => {
            let _ = ready_tx.send(Ok(()));
            engine
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };

    while let Some(job) = job_rx.blocking_recv() {
        let reply = job.reply.clone();
        if let Err(e) = generate_one(&mut engine, job) {
            let _ = reply.send(TokenMsg::Error(e.to_string()));
        }
    }
}

fn build_prompt(model: &mut LoadedModel, req: &GenerateRequest) -> Result<(String, bool)> {
    if let Some(raw) = &req.raw_prompt {
        return Ok((raw.clone(), true));
    }
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|turn| json!({"role": turn.role, "content": turn.content}))
        .collect();
    let rendered = model
        .apply_chat_template_json(vec![messages], None, true)
        .map_err(|e| anyhow!("chat template: {e}"))?;
    match rendered {
        Some(prompt) => Ok((prompt, false)),
        None => {
            let fallback = req
                .messages
                .last()
                .map(|turn| turn.content.clone())
                .unwrap_or_default();
            Ok((fallback, true))
        }
    }
}

fn generate_one(engine: &mut LoadedEngine, job: Job) -> Result<()> {
    let LoadedEngine {
        model,
        stream,
        tokenizer,
        eos,
    } = engine;
    let reply = job.reply;

    let (prompt, add_special) = build_prompt(model, &job.req)?;
    let tokens = model
        .encode_to_array(&prompt, add_special, stream)
        .map_err(|e| anyhow!("encode: {e}"))?;
    let prompt_tokens = tokens.shape()[1] as u32;

    let mut cache = model.new_cache();
    let parts = [InputPart::text_token_ids(&tokens)];
    let input = ModelInput::new(&parts);
    let mut generator = model.generate_input_with_cache_sampler(
        &mut cache,
        0.0,
        input,
        None,
        stream,
        DefaultSampler,
    );

    let mut ids: Vec<u32> = Vec::with_capacity(job.req.max_tokens);
    let mut emitted = String::new();
    let mut finish = FinishReason::Length;

    let mut current = generator.next().transpose().map_err(|e| anyhow!("{e}"))?;
    for index in 0..job.req.max_tokens {
        let Some(token) = current.take() else {
            finish = FinishReason::Stop;
            break;
        };

        // Start the next decode before reading this token back (mlx-lm's
        // one-token async pipeline overlaps compute with host readback).
        let next = if index + 1 < job.req.max_tokens {
            let next = generator.next();
            if let Some(Ok(next_token)) = next.as_ref() {
                async_eval([next_token]).map_err(|e| anyhow!("async_eval: {e}"))?;
            }
            next
        } else {
            None
        };

        let token_id = token.item::<u32>(&*stream);
        if eos.contains(&token_id) {
            finish = FinishReason::Stop;
            break;
        }
        ids.push(token_id);

        // Incremental detokenization: decode the whole id sequence and emit only
        // the newly appended suffix. Robust for byte-level BPE where one char can
        // span multiple tokens.
        let full = tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow!("decode: {e}"))?;
        if let Some(delta) = full.strip_prefix(emitted.as_str()) {
            if !delta.is_empty() {
                if reply.send(TokenMsg::Delta(delta.to_string())).is_err() {
                    return Ok(()); // client hung up
                }
                emitted = full;
            }
        } else {
            emitted = full; // rare re-render; resync silently
        }

        if reply.is_closed() {
            return Ok(());
        }
        current = next.transpose().map_err(|e| anyhow!("{e}"))?;
    }

    let _ = reply.send(TokenMsg::Done {
        finish_reason: finish,
        prompt_tokens,
        completion_tokens: ids.len() as u32,
    });
    Ok(())
}
