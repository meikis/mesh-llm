use std::{fs, time::Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::{
    FlashAttentionType, RuntimeConfig, RuntimeLoadMode, StageModel, parse_cache_type,
};

use crate::cli::BranchBatchLocalArgs;

#[derive(Debug, Serialize)]
struct BranchBatchLocalReport {
    model_path: String,
    prompt_tokens: usize,
    branch_tokens: Vec<i32>,
    serial_predictions: Vec<i32>,
    branch_predictions: Vec<i32>,
    branch_parity: bool,
    commit_next_serial: i32,
    commit_next_branch: i32,
    commit_parity: bool,
    serial_eval_us: u128,
    branch_eval_us: u128,
    raw_speedup: f64,
}

pub fn branch_batch_local(args: BranchBatchLocalArgs) -> Result<()> {
    let serial_model = StageModel::open(&args.model_path, &runtime_config(&args, 2, 0)?)
        .with_context(|| format!("failed to open serial model {}", args.model_path.display()))?;
    let branch_model = StageModel::open(&args.model_path, &runtime_config(&args, 1, 2)?)
        .with_context(|| format!("failed to open branch model {}", args.model_path.display()))?;
    let prompt_tokens = serial_model
        .tokenize(&args.prompt, true)
        .context("failed to tokenize branch parity prompt")?;
    if prompt_tokens.is_empty() {
        bail!("branch parity prompt produced no tokens");
    }

    let mut serial_a = serial_model
        .create_session()
        .context("create serial branch A")?;
    let mut serial_b = serial_model
        .create_session()
        .context("create serial branch B")?;
    let mut branch = branch_model
        .create_session()
        .context("create branch session")?;
    let alternate_seed = prompt_tokens
        .iter()
        .copied()
        .find(|token| *token != prompt_tokens[prompt_tokens.len() - 1])
        .context("prompt did not contain an alternate valid token")?;
    warm_execution_shapes(
        &mut serial_a,
        &mut serial_b,
        &mut branch,
        &prompt_tokens,
        alternate_seed,
    )?;
    serial_a.reset().context("reset warmed serial A")?;
    serial_b.reset().context("reset warmed serial B")?;
    branch.reset().context("reset warmed branch session")?;

    let common = prefill_to_prediction(&mut serial_a, &prompt_tokens, "serial A")?;
    let common_b = prefill_to_prediction(&mut serial_b, &prompt_tokens, "serial B")?;
    let common_branch = prefill_to_prediction(&mut branch, &prompt_tokens, "branch session")?;
    if common != common_b {
        bail!("serial sessions disagree before branch evaluation: {common} != {common_b}");
    }
    if common != common_branch {
        bail!("branch session disagrees before branch evaluation: {common} != {common_branch}");
    }

    let serial_start = Instant::now();
    let second_a = serial_a
        .decode_step(common)
        .context("serial A common token")?;
    let second_b_seed = serial_b
        .decode_step(common)
        .context("serial B common token")?;
    if second_a != second_b_seed {
        bail!("serial sessions disagree after shared token: {second_a} != {second_b_seed}");
    }
    let alternate = prompt_tokens
        .iter()
        .copied()
        .find(|token| *token != second_a)
        .context("prompt did not contain an alternate valid token")?;
    let third_a = serial_a
        .decode_step(second_a)
        .context("serial A second token")?;
    let third_b = serial_b
        .decode_step(alternate)
        .context("serial B alternate token")?;
    let serial_eval_us = serial_start.elapsed().as_micros();
    let serial_predictions = vec![second_a, third_a, third_b];

    // Row 0 is shared by both branches. Rows 1 and 2 diverge at the same
    // logical position, exactly matching llama.cpp's speculative tree shape.
    let branch_tokens = vec![common, second_a, alternate];
    let position_offsets = [0_u32, 1, 1];
    let sequence_offsets = [0_u32, 2, 3, 4];
    let sequence_ids = [0_u32, 1, 0, 1];
    let branch_start = Instant::now();
    let (branch_predictions, output) = branch
        .verify_branch_batch_frame_sampled(
            &branch_tokens,
            &position_offsets,
            &sequence_offsets,
            &sequence_ids,
            2,
            None,
            None,
            0,
        )
        .context("evaluate shared-prefix branch batch")?;
    let branch_eval_us = branch_start.elapsed().as_micros();
    if !output.payload.is_empty() {
        bail!("full-model branch evaluation unexpectedly returned activations");
    }
    let branch_parity = branch_predictions == serial_predictions;
    if !branch_parity {
        bail!(
            "branch predictions differ from serial target runs: branch={branch_predictions:?} serial={serial_predictions:?}"
        );
    }

    branch
        .commit_branch_batch(0, &[common, second_a])
        .context("commit branch A")?;
    let commit_next_branch = branch
        .decode_step(third_a)
        .context("decode after committed branch")?;
    let commit_next_serial = serial_a
        .decode_step(third_a)
        .context("decode serial commit reference")?;
    let commit_parity = commit_next_branch == commit_next_serial;
    if !commit_parity {
        bail!(
            "committed branch KV differs from serial target: branch={commit_next_branch} serial={commit_next_serial}"
        );
    }

    let report = BranchBatchLocalReport {
        model_path: args.model_path.display().to_string(),
        prompt_tokens: prompt_tokens.len(),
        branch_tokens,
        serial_predictions,
        branch_predictions,
        branch_parity,
        commit_next_serial,
        commit_next_branch,
        commit_parity,
        serial_eval_us,
        branch_eval_us,
        raw_speedup: serial_eval_us as f64 / branch_eval_us.max(1) as f64,
    };
    let json = serde_json::to_string_pretty(&report).context("serialize branch report")?;
    if let Some(output) = args.output {
        fs::write(&output, format!("{json}\n"))
            .with_context(|| format!("write {}", output.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn runtime_config(
    args: &BranchBatchLocalArgs,
    lane_count: u32,
    branch_sequence_capacity: u32,
) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count,
        branch_sequence_capacity,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        mmap: None,
        mlock: false,
        selected_backend_device: None,
        cache_type_k: parse_cache_type("f16")?,
        cache_type_v: parse_cache_type("f16")?,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        use_mmap: true,
        use_mmap_prefetch: true,
        use_mmap_buffer: true,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
        glm_dsa_policy: None,
    })
}

fn prefill_to_prediction(
    session: &mut skippy_runtime::StageSession,
    prompt_tokens: &[i32],
    label: &str,
) -> Result<i32> {
    let (last, prefix) = prompt_tokens
        .split_last()
        .context("branch parity prompt produced no tokens")?;
    session
        .prefill_chunked(prefix)
        .with_context(|| format!("prefill {label}"))?;
    session
        .decode_step(*last)
        .with_context(|| format!("decode final prompt token for {label}"))
}

fn warm_execution_shapes(
    serial_a: &mut skippy_runtime::StageSession,
    serial_b: &mut skippy_runtime::StageSession,
    branch: &mut skippy_runtime::StageSession,
    prompt_tokens: &[i32],
    alternate: i32,
) -> Result<()> {
    let common_a = prefill_to_prediction(serial_a, prompt_tokens, "warm serial A")?;
    let common_b = prefill_to_prediction(serial_b, prompt_tokens, "warm serial B")?;
    let common_branch = prefill_to_prediction(branch, prompt_tokens, "warm branch")?;
    if common_a != common_b || common_a != common_branch {
        bail!("warmup sessions disagree before branch evaluation");
    }
    let second_a = serial_a
        .decode_step(common_a)
        .context("warm serial A common token")?;
    let second_b = serial_b
        .decode_step(common_b)
        .context("warm serial B common token")?;
    if second_a != second_b {
        bail!("warmup serial sessions disagree after common token");
    }
    let _ = serial_a
        .decode_step(second_a)
        .context("warm serial A second token")?;
    let _ = serial_b
        .decode_step(alternate)
        .context("warm serial B alternate token")?;
    let tokens = [common_branch, second_a, alternate];
    let positions = [0_u32, 1, 1];
    let offsets = [0_u32, 2, 3, 4];
    let sequence_ids = [0_u32, 1, 0, 1];
    let _ = branch
        .verify_branch_batch_frame_sampled(
            &tokens,
            &positions,
            &offsets,
            &sequence_ids,
            2,
            None,
            None,
            0,
        )
        .context("warm branch batch")?;
    branch
        .discard_branch_batch()
        .context("discard warm branch batch")
}
