use std::{fs, time::Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::{
    FlashAttentionType, RuntimeConfig, RuntimeLoadMode, StageModel, StageSession,
    lookahead::{LookaheadConfig, LookaheadState},
    parse_cache_type,
};

use crate::cli::LookaheadLocalArgs;

#[derive(Debug, Default, Serialize)]
struct LookaheadRunStats {
    target_forwards: usize,
    branch_rows: usize,
    candidate_windows: usize,
    accepted_candidate_tokens: usize,
    emitted_tokens: usize,
    useful_tokens_per_target_forward: f64,
    mean_branch_rows_per_target_forward: f64,
}

#[derive(Debug, Serialize)]
struct LookaheadLocalReport {
    model_path: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    output_parity: bool,
    serial_tokens: Vec<i32>,
    lookahead_tokens: Vec<i32>,
    serial_us: u128,
    lookahead_us: u128,
    raw_speedup: f64,
    stats: LookaheadRunStats,
}

pub fn lookahead_local(args: LookaheadLocalArgs) -> Result<()> {
    let lookahead_config = LookaheadConfig {
        ngram_size: args.ngram_size,
        window_size: args.window_size,
        max_candidates: args.max_candidates,
        candidates_per_token: args.max_candidates.max(1) * 2,
        jacobi_on_miss: args.jacobi_on_miss,
    }
    .validate()?;
    let jacobi_capacity = usize::from(args.jacobi_on_miss).saturating_mul(args.window_size);
    let branch_capacity = u32::try_from(
        args.max_candidates
            .saturating_add(jacobi_capacity)
            .saturating_add(1),
    )
    .context("lookahead branch capacity exceeds u32")?;
    let candidate_rows = args
        .max_candidates
        .saturating_mul(args.window_size.saturating_sub(1));
    let lookahead_rows = jacobi_capacity.saturating_mul(args.ngram_size.saturating_sub(1));
    let branch_batch_size = u32::try_from(candidate_rows.saturating_add(lookahead_rows).max(64))
        .context("lookahead branch batch size exceeds u32")?;
    let serial_model = StageModel::open(&args.model_path, &runtime_config(&args, 0, 64)?)
        .with_context(|| format!("open serial model {}", args.model_path.display()))?;
    let lookahead_model = StageModel::open(
        &args.model_path,
        &runtime_config(&args, branch_capacity, branch_batch_size)?,
    )
    .with_context(|| format!("open lookahead model {}", args.model_path.display()))?;
    let prompt_tokens = serial_model
        .tokenize(&args.prompt, true)
        .context("tokenize lookahead prompt")?;
    if prompt_tokens.is_empty() {
        bail!("lookahead prompt produced no tokens");
    }

    let mut serial = serial_model
        .create_session()
        .context("create serial session")?;
    let mut lookahead = lookahead_model
        .create_session()
        .context("create lookahead session")?;
    let warm_tokens = args.max_tokens.clamp(1, 16);
    let _ = run_serial(&mut serial, &prompt_tokens, warm_tokens)?;
    let _ = run_lookahead(
        &mut lookahead,
        &prompt_tokens,
        warm_tokens,
        lookahead_config,
    )?;
    serial.reset().context("reset warmed serial session")?;
    lookahead
        .reset()
        .context("reset warmed lookahead session")?;

    let serial_started = Instant::now();
    let serial_tokens = run_serial(&mut serial, &prompt_tokens, args.max_tokens)?;
    let serial_us = serial_started.elapsed().as_micros();
    let lookahead_started = Instant::now();
    let (lookahead_tokens, stats) = run_lookahead(
        &mut lookahead,
        &prompt_tokens,
        args.max_tokens,
        lookahead_config,
    )?;
    let lookahead_us = lookahead_started.elapsed().as_micros();
    let output_parity = lookahead_tokens == serial_tokens;
    let report = LookaheadLocalReport {
        model_path: args.model_path.display().to_string(),
        prompt_tokens: prompt_tokens.len(),
        completion_tokens: args.max_tokens,
        output_parity,
        serial_tokens,
        lookahead_tokens,
        serial_us,
        lookahead_us,
        raw_speedup: serial_us as f64 / lookahead_us.max(1) as f64,
        stats,
    };
    let json = serde_json::to_string_pretty(&report).context("serialize lookahead report")?;
    if let Some(path) = args.output.as_deref() {
        fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write {}", path.display()))?;
    }
    println!("{json}");
    if !output_parity {
        bail!("lookahead output differs from serial greedy generation");
    }
    Ok(())
}

fn run_serial(
    session: &mut StageSession,
    prompt_tokens: &[i32],
    max_tokens: usize,
) -> Result<Vec<i32>> {
    let (last, prefix) = prompt_tokens
        .split_last()
        .context("serial prompt produced no tokens")?;
    session
        .prefill_chunked(prefix)
        .context("prefill serial prompt")?;
    let mut current = *last;
    let mut output = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        current = session
            .decode_step(current)
            .context("serial target decode")?;
        output.push(current);
    }
    Ok(output)
}

fn run_lookahead(
    session: &mut StageSession,
    prompt_tokens: &[i32],
    max_tokens: usize,
    config: LookaheadConfig,
) -> Result<(Vec<i32>, LookaheadRunStats)> {
    let (last, prefix) = prompt_tokens
        .split_last()
        .context("lookahead prompt produced no tokens")?;
    session
        .prefill_chunked(prefix)
        .context("prefill lookahead prompt")?;
    let mut state = LookaheadState::new(config, prompt_tokens)?;
    let mut current = *last;
    let mut output = Vec::with_capacity(max_tokens);
    let mut stats = LookaheadRunStats::default();
    while output.len() < max_tokens {
        let remaining = max_tokens - output.len();
        let plan = state.plan(current, remaining)?;
        if plan.candidate_count() == 0 && plan.lookahead_count() == 0 {
            let predicted = session
                .decode_step(current)
                .context("execute target-only serial fallback")?;
            state.observe_serial(current, predicted)?;
            stats.target_forwards += 1;
            stats.branch_rows += 1;
            output.push(predicted);
            current = predicted;
            continue;
        }
        let (predictions, frame) = session
            .verify_branch_batch_frame_sampled(
                &plan.token_ids,
                &plan.position_offsets,
                &plan.sequence_offsets,
                &plan.sequence_ids,
                plan.sequence_count,
                None,
                None,
                0,
            )
            .context("execute target-only lookahead branch batch")?;
        if !frame.payload.is_empty() {
            bail!("full-model lookahead unexpectedly returned activations");
        }
        let decision = state.observe(&plan, &predictions)?;
        session
            .commit_branch_batch(decision.sequence_id, &decision.commit_input_tokens)
            .context("commit selected lookahead branch")?;
        if decision.emitted_target_tokens.is_empty() {
            bail!("lookahead target forward emitted no tokens");
        }
        stats.target_forwards += 1;
        stats.branch_rows += decision.branch_rows;
        stats.candidate_windows += usize::from(decision.candidate_count > 0);
        stats.accepted_candidate_tokens += decision.accepted_candidate_tokens;
        output.extend_from_slice(&decision.emitted_target_tokens);
        current = *output
            .last()
            .context("lookahead emitted no current token")?;
    }
    output.truncate(max_tokens);
    stats.emitted_tokens = output.len();
    stats.useful_tokens_per_target_forward =
        output.len() as f64 / stats.target_forwards.max(1) as f64;
    stats.mean_branch_rows_per_target_forward =
        stats.branch_rows as f64 / stats.target_forwards.max(1) as f64;
    Ok((output, stats))
}

fn runtime_config(
    args: &LookaheadLocalArgs,
    branch_sequence_capacity: u32,
    branch_batch_size: u32,
) -> Result<RuntimeConfig> {
    let required_sequences = 2_u32
        .checked_add(branch_sequence_capacity)
        .context("lookahead sequence reservation overflow")?;
    let batch_size = branch_batch_size.max(required_sequences);
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        branch_sequence_capacity,
        n_batch: Some(batch_size),
        n_ubatch: Some(batch_size),
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
