use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::{
    FlashAttentionType, RuntimeConfig, RuntimeLoadMode, SamplingConfig, StageModel, StageSession,
    parse_cache_type,
};

use crate::cli::VerifySpanLocalArgs;

#[derive(Debug, Serialize)]
struct TimingStats {
    count: usize,
    total_us: u128,
    avg_us: f64,
    min_us: u128,
    p50_us: u128,
    p95_us: u128,
    max_us: u128,
}

#[derive(Debug, Serialize)]
struct VerifySpanLocalReport {
    mode: &'static str,
    model_path: PathBuf,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    cache_type_k: String,
    cache_type_v: String,
    prompt_token_count: usize,
    verify_tokens: Vec<i32>,
    warmup: usize,
    iterations: usize,
    batched_width2: TimingStats,
    serial_two_decode_mtp_n1: TimingStats,
    batched_avg_vs_serial_avg: f64,
    batched_token_per_sec: f64,
    serial_token_per_sec: f64,
    first_batched_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

pub fn verify_span_local(args: VerifySpanLocalArgs) -> Result<()> {
    validate_args(&args)?;
    let output = args.output.clone();
    let config = runtime_config(&args)?;
    let model = StageModel::open(&args.model_path, &config)
        .with_context(|| format!("failed to open {}", args.model_path.display()))?;
    let tokens = model
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    if tokens.is_empty() {
        bail!("prompt produced no tokens");
    }

    let mut session = model.create_session().context("failed to create session")?;
    session
        .prefill_chunked(&tokens)
        .context("failed to prefill prompt")?;
    let base_token_count = session.token_count();
    let verify_tokens = choose_verify_tokens(
        &mut session,
        base_token_count,
        &tokens,
        &args.prompt,
        &config,
    )?;

    let samples = run_samples(
        &mut session,
        base_token_count,
        &verify_tokens,
        args.warmup,
        args.iterations,
    )?;
    let report = build_report(args, tokens.len(), verify_tokens, samples)?;
    let encoded = serde_json::to_vec_pretty(&report)?;

    if let Some(path) = output {
        fs::write(&path, &encoded)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn validate_args(args: &VerifySpanLocalArgs) -> Result<()> {
    if args.layer_end == 0 {
        bail!("layer_end must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("iterations must be greater than zero");
    }
    Ok(())
}

fn runtime_config(args: &VerifySpanLocalArgs) -> Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        lane_count: 1,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers: args.n_gpu_layers,
        selected_backend_device: None,
        cache_type_k: parse_cache_type(&args.cache_type_k)?,
        cache_type_v: parse_cache_type(&args.cache_type_v)?,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    })
}

fn choose_verify_tokens(
    session: &mut StageSession,
    base_token_count: u64,
    prompt_tokens: &[i32],
    prompt: &str,
    config: &RuntimeConfig,
) -> Result<Vec<i32>> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before choosing verify tokens")?;
    let current = *prompt_tokens
        .first()
        .context("prompt produced no token for verify-token seed")?;
    let (_after_current, native_mtp, _frame) = session
        .decode_step_frame_sampled_mtp_n1(current, Some(&SamplingConfig::default()), None, 0)
        .with_context(|| {
            format!(
                "failed to get native MTP draft from {} after prompt {:?}",
                model_description(config),
                prompt
            )
        })?;
    let Some(draft) = native_mtp else {
        bail!("model did not produce a native MTP n=1 draft token");
    };
    session
        .trim_session(base_token_count)
        .context("failed to trim session after choosing verify tokens")?;
    Ok(vec![current, draft.token_id])
}

fn run_samples(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    warmup: usize,
    iterations: usize,
) -> Result<SampleSet> {
    let total = warmup
        .checked_add(iterations)
        .context("sample count overflow")?;
    let mut batched = Vec::with_capacity(iterations);
    let mut serial = Vec::with_capacity(iterations);
    let mut first_batched_prediction = None;
    let mut first_serial_prediction = None;

    {
        let mut targets = SampleRecordTargets {
            batched: &mut batched,
            serial: &mut serial,
            first_batched_prediction: &mut first_batched_prediction,
            first_serial_prediction: &mut first_serial_prediction,
        };
        for index in 0..total {
            let record = index >= warmup;
            if index.is_multiple_of(2) {
                measure_batched_then_serial(
                    session,
                    base_token_count,
                    verify_tokens,
                    record,
                    &mut targets,
                )?;
            } else {
                measure_serial_then_batched(
                    session,
                    base_token_count,
                    verify_tokens,
                    record,
                    &mut targets,
                )?;
            }
        }
    }

    Ok(SampleSet {
        batched,
        serial,
        first_batched_prediction: first_batched_prediction.unwrap_or_default(),
        first_serial_prediction: first_serial_prediction.unwrap_or_default(),
    })
}

fn measure_batched_then_serial(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    record: bool,
    targets: &mut SampleRecordTargets<'_>,
) -> Result<()> {
    let (batched_duration, batched_prediction) =
        measure_batched(session, base_token_count, verify_tokens)?;
    let (serial_duration, serial_prediction) =
        measure_serial(session, base_token_count, verify_tokens)?;
    record_sample(
        record,
        (batched_duration, batched_prediction),
        (serial_duration, serial_prediction),
        targets,
    );
    Ok(())
}

fn measure_serial_then_batched(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
    record: bool,
    targets: &mut SampleRecordTargets<'_>,
) -> Result<()> {
    let (serial_duration, serial_prediction) =
        measure_serial(session, base_token_count, verify_tokens)?;
    let (batched_duration, batched_prediction) =
        measure_batched(session, base_token_count, verify_tokens)?;
    record_sample(
        record,
        (batched_duration, batched_prediction),
        (serial_duration, serial_prediction),
        targets,
    );
    Ok(())
}

struct SampleRecordTargets<'a> {
    batched: &'a mut Vec<Duration>,
    serial: &'a mut Vec<Duration>,
    first_batched_prediction: &'a mut Option<Vec<i32>>,
    first_serial_prediction: &'a mut Option<Vec<i32>>,
}

fn record_sample(
    record: bool,
    batched_sample: (Duration, Vec<i32>),
    serial_sample: (Duration, Vec<i32>),
    targets: &mut SampleRecordTargets<'_>,
) {
    if !record {
        return;
    }
    targets.batched.push(batched_sample.0);
    targets.serial.push(serial_sample.0);
    targets
        .first_batched_prediction
        .get_or_insert(batched_sample.1);
    targets
        .first_serial_prediction
        .get_or_insert(serial_sample.1);
}

fn measure_batched(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
) -> Result<(Duration, Vec<i32>)> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before batched verify")?;
    let start = Instant::now();
    let prediction = session
        .verify_tokens_frame_sampled(verify_tokens, Some(&SamplingConfig::default()), None, 0)
        .context("batched width-2 VerifySpan failed")?
        .0;
    Ok((start.elapsed(), prediction))
}

fn measure_serial(
    session: &mut StageSession,
    base_token_count: u64,
    verify_tokens: &[i32],
) -> Result<(Duration, Vec<i32>)> {
    session
        .trim_session(base_token_count)
        .context("failed to trim session before serial verify")?;
    let start = Instant::now();
    let prediction = serial_decode_mtp_n1(session, verify_tokens)?;
    Ok((start.elapsed(), prediction))
}

fn serial_decode_mtp_n1(session: &mut StageSession, verify_tokens: &[i32]) -> Result<Vec<i32>> {
    let mut predicted_tokens = Vec::with_capacity(verify_tokens.len() + 3);
    let mut last_draft = None;
    for token_id in verify_tokens {
        let (predicted, native_mtp, _frame) = session
            .decode_step_frame_sampled_mtp_n1(*token_id, Some(&SamplingConfig::default()), None, 0)
            .context("serial native MTP n=1 decode failed")?;
        if predicted >= 0 {
            predicted_tokens.push(predicted);
        }
        last_draft = native_mtp;
    }
    if let Some(draft) = last_draft {
        predicted_tokens.push(draft.token_id);
        predicted_tokens.push(i32::try_from(draft.proposal_compute_us.max(0)).unwrap_or(i32::MAX));
        predicted_tokens.push(draft.margin_milli);
    }
    Ok(predicted_tokens)
}

#[derive(Debug)]
struct SampleSet {
    batched: Vec<Duration>,
    serial: Vec<Duration>,
    first_batched_prediction: Vec<i32>,
    first_serial_prediction: Vec<i32>,
}

fn build_report(
    args: VerifySpanLocalArgs,
    prompt_token_count: usize,
    verify_tokens: Vec<i32>,
    samples: SampleSet,
) -> Result<VerifySpanLocalReport> {
    let batched_width2 = timing_stats(&samples.batched)?;
    let serial_two_decode_mtp_n1 = timing_stats(&samples.serial)?;
    let batched_avg = batched_width2.avg_us;
    let serial_avg = serial_two_decode_mtp_n1.avg_us;
    Ok(VerifySpanLocalReport {
        mode: "verify-span-local",
        model_path: args.model_path,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        n_batch: args.n_batch,
        n_ubatch: args.n_ubatch,
        cache_type_k: args.cache_type_k,
        cache_type_v: args.cache_type_v,
        prompt_token_count,
        verify_tokens,
        warmup: args.warmup,
        iterations: args.iterations,
        batched_width2,
        serial_two_decode_mtp_n1,
        batched_avg_vs_serial_avg: batched_avg / serial_avg,
        batched_token_per_sec: verified_tokens_per_sec(batched_avg, 2),
        serial_token_per_sec: verified_tokens_per_sec(serial_avg, 2),
        first_batched_prediction: samples.first_batched_prediction,
        first_serial_prediction: samples.first_serial_prediction,
    })
}

fn timing_stats(samples: &[Duration]) -> Result<TimingStats> {
    if samples.is_empty() {
        bail!("cannot summarize empty timing samples");
    }
    let mut micros = samples.iter().map(Duration::as_micros).collect::<Vec<_>>();
    micros.sort_unstable();
    let total_us = micros.iter().sum::<u128>();
    let avg_us = total_us as f64 / micros.len() as f64;
    Ok(TimingStats {
        count: micros.len(),
        total_us,
        avg_us,
        min_us: micros[0],
        p50_us: percentile(&micros, 0.50),
        p95_us: percentile(&micros, 0.95),
        max_us: *micros.last().context("missing max timing sample")?,
    })
}

fn percentile(sorted_micros: &[u128], percentile: f64) -> u128 {
    let last_index = sorted_micros.len().saturating_sub(1);
    let index = (last_index as f64 * percentile).round() as usize;
    sorted_micros[index.min(last_index)]
}

fn verified_tokens_per_sec(avg_us: f64, token_count: usize) -> f64 {
    token_count as f64 * 1_000_000.0 / avg_us
}

fn model_description(config: &RuntimeConfig) -> String {
    format!(
        "layers={}..{} ctx={} n_gpu_layers={}",
        config.layer_start, config.layer_end, config.ctx_size, config.n_gpu_layers
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{percentile, timing_stats, verified_tokens_per_sec};

    #[test]
    fn timing_stats_sorts_and_summarizes_microseconds() {
        let stats = timing_stats(&[
            Duration::from_micros(30),
            Duration::from_micros(10),
            Duration::from_micros(20),
        ])
        .unwrap();

        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_us, 60);
        assert_eq!(stats.min_us, 10);
        assert_eq!(stats.p50_us, 20);
        assert_eq!(stats.max_us, 30);
    }

    #[test]
    fn percentile_clamps_to_last_sample() {
        assert_eq!(percentile(&[10, 20, 30], 1.0), 30);
    }

    #[test]
    fn token_rate_uses_verified_token_count() {
        assert_eq!(verified_tokens_per_sec(20_000.0, 2), 100.0);
    }
}
