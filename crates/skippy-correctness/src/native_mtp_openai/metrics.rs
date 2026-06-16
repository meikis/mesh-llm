use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde_json::Value;

use crate::report::NativeMtpOpenAiMetricsReport;

pub(super) fn read_metrics(
    stage0_log: &Path,
    stage1_log: &Path,
) -> Result<NativeMtpOpenAiMetricsReport> {
    let mut metrics = NativeMtpOpenAiMetricsReport::default();
    for path in [stage0_log, stage1_log] {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        metrics.fatal_error_events += count_fatal_log_lines(&text);
        for line in text.lines() {
            let Ok(event) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            apply_telemetry_event(&mut metrics, &event);
        }
    }
    Ok(metrics)
}

fn apply_telemetry_event(metrics: &mut NativeMtpOpenAiMetricsReport, event: &Value) {
    let Some(name) = event.get("event").and_then(Value::as_str) else {
        return;
    };
    let attrs = event.get("attributes").unwrap_or(&Value::Null);
    match name {
        "stage.openai_decode_token" => {
            metrics.decode_token_events += 1;
            apply_native_mtp_verification(metrics, attrs);
        }
        "stage.openai_native_mtp_verify" => {
            metrics.batched_verify_events += 1;
            match apply_native_mtp_verification(metrics, attrs) {
                Some("accepted") => metrics.batched_accepted_events += 1,
                Some("rejected") => metrics.batched_rejected_events += 1,
                _ => {}
            }
        }
        "stage.openai_decode" | "stage.openai_generation_summary" => {
            apply_generation_summary(metrics, attrs);
        }
        _ => {}
    }
}

fn apply_generation_summary(metrics: &mut NativeMtpOpenAiMetricsReport, attrs: &Value) {
    metrics.native_mtp_enabled |= attr_bool(attrs, "llama_stage.native_mtp.enabled");
    if let Some(drafted) = attr_u64(attrs, "llama_stage.native_mtp.drafted") {
        metrics.drafted_tokens = drafted;
    }
    if let Some(accepted) = attr_u64(attrs, "llama_stage.native_mtp.accepted") {
        metrics.accepted_tokens = accepted;
    }
    if let Some(rejected) = attr_u64(attrs, "llama_stage.native_mtp.rejected") {
        metrics.rejected_tokens = rejected;
    }
    if let Some(pending) = attr_u64(attrs, "llama_stage.native_mtp.pending") {
        metrics.pending_tokens = pending;
    }
    if let Some(verifications) = attr_u64(attrs, "llama_stage.native_mtp.verifications") {
        metrics.verification_count = verifications;
    }
    if let Some(proposal_compute_us) = attr_i64(attrs, "llama_stage.native_mtp.proposal_compute_us")
    {
        metrics.proposal_compute_us = proposal_compute_us;
    }
    if let Some(verification_compute_us) =
        attr_i64(attrs, "llama_stage.native_mtp.verification_compute_us")
    {
        metrics.verification_compute_us = verification_compute_us;
    }
    if let Some(accept_rate) = attrs
        .get("llama_stage.native_mtp.accept_rate")
        .and_then(Value::as_f64)
    {
        metrics.accept_rate = accept_rate;
    }
}

fn apply_native_mtp_verification<'a>(
    metrics: &mut NativeMtpOpenAiMetricsReport,
    attrs: &'a Value,
) -> Option<&'a str> {
    let verification = attrs
        .get("llama_stage.native_mtp.verification")
        .and_then(Value::as_str)?;
    match verification {
        "accepted" => {
            metrics.native_mtp_enabled = true;
            metrics.accepted_tokens += 1;
            metrics.verification_count += 1;
        }
        "rejected" => {
            metrics.native_mtp_enabled = true;
            metrics.rejected_tokens += 1;
            metrics.verification_count += 1;
        }
        _ => {}
    }
    Some(verification)
}

fn count_fatal_log_lines(text: &str) -> u64 {
    text.lines()
        .filter(|line| {
            line.contains("panicked")
                || line.contains("service_unavailable")
                || line.contains("llama_decode failed for MTP sidecar sync")
        })
        .count() as u64
}

fn attr_bool(attrs: &Value, key: &str) -> bool {
    attrs.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn attr_u64(attrs: &Value, key: &str) -> Option<u64> {
    attrs.get(key).and_then(Value::as_u64)
}

fn attr_i64(attrs: &Value, key: &str) -> Option<i64> {
    attrs.get(key).and_then(Value::as_i64)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn telemetry_parser_extracts_native_mtp_counts() {
        let mut metrics = NativeMtpOpenAiMetricsReport::default();
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_native_mtp_verify",
                "attributes": {
                    "llama_stage.native_mtp.verification": "accepted",
                }
            }),
        );
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_decode_token",
                "attributes": {}
            }),
        );
        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_decode",
                "attributes": {
                    "llama_stage.native_mtp.enabled": true,
                    "llama_stage.native_mtp.drafted": 4,
                    "llama_stage.native_mtp.accepted": 3,
                    "llama_stage.native_mtp.rejected": 1,
                    "llama_stage.native_mtp.pending": 0,
                    "llama_stage.native_mtp.verifications": 4,
                    "llama_stage.native_mtp.accept_rate": 0.75,
                    "llama_stage.native_mtp.proposal_compute_us": 11,
                    "llama_stage.native_mtp.verification_compute_us": 22,
                }
            }),
        );

        assert!(metrics.native_mtp_enabled);
        assert_eq!(metrics.drafted_tokens, 4);
        assert_eq!(metrics.accepted_tokens, 3);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.verification_count, 4);
        assert_eq!(metrics.accept_rate, 0.75);
        assert_eq!(metrics.proposal_compute_us, 11);
        assert_eq!(metrics.verification_compute_us, 22);
        assert_eq!(metrics.batched_verify_events, 1);
        assert_eq!(metrics.batched_accepted_events, 1);
        assert_eq!(metrics.decode_token_events, 1);
    }

    #[test]
    fn telemetry_parser_counts_batched_rejections() {
        let mut metrics = NativeMtpOpenAiMetricsReport::default();

        apply_telemetry_event(
            &mut metrics,
            &json!({
                "event": "stage.openai_native_mtp_verify",
                "attributes": {
                    "llama_stage.native_mtp.verification": "rejected",
                }
            }),
        );

        assert!(metrics.native_mtp_enabled);
        assert_eq!(metrics.rejected_tokens, 1);
        assert_eq!(metrics.verification_count, 1);
        assert_eq!(metrics.batched_verify_events, 1);
        assert_eq!(metrics.batched_rejected_events, 1);
    }

    #[test]
    fn fatal_counter_ignores_connection_retry_noise() {
        let text = "\
downstream connect retry: error=Connection refused (os error 61)
llama_decode failed for MTP sidecar sync
";
        assert_eq!(count_fatal_log_lines(text), 1);
    }
}
