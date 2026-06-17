use std::{fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use crate::cli::SpdOpenAiCheckArgs;

pub fn spd_openai_check(args: SpdOpenAiCheckArgs) -> Result<()> {
    let report = read_report(&args.report)?;
    let result = check_report(&report, &args);
    let output_json = serde_json::to_vec_pretty(&result)?;
    if let Some(output) = args.output.as_ref() {
        fs::write(output, &output_json).with_context(|| format!("write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(output_json)?);
    if !result.pass {
        bail!(
            "SPD OpenAI report check failed: {}",
            result.failures.join("; ")
        );
    }
    Ok(())
}

fn read_report(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("parse {}", path.display()))
}

fn check_report(report: &Value, args: &SpdOpenAiCheckArgs) -> CheckResult {
    let mut checker = ReportChecker {
        report,
        summary: report.get("summary").unwrap_or(&Value::Null),
        args,
        failures: Vec::new(),
    };
    checker.run();
    checker.finish()
}

struct ReportChecker<'a> {
    report: &'a Value,
    summary: &'a Value,
    args: &'a SpdOpenAiCheckArgs,
    failures: Vec<String>,
}

impl ReportChecker<'_> {
    fn run(&mut self) {
        self.require_str("mode", self.report.get("mode"), "spd-openai-smoke");
        self.require_u64_at_least(
            "logical_spd_stage_count",
            self.report.get("logical_spd_stage_count"),
            self.args.expected_logical_stage_count,
        );
        if self.args.require_rolling_executor {
            self.require_bool(
                "spd_rolling_executor",
                self.report.get("spd_rolling_executor"),
            );
        }
        self.check_content_match();
        self.require_u64_at_least(
            "summary.spd_spec_accepted",
            self.summary.get("spd_spec_accepted"),
            self.args.min_accepted,
        );
        self.require_u64_at_least(
            "summary.spd_rolling_executor_max_in_flight",
            self.summary.get("spd_rolling_executor_max_in_flight"),
            self.args.min_max_inflight,
        );
        self.require_u64_at_most(
            "summary.spd_rolling_executor_rejected_oldest",
            self.summary.get("spd_rolling_executor_rejected_oldest"),
            self.args.max_rejected_oldest,
        );
        self.require_u64_at_most(
            "summary.spd_rolling_executor_drained_younger",
            self.summary.get("spd_rolling_executor_drained_younger"),
            self.args.max_drained_younger,
        );
        self.require_u64_equals(
            "summary.tap_return_failures",
            self.summary.get("tap_return_failures"),
            0,
        );
        self.require_u64_equals(
            "summary.tap_record_failures",
            self.summary.get("tap_record_failures"),
            0,
        );
        self.check_rolling_trace();
        self.check_spd_decode_ceiling();
    }

    fn check_content_match(&mut self) {
        if !self.args.require_content_match {
            return;
        }
        let prompt_pairs = value_u64(self.summary.get("prompt_pairs"));
        let matching_content = value_u64(self.summary.get("matching_content"));
        match (prompt_pairs, matching_content) {
            (Some(pairs), Some(matches)) if pairs > 0 && matches == pairs => {}
            (Some(pairs), Some(matches)) => self.failures.push(format!(
                "summary.matching_content {matches} does not match prompt_pairs {pairs}"
            )),
            _ => self
                .failures
                .push("summary prompt pair/content fields are missing".to_string()),
        }
    }

    fn check_rolling_trace(&mut self) {
        let trace = self
            .summary
            .get("rolling_trace_replay")
            .unwrap_or(&Value::Null);
        self.require_u64_at_most(
            "summary.rolling_trace_replay.missing_proposals",
            trace.get("missing_proposals"),
            self.args.max_rolling_trace_missing_proposals,
        );
        self.require_u64_equals(
            "summary.rolling_trace_replay.out_of_order_proposals",
            trace.get("out_of_order_proposals"),
            0,
        );
        self.require_optional_bool_true(
            "summary.rolling_trace_replay.verified_prefix_matches_target",
            trace.get("verified_prefix_matches_target"),
        );
        let observed = value_u64(trace.get("cases_replayed")).unwrap_or(0)
            + value_u64(trace.get("live_cases_observed")).unwrap_or(0);
        if observed == 0 {
            self.failures
                .push("summary.rolling_trace_replay observed no cases".to_string());
        }
    }

    fn check_spd_decode_ceiling(&mut self) {
        let Some(max_spd_decode_ms) = self.args.max_spd_decode_ms else {
            return;
        };
        let max_ms = self
            .summary
            .get("spd_decode_ms")
            .and_then(|value| value.get("max_ms"))
            .and_then(Value::as_f64);
        match max_ms {
            Some(max_ms) if max_ms <= max_spd_decode_ms => {}
            Some(max_ms) => self.failures.push(format!(
                "summary.spd_decode_ms.max_ms {max_ms:.3} exceeds {max_spd_decode_ms:.3}"
            )),
            None => self
                .failures
                .push("summary.spd_decode_ms.max_ms is missing".to_string()),
        }
    }

    fn require_str(&mut self, name: &str, value: Option<&Value>, expected: &str) {
        match value.and_then(Value::as_str) {
            Some(actual) if actual == expected => {}
            Some(actual) => self
                .failures
                .push(format!("{name} is {actual:?}, expected {expected:?}")),
            None => self.failures.push(format!("{name} is missing")),
        }
    }

    fn require_bool(&mut self, name: &str, value: Option<&Value>) {
        match value.and_then(Value::as_bool) {
            Some(true) => {}
            Some(false) => self.failures.push(format!("{name} is false")),
            None => self.failures.push(format!("{name} is missing")),
        }
    }

    fn require_optional_bool_true(&mut self, name: &str, value: Option<&Value>) {
        match value.and_then(Value::as_bool) {
            Some(true) => {}
            Some(false) => self.failures.push(format!("{name} is false")),
            None => self.failures.push(format!("{name} is missing")),
        }
    }

    fn require_u64_at_least(&mut self, name: &str, value: Option<&Value>, minimum: u64) {
        match value_u64(value) {
            Some(actual) if actual >= minimum => {}
            Some(actual) => self
                .failures
                .push(format!("{name} is {actual}, expected at least {minimum}")),
            None => self.failures.push(format!("{name} is missing")),
        }
    }

    fn require_u64_equals(&mut self, name: &str, value: Option<&Value>, expected: u64) {
        match value_u64(value) {
            Some(actual) if actual == expected => {}
            Some(actual) => self
                .failures
                .push(format!("{name} is {actual}, expected {expected}")),
            None => self.failures.push(format!("{name} is missing")),
        }
    }

    fn require_u64_at_most(&mut self, name: &str, value: Option<&Value>, maximum: u64) {
        match value_u64(value) {
            Some(actual) if actual <= maximum => {}
            Some(actual) => self
                .failures
                .push(format!("{name} is {actual}, expected at most {maximum}")),
            None => self.failures.push(format!("{name} is missing")),
        }
    }

    fn finish(self) -> CheckResult {
        let pass = self.failures.is_empty();
        CheckResult {
            pass,
            failures: self.failures,
            observed: observed_summary(self.report),
        }
    }
}

fn value_u64(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

fn observed_summary(report: &Value) -> Value {
    let summary = report.get("summary").unwrap_or(&Value::Null);
    json!({
        "prompt_pairs": summary.get("prompt_pairs"),
        "matching_content": summary.get("matching_content"),
        "spd_spec_accepted": summary.get("spd_spec_accepted"),
        "spd_spec_proposed": summary.get("spd_spec_proposed"),
        "spd_rolling_executor_max_in_flight": summary.get("spd_rolling_executor_max_in_flight"),
        "rolling_trace_replay": summary.get("rolling_trace_replay"),
        "spd_decode_ms": summary.get("spd_decode_ms"),
    })
}

#[derive(Debug, Serialize)]
struct CheckResult {
    pass: bool,
    failures: Vec<String>,
    observed: Value,
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::json;

    use super::*;

    #[test]
    fn accepts_report_that_meets_current_spd_gate() {
        let result = check_report(&passing_report(), &args(None));

        assert!(result.pass, "{:?}", result.failures);
    }

    #[test]
    fn rejects_missing_content_match_and_dirty_rolling_replay() {
        let mut report = passing_report();
        report["summary"]["matching_content"] = json!(0);
        report["summary"]["rolling_trace_replay"]["missing_proposals"] = json!(1);
        report["summary"]["spd_rolling_executor_max_in_flight"] = json!(3);

        let result = check_report(&report, &args(None));

        assert!(!result.pass);
        assert!(
            result
                .failures
                .iter()
                .any(|failure| failure.contains("matching_content"))
        );
        assert!(
            result
                .failures
                .iter()
                .any(|failure| failure.contains("missing_proposals"))
        );
        assert!(
            result
                .failures
                .iter()
                .any(|failure| failure.contains("max_in_flight"))
        );
    }

    #[test]
    fn optional_decode_ceiling_checks_spd_decode_max() {
        let result = check_report(&passing_report(), &args(Some(1500.0)));

        assert!(!result.pass);
        assert!(
            result
                .failures
                .iter()
                .any(|failure| failure.contains("spd_decode_ms.max_ms"))
        );
    }

    #[test]
    fn accepts_bounded_rejection_recovery_when_content_matches() {
        let mut report = passing_report();
        report["summary"]["spd_rolling_executor_rejected_oldest"] = json!(1);
        report["summary"]["spd_rolling_executor_drained_younger"] = json!(3);
        report["summary"]["rolling_trace_replay"]["missing_proposals"] = json!(9);
        let mut args = args(None);
        args.max_rejected_oldest = 1;
        args.max_drained_younger = 3;
        args.max_rolling_trace_missing_proposals = 9;

        let result = check_report(&report, &args);

        assert!(result.pass, "{:?}", result.failures);
    }

    fn args(max_spd_decode_ms: Option<f64>) -> SpdOpenAiCheckArgs {
        SpdOpenAiCheckArgs {
            report: PathBuf::from("/tmp/report.json"),
            min_accepted: 24,
            expected_logical_stage_count: 4,
            min_max_inflight: 4,
            max_rejected_oldest: 0,
            max_drained_younger: 0,
            max_rolling_trace_missing_proposals: 0,
            require_content_match: true,
            require_rolling_executor: true,
            max_spd_decode_ms,
            output: None,
        }
    }

    fn passing_report() -> Value {
        json!({
            "mode": "spd-openai-smoke",
            "logical_spd_stage_count": 4,
            "spd_rolling_executor": true,
            "summary": {
                "prompt_pairs": 1,
                "matching_content": 1,
                "spd_spec_accepted": 24,
                "spd_spec_proposed": 24,
                "spd_rolling_executor_max_in_flight": 4,
                "spd_rolling_executor_rejected_oldest": 0,
                "spd_rolling_executor_drained_younger": 0,
                "tap_return_failures": 0,
                "tap_record_failures": 0,
                "spd_decode_ms": {
                    "max_ms": 1600.0
                },
                "rolling_trace_replay": {
                    "cases_replayed": 1,
                    "live_cases_observed": 0,
                    "missing_proposals": 0,
                    "out_of_order_proposals": 0,
                    "verified_prefix_matches_target": true
                }
            }
        })
    }
}
