use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{
    NativeMtpDraftOrigin, native_mtp_ngram_hybrid_enabled, native_mtp_ngram_max_proposal_tokens,
    native_mtp_ngram_size, native_mtp_reject_cooldown_tokens,
    native_mtp_suppress_cooldown_draft_limit, native_mtp_suppress_cooldown_drafts_enabled,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) struct NativeMtpDecodeOptions {
    pub(in crate::frontend) max_draft_tokens: usize,
    pub(in crate::frontend) min_draft_tokens: usize,
    pub(in crate::frontend) reject_cooldown_tokens: usize,
    pub(in crate::frontend) suppress_cooldown_drafts: bool,
    pub(in crate::frontend) suppress_cooldown_draft_limit: usize,
    pub(in crate::frontend) ngram_hybrid: bool,
    pub(in crate::frontend) ngram_size: usize,
    pub(in crate::frontend) ngram_max_proposal_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) enum NativeMtpTrimAction {
    None,
    FullSession,
}

pub(in crate::frontend) fn native_mtp_trim_action(
    committed_positions: usize,
    consumed_positions: usize,
) -> NativeMtpTrimAction {
    if committed_positions == consumed_positions {
        NativeMtpTrimAction::None
    } else {
        NativeMtpTrimAction::FullSession
    }
}

impl NativeMtpDecodeOptions {
    pub(in crate::frontend) fn from_env() -> Self {
        Self {
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: native_mtp_reject_cooldown_tokens(),
            suppress_cooldown_drafts: native_mtp_suppress_cooldown_drafts_enabled(),
            suppress_cooldown_draft_limit: native_mtp_suppress_cooldown_draft_limit(),
            ngram_hybrid: native_mtp_ngram_hybrid_enabled(),
            ngram_size: native_mtp_ngram_size(),
            ngram_max_proposal_tokens: native_mtp_ngram_max_proposal_tokens(),
        }
    }

    pub(in crate::frontend) fn with_window(
        mut self,
        max_draft_tokens: usize,
        min_draft_tokens: usize,
    ) -> Self {
        self.max_draft_tokens = max_draft_tokens.max(1);
        self.min_draft_tokens = min_draft_tokens.min(self.max_draft_tokens);
        self
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::frontend) struct NativeMtpDecodeCounters {
    suppressed_cooldown_draft_count: usize,
    verify_window_verification_count: usize,
    initial_serial_verification_count: usize,
    initial_serial_accepted_count: usize,
    serial_after_gap_verification_count: usize,
    serial_after_gap_accepted_count: usize,
    verify_next_verification_count: usize,
    verify_next_accepted_count: usize,
    verify_next_draft_available_count: usize,
    verify_next_draft_adopted_count: usize,
    hybrid_anchor_available_count: usize,
    hybrid_ngram_span_available_count: usize,
    hybrid_anchor_agreement_count: usize,
    hybrid_anchor_disagreement_count: usize,
    hybrid_proposal_token_count: usize,
    hybrid_accepted_token_count: usize,
    hybrid_accepted_tail_token_count: usize,
}

impl NativeMtpDecodeCounters {
    pub(in crate::frontend) fn verify_window_verification_count(&self) -> usize {
        self.verify_window_verification_count
    }

    pub(in crate::frontend) fn observe_suppressed_cooldown_draft(&mut self) {
        self.suppressed_cooldown_draft_count += 1;
    }

    pub(in crate::frontend) fn observe_verify_window_verification(
        &mut self,
        origin: NativeMtpDraftOrigin,
        accepted: bool,
    ) {
        self.verify_window_verification_count += 1;
        match origin {
            NativeMtpDraftOrigin::InitialSerial => {
                self.initial_serial_verification_count += 1;
                if accepted {
                    self.initial_serial_accepted_count += 1;
                }
            }
            NativeMtpDraftOrigin::SerialAfterGap => {
                self.serial_after_gap_verification_count += 1;
                if accepted {
                    self.serial_after_gap_accepted_count += 1;
                }
            }
            NativeMtpDraftOrigin::VerifyNext => {
                self.verify_next_verification_count += 1;
                if accepted {
                    self.verify_next_accepted_count += 1;
                }
            }
        }
    }

    pub(in crate::frontend) fn observe_verify_next_draft(
        &mut self,
        available: bool,
        adopted: bool,
    ) {
        if available {
            self.verify_next_draft_available_count += 1;
        }
        if adopted {
            self.verify_next_draft_adopted_count += 1;
        }
    }

    pub(in crate::frontend) fn observe_hybrid_proposal(
        &mut self,
        ngram_span_available: bool,
        ngram_anchor_agreed: bool,
        ngram_anchor_disagreed: bool,
        proposal_token_count: usize,
        accepted_token_count: usize,
    ) {
        self.hybrid_anchor_available_count += 1;
        self.hybrid_ngram_span_available_count += usize::from(ngram_span_available);
        self.hybrid_anchor_agreement_count += usize::from(ngram_anchor_agreed);
        self.hybrid_anchor_disagreement_count += usize::from(ngram_anchor_disagreed);
        self.hybrid_proposal_token_count += proposal_token_count;
        self.hybrid_accepted_token_count += accepted_token_count;
        self.hybrid_accepted_tail_token_count += accepted_token_count.saturating_sub(1);
    }

    pub(in crate::frontend) fn insert_summary_attrs(
        &self,
        attrs: &mut BTreeMap<String, Value>,
        options: NativeMtpDecodeOptions,
    ) {
        attrs.insert(
            "llama_stage.native_mtp.reject_cooldown_tokens".to_string(),
            json!(options.reject_cooldown_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.max_draft_tokens".to_string(),
            json!(options.max_draft_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.min_draft_tokens".to_string(),
            json!(options.min_draft_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppress_cooldown_drafts".to_string(),
            json!(options.suppress_cooldown_drafts),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppress_cooldown_draft_limit".to_string(),
            json!(options.suppress_cooldown_draft_limit),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_hybrid".to_string(),
            json!(options.ngram_hybrid),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_size".to_string(),
            json!(options.ngram_size),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_max_proposal_tokens".to_string(),
            json!(options.ngram_max_proposal_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppressed_cooldown_draft_count".to_string(),
            json!(self.suppressed_cooldown_draft_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_window_verification_count".to_string(),
            json!(self.verify_window_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.initial_serial_verification_count".to_string(),
            json!(self.initial_serial_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.initial_serial_accepted_count".to_string(),
            json!(self.initial_serial_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.serial_after_gap_verification_count".to_string(),
            json!(self.serial_after_gap_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.serial_after_gap_accepted_count".to_string(),
            json!(self.serial_after_gap_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_verification_count".to_string(),
            json!(self.verify_next_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_accepted_count".to_string(),
            json!(self.verify_next_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_draft_available_count".to_string(),
            json!(self.verify_next_draft_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_draft_adopted_count".to_string(),
            json!(self.verify_next_draft_adopted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_anchor_available_count".to_string(),
            json!(self.hybrid_anchor_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_span_available_count".to_string(),
            json!(self.hybrid_ngram_span_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_anchor_agreement_count".to_string(),
            json!(self.hybrid_anchor_agreement_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_anchor_disagreement_count".to_string(),
            json!(self.hybrid_anchor_disagreement_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_proposal_token_count".to_string(),
            json!(self.hybrid_proposal_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_accepted_token_count".to_string(),
            json!(self.hybrid_accepted_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_accepted_tail_token_count".to_string(),
            json!(self.hybrid_accepted_tail_token_count),
        );
    }

    fn insert_response_timings(&self, timings: &mut BTreeMap<String, Value>) {
        timings.insert(
            "native_mtp_verify_window_verifications".to_string(),
            json!(self.verify_window_verification_count),
        );
        timings.insert(
            "native_mtp_ngram_anchor_available".to_string(),
            json!(self.hybrid_anchor_available_count),
        );
        timings.insert(
            "native_mtp_ngram_span_available".to_string(),
            json!(self.hybrid_ngram_span_available_count),
        );
        timings.insert(
            "native_mtp_ngram_anchor_agreed".to_string(),
            json!(self.hybrid_anchor_agreement_count),
        );
        timings.insert(
            "native_mtp_ngram_anchor_disagreed".to_string(),
            json!(self.hybrid_anchor_disagreement_count),
        );
        timings.insert(
            "native_mtp_hybrid_proposed_tokens".to_string(),
            json!(self.hybrid_proposal_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_accepted_tokens".to_string(),
            json!(self.hybrid_accepted_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_accepted_tail_tokens".to_string(),
            json!(self.hybrid_accepted_tail_token_count),
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::frontend) struct NativeMtpDecodeTelemetry {
    options: NativeMtpDecodeOptions,
    counters: NativeMtpDecodeCounters,
}

impl NativeMtpDecodeTelemetry {
    pub(in crate::frontend) fn new(
        options: NativeMtpDecodeOptions,
        counters: NativeMtpDecodeCounters,
    ) -> Self {
        Self { options, counters }
    }

    pub(in crate::frontend) fn insert_response_timings(
        self,
        timings: &mut BTreeMap<String, Value>,
    ) {
        timings.insert(
            "native_mtp_ngram_hybrid_enabled".to_string(),
            json!(self.options.ngram_hybrid),
        );
        timings.insert(
            "native_mtp_ngram_size".to_string(),
            json!(self.options.ngram_size),
        );
        timings.insert(
            "native_mtp_ngram_max_proposal_tokens".to_string(),
            json!(self.options.ngram_max_proposal_tokens),
        );
        self.counters.insert_response_timings(timings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_track_verify_window_verification_by_origin() {
        let mut counters = NativeMtpDecodeCounters::default();
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::InitialSerial, true);
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::SerialAfterGap, false);
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::VerifyNext, true);
        counters.observe_verify_next_draft(true, false);
        counters.observe_verify_next_draft(true, true);
        counters.observe_suppressed_cooldown_draft();
        counters.observe_hybrid_proposal(true, true, false, 4, 3);

        let mut attrs = BTreeMap::new();
        counters.insert_summary_attrs(
            &mut attrs,
            NativeMtpDecodeOptions {
                max_draft_tokens: 3,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 6,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 2,
                ngram_hybrid: true,
                ngram_size: 8,
                ngram_max_proposal_tokens: 4,
            },
        );

        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_window_verification_count"),
            Some(&json!(3))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.initial_serial_accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.serial_after_gap_accepted_count"),
            Some(&json!(0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_draft_available_count"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_draft_adopted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.suppressed_cooldown_draft_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.reject_cooldown_tokens"),
            Some(&json!(6))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_accepted_tail_token_count"),
            Some(&json!(2))
        );
    }

    #[test]
    fn response_timings_show_hybrid_widening_evidence() {
        let mut counters = NativeMtpDecodeCounters::default();
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::InitialSerial, true);
        counters.observe_hybrid_proposal(true, true, false, 4, 3);
        let telemetry = NativeMtpDecodeTelemetry::new(
            NativeMtpDecodeOptions {
                max_draft_tokens: 1,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
                ngram_hybrid: true,
                ngram_size: 8,
                ngram_max_proposal_tokens: 4,
            },
            counters,
        );

        let mut timings = BTreeMap::new();
        telemetry.insert_response_timings(&mut timings);

        assert_eq!(
            timings.get("native_mtp_ngram_hybrid_enabled"),
            Some(&json!(true))
        );
        assert_eq!(
            timings.get("native_mtp_ngram_anchor_agreed"),
            Some(&json!(1))
        );
        assert_eq!(
            timings.get("native_mtp_hybrid_accepted_tail_tokens"),
            Some(&json!(2))
        );
    }
}
