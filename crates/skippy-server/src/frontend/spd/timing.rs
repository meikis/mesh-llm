use std::collections::BTreeMap;

use serde_json::{Value, json};
use skippy_runtime::spd::{SpdQwen3FixtureTopK, SpdQwen3ForwardTiming, SpdQwen3TimedForward};

#[derive(Debug, Clone, Default, PartialEq)]
pub(in crate::frontend) struct SpdHeadForwardTiming {
    pub(in crate::frontend) cache_prefill_ms: f64,
    pub(in crate::frontend) fixed_stage_projection_ms: f64,
    pub(in crate::frontend) decoder_layer_ms: Vec<f64>,
    pub(in crate::frontend) final_norm_ms: f64,
    pub(in crate::frontend) lm_head_topk_ms: f64,
    pub(in crate::frontend) total_ms: f64,
}

impl SpdHeadForwardTiming {
    pub(in crate::frontend) fn from_runtime(
        cache_prefill_ms: f64,
        timing: SpdQwen3ForwardTiming,
    ) -> Self {
        Self {
            cache_prefill_ms,
            fixed_stage_projection_ms: timing.fixed_stage_projection_ms,
            decoder_layer_ms: timing.decoder_layer_ms,
            final_norm_ms: timing.final_norm_ms,
            lm_head_topk_ms: timing.lm_head_topk_ms,
            total_ms: timing.total_ms,
        }
    }

    pub(in crate::frontend) fn decoder_total_ms(&self) -> f64 {
        self.decoder_layer_ms.iter().sum()
    }
}

pub(super) struct SpdHeadForwardOutcome {
    pub(super) topk: SpdQwen3FixtureTopK,
    pub(super) cache_used: bool,
    pub(super) cache_prefix_len: Option<usize>,
    pub(super) timing: SpdHeadForwardTiming,
}

impl SpdHeadForwardOutcome {
    pub(super) fn from_timed_forward(
        timed: SpdQwen3TimedForward,
        cache_used: bool,
        cache_prefix_len: Option<usize>,
        cache_prefill_ms: f64,
    ) -> Self {
        Self {
            topk: timed.topk,
            cache_used,
            cache_prefix_len,
            timing: SpdHeadForwardTiming::from_runtime(cache_prefill_ms, timed.timing),
        }
    }
}

pub(super) fn insert_head_forward_timing_attrs(
    prefix: &str,
    timing: &SpdHeadForwardTiming,
    attrs: &mut BTreeMap<String, Value>,
) {
    attrs.insert(
        format!("{prefix}_cache_prefill_ms"),
        json!(timing.cache_prefill_ms),
    );
    attrs.insert(
        format!("{prefix}_head_fixed_stage_projection_ms"),
        json!(timing.fixed_stage_projection_ms),
    );
    attrs.insert(
        format!("{prefix}_head_decoder_ms"),
        json!(timing.decoder_total_ms()),
    );
    attrs.insert(
        format!("{prefix}_head_decoder_layer_ms"),
        json!(&timing.decoder_layer_ms),
    );
    attrs.insert(
        format!("{prefix}_head_final_norm_ms"),
        json!(timing.final_norm_ms),
    );
    attrs.insert(
        format!("{prefix}_head_lm_head_topk_ms"),
        json!(timing.lm_head_topk_ms),
    );
    attrs.insert(format!("{prefix}_head_total_ms"), json!(timing.total_ms));
}
