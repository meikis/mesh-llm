use super::{
    QuantGroup, QuantLayoutCandidate, QuantSelector, StageBalancedFfnPartCandidateInput,
    compression_quant_for_source, largest_unprotected_stage, stage_ffn_parts_patterns,
    with_layout_hash,
};

pub(super) fn stage_balanced_layer_ffn_sensitivity_candidates(
    input: StageBalancedFfnPartCandidateInput<'_>,
) -> Vec<QuantLayoutCandidate> {
    let Some(stage) =
        largest_unprotected_stage(input.stage_hints, input.protected_width, input.layer_count)
    else {
        return Vec::new();
    };

    let mut candidates = Vec::new();
    for layer in stage.layer_start..stage.layer_end {
        candidates.push(layer_ffn_candidate(
            input.clone(),
            layer,
            &["down"],
            "down",
            "down",
        ));
        candidates.push(layer_ffn_candidate(
            input.clone(),
            layer,
            &["gate", "up"],
            "gate-up",
            "gate/up",
        ));
    }
    candidates
}

fn layer_ffn_candidate(
    mut input: StageBalancedFfnPartCandidateInput<'_>,
    layer: u32,
    parts: &[&'static str],
    candidate_part: &'static str,
    label_part: &'static str,
) -> QuantLayoutCandidate {
    input.groups.push(QuantGroup {
        name: format!("layer-{layer}-ffn-{candidate_part}-sensitivity"),
        quant: compression_quant_for_source(input.source_quant, "Q3_K_M").to_string(),
        selector: QuantSelector::TensorNamePattern {
            patterns: stage_ffn_parts_patterns(layer, layer + 1, parts, input.has_moe),
        },
        reason: format!(
            "Compresses layer {layer} FFN {label_part} tensors to isolate per-layer decode and memory sensitivity."
        ),
    });
    with_layout_hash(QuantLayoutCandidate {
        id: format!("stage-balanced-layer-{layer}-ffn-{candidate_part}-proxy"),
        layout_hash: String::new(),
        name: format!("Layer {layer} FFN {label_part} sensitivity probe"),
        status: "experimental".to_string(),
        strategy: format!("stage-balanced-layer-ffn-{candidate_part}-sensitivity-proxy"),
        default_quant: input.default_quant.to_string(),
        groups: input.groups,
        stage_hints: input.stage_hints.to_vec(),
        notes: vec![format!(
            "Tests whether lowering only layer {layer} FFN {label_part} preserves decode latency while reducing the largest stage."
        )],
    })
}
