use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::write_json_file;

#[derive(Debug, clap::Args)]
pub(crate) struct QuantPackComposeCandidateArgs {
    /// Existing quant-plan JSON to extend with a composed candidate.
    pub(crate) plan: PathBuf,
    /// Candidate IDs whose quant groups should be combined in order.
    #[arg(long = "from", required = true)]
    pub(crate) from_candidates: Vec<String>,
    /// New candidate ID to append to the output plan.
    #[arg(long)]
    pub(crate) id: String,
    /// Human-readable candidate name.
    #[arg(long)]
    pub(crate) name: Option<String>,
    /// Strategy label recorded in the quant layout.
    #[arg(long, default_value = "evidence-composed-layer-sensitivity")]
    pub(crate) strategy: String,
    /// Extra note to attach to the composed candidate.
    #[arg(long = "note")]
    pub(crate) notes: Vec<String>,
    /// Output quant-plan JSON containing the appended candidate.
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QuantPlanCandidate {
    id: String,
    layout_hash: String,
    name: String,
    status: String,
    strategy: String,
    default_quant: String,
    groups: Vec<QuantGroup>,
    stage_hints: Vec<StageHint>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct QuantGroup {
    name: String,
    quant: String,
    selector: QuantSelector,
    reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum QuantSelector {
    Role { roles: Vec<String> },
    LayerRange { start: u32, end: u32 },
    TensorNamePattern { patterns: Vec<String> },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct StageHint {
    stage_index: usize,
    layer_start: u32,
    layer_end: u32,
    tensor_bytes: u64,
    role: String,
}

#[derive(Debug, Serialize)]
struct ComposeReport {
    schema_version: u32,
    kind: String,
    plan: String,
    out: String,
    candidate: QuantPlanCandidate,
    source_candidates: Vec<String>,
    deduped_group_count: usize,
}

pub(crate) fn run_quant_pack_compose_candidate(args: QuantPackComposeCandidateArgs) -> Result<()> {
    let mut plan = read_plan_value(&args.plan)?;
    let candidates = read_candidates(&plan)?;
    let candidate = compose_candidate(&candidates, &args)?;
    append_candidate(&mut plan, candidate.clone())?;
    write_json_file(&args.out, &plan)?;

    let source_group_count = selected_candidates(&candidates, &args.from_candidates)?
        .iter()
        .map(|candidate| candidate.groups.len())
        .sum::<usize>();
    let report = ComposeReport {
        schema_version: 1,
        kind: "skippy_quant_pack_compose_candidate".to_string(),
        plan: args.plan.display().to_string(),
        out: args.out.display().to_string(),
        deduped_group_count: source_group_count.saturating_sub(candidate.groups.len()),
        candidate,
        source_candidates: args.from_candidates,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn read_plan_value(path: &Path) -> Result<Value> {
    let contents = fs::read(path).with_context(|| format!("read quant plan {}", path.display()))?;
    serde_json::from_slice(&contents)
        .with_context(|| format!("parse quant plan {}", path.display()))
}

fn read_candidates(plan: &Value) -> Result<Vec<QuantPlanCandidate>> {
    let candidates = plan
        .get("candidates")
        .and_then(Value::as_array)
        .with_context(|| "quant plan is missing candidates array")?;
    candidates
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| "parse quant plan candidates")
}

fn compose_candidate(
    candidates: &[QuantPlanCandidate],
    args: &QuantPackComposeCandidateArgs,
) -> Result<QuantPlanCandidate> {
    ensure!(
        args.from_candidates.len() >= 2,
        "compose-candidate requires at least two --from candidates"
    );
    ensure!(
        !candidates.iter().any(|candidate| candidate.id == args.id),
        "quant plan already contains candidate {:?}",
        args.id
    );
    let selected = selected_candidates(candidates, &args.from_candidates)?;
    let first = selected
        .first()
        .with_context(|| "compose-candidate requires selected candidates")?;
    ensure_same_default_quant(&selected)?;
    ensure_same_stage_hints(&selected)?;

    let mut groups = dedupe_groups(&selected)?;
    let notes = composed_notes(&selected, &args.notes);
    let mut candidate = QuantPlanCandidate {
        id: args.id.clone(),
        layout_hash: String::new(),
        name: args
            .name
            .clone()
            .unwrap_or_else(|| format!("Evidence-composed candidate {}", args.id)),
        status: "experimental".to_string(),
        strategy: args.strategy.clone(),
        default_quant: first.default_quant.clone(),
        groups: std::mem::take(&mut groups),
        stage_hints: first.stage_hints.clone(),
        notes,
    };
    candidate.layout_hash = quant_layout_hash(&candidate);
    Ok(candidate)
}

fn selected_candidates<'a>(
    candidates: &'a [QuantPlanCandidate],
    requested: &[String],
) -> Result<Vec<&'a QuantPlanCandidate>> {
    let by_id = candidates
        .iter()
        .map(|candidate| (candidate.id.as_str(), candidate))
        .collect::<BTreeMap<_, _>>();
    requested
        .iter()
        .map(|id| {
            by_id
                .get(id.as_str())
                .copied()
                .with_context(|| format!("quant plan does not contain candidate {id:?}"))
        })
        .collect()
}

fn ensure_same_default_quant(candidates: &[&QuantPlanCandidate]) -> Result<()> {
    let first = &candidates[0].default_quant;
    if let Some(candidate) = candidates
        .iter()
        .find(|candidate| candidate.default_quant != *first)
    {
        bail!(
            "candidate {:?} default quant {:?} does not match {:?}",
            candidate.id,
            candidate.default_quant,
            first
        );
    }
    Ok(())
}

fn ensure_same_stage_hints(candidates: &[&QuantPlanCandidate]) -> Result<()> {
    let first = stage_hint_key(&candidates[0].stage_hints)?;
    for candidate in candidates.iter().skip(1) {
        let key = stage_hint_key(&candidate.stage_hints)?;
        if key != first {
            bail!("candidate {:?} uses different stage hints", candidate.id);
        }
    }
    Ok(())
}

fn stage_hint_key(stage_hints: &[StageHint]) -> Result<String> {
    serde_json::to_string(stage_hints).with_context(|| "serialize stage hints")
}

fn dedupe_groups(candidates: &[&QuantPlanCandidate]) -> Result<Vec<QuantGroup>> {
    let mut groups = Vec::new();
    let mut seen_layouts = BTreeSet::new();
    let mut names = BTreeMap::new();
    for candidate in candidates {
        for group in &candidate.groups {
            let layout_key = group_layout_key(group)?;
            if let Some(existing_key) = names.get(group.name.as_str()) {
                ensure!(
                    existing_key == &layout_key,
                    "group {:?} appears with conflicting layouts",
                    group.name
                );
            }
            names.insert(group.name.as_str(), layout_key.clone());
            if seen_layouts.insert(layout_key) {
                groups.push(group.clone());
            }
        }
    }
    Ok(groups)
}

fn group_layout_key(group: &QuantGroup) -> Result<String> {
    serde_json::to_string(&(&group.name, &group.quant, &group.selector))
        .with_context(|| "serialize quant group layout")
}

fn composed_notes(candidates: &[&QuantPlanCandidate], extra_notes: &[String]) -> Vec<String> {
    let mut notes = vec![format!(
        "Composes quant groups from evidence-selected candidates: {}.",
        candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )];
    notes.extend(extra_notes.iter().cloned());
    notes
}

fn append_candidate(plan: &mut Value, candidate: QuantPlanCandidate) -> Result<()> {
    let candidates = plan
        .get_mut("candidates")
        .and_then(Value::as_array_mut)
        .with_context(|| "quant plan is missing mutable candidates array")?;
    candidates.push(serde_json::to_value(candidate)?);
    Ok(())
}

fn quant_layout_hash(candidate: &QuantPlanCandidate) -> String {
    let mut hasher = Sha256::new();
    hash_str(&mut hasher, "skippy-quant-layout-v3");
    hash_str(&mut hasher, &candidate.id);
    hash_str(&mut hasher, &candidate.strategy);
    hash_str(&mut hasher, &candidate.default_quant);
    for group in &candidate.groups {
        hash_str(&mut hasher, &group.name);
        hash_str(&mut hasher, &group.quant);
        hash_selector(&mut hasher, &group.selector);
    }
    format!("{:x}", hasher.finalize())
}

fn hash_selector(hasher: &mut Sha256, selector: &QuantSelector) {
    match selector {
        QuantSelector::Role { roles } => {
            hash_str(hasher, "role");
            for role in roles {
                hash_str(hasher, role);
            }
        }
        QuantSelector::LayerRange { start, end } => {
            hash_str(hasher, "layer_range");
            hasher.update(start.to_le_bytes());
            hasher.update(end.to_le_bytes());
        }
        QuantSelector::TensorNamePattern { patterns } => {
            hash_str(hasher, "tensor_name_pattern");
            for pattern in patterns {
                hash_str(hasher, pattern);
            }
        }
    }
}

fn hash_str(hasher: &mut Sha256, value: &str) {
    hasher.update(value.len().to_le_bytes());
    hasher.update(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_candidate_dedupes_shared_groups_and_hashes_layout() {
        let candidates = vec![
            candidate(
                "stage-balanced-layer-22-ffn-gate-up-proxy",
                vec![boundary_group(), layer_group("22", "gate-up")],
            ),
            candidate(
                "stage-balanced-layer-20-ffn-down-proxy",
                vec![boundary_group(), layer_group("20", "down")],
            ),
        ];
        let args = args(&[
            "stage-balanced-layer-22-ffn-gate-up-proxy",
            "stage-balanced-layer-20-ffn-down-proxy",
        ]);

        let composed = compose_candidate(&candidates, &args).expect("compose candidate");

        assert_eq!(composed.id, "mixed-layer-proxy");
        assert_eq!(composed.groups.len(), 3);
        assert_eq!(composed.groups[0].name, "boundary");
        assert_eq!(composed.groups[1].name, "layer-22-ffn-gate-up-sensitivity");
        assert_eq!(composed.groups[2].name, "layer-20-ffn-down-sensitivity");
        assert_eq!(composed.layout_hash.len(), 64);
        assert!(composed.notes[0].contains("stage-balanced-layer-22-ffn-gate-up-proxy"));
    }

    #[test]
    fn compose_candidate_rejects_conflicting_named_groups() {
        let mut changed = boundary_group();
        changed.quant = "Q2_K".to_string();
        let candidates = vec![
            candidate("a", vec![boundary_group()]),
            candidate("b", vec![changed]),
        ];
        let args = args(&["a", "b"]);

        let err = compose_candidate(&candidates, &args).expect_err("conflict should fail");

        assert!(err.to_string().contains("conflicting layouts"));
    }

    fn args(from: &[&str]) -> QuantPackComposeCandidateArgs {
        QuantPackComposeCandidateArgs {
            plan: PathBuf::from("plan.json"),
            from_candidates: from.iter().map(|id| (*id).to_string()).collect(),
            id: "mixed-layer-proxy".to_string(),
            name: None,
            strategy: "evidence-composed-layer-sensitivity".to_string(),
            notes: Vec::new(),
            out: PathBuf::from("out.json"),
        }
    }

    fn candidate(id: &str, groups: Vec<QuantGroup>) -> QuantPlanCandidate {
        QuantPlanCandidate {
            id: id.to_string(),
            layout_hash: "old-hash".to_string(),
            name: id.to_string(),
            status: "experimental".to_string(),
            strategy: "probe".to_string(),
            default_quant: "COPY".to_string(),
            groups,
            stage_hints: vec![StageHint {
                stage_index: 0,
                layer_start: 0,
                layer_end: 4,
                tensor_bytes: 100,
                role: "stage".to_string(),
            }],
            notes: Vec::new(),
        }
    }

    fn boundary_group() -> QuantGroup {
        QuantGroup {
            name: "boundary".to_string(),
            quant: "Q4_K_M".to_string(),
            selector: QuantSelector::LayerRange { start: 0, end: 1 },
            reason: "shared protection".to_string(),
        }
    }

    fn layer_group(layer: &str, part: &str) -> QuantGroup {
        QuantGroup {
            name: format!("layer-{layer}-ffn-{part}-sensitivity"),
            quant: "Q3_K_M".to_string(),
            selector: QuantSelector::TensorNamePattern {
                patterns: vec![format!("blk\\.({layer})\\.ffn_{part}\\.weight")],
            },
            reason: "measured layer sensitivity".to_string(),
        }
    }
}
