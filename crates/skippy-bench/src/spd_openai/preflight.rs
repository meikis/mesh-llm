use std::{collections::BTreeSet, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::spd::{SpdHeadManifest, SpdSafetensorsIndex};

use crate::cli::SpdOpenAiSmokeArgs;

use super::remote::{RemotePreflightPlan, preflight_stage_placement};

pub(super) fn write_spd_openai_preflight(
    args: &SpdOpenAiSmokeArgs,
    manifest: &SpdHeadManifest,
    stage_ranges: &[(u32, u32)],
    tap_allowlist: &[u32],
    prompt_count: usize,
) -> Result<()> {
    let report = build_preflight_report(args, manifest, stage_ranges, tap_allowlist, prompt_count)?;
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output.as_ref() {
        fs::write(output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}

pub(super) fn validate_tap_coverage(
    stage_ranges: &[(u32, u32)],
    tap_allowlist: &[u32],
) -> Result<()> {
    let stage_boundary_hf_indices = stage_ranges
        .iter()
        .map(|(_, layer_end)| *layer_end)
        .collect::<BTreeSet<_>>();
    let missing = tap_allowlist
        .iter()
        .copied()
        .filter(|hf_index| !stage_boundary_hf_indices.contains(hf_index))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "SPD tap-return indices {:?} are not produced by --splits; stage boundary hf indices are {:?}",
            missing,
            stage_boundary_hf_indices.into_iter().collect::<Vec<_>>()
        );
    }
    Ok(())
}

fn build_preflight_report(
    args: &SpdOpenAiSmokeArgs,
    manifest: &SpdHeadManifest,
    stage_ranges: &[(u32, u32)],
    tap_allowlist: &[u32],
    prompt_count: usize,
) -> Result<SpdOpenAiPreflightReport> {
    let serving_checkpoint = manifest.ensure_serving_checkpoint_for_runtime(&args.manifest)?;
    let fixture_index = SpdSafetensorsIndex::from_path(&args.fixture)
        .with_context(|| format!("parse SPD fixture {}", args.fixture.display()))?;
    let remote = preflight_stage_placement(args, stage_ranges.len())?;
    Ok(SpdOpenAiPreflightReport {
        mode: "spd-openai-preflight",
        model_id: args.model_id.clone(),
        prompt_count,
        artifacts: ArtifactPreflight {
            stage_server_bin: sized_path(&args.stage_server_bin)?,
            model_path: sized_path(&args.model_path)?,
            manifest_path: path_string(&args.manifest),
            serving_checkpoint_path: path_string(manifest.serving_checkpoint_path(&args.manifest)?),
            serving_checkpoint_tensor_count: serving_checkpoint.tensors.len(),
            fixture_path: sized_path(&args.fixture)?,
            fixture_tensor_count: fixture_index.tensors.len(),
        },
        topology: TopologyPreflight {
            logical_spd_stage_count: manifest.topology.num_stages,
            physical_stage_count: stage_ranges.len(),
            splits: args.splits.clone(),
            layer_end: args.layer_end,
            stage_ranges: stage_ranges
                .iter()
                .map(|(layer_start, layer_end)| StageRangePreflight {
                    layer_start: *layer_start,
                    layer_end: *layer_end,
                })
                .collect(),
        },
        tap_plan: TapPreflight {
            tap_return_hf_indices: tap_allowlist.to_vec(),
            stage_boundary_hf_indices: stage_ranges
                .iter()
                .map(|(_, layer_end)| *layer_end)
                .collect(),
        },
        remote,
    })
}

fn sized_path(path: &Path) -> Result<SizedPathPreflight> {
    let bytes = fs::metadata(path)
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    Ok(SizedPathPreflight {
        path: path_string(path),
        bytes,
    })
}

fn path_string(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string()
}

#[derive(Debug, Serialize)]
struct SpdOpenAiPreflightReport {
    mode: &'static str,
    model_id: String,
    prompt_count: usize,
    artifacts: ArtifactPreflight,
    topology: TopologyPreflight,
    tap_plan: TapPreflight,
    remote: RemotePreflightPlan,
}

#[derive(Debug, Serialize)]
struct ArtifactPreflight {
    stage_server_bin: SizedPathPreflight,
    model_path: SizedPathPreflight,
    manifest_path: String,
    serving_checkpoint_path: String,
    serving_checkpoint_tensor_count: usize,
    fixture_path: SizedPathPreflight,
    fixture_tensor_count: usize,
}

#[derive(Debug, Serialize)]
struct SizedPathPreflight {
    path: String,
    bytes: u64,
}

#[derive(Debug, Serialize)]
struct TopologyPreflight {
    logical_spd_stage_count: u32,
    physical_stage_count: usize,
    splits: Vec<u32>,
    layer_end: u32,
    stage_ranges: Vec<StageRangePreflight>,
}

#[derive(Debug, Serialize)]
struct StageRangePreflight {
    layer_start: u32,
    layer_end: u32,
}

#[derive(Debug, Serialize)]
struct TapPreflight {
    tap_return_hf_indices: Vec<u32>,
    stage_boundary_hf_indices: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_coverage_accepts_stage_boundaries() {
        validate_tap_coverage(&[(0, 8), (8, 10), (10, 16)], &[8, 10]).unwrap();
    }

    #[test]
    fn tap_coverage_rejects_missing_boundary() {
        let err =
            validate_tap_coverage(&[(0, 8), (8, 16)], &[8, 10]).expect_err("missing tap boundary");

        assert!(err.to_string().contains("[10]"));
    }
}
