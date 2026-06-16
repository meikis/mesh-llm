use std::fs;

use anyhow::{Context, Result};
use serde_json::json;
use skippy_runtime::spd::{run_qwen3_fixture_parity, run_spd_tap_input_fixture_parity};

use crate::cli::SpdFixtureParityArgs;

pub fn spd_fixture_parity(args: SpdFixtureParityArgs) -> Result<()> {
    let tap_input = run_spd_tap_input_fixture_parity(&args.manifest, &args.fixture)
        .context("failed to reconstruct SPD fixture cur_in from tap inputs")?;
    let forward = run_qwen3_fixture_parity(&args.manifest, &args.fixture, args.top_k)
        .context("failed to run Qwen3 SPD fixture forward parity")?;
    let report = json!({
        "mode": "spd-fixture-parity",
        "manifest": args.manifest,
        "fixture": args.fixture,
        "tap_input": {
            "max_abs_diff": tap_input.max_abs_diff,
            "rows": tap_input.rows.iter().map(|row| {
                json!({
                    "row_index": row.row_index,
                    "position_id": row.position_id,
                    "stage_id": row.stage_id,
                    "projection_name": row.projection_name,
                    "hf_indices": row.hf_indices,
                    "max_abs_diff": row.max_abs_diff,
                })
            }).collect::<Vec<_>>(),
        },
        "forward": {
            "rust": {
                "draft_indices": forward.rust.draft_indices,
                "token_ids": forward.rust.token_ids,
                "logits": forward.rust.logits,
            },
            "python": {
                "draft_indices": forward.python.draft_indices,
                "token_ids": forward.python.token_ids,
                "logits": forward.python.logits,
            },
            "diagnostics": {
                "layer_input_max_abs_diff": forward.diagnostics.layer_input_max_abs_diff,
                "layer_query_max_abs_diff": forward.diagnostics.layer_query_max_abs_diff,
                "spec_query_max_abs_diff": forward.diagnostics.spec_query_max_abs_diff,
                "final_hidden_max_abs_diff": forward.diagnostics.final_hidden_max_abs_diff,
                "python_top_logit_values_at_rust_indices": forward.diagnostics.python_top_logit_values_at_rust_indices,
            }
        }
    });
    let json = serde_json::to_vec_pretty(&report)?;
    if let Some(output) = args.output {
        fs::write(&output, &json)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    println!("{}", String::from_utf8(json)?);
    Ok(())
}
