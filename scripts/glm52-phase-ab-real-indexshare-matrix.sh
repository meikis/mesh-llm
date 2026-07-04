#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STAGE_MODEL="${STAGE_MODEL:-/Volumes/External/models/huggingface/hub/models--meshllm--GLM-5.2-Q2_K-MTP-Q8-layers/snapshots/main}"
MODEL_ID="${MODEL_ID:-meshllm/GLM-5.2-Q2_K-MTP-Q8-layers}"
SKIPPY_MODEL_PACKAGE_BIN="${SKIPPY_MODEL_PACKAGE_BIN:-$ROOT/target/debug/skippy-model-package}"
SKIPPY_BENCH_BIN="${SKIPPY_BENCH_BIN:-$ROOT/target/debug/skippy-bench}"
OUT_DIR="${OUT_DIR:-/tmp/glm52-phase-ab-real-indexshare-matrix}"
CTX_SIZE="${CTX_SIZE:-128}"
TOKENS="${TOKENS:-1}"
POSITION_START="${POSITION_START:-16}"
N_BATCH="${N_BATCH:-16}"
N_UBATCH="${N_UBATCH:-16}"
KV_WARMUP_TOKENS="${KV_WARMUP_TOKENS:-16}"
KV_WARMUP_CHUNK_TOKENS="${KV_WARMUP_CHUNK_TOKENS:-16}"
SPANS="${SPANS:-2:6:initial,6:10:early,30:34:middle,74:78:late}"
COMPACT_FLASH_ATTN="${COMPACT_FLASH_ATTN:-0}"
ALLOW_COMPACT_FLASH_AUTO="${ALLOW_COMPACT_FLASH_AUTO:-0}"
COMPACT_FLASH_NO_MASK="${COMPACT_FLASH_NO_MASK:-0}"
SELECTED_ROW_FLASH="${SELECTED_ROW_FLASH:-0}"
REQUIRE_COMPACT_FLASH_PROOF="${REQUIRE_COMPACT_FLASH_PROOF:-0}"
METAL_DISPATCH_LOG="${METAL_DISPATCH_LOG:-0}"
RUN_NEGATIVE="${RUN_NEGATIVE:-1}"

usage() {
  cat <<'EOF'
Usage: scripts/glm52-phase-ab-real-indexshare-matrix.sh [options]

Runs the real GLM-5.2 A/B-only IndexShare proof matrix:

  A. Validate the llama-native GLM-DSA runtime contract and generation policy
     from the real layer package.
  B. Validate representative Full producer -> Shared consumer spans with parity.
  B-. Validate that starting on a Shared layer without top-k sideband fails.

This script deliberately keeps direct sparse attention and compact flash disabled.
It is not a sparse-attention tuning run.

Options:
  --stage-model PATH      Layer package path.
  --model-id ID           Model id recorded in reports.
  --skippy-model-package PATH
                           skippy-model-package binary.
  --skippy-bench PATH     skippy-bench binary.
  --out-dir PATH          Output directory.
  --spans LIST            Comma-separated START:END:LABEL list. Default:
                           2:6:initial,6:10:early,30:34:middle,74:78:late
  --ctx-size N            Context size. Default: 128
  --tokens N              Tokens. Default: 1
  --position-start N      Decode position start. Default: 16
  --n-batch N             Batch size. Default: 16
  --n-ubatch N            Microbatch size. Default: 16
  --kv-warmup-tokens N    KV prefix tokens. Default: 16
  --kv-warmup-chunk-tokens N
                           KV warmup chunk size. Default: 16
  --compact-flash-attn     Enable compact top-k K/V + flash-attention candidate path.
  --allow-compact-flash-auto
                           Allow native compact flash policy to select compact path.
  --compact-flash-no-mask  Skip compact top-k mask gather for compact flash.
  --selected-row-flash     Enable native Metal selected-row flash gather path.
  --require-compact-flash-proof
                           Fail unless compact flash eliminated the old sparse path.
  --metal-dispatch-log     Capture Metal dispatch records.
  --skip-negative         Do not run the Shared-only missing-top-k negative case.
  -h, --help              Show this help.

Environment overrides mirror upper-case option names.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-model)
      STAGE_MODEL="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --skippy-model-package)
      SKIPPY_MODEL_PACKAGE_BIN="$2"
      shift 2
      ;;
    --skippy-bench)
      SKIPPY_BENCH_BIN="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --spans)
      SPANS="$2"
      shift 2
      ;;
    --ctx-size)
      CTX_SIZE="$2"
      shift 2
      ;;
    --tokens)
      TOKENS="$2"
      shift 2
      ;;
    --position-start)
      POSITION_START="$2"
      shift 2
      ;;
    --n-batch)
      N_BATCH="$2"
      shift 2
      ;;
    --n-ubatch)
      N_UBATCH="$2"
      shift 2
      ;;
    --kv-warmup-tokens)
      KV_WARMUP_TOKENS="$2"
      shift 2
      ;;
    --kv-warmup-chunk-tokens)
      KV_WARMUP_CHUNK_TOKENS="$2"
      shift 2
      ;;
    --compact-flash-attn)
      COMPACT_FLASH_ATTN=1
      shift
      ;;
    --allow-compact-flash-auto)
      ALLOW_COMPACT_FLASH_AUTO=1
      shift
      ;;
    --compact-flash-no-mask)
      COMPACT_FLASH_NO_MASK=1
      COMPACT_FLASH_ATTN=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --selected-row-flash)
      SELECTED_ROW_FLASH=1
      COMPACT_FLASH_ATTN=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --require-compact-flash-proof)
      REQUIRE_COMPACT_FLASH_PROOF=1
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --metal-dispatch-log)
      METAL_DISPATCH_LOG=1
      shift
      ;;
    --skip-negative)
      RUN_NEGATIVE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$STAGE_MODEL" ]]; then
  echo "stage model package not found: $STAGE_MODEL" >&2
  exit 1
fi
if [[ ! -x "$SKIPPY_MODEL_PACKAGE_BIN" ]]; then
  echo "skippy-model-package binary not executable: $SKIPPY_MODEL_PACKAGE_BIN" >&2
  exit 1
fi
if [[ ! -x "$SKIPPY_BENCH_BIN" ]]; then
  echo "skippy-bench binary not executable: $SKIPPY_BENCH_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
CONTRACT_JSON="$OUT_DIR/contract.json"
MATRIX_JSON="$OUT_DIR/matrix.json"

"$SKIPPY_MODEL_PACKAGE_BIN" glm-dsa-contract --require-generation-policy "$STAGE_MODEL" >"$CONTRACT_JSON"

python3 - "$CONTRACT_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
contract = json.loads(path.read_text())
failures = []
if not contract.get("valid"):
    failures.append("contract valid=false")
if not contract.get("generation_policy_required"):
    failures.append("generation_policy_required=false")
if contract.get("architecture") != "glm-dsa":
    failures.append(f"unexpected architecture {contract.get('architecture')!r}")
if contract.get("role_source") != "metadata_types":
    failures.append(f"unexpected role_source {contract.get('role_source')!r}")
if contract.get("effective_decoder_layers") != 78:
    failures.append(f"unexpected effective_decoder_layers {contract.get('effective_decoder_layers')!r}")
if contract.get("nextn_predict_layers") != 1:
    failures.append(f"unexpected nextn_predict_layers {contract.get('nextn_predict_layers')!r}")
if not contract.get("full_layers"):
    failures.append("missing full_layers")
if not contract.get("shared_layers"):
    failures.append("missing shared_layers")
if contract.get("metadata_errors"):
    failures.append(f"metadata_errors={contract.get('metadata_errors')}")
if contract.get("tensor_errors"):
    failures.append(f"tensor_errors={contract.get('tensor_errors')}")
if contract.get("warnings"):
    failures.append(f"warnings={contract.get('warnings')}")
policy = contract.get("generation_policy") or {}
expected_policy = {
    "profile": "glm-dsa-v1",
    "decode": "compact-flash",
    "short_prefill": "dense",
    "long_prefill": "sparse-chunked",
    "verify": "auto",
    "indexshare": "required",
    "selected_row_flash": "evidence-gated",
}
for key, expected in expected_policy.items():
    if policy.get(key) != expected:
        failures.append(f"generation_policy.{key}={policy.get(key)!r}, expected {expected!r}")
thresholds = contract.get("generation_thresholds") or {}
expected_thresholds = {
    "short_prefill_max_tokens": 2048,
    "direct_sparse_decode_max_top_k": 256,
    "compact_flash_min_kv": 1,
    "dense_mask_max_bytes": 268435456,
}
for key, expected in expected_thresholds.items():
    if thresholds.get(key) != expected:
        failures.append(f"generation_thresholds.{key}={thresholds.get(key)!r}, expected {expected!r}")
if contract.get("generation_policy_errors"):
    failures.append(f"generation_policy_errors={contract.get('generation_policy_errors')}")
if contract.get("generation_threshold_errors"):
    failures.append(f"generation_threshold_errors={contract.get('generation_threshold_errors')}")
if failures:
    print("GLM-5.2 contract validation failed", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    raise SystemExit(1)
PY

IFS=',' read -r -a SPAN_ITEMS <<<"$SPANS"
REPORTS=()
for item in "${SPAN_ITEMS[@]}"; do
  IFS=':' read -r layer_start layer_end label <<<"$item"
  if [[ -z "${layer_start:-}" || -z "${layer_end:-}" || -z "${label:-}" ]]; then
    echo "invalid span item '$item'; expected START:END:LABEL" >&2
    exit 1
  fi

  span_dir="$OUT_DIR/$label"
  mkdir -p "$span_dir"
  report="$span_dir/report.json"
  log="$span_dir/run.log"
  REPORT="$report" LOG="$log" \
  KV_WARMUP_TOKENS="$KV_WARMUP_TOKENS" KV_WARMUP_CHUNK_TOKENS="$KV_WARMUP_CHUNK_TOKENS" \
  DIRECT_SPARSE_ATTN=0 DIRECT_SPARSE_PREFILL=0 \
  COMPACT_FLASH_ATTN="$COMPACT_FLASH_ATTN" \
  ALLOW_COMPACT_FLASH_AUTO="$ALLOW_COMPACT_FLASH_AUTO" \
  COMPACT_FLASH_NO_MASK="$COMPACT_FLASH_NO_MASK" \
  SELECTED_ROW_FLASH="$SELECTED_ROW_FLASH" \
  REQUIRE_COMPACT_FLASH_PROOF="$REQUIRE_COMPACT_FLASH_PROOF" \
  METAL_DISPATCH_LOG="$METAL_DISPATCH_LOG" \
  "$ROOT/scripts/glm52-phase-b-real-indexshare-parity.sh" \
    --stage-model "$STAGE_MODEL" \
    --model-id "$MODEL_ID" \
    --skippy-bench "$SKIPPY_BENCH_BIN" \
    --layer-start "$layer_start" \
    --layer-end "$layer_end" \
    --ctx-size "$CTX_SIZE" \
    --tokens "$TOKENS" \
    --position-start "$POSITION_START" \
    --n-batch "$N_BATCH" \
    --n-ubatch "$N_UBATCH" \
    --kv-warmup-tokens "$KV_WARMUP_TOKENS" \
    --kv-warmup-chunk-tokens "$KV_WARMUP_CHUNK_TOKENS" \
    --out-dir "$span_dir" \
    >"$span_dir/stdout.txt" \
    2>"$span_dir/stderr.txt"
  REPORTS+=("$label:$report")
done

NEGATIVE_STATUS="skipped"
NEGATIVE_STDERR=""
if [[ "$RUN_NEGATIVE" == "1" ]]; then
  negative_dir="$OUT_DIR/shared-only-negative"
  mkdir -p "$negative_dir"
  set +e
  NEGATIVE_ARGS=(
    glm-dsa-layer-microbench
    --stage-model "$STAGE_MODEL"
    --model-id "$MODEL_ID"
    --layer-start 3
    --layer-end 4
    --ctx-size "$CTX_SIZE"
    --tokens "$TOKENS"
    --position-start "$POSITION_START"
    --iterations 1
    --warmup 0
    --n-batch "$N_BATCH"
    --n-ubatch "$N_UBATCH"
    --direct-sparse-attn false
    --op-timing true
    --output "$negative_dir/report.json"
  )
  if [[ "$KV_WARMUP_TOKENS" != "0" ]]; then
    NEGATIVE_ARGS+=(--kv-warmup-tokens "$KV_WARMUP_TOKENS")
    if [[ -n "$KV_WARMUP_CHUNK_TOKENS" ]]; then
      NEGATIVE_ARGS+=(--kv-warmup-chunk-tokens "$KV_WARMUP_CHUNK_TOKENS")
    fi
  fi
  "$SKIPPY_BENCH_BIN" "${NEGATIVE_ARGS[@]}" \
    >"$negative_dir/stdout.txt" \
    2>"$negative_dir/stderr.txt"
  negative_rc=$?
  set -e
  NEGATIVE_STDERR="$negative_dir/stderr.txt"
  if [[ "$negative_rc" == "0" ]]; then
    echo "Shared-only missing-top-k negative case unexpectedly passed" >&2
    exit 1
  fi
  if ! grep -Eq \
    "GLM-DSA consumer slices require top-k sideband input|GLM_DSA split starts inside an IndexShare consumer group without top-k sideband input" \
    "$negative_dir/stderr.txt"; then
    echo "Shared-only missing-top-k negative case failed with unexpected error" >&2
    echo "stderr=$negative_dir/stderr.txt" >&2
    exit 1
  fi
  NEGATIVE_STATUS="passed"
fi

python3 - "$CONTRACT_JSON" "$MATRIX_JSON" "$NEGATIVE_STATUS" "$NEGATIVE_STDERR" "${REPORTS[@]}" <<'PY'
import json
import pathlib
import sys

contract_path = pathlib.Path(sys.argv[1])
matrix_path = pathlib.Path(sys.argv[2])
negative_status = sys.argv[3]
negative_stderr = sys.argv[4]
items = sys.argv[5:]

contract = json.loads(contract_path.read_text())
rows = []
failures = []
for item in items:
    label, path_str = item.split(":", 1)
    path = pathlib.Path(path_str)
    report = json.loads(path.read_text())
    comparison = report.get("comparison") or {}
    parity = comparison.get("parity") or {}
    guard = report.get("native_indexshare_guard") or {}
    baseline = comparison.get("baseline") or {}
    candidate = comparison.get("candidate") or {}
    candidate_ops = candidate.get("op_timing_summary") or {}
    baseline_timing = baseline.get("timing_summary") or {}
    candidate_timing = candidate.get("timing_summary") or {}
    row = {
        "label": label,
        "report": str(path),
        "parity_passed": bool(parity.get("passed")),
        "hidden_mismatches": parity.get("hidden_mismatches"),
        "sideband_mismatched_bytes": parity.get("sideband_mismatched_bytes"),
        "guard_passed": bool(guard.get("passed")),
        "full_layers": guard.get("full_layers"),
        "shared_layers": guard.get("shared_layers"),
        "shared_exec_with_input_top_k": guard.get("shared_exec_with_input_top_k"),
        "shared_exec_missing_input_top_k": guard.get("shared_exec_missing_input_top_k"),
        "candidate_indexer_topk_nodes": (candidate_ops.get("indexer_topk") or {}).get("nodes"),
        "candidate_indexer_nodes": (candidate_ops.get("indexer") or {}).get("nodes"),
        "candidate_top_k_nodes": (candidate_ops.get("top_k") or {}).get("nodes"),
        "baseline_mean_ms": baseline_timing.get("mean_ms"),
        "candidate_mean_ms": candidate_timing.get("mean_ms"),
    }
    if row["baseline_mean_ms"] and row["candidate_mean_ms"]:
        row["diagnostic_ratio"] = row["baseline_mean_ms"] / row["candidate_mean_ms"]
    rows.append(row)
    if not row["parity_passed"]:
        failures.append(f"{label}: parity failed")
    if row["hidden_mismatches"] not in (0, None):
        failures.append(f"{label}: hidden mismatches {row['hidden_mismatches']}")
    if row["sideband_mismatched_bytes"] not in (0, None):
        failures.append(f"{label}: sideband mismatch {row['sideband_mismatched_bytes']}")
    if not row["guard_passed"]:
        failures.append(f"{label}: native guard failed")
    if row["shared_exec_missing_input_top_k"] not in (0, None):
        failures.append(f"{label}: shared consumer missed top-k")
    if row["candidate_indexer_topk_nodes"] not in (0, None):
        failures.append(f"{label}: candidate recomputed indexer_topk")
    if row["candidate_indexer_nodes"] not in (0, None):
        failures.append(f"{label}: candidate recomputed indexer")
    if row["candidate_top_k_nodes"] not in (0, None):
        failures.append(f"{label}: candidate recomputed top_k")

summary = {
    "contract": {
        "path": str(contract_path),
        "architecture": contract.get("architecture"),
        "role_source": contract.get("role_source"),
        "layer_count": contract.get("layer_count"),
        "effective_decoder_layers": contract.get("effective_decoder_layers"),
        "nextn_predict_layers": contract.get("nextn_predict_layers"),
        "full_layers": contract.get("full_layers"),
        "shared_layers": contract.get("shared_layers"),
        "generation_policy_required": contract.get("generation_policy_required"),
        "generation_policy": contract.get("generation_policy"),
        "generation_thresholds": contract.get("generation_thresholds"),
    },
    "rows": rows,
    "negative_missing_top_k": {
        "status": negative_status,
        "stderr": negative_stderr or None,
    },
    "passed": not failures and negative_status in ("passed", "skipped"),
    "failures": failures,
}
matrix_path.write_text(json.dumps(summary, indent=2) + "\n")
if failures:
    for failure in failures:
        print(failure, file=sys.stderr)
    raise SystemExit(1)

print("GLM-5.2 Phase A/B real-artifact matrix passed")
print(f"contract={contract_path}")
print(f"matrix={matrix_path}")
for row in rows:
    ratio = row.get("diagnostic_ratio")
    ratio_text = f" ratio={ratio:.3f}x" if ratio else ""
    print(
        f"{row['label']}: full={row['full_layers']} shared={row['shared_layers']} "
        f"topk_recompute={row['candidate_indexer_topk_nodes']} "
        f"shared_with_topk={row['shared_exec_with_input_top_k']}{ratio_text}"
    )
print(f"negative_missing_top_k={negative_status}")
PY
