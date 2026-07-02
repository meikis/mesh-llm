#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/glm-dsa-moe-producer-sweep.sh --stage-model PATH --output-dir DIR [options]

Runs GLM-DSA routed-MoE weighted-parallel comparisons over producer-looking
layers in a Skippy layer package and writes an aggregate summary.json.

Options:
  --stage-model PATH       Layer package directory containing model-package.json.
  --output-dir DIR         Directory for per-layer compare.json files and summary.json.
  --layers CSV             Explicit layer list, e.g. 6,10,30,34. Defaults to producer-looking layers.
  --max-layers N           Limit the number of selected layers after discovery.
  --iterations N           Measured iterations per layer. Default: 6.
  --warmup N               Warmup iterations per layer. Default: 1.
  --ctx-size N             Context size. Default: 131072.
  --tokens N               Decode tokens. Default: 1.
  --position-start N       Decode position. Default: 4096.
  --kv-warmup-tokens N     KV warmup tokens. Default: 4096.
  --n-batch N              llama n_batch. Default: 512.
  --n-ubatch N             llama n_ubatch. Default: 512.
  --bench-bin PATH         skippy-bench binary. Default: target/debug/skippy-bench.
  --dry-run                Print discovered commands without running them.
  -h, --help               Show this help.
EOF
}

stage_model=""
output_dir=""
layers_csv=""
max_layers=""
iterations=6
warmup=1
ctx_size=131072
tokens=1
position_start=4096
kv_warmup_tokens=4096
n_batch=512
n_ubatch=512
bench_bin="target/debug/skippy-bench"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage-model)
      stage_model="${2:?missing --stage-model value}"
      shift 2
      ;;
    --output-dir)
      output_dir="${2:?missing --output-dir value}"
      shift 2
      ;;
    --layers)
      layers_csv="${2:?missing --layers value}"
      shift 2
      ;;
    --max-layers)
      max_layers="${2:?missing --max-layers value}"
      shift 2
      ;;
    --iterations)
      iterations="${2:?missing --iterations value}"
      shift 2
      ;;
    --warmup)
      warmup="${2:?missing --warmup value}"
      shift 2
      ;;
    --ctx-size)
      ctx_size="${2:?missing --ctx-size value}"
      shift 2
      ;;
    --tokens)
      tokens="${2:?missing --tokens value}"
      shift 2
      ;;
    --position-start)
      position_start="${2:?missing --position-start value}"
      shift 2
      ;;
    --kv-warmup-tokens)
      kv_warmup_tokens="${2:?missing --kv-warmup-tokens value}"
      shift 2
      ;;
    --n-batch)
      n_batch="${2:?missing --n-batch value}"
      shift 2
      ;;
    --n-ubatch)
      n_ubatch="${2:?missing --n-ubatch value}"
      shift 2
      ;;
    --bench-bin)
      bench_bin="${2:?missing --bench-bin value}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
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

if [[ -z "$stage_model" || -z "$output_dir" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$stage_model/model-package.json" ]]; then
  echo "missing model-package.json under stage model: $stage_model" >&2
  exit 1
fi

if [[ ! -x "$bench_bin" ]]; then
  echo "skippy-bench binary is not executable: $bench_bin" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

if [[ -n "$layers_csv" ]]; then
  IFS=',' read -r -a layers <<<"$layers_csv"
else
  while IFS= read -r layer; do
    layers+=("$layer")
  done < <(
    jq -r '
      .layers[]
      | select(.layer_index != null)
      | select((.tensor_count // 0) > 18)
      | .layer_index
    ' "$stage_model/model-package.json"
  )
fi

if [[ -n "$max_layers" ]]; then
  layers=("${layers[@]:0:max_layers}")
fi

if [[ ${#layers[@]} -eq 0 ]]; then
  echo "no GLM-DSA producer-looking layers selected" >&2
  exit 1
fi

mkdir -p "$output_dir"

build_dir="${LLAMA_STAGE_BUILD_DIR:-}"
if [[ -z "$build_dir" ]]; then
  build_dir="$(LLAMA_STAGE_BACKEND=metal scripts/build-llama.sh --print-build-dir)"
fi

printf '%s\n' "${layers[@]}" >"$output_dir/layers.txt"

for layer in "${layers[@]}"; do
  layer_dir="$output_dir/layer-${layer}"
  mkdir -p "$layer_dir"
  cmd=(
    "$bench_bin" glm-dsa-layer-microbench
    --stage-model "$stage_model"
    --layer-start "$layer"
    --layer-end "$((layer + 1))"
    --ctx-size "$ctx_size"
    --tokens "$tokens"
    --position-start "$position_start"
    --kv-warmup-tokens "$kv_warmup_tokens"
    --iterations "$iterations"
    --warmup "$warmup"
    --n-batch "$n_batch"
    --n-ubatch "$n_ubatch"
    --direct-sparse-attn true
    --compact-flash-attn true
    --allow-compact-flash-auto
    --direct-sparse-prefill true
    --fused-sparse-mask true
    --metal-topk-moe-route-fusion true
    --op-timing true
    --metal-dispatch-log true
    --compare-moe-down-weighted-parallel
    --output "$layer_dir/compare.json"
  )
  printf '%q ' "${cmd[@]}" >"$layer_dir/command.sh"
  printf '\n' >>"$layer_dir/command.sh"
  echo "layer $layer -> $layer_dir/compare.json"
  if [[ "$dry_run" == 0 ]]; then
    LLAMA_STAGE_BACKEND=metal LLAMA_STAGE_BUILD_DIR="$build_dir" "${cmd[@]}"
  fi
done

if [[ "$dry_run" == 1 ]]; then
  echo "dry run complete: $output_dir"
  exit 0
fi

jq -s '
  def mean($xs): if ($xs | length) == 0 then null else (($xs | add) / ($xs | length)) end;
  def layer_result:
    .comparison as $c
    | {
        layer: .layer_start,
        layer_end: .layer_end,
        baseline_ms: $c.baseline.timing_summary.mean_ms,
        candidate_ms: $c.candidate.timing_summary.mean_ms,
        delta_ms: ($c.baseline.timing_summary.mean_ms - $c.candidate.timing_summary.mean_ms),
        win_pct: (
          (($c.baseline.timing_summary.mean_ms - $c.candidate.timing_summary.mean_ms)
          / $c.baseline.timing_summary.mean_ms) * 100.0
        ),
        parity_passed: $c.parity.passed,
        hidden_mismatches: $c.parity.hidden_mismatches
      };
  (map(layer_result) | sort_by(.layer)) as $layers
  | {
      candidate: "moe_down_weighted_parallel",
      layer_count: ($layers | length),
      parity_passed: (all($layers[]; .parity_passed)),
      hidden_mismatches: ($layers | map(.hidden_mismatches) | add),
      mean_win_pct: mean($layers | map(.win_pct)),
      total_baseline_ms: ($layers | map(.baseline_ms) | add),
      total_candidate_ms: ($layers | map(.candidate_ms) | add),
      total_win_pct: (
        (($layers | map(.baseline_ms) | add) - ($layers | map(.candidate_ms) | add))
        / ($layers | map(.baseline_ms) | add) * 100.0
      ),
      winning_layers: ($layers | map(select(.win_pct > 0)) | length),
      losing_layers: ($layers | map(select(.win_pct < 0)) | length),
      layers: $layers
    }
' "$output_dir"/layer-*/compare.json >"$output_dir/summary.json"

jq -r '
  "candidate=\(.candidate) layers=\(.layer_count) parity=\(.parity_passed) total_win_pct=\(.total_win_pct) mean_win_pct=\(.mean_win_pct) wins=\(.winning_layers) losses=\(.losing_layers) hidden_mismatches=\(.hidden_mismatches)"
' "$output_dir/summary.json"

echo "$output_dir"
