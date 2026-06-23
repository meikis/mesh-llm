#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/skippy-bf16-to-quant-layer-package.sh \
    --bf16-root PATH \
    --quant-root PATH \
    --package-dir PATH \
    --work-dir PATH \
    --quant Q2_K-MTP-Q8 \
    --output-basename GLM-5.2-Q2_K-MTP-Q8 \
    --model-id meshllm/GLM-5.2-Q2_K-MTP-Q8-GGUF:Q2_K-MTP-Q8 \
    --source-repo meshllm/GLM-5.2-Q2_K-MTP-Q8-GGUF \
    --source-revision local \
    [--bf16-prefix BF16] \
    [--quant-prefix Q2_K-MTP-Q8] \
    [--source-file Q2_K-MTP-Q8/GLM-5.2-Q2_K-MTP-Q8-00001-of-00306.gguf] \
    [--backend llama-api] \
    [--stages 2] \
    [--nthreads N] \
    [--watchdog-seconds N] \
    [--verify-llama-load] \
    [--keep-quant]

Builds a disposable quantized GGUF artifact from a reusable split BF16 GGUF
source, then writes and preflights a Skippy layer package from the quant.

Environment overrides:
  SKIPPY_QUANTIZE_BIN       default: target/release/skippy-quantize
  SKIPPY_MODEL_PACKAGE_BIN  default: target/release/skippy-model-package
EOF
}

bf16_root=""
bf16_prefix="BF16"
quant_root=""
quant_prefix=""
package_dir=""
work_dir=""
quant=""
output_basename=""
model_id=""
source_repo=""
source_revision=""
source_file=""
backend="llama-api"
stages=""
nthreads=""
watchdog_seconds=""
verify_llama_load=false
keep_quant=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bf16-root)
      bf16_root="$2"
      shift 2
      ;;
    --bf16-prefix)
      bf16_prefix="$2"
      shift 2
      ;;
    --quant-root)
      quant_root="$2"
      shift 2
      ;;
    --quant-prefix)
      quant_prefix="$2"
      shift 2
      ;;
    --package-dir)
      package_dir="$2"
      shift 2
      ;;
    --work-dir)
      work_dir="$2"
      shift 2
      ;;
    --quant)
      quant="$2"
      shift 2
      ;;
    --output-basename)
      output_basename="$2"
      shift 2
      ;;
    --model-id)
      model_id="$2"
      shift 2
      ;;
    --source-repo)
      source_repo="$2"
      shift 2
      ;;
    --source-revision)
      source_revision="$2"
      shift 2
      ;;
    --source-file)
      source_file="$2"
      shift 2
      ;;
    --backend)
      backend="$2"
      shift 2
      ;;
    --stages)
      stages="$2"
      shift 2
      ;;
    --nthreads)
      nthreads="$2"
      shift 2
      ;;
    --watchdog-seconds)
      watchdog_seconds="$2"
      shift 2
      ;;
    --verify-llama-load)
      verify_llama_load=true
      shift
      ;;
    --keep-quant)
      keep_quant=true
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

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "missing required argument: $name" >&2
    usage >&2
    exit 2
  fi
}

require_value "--bf16-root" "$bf16_root"
require_value "--quant-root" "$quant_root"
require_value "--package-dir" "$package_dir"
require_value "--work-dir" "$work_dir"
require_value "--quant" "$quant"
require_value "--output-basename" "$output_basename"
require_value "--model-id" "$model_id"
require_value "--source-repo" "$source_repo"
require_value "--source-revision" "$source_revision"

if [[ -z "$quant_prefix" ]]; then
  quant_prefix="$quant"
fi

skippy_quantize_bin="${SKIPPY_QUANTIZE_BIN:-target/release/skippy-quantize}"
skippy_model_package_bin="${SKIPPY_MODEL_PACKAGE_BIN:-target/release/skippy-model-package}"

if [[ ! -x "$skippy_quantize_bin" ]]; then
  echo "missing executable: $skippy_quantize_bin" >&2
  echo "build it with: just skippy-quantize-standalone-release-build" >&2
  exit 1
fi

if [[ ! -x "$skippy_model_package_bin" ]]; then
  echo "missing executable: $skippy_model_package_bin" >&2
  echo "build it with: cargo build --release --locked -p skippy-model-package" >&2
  exit 1
fi

mkdir -p "$quant_root" "$package_dir" "$work_dir"

manifest="$work_dir/quant-manifest.json"
status_file="$work_dir/quant-status.json"
record_dir="$work_dir/records"
spool_dir="$work_dir/spool"
runner_work_dir="$work_dir/native-work"

echo "== init quant manifest =="
"$skippy_quantize_bin" init-quant \
  --source "$bf16_root" \
  --source-prefix "$bf16_prefix" \
  --target "$quant_root" \
  --target-prefix "$quant_prefix" \
  --output-basename "$output_basename" \
  --quant "$quant" \
  --window-size 1 \
  --manifest "$manifest"

echo "== quantize BF16 -> $quant =="
run_quant_args=(
  run-quant
  --manifest "$manifest"
  --backend "$backend"
  --work-dir "$runner_work_dir"
  --spool-dir "$spool_dir"
  --record-dir "$record_dir"
  --json-event-file "$status_file"
  --json-event-interval-seconds 120
  --json-event-window 8
)

if [[ -n "$nthreads" ]]; then
  run_quant_args+=(--nthreads "$nthreads")
fi

if [[ -n "$watchdog_seconds" ]]; then
  run_quant_args+=(--watchdog-seconds "$watchdog_seconds")
fi

"$skippy_quantize_bin" "${run_quant_args[@]}"

echo "== verify quant artifact =="
verify_args=(verify-job --manifest "$manifest")
if [[ "$verify_llama_load" == true ]]; then
  verify_args+=(--llama-load)
fi
"$skippy_quantize_bin" "${verify_args[@]}"

first_quant_shard="$quant_root/$quant_prefix/$output_basename-00001-of-"*
first_quant_shard_matches=($first_quant_shard)
if [[ ${#first_quant_shard_matches[@]} -ne 1 || ! -f "${first_quant_shard_matches[0]}" ]]; then
  echo "expected exactly one first quant shard matching: $first_quant_shard" >&2
  exit 1
fi
first_quant_shard="${first_quant_shard_matches[0]}"
first_quant_filename="$(basename "$first_quant_shard")"

if [[ -z "$source_file" ]]; then
  source_file="$quant_prefix/$first_quant_filename"
fi

echo "== write layer package =="
"$skippy_model_package_bin" write-package "$first_quant_shard" \
  --out-dir "$package_dir" \
  --model-id "$model_id" \
  --source-repo "$source_repo" \
  --source-revision "$source_revision" \
  --source-file "$source_file"

echo "== preflight layer package =="
preflight_args=(preflight "$package_dir" --verify-sha256)
if [[ -n "$stages" ]]; then
  preflight_args+=(--stages "$stages")
fi
"$skippy_model_package_bin" "${preflight_args[@]}"

if [[ "$keep_quant" == false ]]; then
  echo "== clean disposable quant staging =="
  rm -rf "$quant_root/$quant_prefix"
else
  echo "== kept quant artifact =="
  echo "$quant_root/$quant_prefix"
fi

echo "== done =="
echo "BF16 source:      $bf16_root/$bf16_prefix"
echo "Layer package:    $package_dir"
echo "Quant manifest:   $manifest"
echo "Quant status:     $status_file"
