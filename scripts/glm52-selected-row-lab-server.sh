#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/glm52-selected-row-lab-server.sh STAGE VARIANT

Launch one stage of the fixed GLM-5.2 selected-row flash A/B topology.

  STAGE    0 on micstudio, 1 on studio54
  VARIANT  baseline, selected-row, selected-row-ngram, indirect-tiled,
           or retained

Runtime overrides:
  CONFIG_ROOT, STAGE_MAX_INFLIGHT, OPENAI_GENERATION_CONCURRENCY,
  OPENAI_DEFAULT_MAX_TOKENS, SKIPPY_DECODE_BATCH_COLLECTION_US,
  ROOFLINE_BYPASS_ATTENTION, ROOFLINE_BYPASS_AFTER_Q_B,
  ROOFLINE_BYPASS_BEFORE_CACHE,
  ROOFLINE_BYPASS_ATTN_TERMINAL,
  ROOFLINE_BYPASS_ATTN_VALUE_TAIL, ROOFLINE_BYPASS_ATTN_OUTPUT,
  ROOFLINE_BYPASS_DENSE_FFN,
  ROOFLINE_BYPASS_ROUTED_MOE, ROOFLINE_BYPASS_SHARED_EXPERT
EOF
}

if [[ $# -ne 2 ]]; then
  usage >&2
  exit 2
fi

stage="$1"
variant="$2"

case "$variant" in
  baseline|selected-row|selected-row-ngram|indirect-tiled|retained) ;;
  *)
    echo "invalid variant: $variant" >&2
    usage >&2
    exit 2
    ;;
esac

case "$stage" in
  0)
    expected_host="micstudio"
    root="/Users/lab/src/mesh-llm-codex"
    hf_home="/Users/lab/models/huggingface"
    default_config_root="/Users/lab/models/skippy-runtime-bench/glm52-moe-baseline-manual/stage-0"
    ;;
  1)
    expected_host="studio54"
    root="/Users/jdumay/.codex/worktrees/c72f/mesh-llm"
    hf_home="/Volumes/External/models/huggingface"
    default_config_root="/Volumes/External/skippy-runtime-bench/glm52-moe-baseline-manual/stage-1"
    ;;
  *)
    echo "invalid stage: $stage" >&2
    usage >&2
    exit 2
    ;;
esac

config_root="${CONFIG_ROOT:-$default_config_root}"
stage_max_inflight="${STAGE_MAX_INFLIGHT:-1}"
openai_generation_concurrency="${OPENAI_GENERATION_CONCURRENCY:-1}"
openai_default_max_tokens="${OPENAI_DEFAULT_MAX_TOKENS:-256}"
roofline_bypass_attention="${ROOFLINE_BYPASS_ATTENTION:-0}"
roofline_bypass_after_q_b="${ROOFLINE_BYPASS_AFTER_Q_B:-0}"
roofline_bypass_before_cache="${ROOFLINE_BYPASS_BEFORE_CACHE:-0}"
roofline_bypass_attn_terminal="${ROOFLINE_BYPASS_ATTN_TERMINAL:-0}"
roofline_bypass_attn_value_tail="${ROOFLINE_BYPASS_ATTN_VALUE_TAIL:-0}"
roofline_bypass_attn_output="${ROOFLINE_BYPASS_ATTN_OUTPUT:-0}"
roofline_bypass_dense_ffn="${ROOFLINE_BYPASS_DENSE_FFN:-0}"
roofline_bypass_routed_moe="${ROOFLINE_BYPASS_ROUTED_MOE:-0}"
roofline_bypass_shared_expert="${ROOFLINE_BYPASS_SHARED_EXPERT:-0}"

for flag in \
  "$roofline_bypass_attention" \
  "$roofline_bypass_after_q_b" \
  "$roofline_bypass_before_cache" \
  "$roofline_bypass_attn_terminal" \
  "$roofline_bypass_attn_value_tail" \
  "$roofline_bypass_attn_output" \
  "$roofline_bypass_dense_ffn" \
  "$roofline_bypass_routed_moe" \
  "$roofline_bypass_shared_expert"; do
  if [[ "$flag" != "0" && "$flag" != "1" ]]; then
    echo "roofline bypass values must be 0 or 1, got: $flag" >&2
    exit 2
  fi
done

host_name="$(hostname -s | tr '[:upper:]' '[:lower:]')"
if [[ "$host_name" != "$expected_host"* ]]; then
  echo "stage $stage must run on $expected_host, not $host_name" >&2
  exit 1
fi

cd "$root"
export PATH="/opt/homebrew/bin:$PATH"
export HF_HOME="$hf_home"
export SKIPPY_BINARY_WARM_PRECONNECT=1
export SKIPPY_NATIVE_MTP_ENABLED=0

# Throughput runs must not inherit diagnostics or rejected approximation knobs.
unset \
  GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS \
  LLAMA_GLM_DSA_INDEXSHARE_EXEC_LOG \
  SKIPPY_GLM_DSA_DISABLE_SELECTED_ROW_FLASH \
  SKIPPY_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH \
  LLAMA_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH_TILED \
  LLAMA_GLM_DSA_EXPERIMENTAL_MUL_MV_SHAPE_POLICY \
  LLAMA_GLM_DSA_MUL_MV_Q8_0_NSG \
  LLAMA_GLM_DSA_MUL_MV_Q3_K_NSG \
  LLAMA_GLM_DSA_MUL_MV_Q4_K_NSG \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTENTION \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_AFTER_Q_B \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_BEFORE_CACHE \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_TERMINAL \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_VALUE_TAIL \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_OUTPUT \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_DENSE_FFN \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_ROUTED_MOE \
  LLAMA_GLM_DSA_ROOFLINE_BYPASS_SHARED_EXPERT \
  GGML_GLM_DSA_EXPERIMENTAL_NATIVE_MOE_DOWN \
  SKIPPY_GLM_DSA_LOG_COMPACT_FLASH_POLICY \
  SKIPPY_GLM_DSA_LOG_DIRECT_SPARSE_DECISIONS \
  SKIPPY_GLM_DSA_LOG_METAL_DISPATCH \
  SKIPPY_GLM_DSA_OP_TIMING \
  SKIPPY_GLM_DSA_TENSOR_TRACE \
  SKIPPY_GLM_DSA_TENSOR_TRACE_STATS \
  SKIPPY_GLM_DSA_TENSOR_TRACE_FILTER \
  SKIPPY_GLM_DSA_TENSOR_TRACE_VALUES \
  SKIPPY_GLM_DSA_TENSOR_TRACE_NODES

if [[ "${TRACE_ROUTE_TENSORS:-0}" == "1" ]]; then
  export SKIPPY_GLM_DSA_TENSOR_TRACE=1
  export SKIPPY_GLM_DSA_TENSOR_TRACE_STATS=0
  export SKIPPY_GLM_DSA_TENSOR_TRACE_FILTER=ffn_moe_topk,ffn_moe_route_weights
  export SKIPPY_GLM_DSA_TENSOR_TRACE_VALUES=8
  export SKIPPY_GLM_DSA_TENSOR_TRACE_NODES=128
fi

# Keep every A/B arm explicit. Selected-row and tiled execution now default on
# in llama.cpp, so an unset environment would no longer be a valid control.
export SKIPPY_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH=0
export LLAMA_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH_TILED=0
export LLAMA_GLM_DSA_EXPERIMENTAL_MUL_MV_SHAPE_POLICY=0
export GGML_GLM_DSA_EXPERIMENTAL_NATIVE_MOE_DOWN=0
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTENTION="$roofline_bypass_attention"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_AFTER_Q_B="$roofline_bypass_after_q_b"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_BEFORE_CACHE="$roofline_bypass_before_cache"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_TERMINAL="$roofline_bypass_attn_terminal"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_VALUE_TAIL="$roofline_bypass_attn_value_tail"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_OUTPUT="$roofline_bypass_attn_output"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_DENSE_FFN="$roofline_bypass_dense_ffn"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_ROUTED_MOE="$roofline_bypass_routed_moe"
export LLAMA_GLM_DSA_ROOFLINE_BYPASS_SHARED_EXPERT="$roofline_bypass_shared_expert"

case "$variant" in
  selected-row|selected-row-ngram)
    export SKIPPY_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH=1
    ;;
  indirect-tiled)
    export SKIPPY_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH=1
    export LLAMA_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH_TILED=1
    ;;
  retained)
    export SKIPPY_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH=1
    export LLAMA_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH_TILED=1
    export LLAMA_GLM_DSA_EXPERIMENTAL_MUL_MV_SHAPE_POLICY=3
    export GGML_GLM_DSA_EXPERIMENTAL_NATIVE_MOE_DOWN=1
    ;;
esac

args=(
  target/release/skippy-server serve-binary
  --config "$config_root/stage.json"
  --topology "$config_root/topology.json"
  --activation-width 6144
  --activation-wire-dtype f16
  --telemetry-queue-capacity 32768
  --telemetry-level summary
  --max-inflight "$stage_max_inflight"
  --downstream-wire-delay-ms 0
)

if [[ "$stage" == "0" ]]; then
  args+=(
    --openai-bind-addr 127.0.0.1:9337
    --openai-model-id meshllm/GLM-5.2-Q2_K-MTP-Q8-layers
    --openai-default-max-tokens "$openai_default_max_tokens"
    --openai-generation-concurrency "$openai_generation_concurrency"
  )
  if [[ "$variant" == "selected-row-ngram" ]]; then
    args+=(
      --openai-ngram-simple
      --openai-ngram-size-n "${NGRAM_SIZE_N:-12}"
      --openai-speculative-window-min "${NGRAM_WINDOW:-8}"
      --openai-speculative-window "${NGRAM_WINDOW:-8}"
    )
  fi
fi

printf 'host=%s stage=%s variant=%s config_root=%s max_inflight=%s generation_concurrency=%s decode_batch_collection_us=%s selected_row_flash=%s indirect_tiled=%s projection_policy=%s native_moe_down=%s ngram_size_n=%s ngram_window=%s native_mtp=%s trace_route_tensors=%s roofline_attention=%s roofline_after_q_b=%s roofline_before_cache=%s roofline_attn_terminal=%s roofline_attn_value_tail=%s roofline_attn_output=%s roofline_dense_ffn=%s roofline_routed_moe=%s roofline_shared_expert=%s\n' \
  "$host_name" "$stage" "$variant" \
  "$config_root" \
  "$stage_max_inflight" \
  "$openai_generation_concurrency" \
  "${SKIPPY_DECODE_BATCH_COLLECTION_US:-0}" \
  "${SKIPPY_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH:-0}" \
  "${LLAMA_GLM_DSA_EXPERIMENTAL_SELECTED_ROW_FLASH_TILED:-0}" \
  "$LLAMA_GLM_DSA_EXPERIMENTAL_MUL_MV_SHAPE_POLICY" \
  "$GGML_GLM_DSA_EXPERIMENTAL_NATIVE_MOE_DOWN" \
  "${NGRAM_SIZE_N:-0}" \
  "${NGRAM_WINDOW:-0}" \
  "$SKIPPY_NATIVE_MTP_ENABLED" \
  "${SKIPPY_GLM_DSA_TENSOR_TRACE:-0}" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTENTION" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_AFTER_Q_B" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_BEFORE_CACHE" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_TERMINAL" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_VALUE_TAIL" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_ATTN_OUTPUT" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_DENSE_FFN" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_ROUTED_MOE" \
  "$LLAMA_GLM_DSA_ROOFLINE_BYPASS_SHARED_EXPERT"
exec "${args[@]}"
