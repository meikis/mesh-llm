#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/glm52-selected-row-openai-bench.sh VARIANT [OUTPUT_DIR]

Run one warm-up and three measured OpenAI samples against the fixed GLM-5.2
two-node lab. This must run on micstudio while stage 0 owns 127.0.0.1:9337.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

variant="$1"
output_dir="${2:-/Users/lab/models/skippy-runtime-bench/glm52-selected-row-${variant}-$(date -u +%Y%m%dT%H%M%SZ)}"
root="/Users/lab/src/mesh-llm-codex"

host_name="$(hostname -s | tr '[:upper:]' '[:lower:]')"
if [[ "$host_name" != micstudio* ]]; then
  echo "benchmark must run on micstudio, not $host_name" >&2
  exit 1
fi

cd "$root"
mkdir -p "$output_dir"

bench=(
  target/release/skippy-bench chat-corpus
  --base-url http://127.0.0.1:9337/v1
  --model meshllm/GLM-5.2-Q2_K-MTP-Q8-layers
  --prompt-corpus crates/skippy-bench/corpora/glm_dsa_long_context_coding_prompts.jsonl
  --prompt-id glm-dsa-long-code-8k-001
  --max-tokens 64
  --concurrency-depth 1
  --stream
  --include-usage true
  --request-timeout-secs 1800
  --session-prefix glm52-selected-row-fixed
  --temperature 0
  --top-p 1
  --top-k 1
  --seed 42
  --enable-thinking false
)

diagnostic_bench=(
  target/release/skippy-bench chat-corpus
  --base-url http://127.0.0.1:9337/v1
  --model meshllm/GLM-5.2-Q2_K-MTP-Q8-layers
  --prompt-corpus crates/skippy-bench/corpora/speculative_coding_prompts.jsonl
  --prompt-id spec-code-001
  --max-tokens 64
  --concurrency-depth 1
  --include-usage true
  --request-timeout-secs 1800
  --session-prefix glm52-selected-row-diagnostic
  --temperature 0
  --top-p 1
  --top-k 1
  --seed 42
  --enable-thinking false
)

run_diagnostic() {
  local report="$output_dir/glm52-selected-row-${variant}-diagnostic.json"

  printf 'sample=diagnostic variant=%s started=%s\n' "$variant" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${diagnostic_bench[@]}" --output "$report"
  jq '{
    prompt_tokens: .results[0].prompt_tokens,
    completion_tokens: .results[0].completion_tokens,
    elapsed_ms: .results[0].elapsed_ms,
    output_sha256: .results[0].output_sha256,
    spec_proposal_attempts: .results[0].spec_proposal_attempts,
    spec_proposal_misses: .results[0].spec_proposal_misses,
    spec_windows: .results[0].spec_windows,
    spec_proposed: .results[0].spec_proposed,
    spec_accepted: .results[0].spec_accepted,
    spec_rejected: .results[0].spec_rejected,
    spec_accept_rate: .results[0].spec_accept_rate,
    errors: .summary.errors
  }' "$report"
  printf 'sample=diagnostic variant=%s finished=%s\n' "$variant" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_sample() {
  local sample="$1"
  local report="$output_dir/glm52-selected-row-${variant}-${sample}.json"

  printf 'sample=%s variant=%s started=%s\n' "$sample" "$variant" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "${bench[@]}" --output "$report"
  jq '{
    prompt_tokens: .results[0].prompt_tokens,
    completion_tokens: .results[0].completion_tokens,
    elapsed_ms: .results[0].elapsed_ms,
    ttft_ms: .results[0].ttft_ms,
    output_sha256: .results[0].output_sha256,
    decode_tok_s: (
      if .results[0].completion_tokens != null
          and .results[0].completion_tokens > 1
          and .results[0].elapsed_ms != null
          and .results[0].ttft_ms != null
          and .results[0].elapsed_ms > .results[0].ttft_ms
      then (.results[0].completion_tokens - 1) / ((.results[0].elapsed_ms - .results[0].ttft_ms) / 1000)
      else null
      end
    ),
    spec_windows: .results[0].spec_windows,
    spec_proposed: .results[0].spec_proposed,
    spec_accepted: .results[0].spec_accepted,
    spec_rejected: .results[0].spec_rejected,
    spec_accept_rate: .results[0].spec_accept_rate,
    errors: .summary.errors,
    error: .results[0].error
  }' "$report"
  printf 'sample=%s variant=%s finished=%s\n' "$sample" "$variant" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

run_diagnostic
run_sample warm
for sample in 1 2 3; do
  run_sample "$sample"
done

jq -s '{
  variant: $variant,
  samples: map({
    prompt_tokens: .results[0].prompt_tokens,
    completion_tokens: .results[0].completion_tokens,
    elapsed_ms: .results[0].elapsed_ms,
    ttft_ms: .results[0].ttft_ms,
    output_sha256: .results[0].output_sha256,
    decode_tok_s: (
      if .results[0].completion_tokens != null
          and .results[0].completion_tokens > 1
          and .results[0].elapsed_ms != null
          and .results[0].ttft_ms != null
          and .results[0].elapsed_ms > .results[0].ttft_ms
      then (.results[0].completion_tokens - 1) / ((.results[0].elapsed_ms - .results[0].ttft_ms) / 1000)
      else null
      end
    ),
    spec_windows: .results[0].spec_windows,
    spec_proposed: .results[0].spec_proposed,
    spec_accepted: .results[0].spec_accepted,
    spec_rejected: .results[0].spec_rejected,
    spec_accept_rate: .results[0].spec_accept_rate,
    errors: .summary.errors,
    error: .results[0].error
  }),
  diagnostic: ($diagnostic[0].results[0] | {
    prompt_tokens,
    completion_tokens,
    elapsed_ms,
    output_sha256,
    spec_proposal_attempts,
    spec_proposal_misses,
    spec_windows,
    spec_proposed,
    spec_accepted,
    spec_rejected,
    spec_accept_rate,
    error
  }),
  decode_tok_s_mean: (
    map(
      if .results[0].completion_tokens != null
          and .results[0].completion_tokens > 1
          and .results[0].elapsed_ms != null
          and .results[0].ttft_ms != null
          and .results[0].elapsed_ms > .results[0].ttft_ms
      then (.results[0].completion_tokens - 1) / ((.results[0].elapsed_ms - .results[0].ttft_ms) / 1000)
      else empty
      end
    )
    | if length > 0 then add / length else null end
  )
}' --arg variant "$variant" --slurpfile diagnostic "$output_dir/glm52-selected-row-${variant}-diagnostic.json" \
  "$output_dir/glm52-selected-row-${variant}-1.json" \
  "$output_dir/glm52-selected-row-${variant}-2.json" \
  "$output_dir/glm52-selected-row-${variant}-3.json" \
  | tee "$output_dir/summary.json"

printf 'output_dir=%s\n' "$output_dir"
