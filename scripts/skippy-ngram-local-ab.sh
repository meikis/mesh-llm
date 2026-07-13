#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIPPY_SERVER_BIN="${SKIPPY_SERVER_BIN:-$ROOT_DIR/target/debug/skippy-server}"
: "${MODEL_PATH:?set MODEL_PATH to a complete GGUF}"
: "${LAYER_COUNT:?set LAYER_COUNT to the model block count}"
: "${SPLIT_LAYER:?set SPLIT_LAYER to an internal layer boundary}"
: "${ACTIVATION_WIDTH:?set ACTIVATION_WIDTH to the model embedding width}"

MODEL_ID="${MODEL_ID:-local/ngram-verify-smoke}"
CTX_SIZE="${CTX_SIZE:-4096}"
MAX_TOKENS="${MAX_TOKENS:-64}"
NGRAM_SIZE="${NGRAM_SIZE:-3}"
NGRAM_WINDOW_MIN="${NGRAM_WINDOW_MIN:-16}"
NGRAM_WINDOW_MAX="${NGRAM_WINDOW_MAX:-16}"
STAGE0_PORT="${STAGE0_PORT:-19131}"
STAGE1_PORT="${STAGE1_PORT:-19132}"
OPENAI_PORT="${OPENAI_PORT:-19337}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skippy-ngram-ab.XXXXXX")"
STAGE0_PID=""
STAGE1_PID=""

cleanup_processes() {
  if [[ -n "$STAGE0_PID" ]]; then
    kill -TERM "$STAGE0_PID" 2>/dev/null || true
    wait "$STAGE0_PID" 2>/dev/null || true
    STAGE0_PID=""
  fi
  if [[ -n "$STAGE1_PID" ]]; then
    kill -TERM "$STAGE1_PID" 2>/dev/null || true
    wait "$STAGE1_PID" 2>/dev/null || true
    STAGE1_PID=""
  fi
}

cleanup() {
  cleanup_processes
  if [[ "${KEEP_WORK_DIR:-0}" != "1" ]]; then
    rm -rf "$WORK_DIR"
  else
    printf 'kept work directory: %s\n' "$WORK_DIR" >&2
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if [[ ! -x "$SKIPPY_SERVER_BIN" ]]; then
  printf 'skippy-server binary is not executable: %s\n' "$SKIPPY_SERVER_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  printf 'model does not exist: %s\n' "$MODEL_PATH" >&2
  exit 1
fi
if (( SPLIT_LAYER <= 0 || SPLIT_LAYER >= LAYER_COUNT )); then
  printf 'SPLIT_LAYER must satisfy 0 < split < LAYER_COUNT\n' >&2
  exit 1
fi
if (( NGRAM_WINDOW_MIN <= 0 || NGRAM_WINDOW_MIN > NGRAM_WINDOW_MAX )); then
  printf 'ngram window must satisfy 0 < min <= max\n' >&2
  exit 1
fi

cat > "$WORK_DIR/stage0.json" <<EOF
{
  "bind_addr": "127.0.0.1:$STAGE0_PORT",
  "cache_type_k": "f16",
  "cache_type_v": "f16",
  "ctx_size": $CTX_SIZE,
  "downstream": {
    "stage_id": "stage-1",
    "stage_index": 1,
    "endpoint": "127.0.0.1:$STAGE1_PORT"
  },
  "filter_tensors_on_load": true,
  "lane_count": 1,
  "layer_end": $SPLIT_LAYER,
  "layer_start": 0,
  "load_mode": "runtime-slice",
  "model_id": "$MODEL_ID",
  "model_path": "$MODEL_PATH",
  "n_gpu_layers": -1,
  "projector_path": null,
  "run_id": "ngram-local-ab",
  "stage_id": "stage-0",
  "stage_index": 0,
  "topology_id": "ngram-two-stage-loopback",
  "upstream": null
}
EOF

cat > "$WORK_DIR/stage1.json" <<EOF
{
  "bind_addr": "127.0.0.1:$STAGE1_PORT",
  "cache_type_k": "f16",
  "cache_type_v": "f16",
  "ctx_size": $CTX_SIZE,
  "downstream": null,
  "filter_tensors_on_load": true,
  "lane_count": 1,
  "layer_end": $LAYER_COUNT,
  "layer_start": $SPLIT_LAYER,
  "load_mode": "runtime-slice",
  "model_id": "$MODEL_ID",
  "model_path": "$MODEL_PATH",
  "n_gpu_layers": -1,
  "projector_path": null,
  "run_id": "ngram-local-ab",
  "stage_id": "stage-1",
  "stage_index": 1,
  "topology_id": "ngram-two-stage-loopback",
  "upstream": {
    "stage_id": "stage-0",
    "stage_index": 0,
    "endpoint": "127.0.0.1:$STAGE0_PORT"
  }
}
EOF

python3 - "$MODEL_ID" "$MAX_TOKENS" "$WORK_DIR/request.json" "$WORK_DIR/warmup-request.json" <<'PY'
import json
import sys

model_id, max_tokens, path, warmup_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
prompt = """Complete with exactly twelve more identical lines:
let result = source + 7;
let result = source + 7;
let result = source + 7;
let result = source + 7;
let result = source + 7;
"""
request = {
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "seed": 42,
    "max_tokens": max_tokens,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(request, handle)

warmup = {
    "model": model_id,
    "messages": [{
        "role": "user",
        "content": """Continue the repeated pattern:
let warm = input * 3;
let warm = input * 3;
let warm = input * 3;
let warm = input * 3;
let warm = input * 3;
""",
    }],
    "temperature": 0,
    "seed": 7,
    "max_tokens": min(max_tokens, 32),
}
with open(warmup_path, "w", encoding="utf-8") as handle:
    json.dump(warmup, handle)
PY

wait_for_port() {
  local port="$1"
  local pid="$2"
  local log="$3"
  local attempt
  for attempt in $(seq 1 300); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -80 "$log" >&2 || true
      return 1
    fi
    if nc -z 127.0.0.1 "$port" 2>/dev/null; then
      return 0
    fi
    sleep 0.1
  done
  tail -80 "$log" >&2 || true
  return 1
}

wait_for_openai() {
  local pid="$1"
  local log="$2"
  local attempt
  for attempt in $(seq 1 300); do
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -80 "$log" >&2 || true
      return 1
    fi
    if curl -fsS --max-time 1 "http://127.0.0.1:$OPENAI_PORT/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.1
  done
  tail -80 "$log" >&2 || true
  return 1
}

run_condition() {
  local label="$1"
  local enabled="$2"
  local stage0_log="$WORK_DIR/$label-stage0.log"
  local stage1_log="$WORK_DIR/$label-stage1.log"
  set --

  "$SKIPPY_SERVER_BIN" serve-binary \
    --config "$WORK_DIR/stage1.json" \
    --activation-width "$ACTIVATION_WIDTH" \
    --bind-addr "127.0.0.1:$STAGE1_PORT" \
    --telemetry-level summary >"$stage1_log" 2>&1 &
  STAGE1_PID=$!
  wait_for_port "$STAGE1_PORT" "$STAGE1_PID" "$stage1_log"

  if [[ "$enabled" == "1" ]]; then
    set -- "$@" --openai-ngram-simple --openai-ngram-size-n "$NGRAM_SIZE"
    if (( NGRAM_WINDOW_MIN < NGRAM_WINDOW_MAX )); then
      set -- "$@" --openai-adaptive-speculative-window
    fi
  fi

  "$SKIPPY_SERVER_BIN" serve-binary \
    --config "$WORK_DIR/stage0.json" \
    --activation-width "$ACTIVATION_WIDTH" \
    --bind-addr "127.0.0.1:$STAGE0_PORT" \
    --openai-bind-addr "127.0.0.1:$OPENAI_PORT" \
    --openai-default-max-tokens "$MAX_TOKENS" \
    --openai-prefill-chunk-policy fixed \
    --openai-prefill-chunk-size 256 \
    --openai-speculative-window-min "$NGRAM_WINDOW_MIN" \
    --openai-speculative-window "$NGRAM_WINDOW_MAX" \
    "$@" \
    --telemetry-level summary >"$stage0_log" 2>&1 &
  STAGE0_PID=$!
  wait_for_openai "$STAGE0_PID" "$stage0_log"

  curl -fsS --max-time 120 \
    "http://127.0.0.1:$OPENAI_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$WORK_DIR/warmup-request.json" >/dev/null

  curl -fsS --max-time 120 \
    -o "$WORK_DIR/$label-response.json" \
    -w '%{time_total}\n' \
    "http://127.0.0.1:$OPENAI_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$WORK_DIR/request.json" > "$WORK_DIR/$label-time.txt"
  cleanup_processes
}

run_condition baseline 0
run_condition ngram 1

python3 - "$WORK_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
baseline = json.loads((root / "baseline-response.json").read_text())
ngram = json.loads((root / "ngram-response.json").read_text())
baseline_time = float((root / "baseline-time.txt").read_text().strip())
ngram_time = float((root / "ngram-time.txt").read_text().strip())

def outcome(response):
    choice = response["choices"][0]
    return {
        "content": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "completion_tokens": response["usage"]["completion_tokens"],
    }

baseline_outcome = outcome(baseline)
ngram_outcome = outcome(ngram)
if baseline_outcome != ngram_outcome:
    raise SystemExit("baseline and ngram outputs differ")

tokens = baseline_outcome["completion_tokens"]
summary = {
    "output_parity": True,
    "completion_tokens": tokens,
    "baseline_seconds": baseline_time,
    "ngram_seconds": ngram_time,
    "baseline_tokens_per_second": tokens / baseline_time,
    "ngram_tokens_per_second": tokens / ngram_time,
    "speedup": baseline_time / ngram_time,
    "spec": ngram.get("skippy", {}).get("spec"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
