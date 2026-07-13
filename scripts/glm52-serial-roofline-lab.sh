#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-lab@192.168.0.10}"
REMOTE_ROOT="${REMOTE_ROOT:-/Users/lab/src/mesh-llm-codex}"
REMOTE_STAGE_CONFIG="${REMOTE_STAGE_CONFIG:-/Users/lab/models/skippy-runtime-bench/glm52-moe-baseline-manual/stage-0}"
LOCAL_STAGE_CONFIG="${LOCAL_STAGE_CONFIG:-/Volumes/External/skippy-runtime-bench/glm52-moe-baseline-manual/stage-1}"
REMOTE_RESULT_ROOT="${REMOTE_RESULT_ROOT:-/Users/lab/models/skippy-runtime-bench/glm52-serial-roofline-20260713}"
TMUX_SOCKET="${TMUX_SOCKET:-glm52-roofline}"
WAIT_SECONDS="${WAIT_SECONDS:-1200}"
MAX_TOKENS="${MAX_TOKENS:-256}"

usage() {
  cat <<'EOF'
Usage: scripts/glm52-serial-roofline-lab.sh COMMAND [ARM]

Commands:
  start ARM  Start micstudio stage 0, then studio54 stage 1, and wait for OpenAI readiness.
  bench ARM  Warm the running arm and record one depth-one OpenAI measurement.
  run ARM    Stop, start, and benchmark one arm.
  status     Show both stage processes and recent logs.
  stop       Stop only this script's two dedicated tmux servers.

Arms:
  baseline, attention, attn-q-b, attn-pre-cache, attn-terminal, attn-value-tail, attn-output,
  dense-ffn, routed-moe, shared-expert, all-ffn, attention-routed, all

The topology, model package, layer ranges, context, F16 activation wire,
OpenAI route, and one-lane execution remain fixed. Bypass arms are intentionally
output-inexact diagnostics and must never be used as quality evidence.
EOF
}

arm_flags() {
  case "$1" in
    baseline)          printf '0 0 0 0 0 0 0 0 0\n' ;;
    attention)         printf '1 0 0 0 0 0 0 0 0\n' ;;
    attn-q-b)          printf '0 1 0 0 0 0 0 0 0\n' ;;
    attn-pre-cache)    printf '0 0 1 0 0 0 0 0 0\n' ;;
    attn-terminal)     printf '0 0 0 1 0 0 0 0 0\n' ;;
    attn-value-tail)   printf '0 0 0 0 1 0 0 0 0\n' ;;
    attn-output)       printf '0 0 0 0 0 1 0 0 0\n' ;;
    dense-ffn)         printf '0 0 0 0 0 0 1 0 0\n' ;;
    routed-moe)        printf '0 0 0 0 0 0 0 1 0\n' ;;
    shared-expert)     printf '0 0 0 0 0 0 0 0 1\n' ;;
    all-ffn)           printf '0 0 0 0 0 0 1 1 1\n' ;;
    attention-routed) printf '1 0 0 0 0 0 0 1 0\n' ;;
    all)               printf '1 0 0 0 0 0 1 1 1\n' ;;
    *)
      echo "invalid roofline arm: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

remote_login() {
  ssh -tt "$REMOTE_HOST" "/bin/zsh -ilc '$1'"
}

stop_stages() {
  /opt/homebrew/bin/tmux -L "$TMUX_SOCKET" kill-server 2>/dev/null || true
  remote_login "/opt/homebrew/bin/tmux -L $TMUX_SOCKET kill-server 2>/dev/null || true" || true
  sleep 2

  if pgrep -f "skippy-server serve-binary.*${LOCAL_STAGE_CONFIG}/stage.json" >/dev/null; then
    echo "studio54 stage survived tmux shutdown" >&2
    return 1
  fi
  # The fixed remote config path is intentionally interpolated by this controller.
  # shellcheck disable=SC2029
  if ssh "$REMOTE_HOST" "pgrep -f 'skippy-server serve-binary.*${REMOTE_STAGE_CONFIG}/stage.json'" >/dev/null; then
    echo "micstudio stage survived tmux shutdown" >&2
    return 1
  fi
}

network_preflight() {
  ping -c 2 -W 1000 192.168.0.10 >/dev/null
  remote_login "ping -c 2 -W 1000 192.168.0.5 >/dev/null"
}

write_remote_start() {
  local arm="$1"
  local attention="$2"
  local after_q_b="$3"
  local before_cache="$4"
  local attn_terminal="$5"
  local attn_value_tail="$6"
  local attn_output="$7"
  local dense_ffn="$8"
  local routed_moe="$9"
  local shared_expert="${10}"

  # This writes a fully resolved, operator-inspectable remote launch script.
  # shellcheck disable=SC2087
  ssh "$REMOTE_HOST" 'cat > /tmp/codex-glm52-roofline-start.sh && chmod +x /tmp/codex-glm52-roofline-start.sh' <<EOF
#!/usr/bin/env zsh
set -euo pipefail
export TERM=xterm-256color
tmux_bin=/opt/homebrew/bin/tmux
"\$tmux_bin" -L "$TMUX_SOCKET" kill-server 2>/dev/null || true
"\$tmux_bin" -L "$TMUX_SOCKET" new-session -d -s stage0 "/bin/zsh -ilc 'cd $REMOTE_ROOT && CONFIG_ROOT=$REMOTE_STAGE_CONFIG ROOFLINE_BYPASS_ATTENTION=$attention ROOFLINE_BYPASS_AFTER_Q_B=$after_q_b ROOFLINE_BYPASS_BEFORE_CACHE=$before_cache ROOFLINE_BYPASS_ATTN_TERMINAL=$attn_terminal ROOFLINE_BYPASS_ATTN_VALUE_TAIL=$attn_value_tail ROOFLINE_BYPASS_ATTN_OUTPUT=$attn_output ROOFLINE_BYPASS_DENSE_FFN=$dense_ffn ROOFLINE_BYPASS_ROUTED_MOE=$routed_moe ROOFLINE_BYPASS_SHARED_EXPERT=$shared_expert scripts/glm52-selected-row-lab-server.sh 0 indirect-tiled 2>&1 | tee /tmp/glm52-roofline-$arm-stage0.log'"
"\$tmux_bin" -L "$TMUX_SOCKET" has-session -t stage0
EOF
}

start_local_stage() {
  local arm="$1"
  local attention="$2"
  local after_q_b="$3"
  local before_cache="$4"
  local attn_terminal="$5"
  local attn_value_tail="$6"
  local attn_output="$7"
  local dense_ffn="$8"
  local routed_moe="$9"
  local shared_expert="${10}"

  /opt/homebrew/bin/tmux -L "$TMUX_SOCKET" kill-server 2>/dev/null || true
  /opt/homebrew/bin/tmux -L "$TMUX_SOCKET" new-session -d -s stage1 \
    "/bin/zsh -ilc 'cd $ROOT && CONFIG_ROOT=$LOCAL_STAGE_CONFIG ROOFLINE_BYPASS_ATTENTION=$attention ROOFLINE_BYPASS_AFTER_Q_B=$after_q_b ROOFLINE_BYPASS_BEFORE_CACHE=$before_cache ROOFLINE_BYPASS_ATTN_TERMINAL=$attn_terminal ROOFLINE_BYPASS_ATTN_VALUE_TAIL=$attn_value_tail ROOFLINE_BYPASS_ATTN_OUTPUT=$attn_output ROOFLINE_BYPASS_DENSE_FFN=$dense_ffn ROOFLINE_BYPASS_ROUTED_MOE=$routed_moe ROOFLINE_BYPASS_SHARED_EXPERT=$shared_expert scripts/glm52-selected-row-lab-server.sh 1 indirect-tiled 2>&1 | tee /tmp/glm52-roofline-$arm-stage1.log'"
}

wait_ready() {
  local arm="$1"
  local deadline=$((SECONDS + WAIT_SECONDS))

  while ((SECONDS < deadline)); do
    if ssh "$REMOTE_HOST" 'curl -fsS --max-time 2 http://127.0.0.1:9337/v1/models' \
      >"/tmp/glm52-roofline-$arm-models.json" 2>/dev/null; then
      if ! rg -q 'peer=Some\(192\.168\.0\.10:' "/tmp/glm52-roofline-$arm-stage1.log"; then
        echo "stage 1 did not record a direct private-LAN stage-0 peer" >&2
        return 1
      fi
      echo "arm=$arm openai=ready route=192.168.0.10->192.168.0.5"
      return 0
    fi

    if ! /opt/homebrew/bin/tmux -L "$TMUX_SOCKET" has-session -t stage1 2>/dev/null; then
      echo "studio54 stage exited before readiness" >&2
      tail -n 100 "/tmp/glm52-roofline-$arm-stage1.log" >&2
      return 1
    fi
    # The dedicated socket name is intentionally interpolated locally.
    # shellcheck disable=SC2029
    if ! ssh "$REMOTE_HOST" "/opt/homebrew/bin/tmux -L '$TMUX_SOCKET' has-session -t stage0" 2>/dev/null; then
      echo "micstudio stage exited before readiness" >&2
      remote_login "tail -n 100 /tmp/glm52-roofline-$arm-stage0.log" >&2
      return 1
    fi
    sleep 15
  done

  echo "timed out waiting for arm $arm" >&2
  return 1
}

start_arm() {
  local arm="$1"
  local attention after_q_b before_cache attn_terminal attn_value_tail attn_output dense_ffn routed_moe shared_expert
  read -r attention after_q_b before_cache attn_terminal attn_value_tail attn_output dense_ffn routed_moe shared_expert < <(arm_flags "$arm")

  network_preflight
  write_remote_start "$arm" "$attention" "$after_q_b" "$before_cache" "$attn_terminal" "$attn_value_tail" "$attn_output" "$dense_ffn" "$routed_moe" "$shared_expert"
  remote_login "/tmp/codex-glm52-roofline-start.sh"
  start_local_stage "$arm" "$attention" "$after_q_b" "$before_cache" "$attn_terminal" "$attn_value_tail" "$attn_output" "$dense_ffn" "$routed_moe" "$shared_expert"
  wait_ready "$arm"
}

bench_arm() {
  local arm="$1"
  arm_flags "$arm" >/dev/null
  local run_stamp
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"

  # This writes a fully resolved, operator-inspectable remote benchmark script.
  # shellcheck disable=SC2087
  ssh "$REMOTE_HOST" 'cat > /tmp/codex-glm52-roofline-bench.sh && chmod +x /tmp/codex-glm52-roofline-bench.sh' <<EOF
#!/usr/bin/env zsh
set -euo pipefail
cd "$REMOTE_ROOT"
mkdir -p "$REMOTE_RESULT_ROOT"
PROMPT_LIMIT=1 MAX_TOKENS=16 CONCURRENCY_DEPTH=1 \\
  SESSION_PREFIX="glm52-roofline-$arm-warm-$run_stamp" \\
  scripts/glm52-multi-session-openai-bench.sh \\
  "$REMOTE_RESULT_ROOT/$arm-warmup-$run_stamp.json"
PROMPT_LIMIT=1 MAX_TOKENS="$MAX_TOKENS" CONCURRENCY_DEPTH=1 \\
  SESSION_PREFIX="glm52-roofline-$arm-measured-$run_stamp" \\
  scripts/glm52-multi-session-openai-bench.sh \\
  "$REMOTE_RESULT_ROOT/$arm-$run_stamp.json"
jq '{
  arm: "$arm",
  artifact: "$REMOTE_RESULT_ROOT/$arm-$run_stamp.json",
  completion_tokens: .results[0].completion_tokens,
  finish_reason: .results[0].finish_reason,
  elapsed_ms: .results[0].elapsed_ms,
  ttft_ms: .results[0].ttft_ms,
  completion_tok_s: .summary.completion_tok_s,
  post_ttft_tok_s: (
    if (.results[0].completion_tokens > 1 and .results[0].elapsed_ms > .results[0].ttft_ms)
    then ((.results[0].completion_tokens - 1) / ((.results[0].elapsed_ms - .results[0].ttft_ms) / 1000))
    else null end
  ),
  post_ttft_ms_per_token: (
    if (.results[0].completion_tokens > 1 and .results[0].elapsed_ms > .results[0].ttft_ms)
    then ((.results[0].elapsed_ms - .results[0].ttft_ms) / (.results[0].completion_tokens - 1))
    else null end
  ),
  error: .results[0].error
}' "$REMOTE_RESULT_ROOT/$arm-$run_stamp.json"
EOF
  remote_login "/tmp/codex-glm52-roofline-bench.sh"
}

show_status() {
  echo "studio54"
  pgrep -fl "skippy-server serve-binary.*${LOCAL_STAGE_CONFIG}/stage.json" || true
  # Paths are fixed diagnostic names without whitespace.
  # shellcheck disable=SC2012
  ls -t /tmp/glm52-roofline-*-stage1.log 2>/dev/null | head -1 | xargs tail -n 20 2>/dev/null || true
  echo "micstudio"
  remote_login "pgrep -fl skippy-server || true; latest=\$(ls -t /tmp/glm52-roofline-*-stage0.log 2>/dev/null | head -1); [[ -n \$latest ]] && tail -n 20 \$latest || true"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

command_name="$1"
arm="${2:-}"

case "$command_name" in
  start)
    [[ -n "$arm" ]] || { usage >&2; exit 2; }
    start_arm "$arm"
    ;;
  bench)
    [[ -n "$arm" ]] || { usage >&2; exit 2; }
    bench_arm "$arm"
    ;;
  run)
    [[ -n "$arm" ]] || { usage >&2; exit 2; }
    stop_stages
    start_arm "$arm"
    bench_arm "$arm"
    ;;
  status)
    show_status
    ;;
  stop)
    stop_stages
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
