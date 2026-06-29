#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/sync-lab-source.sh [--dry-run] [--source PATH] HOST [REMOTE_DIR]

Synchronize the current mesh-llm source tree to a lab host while preserving
remote build/runtime caches. The default remote directory is:

  /Users/lab/src/mesh-llm-codex

Examples:

  scripts/sync-lab-source.sh micstudio
  scripts/sync-lab-source.sh --dry-run micstudio /Users/lab/src/mesh-llm-codex
EOF
}

DRY_RUN=0
SOURCE="$ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --source)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
      SOURCE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

HOST="$1"
REMOTE_DIR="${2:-/Users/lab/src/mesh-llm-codex}"

SOURCE="$(cd "$SOURCE" && pwd -P)"

if [[ ! -f "$SOURCE/Cargo.toml" || ! -f "$SOURCE/justfile" ]]; then
  echo "source does not look like a mesh-llm checkout: $SOURCE" >&2
  exit 1
fi

RSYNC_ARGS=(
  -az
  --delete
  --delete-after
  --itemize-changes
  --stats
  --human-readable
)

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

# Protect prevents deleting existing remote cache trees. Hide prevents sending
# local cache trees. Keep protect before hide so receiver-side deletion checks
# see the cache rule before any sender-side exclusion.
CACHE_FILTERS=(
  '/target/***'
  '/.deps/***'
  '/.sccache/***'
  '/node_modules/***'
  '/website/node_modules/***'
  '/crates/mesh-llm-ui/node_modules/***'
)

FILTER_ARGS=()
for pattern in "${CACHE_FILTERS[@]}"; do
  FILTER_ARGS+=(--filter="P ${pattern}")
  FILTER_ARGS+=(--filter="H ${pattern}")
done
FILTER_ARGS+=(--filter='H /.git')
FILTER_ARGS+=(--filter='H /.git/***')

echo "syncing mesh-llm source"
echo "  source: $SOURCE/"
echo "  target: $HOST:$REMOTE_DIR/"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  mode:   dry run"
fi

REMOTE_DIR_QUOTED="$(printf "%q" "$REMOTE_DIR")"
ssh "$HOST" "mkdir -p ${REMOTE_DIR_QUOTED} && rm -rf ${REMOTE_DIR_QUOTED}/.git"

rsync "${RSYNC_ARGS[@]}" "${FILTER_ARGS[@]}" "$SOURCE/" "$HOST:$REMOTE_DIR/"
