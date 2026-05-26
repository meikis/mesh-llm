#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: scripts/publish-crates.sh [--dry-run] [--allow-dirty] [--sleep-seconds N]

Publishes the crates.io package chain in dependency order. Use --dry-run for
local and CI validation without uploading packages. --allow-dirty is accepted
only with --dry-run so local pre-commit validation can include uncommitted
manifest changes; real publishing always requires Cargo's clean-tree check.
USAGE
}

dry_run=0
allow_dirty=0
sleep_seconds=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            dry_run=1
            shift
            ;;
        --allow-dirty)
            allow_dirty=1
            shift
            ;;
        --sleep-seconds)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                usage
                exit 1
            fi
            sleep_seconds="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ "$allow_dirty" -eq 1 && "$dry_run" -eq 0 ]]; then
    echo "--allow-dirty is only supported together with --dry-run" >&2
    exit 1
fi

if [[ -z "$sleep_seconds" ]]; then
    if [[ "$dry_run" -eq 1 ]]; then
        sleep_seconds=0
    else
        sleep_seconds="${CRATES_IO_PUBLISH_SETTLE_SECONDS:-30}"
    fi
fi

if [[ "$dry_run" -eq 0 && -z "${CARGO_REGISTRY_TOKEN:-}" ]]; then
    echo "CARGO_REGISTRY_TOKEN is required for real crates.io publishing" >&2
    exit 1
fi

workspace_version="$(
    perl -ne '
        $in_workspace_package = 1 if /^\[workspace\.package\]/;
        $in_workspace_package = 0 if /^\[/ && !/^\[workspace\.package\]/;
        if ($in_workspace_package && /^\s*version\s*=\s*"([^"]+)"/) {
            print $1;
            exit;
        }
    ' Cargo.toml
)"

if [[ -z "$workspace_version" ]]; then
    echo "failed to read [workspace.package].version from Cargo.toml" >&2
    exit 1
fi

crate_version_published() {
    local crate="$1"
    local status
    if ! command -v curl >/dev/null 2>&1; then
        return 1
    fi
    status="$(
        curl \
            --fail \
            --silent \
            --show-error \
            --output /dev/null \
            --write-out '%{http_code}' \
            "https://crates.io/api/v1/crates/${crate}/${workspace_version}" \
            2>/dev/null || true
    )"
    [[ "$status" == "200" ]]
}

# Resolve a crate name to its `crates/<dir>/Cargo.toml` path. The dir
# name often equals the crate name but not always (e.g. mesh-client/
# hosts the `mesh-llm-client` crate).
crate_manifest_path() {
    local crate="$1"
    local direct="crates/${crate}/Cargo.toml"
    if [[ -f "$direct" ]]; then
        printf '%s\n' "$direct"
        return 0
    fi
    grep -l -E "^\s*name\s*=\s*\"${crate}\"\s*$" crates/*/Cargo.toml 2>/dev/null | head -n 1
}

# List the workspace-internal registry deps of $1 (one crate name per
# line). Driven directly off the crate's own Cargo.toml so it stays in
# sync as deps change. Reads `[dependencies]` and `[build-dependencies]`
# entries that carry both a `path = "../<dir>"` and a workspace-internal
# crate name on the same line.
unpublished_registry_deps() {
    local crate="$1"
    local cargo
    cargo="$(crate_manifest_path "$crate")"
    if [[ -z "$cargo" || ! -f "$cargo" ]]; then
        return 0
    fi
    python3 - "$cargo" <<'PY'
import re
import sys
import pathlib

cargo = pathlib.Path(sys.argv[1])
text = cargo.read_text()
section = None
in_deps = False
pkg_re = re.compile(r'package\s*=\s*"([^"]+)"')
path_re = re.compile(r'path\s*=\s*"\.\./([^"]+)"')
dep_line_re = re.compile(r'^\s*([a-zA-Z0-9_-]+)\s*=\s*\{(.*)\}\s*$')
section_re = re.compile(r'^\[([^\]]+)\]')
for line in text.splitlines():
    s = line.rstrip()
    m = section_re.match(s)
    if m:
        section = m.group(1)
        in_deps = (
            section in ("dependencies", "build-dependencies")
            or (section.startswith("target.") and ".dependencies" in section)
        )
        continue
    if not in_deps:
        continue
    dm = dep_line_re.match(s)
    if not dm:
        continue
    body = dm.group(2)
    pm = path_re.search(body)
    if not pm:
        continue
    pkg_m = pkg_re.search(body)
    name = pkg_m.group(1) if pkg_m else dm.group(1)
    print(name)
PY
}

should_skip_initial_dry_run() {
    local crate="$1"
    local dep
    while IFS= read -r dep; do
        [[ -n "$dep" ]] || continue
        if ! crate_version_published "$dep"; then
            echo "dry-run cannot verify ${crate} until ${dep}@${workspace_version} exists in crates.io"
            return 0
        fi
    done < <(unpublished_registry_deps "$crate")
    return 1
}

publish_crates=(
    mesh-llm-gpu-bench
    mesh-llm-guardrails
    mesh-llm-identity
    mesh-llm-plugin
    mesh-llm-protocol
    mesh-llm-routing
    mesh-llm-types
    mesh-llm-ui
    model-ref
    skippy-coordinator
    skippy-ffi
    skippy-metrics
    skippy-protocol
    skippy-topology
    mesh-mixture-of-agents
    model-artifact
    openai-frontend
    skippy-cache
    skippy-runtime
    mesh-llm-client
    mesh-llm-system
    model-hf
    model-resolver
    skippy-server
    mesh-llm-api-client
    mesh-llm-node
    model-package
    mesh-llm-api-server
    mesh-llm-host-runtime
)

# Crates whose `cargo publish` verify step builds native code that
# needs the patched llama.cpp static archives. We skip the verify step
# for these because the packaged tarball's build.rs can't find
# .deps/llama-build from inside target/package/. The release pipeline
# guards against actual build breakage by running
# `cargo build -p mesh-llm-ffi` etc. before this script ever runs.
crate_needs_no_verify() {
    case "$1" in
        skippy-ffi|skippy-runtime|skippy-server|skippy-cache|skippy-coordinator|skippy-topology|skippy-protocol|skippy-metrics|mesh-llm-system|mesh-llm-host-runtime|mesh-llm-node|model-package|model-resolver)
            return 0
            ;;
    esac
    return 1
}

for index in "${!publish_crates[@]}"; do
    crate="${publish_crates[$index]}"
    if [[ "$dry_run" -eq 1 ]] && should_skip_initial_dry_run "$crate"; then
        continue
    fi

    args=(publish --locked -p "$crate")
    if [[ "$dry_run" -eq 1 ]]; then
        args+=(--dry-run)
    fi
    if [[ "$allow_dirty" -eq 1 ]]; then
        args+=(--allow-dirty)
    fi
    if crate_needs_no_verify "$crate"; then
        args+=(--no-verify)
    fi

    echo "cargo ${args[*]}"
    cargo "${args[@]}"

    if [[ "$index" -lt "$((${#publish_crates[@]} - 1))" && "$sleep_seconds" -gt 0 ]]; then
        echo "waiting ${sleep_seconds}s for crates.io index propagation"
        sleep "$sleep_seconds"
    fi
done
