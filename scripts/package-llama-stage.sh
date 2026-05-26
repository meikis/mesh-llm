#!/usr/bin/env bash
#
# Package the patched llama.cpp static archives for the current backend
# into a release-asset-shaped tarball that `skippy-ffi/build.rs` can
# fetch from a GitHub release at consumer build time.
#
# Layout produced:
#
#   llama-stage-<target_triple>-<flavor>.tar.gz
#   llama-stage-<target_triple>-<flavor>.tar.gz.sha256
#
# Tarball contents (a single top-level directory named
# `<target_triple>-<flavor>` containing the cmake build outputs
# `skippy-ffi/build.rs` reads):
#
#   <target_triple>-<flavor>/
#     CMakeCache.txt
#     src/libllama.a
#     tools/mtmd/libmtmd.a
#     common/libllama-common.a
#     common/libllama-common-base.a
#     ggml/src/libggml.a
#     ggml/src/libggml-base.a
#     ggml/src/libggml-cpu.a
#     ggml/src/ggml-<backend>/libggml-<backend>.a    (if present)
#     ...
#
# Required inputs (env vars or args):
#   --backend <name>          metal | cpu | cuda | rocm | vulkan
#   --target  <triple>        e.g. aarch64-apple-darwin (default: host triple)
#   --build-dir <path>        path to the cmake build dir
#                             (default: $LLAMA_STAGE_BUILD_DIR or
#                              .deps/llama-build/build-stage-abi-<backend>)
#   --out <dir>               where to write the tarball
#                             (default: dist/llama-stage-static)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND="${LLAMA_STAGE_BACKEND:-cpu}"
TARGET_TRIPLE="${MESH_LLAMA_STAGE_TARGET:-}"
BUILD_DIR_INPUT=""
OUT_DIR="$REPO_ROOT/dist/llama-stage-static"

usage() {
    cat >&2 <<'EOF'
Usage: scripts/package-llama-stage.sh [options]

Options:
  --backend NAME       metal | cpu | cuda | rocm | vulkan
  --target TRIPLE      Rust target triple. Defaults to host triple.
  --build-dir PATH     CMake build dir. Defaults to
                       $LLAMA_STAGE_BUILD_DIR or
                       .deps/llama-build/build-stage-abi-<backend>.
  --out DIR            Output directory. Defaults to dist/llama-stage-static.
  -h, --help           Show this help.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="${2:?missing backend}"
            shift 2
            ;;
        --target)
            TARGET_TRIPLE="${2:?missing target}"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR_INPUT="${2:?missing build dir}"
            shift 2
            ;;
        --out)
            OUT_DIR="${2:?missing out dir}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

default_host_triple() {
    if command -v rustc >/dev/null 2>&1; then
        rustc -vV | awk '/^host:/ { print $2 }'
    fi
}

if [[ -z "$TARGET_TRIPLE" ]]; then
    TARGET_TRIPLE="$(default_host_triple)"
fi
if [[ -z "$TARGET_TRIPLE" ]]; then
    echo "could not infer target triple; pass --target" >&2
    exit 1
fi

if [[ -n "$BUILD_DIR_INPUT" ]]; then
    BUILD_DIR="$BUILD_DIR_INPUT"
elif [[ -n "${LLAMA_STAGE_BUILD_DIR:-}" ]]; then
    BUILD_DIR="$LLAMA_STAGE_BUILD_DIR"
else
    BUILD_DIR="$REPO_ROOT/.deps/llama-build/build-stage-abi-$BACKEND"
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "build dir not found: $BUILD_DIR" >&2
    echo "run 'just llama-build' (or pass --build-dir) before packaging." >&2
    exit 1
fi

artifact_id="$TARGET_TRIPLE-$BACKEND"
stage_dir="$OUT_DIR/$artifact_id"

rm -rf "$stage_dir"
mkdir -p "$stage_dir"

# Mandatory files (build will fail without these).
mandatory=(
    "CMakeCache.txt"
    "src/libllama.a"
    "common/libllama-common.a"
    "common/libllama-common-base.a"
    "ggml/src/libggml.a"
    "ggml/src/libggml-base.a"
    "ggml/src/libggml-cpu.a"
)

# Optional files (present only for some backends/configurations).
optional=(
    "tools/mtmd/libmtmd.a"
    "ggml/src/ggml-blas/libggml-blas.a"
    "ggml/src/ggml-metal/libggml-metal.a"
    "ggml/src/ggml-cuda/libggml-cuda.a"
    "ggml/src/ggml-hip/libggml-hip.a"
    "ggml/src/ggml-vulkan/libggml-vulkan.a"
)

for f in "${mandatory[@]}"; do
    src="$BUILD_DIR/$f"
    if [[ ! -f "$src" ]]; then
        echo "missing required artifact: $src" >&2
        exit 1
    fi
    dest="$stage_dir/$f"
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
done

for f in "${optional[@]}"; do
    src="$BUILD_DIR/$f"
    if [[ -f "$src" ]]; then
        dest="$stage_dir/$f"
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
    fi
done

mkdir -p "$OUT_DIR"
tarball_name="llama-stage-$artifact_id.tar.gz"
tarball_path="$OUT_DIR/$tarball_name"

(
    cd "$OUT_DIR"
    tar czf "$tarball_name" "$artifact_id"
)

(
    cd "$OUT_DIR"
    shasum -a 256 "$tarball_name" > "$tarball_name.sha256"
)

echo "packaged llama-stage tarball:"
echo "  artifact_id: $artifact_id"
echo "  tarball:     $tarball_path"
echo "  sha256:      $tarball_path.sha256"
echo "  size:        $(du -h "$tarball_path" | awk '{print $1}')"
