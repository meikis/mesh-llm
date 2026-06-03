#!/usr/bin/env bash

set -euo pipefail

# Runtime releases currently live on the prerelease channel. Keep install.sh as
# the stable bundled-runtime installer and make this script runtime-first.
MESH_LLM_INSTALL_PRERELEASE="${MESH_LLM_INSTALL_PRERELEASE:-1}"

source_installer_helpers() {
    if [[ -n "${MESH_LLM_RUNTIME_INSTALL_HELPERS:-}" ]]; then
        # shellcheck source=/dev/null
        . "$MESH_LLM_RUNTIME_INSTALL_HELPERS"
        return 0
    fi

    local source_path="${BASH_SOURCE[0]-}"
    local script_dir
    if [[ -n "$source_path" && "$source_path" == */* ]]; then
        script_dir="$(cd "$(dirname "$source_path")" && pwd)"
        if [[ -f "$script_dir/install.sh" ]]; then
            # shellcheck source=install.sh
            . "$script_dir/install.sh"
            return 0
        fi
    fi

    local repo="${MESH_LLM_INSTALL_REPO:-Mesh-LLM/mesh-llm}"
    local ref="${MESH_LLM_INSTALL_REF:-main}"
    local helper_file

    if ! command -v curl >/dev/null 2>&1; then
        echo "error: required command not found: curl" >&2
        exit 1
    fi
    if ! command -v mktemp >/dev/null 2>&1; then
        echo "error: required command not found: mktemp" >&2
        exit 1
    fi

    helper_file="$(mktemp)"
    curl -fsSL "https://raw.githubusercontent.com/${repo}/${ref}/install.sh" -o "$helper_file"
    # shellcheck source=/dev/null
    . "$helper_file"
    rm -f "$helper_file"
}

source_installer_helpers

usage_runtime() {
    cat <<EOF
Usage: install-runtime.sh [--pre-release] [--stable] [--service] [--no-start-service]

Options:
  --pre-release              Install the latest published GitHub prerelease. This is the default.
  --stable                   Install the latest stable release. Requires runtime release assets.
  --service                  Install a per-user background service for this platform.
  --no-start-service         Install the service files but do not start them yet.
  -h, --help                 Show this help text.

Environment overrides:
  MESH_LLM_INSTALL_DIR
  MESH_LLM_INSTALL_FLAVOR
  MESH_LLM_INSTALL_PRERELEASE=0
  MESH_LLM_INSTALL_REF=main
  MESH_LLM_INSTALL_REPO=Mesh-LLM/mesh-llm
  MESH_LLM_INSTALL_SERVICE=1
  MESH_LLM_INSTALL_SERVICE_START=0
EOF
}

parse_runtime_args() {
    while (($# > 0)); do
        case "$1" in
            --pre-release)
                INSTALL_PRERELEASE=1
                ;;
            --stable)
                INSTALL_PRERELEASE=0
                ;;
            --service)
                INSTALL_SERVICE=1
                ;;
            --service-args)
                echo "error: background services now run \`mesh-llm serve\` and load startup models from $MESH_CONFIG_FILE" >&2
                echo "Add startup models under [[models]] instead of passing custom service args." >&2
                exit 1
                ;;
            --no-start-service)
                INSTALL_SERVICE_START=0
                ;;
            -h|--help)
                usage_runtime
                exit 0
                ;;
            *)
                echo "error: unknown argument: $1" >&2
                echo >&2
                usage_runtime >&2
                exit 1
                ;;
        esac
        shift
    done
}

download_native_runtime_manifest() {
    local tmp_dir="$1"
    local manifest_path="$tmp_dir/native-runtimes.json"
    local manifest_url

    if [[ -f "$manifest_path" ]]; then
        return 0
    fi
    if ! manifest_url="$(release_url "native-runtimes.json")"; then
        return 1
    fi

    echo "Downloading $manifest_url"
    curl -fsSL "$manifest_url" -o "$manifest_path" 2>/dev/null
}

download_release_asset() {
    local asset="$1"
    local archive="$2"
    local url

    if ! url="$(release_url "$asset")"; then
        return 1
    fi

    echo "Downloading $url"
    curl -fsSL "$url" -o "$archive" 2>/dev/null
}

download_runtime_binary_archive() {
    local tmp_dir="$1"
    local requested_asset="$2"
    local requested_archive="$tmp_dir/$requested_asset"
    local fallback_asset="mesh-bundle.tar.gz"
    local fallback_archive="$tmp_dir/$fallback_asset"

    DOWNLOADED_ASSET="$requested_asset"
    DOWNLOADED_ARCHIVE="$requested_archive"

    if download_release_asset "$requested_asset" "$requested_archive"; then
        return 0
    fi

    if [[ "$requested_asset" != "$fallback_asset" ]] &&
        download_release_asset "$fallback_asset" "$fallback_archive"; then
        echo "Using runtime-enabled mesh bundle because $requested_asset was not available."
        DOWNLOADED_ASSET="$fallback_asset"
        DOWNLOADED_ARCHIVE="$fallback_archive"
        return 0
    fi

    echo "error: could not download runtime release archive: $requested_asset or $fallback_asset" >&2
    return 1
}

install_native_runtime_required() {
    local tmp_dir="$1"
    local manifest_path="$tmp_dir/native-runtimes.json"
    local binary="$INSTALL_DIR/mesh-llm"

    if [[ ! -x "$binary" ]]; then
        echo "error: mesh-llm binary was not installed at $binary" >&2
        return 1
    fi
    if ! "$binary" runtime install --help >/dev/null 2>&1; then
        echo "error: installed mesh-llm does not support native runtime install" >&2
        return 1
    fi
    if [[ ! -f "$manifest_path" ]]; then
        echo "error: native runtime manifest was not downloaded" >&2
        return 1
    fi

    "$binary" runtime install --manifest "$manifest_path"
    "$binary" runtime prune --active-only || true
}

main_runtime() {
    parse_runtime_args "$@"
    if [[ -n "$INSTALL_SERVICE_ARGS" ]]; then
        echo "error: background services now run \`mesh-llm serve\` and load startup models from $MESH_CONFIG_FILE" >&2
        echo "Add startup models under [[models]] instead of using MESH_LLM_INSTALL_SERVICE_ARGS." >&2
        exit 1
    fi
    need_cmd curl
    need_cmd tar
    need_cmd mktemp

    local flavor
    flavor="$(choose_flavor)"
    local asset
    asset="$(asset_name "$flavor")"

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    local tmp_dir_escaped
    printf -v tmp_dir_escaped '%q' "$tmp_dir"
    trap "rm -rf -- $tmp_dir_escaped" EXIT

    echo "Installing runtime flavor: $flavor"
    if bool_is_true "$INSTALL_PRERELEASE"; then
        echo "Release channel: prerelease"
    else
        echo "Release channel: stable"
    fi

    if ! download_native_runtime_manifest "$tmp_dir"; then
        echo "error: native runtime manifest was not available for this release." >&2
        echo "Use install.sh for bundled-runtime stable releases, or retry with --pre-release." >&2
        exit 1
    fi
    download_runtime_binary_archive "$tmp_dir" "$asset"

    tar -xzf "$DOWNLOADED_ARCHIVE" -C "$tmp_dir"

    if [[ ! -d "$tmp_dir/mesh-bundle" ]]; then
        echo "error: release archive did not contain mesh-bundle/" >&2
        exit 1
    fi

    install_bundle "$tmp_dir/mesh-bundle"
    install_native_runtime_required "$tmp_dir"

    echo "Installed $DOWNLOADED_ASSET and native runtime to $INSTALL_DIR"

    if bool_is_true "$INSTALL_SERVICE"; then
        echo
        install_service
    fi

    if ! path_contains_install_dir; then
        echo
        echo "$INSTALL_DIR is not on your PATH."
        echo "Add it with one of these commands:"
        echo
        echo "bash:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
        echo
        echo "zsh:"
        echo "  echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc"
        echo "  source ~/.zshrc"
    fi
}

if [[ "${BASH_SOURCE[0]-}" == "$0" || ( -z "${BASH_SOURCE[0]-}" && "$0" == "bash" ) ]]; then
    main_runtime "$@"
fi
