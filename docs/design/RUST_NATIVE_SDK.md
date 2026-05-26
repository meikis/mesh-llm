# Rust Native SDK: in-process mesh node from cargo

## Status

Trial implementation working on `micn/native-sdk-cargo-publish`. A Rust
app outside the workspace calls `mesh_llm_api_server::MeshNode::builder()`
and `OwnerKeypair::generate()` directly, with skippy-ffi fetching
prebuilt patched-llama.cpp static archives from a tarball URL at build
time. End-to-end verified locally.

What remains is publish-side plumbing (release pipeline ships the
tarballs as assets per matrix cell; crates.io publish chain completes
for `mesh-llm-host-runtime` / skippy crates / `mesh-llm-api-server`).

## Goal

A Rust application adds mesh-llm to `Cargo.toml`, runs `cargo build`, and
gets a real in-process mesh node — same outcome as Swift and Kotlin SDKs:

```toml
[dependencies]
mesh-llm-api-server = { version = "0.66", features = ["host-runtime"] }
```

```rust
let node = mesh_llm_api_server::MeshNode::builder()
    .identity(owner)
    .join(invite)
    .build()?;
node.start().await?;
// real iroh peer in this process, optional local serving
```

No CMake on the consumer's machine. No source build of patched llama.cpp.
No separate `mesh-llm` daemon. No FFI wrappers in consumer code.

## How Swift and Kotlin do it

Both ship a prebuilt artifact containing patched llama.cpp + skippy +
mesh-llm host runtime, compiled as a **static archive** that exposes a
UniFFI-generated C ABI:

- **Swift:** `MeshLLMFFI.xcframework.zip` on each GitHub release.
  SwiftPM `.binaryTarget(url:, checksum:)` downloads at resolve time;
  inside is a Mach-O `.a` static archive per Apple slice. Static-linked
  into the consumer's app binary.
- **Kotlin:** `libmeshllm_ffi.so` per Android ABI inside an AAR on
  GitHub Packages Maven. JVM loads at runtime.
- **Node.js:** prebuilt N-API `.node` addon via npm.

In all three, the consumer's app *links/loads a single prebuilt
artifact* through their language's native package channel. The native
runtime runs in-process inside the consumer's app.

## Why Rust does it differently (and better)

Rust has one option Swift and Kotlin don't: **it can link Rust source
to Rust source through cargo directly.** The mesh-llm public API surface
(`mesh-llm-api-server`, `mesh-llm-host-runtime`, `MeshNode::builder()`,
…) is already normal Rust code. A Rust consumer doesn't need any C ABI,
UniFFI, or hand-written FFI wrappers to call it — they just `cargo add`
the crate and call Rust functions.

What Rust *does* need from a published artifact is the same thing Swift
and Kotlin need: **prebuilt patched llama.cpp + skippy static
archives**, so the consumer's `cargo build` doesn't have to run cmake
and rebuild llama.cpp from source on their machine.

So the Rust SDK shape is:

- **All Rust source code lives on crates.io** as normal Rust crates
  (`mesh-llm-api-server`, `mesh-llm-host-runtime`, `skippy-*`, etc.).
  Pure-Rust source, normal cargo dependency resolution.
- **The native build artifacts** (the patched llama.cpp static
  archives — `libllama.a`, `libggml.a`, `libggml-metal.a`/cuda/etc.,
  `libmtmd.a`, `libllama-common.a`) **are published per
  platform/backend as a GitHub release asset**.
- **`skippy-ffi`'s `build.rs` fetches the matching tarball** at consumer
  build time, verifies its sha256, extracts it into a per-user cache,
  and points its existing link directives at the extracted archives.

Consumer experience:

```toml
mesh-llm-api-server = { version = "0.66", features = ["host-runtime"] }
```

`cargo build`:

1. Cargo resolves the dep tree from crates.io. All pure Rust source.
2. Compiles each crate. When it gets to `skippy-ffi`, the build script
   detects target triple + backend, fetches
   `llama-stage-<triple>-<backend>.tar.gz` from the GitHub release URL,
   verifies sha256, extracts into `~/.cache/skippy-llama-stage/`.
3. `skippy-ffi/build.rs` emits the same `cargo:rustc-link-search` and
   `cargo:rustc-link-lib=static=...` directives it already does today
   for the workspace-internal `.deps/llama-build/` path.
4. Cargo finishes the Rust compile, statically linking the patched
   llama.cpp archives into the consumer's final binary.

Final consumer binary: one self-contained Rust executable. patched
llama.cpp + skippy + mesh-llm host runtime all statically inside.
Dynamically linked only against system libraries (`libSystem`, Apple
frameworks on macOS, libpthread/libdl on Linux, etc.) — same as the
shipped `mesh-llm` binary, same as a Swift app from `.binaryTarget`.

## What's on this branch right now

### Working

- **`crates/skippy-ffi/build.rs`** has a new additive code path: when
  `SKIPPY_LLAMA_TARBALL_URL` env var is set, the script fetches the
  tarball, verifies sha256 (against `.sha256` sidecar or
  `SKIPPY_LLAMA_TARBALL_SHA256`), extracts into a per-user cache, and
  sets `SKIPPY_LLAMA_BUILD_DIR` to point at the extracted root. The
  rest of `build.rs` is unchanged and links the static archives the
  same way it does today. When the env var is unset, behavior is
  identical to before — workspace-internal builds untouched.

  Override env vars:
  - `SKIPPY_LLAMA_TARBALL_URL` — `file://` or `https://` URL.
  - `SKIPPY_LLAMA_TARBALL_SHA256` — expected hex sha256, optional.
  - `SKIPPY_LLAMA_CACHE_DIR` — cache root (default
    `~/.cache/skippy-llama-stage/`).
  - `SKIPPY_LLAMA_TARBALL_FLAVOR` — `cpu` / `metal` / `cuda` / `rocm` /
    `vulkan`. Inferred from target triple if not set.

- **Local trial reproducible** with a manually-packaged tarball:

  ```bash
  # Inside the mesh-llm workspace, produce the static archives
  # (one-time per backend; already produced by `just llama-build`).
  just llama-prepare
  just llama-build
  # ... static archives now in .deps/llama-build/build-stage-abi-metal/

  # Package just the .a archives + CMakeCache.txt into a tarball.
  mkdir -p dist/llama-stage-static/aarch64-apple-darwin-metal
  cd .deps/llama-build/build-stage-abi-metal && \
    for f in CMakeCache.txt src/libllama.a tools/mtmd/libmtmd.a \
             common/libllama-common.a common/libllama-common-base.a \
             ggml/src/libggml.a ggml/src/libggml-base.a \
             ggml/src/libggml-cpu.a \
             ggml/src/ggml-metal/libggml-metal.a; do
      [ -f "$f" ] && cp --parents "$f" \
        ../../dist/llama-stage-static/aarch64-apple-darwin-metal/
    done
  cd dist/llama-stage-static && \
    tar czf llama-stage-aarch64-apple-darwin-metal.tar.gz \
        aarch64-apple-darwin-metal/ && \
    shasum -a 256 llama-stage-aarch64-apple-darwin-metal.tar.gz \
      > llama-stage-aarch64-apple-darwin-metal.tar.gz.sha256

  # Consumer (Rust app, anywhere on disk).
  cd /tmp/sprout-faux2
  SKIPPY_LLAMA_TARBALL_URL=\
"file:///Users/.../dist/llama-stage-static/llama-stage-aarch64-apple-darwin-metal.tar.gz" \
    cargo build
  ./target/debug/sprout-faux2
  # -> linked mesh-llm-api-server OK
  # -> owner keypair hex (len=128, first 16) = <random ed25519>
  # -> MeshNode::builder() typed OK
  ```

  Tarball size: **~5 MB** compressed. Cache populated after first run.
  Final binary: **1.7 MB**, dynamically linked only to macOS system
  frameworks.

### Not done yet

Three things to make this consumable by an external Rust app from
crates.io alone:

1. **Fix the pure-Rust publish chain.** Today the v0.66.0 publish run
   failed at `model-artifact` with crates.io HTTP 429 ("too many new
   crates in a short period"). `mesh-llm-api-server` has never reached
   crates.io. Fix: retry-on-429 in `scripts/publish-crates.sh`, and/or
   request a rate-limit increase from crates.io for the publishing
   account. Tracked in issue #691. **This grew more important on this
   branch** because the publish list now adds 18 new crate names —
   ~3× the new-crate-name volume of v0.66.0 — which would amplify the
   429 issue without a fix.

2. **Add `mesh-llm-host-runtime`, `skippy-ffi`, `skippy-runtime`,
   `skippy-server`, and the other internal crates that
   `mesh-llm-api-server`'s `host-runtime` feature transitively requires
   to the publish chain.** **Done on this branch.** The 18 new crates
   are added to `scripts/publish-crates.sh` in topological order. All
   path-deps across the workspace now carry both `path = "..."` and
   `version = "..."` so they resolve from crates.io for external
   consumers. A `cargo publish --dry-run` walks the entire chain
   cleanly: 15 crates fully verify, the rest correctly skip with a
   "depends on X@0.66.0 not yet on crates.io" message that goes away
   once each predecessor lands. `skippy-ffi` and other native-linking
   crates use `--no-verify` because the packaged tarball's `build.rs`
   can't find `.deps/llama-build` from `target/package/` (the release
   pipeline's pre-publish `cargo build` is the actual gate).

3. **Release pipeline produces and uploads `llama-stage-<triple>-<flavor>.tar.gz`
   per matrix cell.** **Done on this branch for Linux + macOS** (the
   matrix cells most relevant for sprout-shape consumers). Each
   relevant `build_*` job in `.github/workflows/release.yml` now runs
   `scripts/package-llama-stage.sh` after the existing
   `scripts/build-llama.sh` step. The script packages the patched
   llama.cpp static archives from `.deps/llama-build/build-stage-abi-<backend>/`
   into a release-asset-shaped tarball plus sha256 sidecar. Naming
   matches `skippy-ffi/build.rs`'s default URL construction
   (`llama-stage-<target_triple>-<flavor>.tar.gz`).

   Wired into: `build` (macOS metal, Linux x86_64 CPU), `build_linux_arm64`,
   `build_linux_cuda`, `build_linux_cuda_blackwell` (uses
   `--backend cuda-blackwell` so the asset name is distinct from
   regular CUDA), `build_linux_rocm`, `build_linux_vulkan`.

   Not wired: Windows. The package script is bash; Windows release
   jobs run PowerShell. Adding Windows requires either a `.ps1`
   equivalent of the package script or invoking bash from PowerShell.
   Tractable but skipped here to keep scope manageable; Windows Rust
   consumers can still consume the SDK by overriding
   `SKIPPY_LLAMA_TARBALL_URL` until this is closed.

## Pure-Rust source linking — verified

What's on this branch shows the **build mechanic** works
(`skippy-ffi/build.rs` fetches and extracts a tarball at consumer build
time). The trial consumer compiles fully from source — including the
Rust crate graph — and links the prebuilt llama.cpp archives.

What's *not* yet proven on this branch:

- **A running `node.start().await?`** — the trial builds the type and
  generates a keypair but doesn't actually start a node, because that
  requires real network setup and a real invite token. The static
  linkage path is the part that was uncertain; that's the part that's
  now verified. Calling `start()` is just normal mesh-llm code that
  already works in the workspace binary.

## Risks / honest caveats

- **First consumer build is slow** (~2 min in my measurement; will be
  longer with cold sccache and on slower machines). The Rust crate
  graph is large. Cached after first build.

- **Consumer's CI build environment** needs a Rust toolchain plus the
  system frameworks/libs the static archives reference — Metal /
  Accelerate / Foundation on macOS, libstdc++ / libdl / libpthread on
  Linux, etc. These are already required by the standalone `mesh-llm`
  binary today; nothing new for the consumer side.

- **Network at consumer build time** — the tarball URL is fetched by
  `skippy-ffi/build.rs`. Override env vars are documented above for
  offline / air-gapped consumers.

- **Static archive size per backend.** macOS Metal is small (~5 MB
  compressed). CUDA will be larger — single-digit hundreds of MB
  compressed, because nvcc emits one set of compiled kernels per
  CUDA arch. Still much smaller than the equivalent
  `libmeshllm_ffi.a`, since this is just llama.cpp, not the full Rust
  graph.

- **Per-platform/backend matrix is mesh-llm's problem, not the
  consumer's.** Sprout's CI just runs `cargo build`; the tarball it
  needs is published by mesh-llm's release pipeline. Sprout never sees
  cmake, never installs CUDA SDK, never compiles llama.cpp.

## Out of scope for this proposal

- An FFI / UniFFI surface for Rust consumers. Not needed; Rust calls
  Rust directly.
- Bundling a separate dylib/so/dll. Not needed; everything is statically
  linked.
- A workspace-internal `host-runtime` feature shape change. Existing
  workspace builds (no env var set) are unaffected.
