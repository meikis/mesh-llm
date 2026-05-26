# Rust Native SDK: in-process mesh node from cargo

## Status: Trial implementation landed; pipeline + API surface work follow

## Goal

A Rust application adds mesh-llm to `Cargo.toml`, runs `cargo build`, and
gets a real in-process mesh node — same shape as the Swift and Kotlin SDKs:

```toml
[dependencies]
mesh-llm-api-server = { version = "0.66", features = ["native-metal"] }
```

No CMake on the consumer's machine. No source build of patched llama.cpp.
No separate `mesh-llm` daemon. The native bits arrive with the crate at
build time, link statically into the consumer's final binary.

## How Swift and Kotlin do it (the model we match)

Both ship a prebuilt artifact containing patched llama.cpp + skippy +
mesh-llm host runtime, compiled as a **static archive** with a
UniFFI-generated C ABI. The language SDK code calls into it; everything
runs in-process.

- **Swift:** `MeshLLMFFI.xcframework.zip` (~168 MB zipped, ~140 MB
  unzipped per macOS slice) on each GitHub release. SwiftPM
  `.binaryTarget(url:, checksum:)` in `Package.swift` downloads the zip
  at resolve time. Inside the framework is a **static archive** (Mach-O
  `.a` format), one per Apple architecture/SDK slice. SwiftPM
  static-links it into the consumer's app binary. No separate dylib to
  bundle.
- **Kotlin:** `libmeshllm_ffi.so` per Android ABI inside an AAR on GitHub
  Packages Maven. JVM/Android loads it at runtime.
- **Node.js:** prebuilt N-API `.node` addon via npm/GitHub Packages.

In all three the *prebuilt artifact arrives through the language's
native package channel* and the consumer's app links/loads it directly.
No daemon, no child process, no out-of-process IPC.

## The Rust equivalent (this proposal)

Rust gets the same shape: a small crate published to crates.io, whose
`build.rs` fetches the matching prebuilt **static archive**
(`libmeshllm_ffi.a`) for the consumer's target platform + selected
backend from a GitHub release, verifies its sha256, and emits link
directives so cargo links it statically into the consumer's binary.

The model is closest to Swift's `.binaryTarget(url:, checksum:)`. The
small crate on crates.io is the equivalent of `Package.swift`; the
prebuilt static archive lives on the GitHub release; the consumer's
build links everything statically into their final binary.

### Crate

`mesh-llm-native-sdk` — small Rust crate, no native bytes inside the
`.crate` payload. Just `build.rs` + a few lines of source.

Features select the backend (mutually exclusive):

```toml
mesh-llm-native-sdk = { version = "0.66", features = ["metal"] }
```

Available features: `metal`, `cpu`, `cuda`, `rocm`, `vulkan`.

### `build.rs` behaviour

1. Read selected backend feature (refuses to build if zero or more than one).
2. Read `CARGO_CFG_TARGET_OS` and `CARGO_CFG_TARGET_ARCH`.
3. Compose artifact ID: `meshllm-native-<platform>-<arch>-<backend>`,
   matching the naming `scripts/package-native-sdk.sh` already emits.
4. Compose default tarball URL from `CARGO_PKG_VERSION`:
   `https://github.com/Mesh-LLM/mesh-llm/releases/download/v<version>/<artifact_id>.tar.gz`.
5. Override default with `MESH_LLM_NATIVE_TARBALL_URL` env var (accepts
   `file://` for local trials, offline builds, and air-gapped mirrors).
6. Fetch the tarball into a stable per-user cache
   (`~/.cache/mesh-llm-native-sdk/<version>/<artifact_id>/` by default;
   override with `MESH_LLM_NATIVE_CACHE_DIR`).
7. Verify sha256 against the `.sha256` sidecar fetched from the same URL,
   or against `MESH_LLM_NATIVE_TARBALL_SHA256` if explicitly set.
8. Extract `libmeshllm_ffi.a` into `OUT_DIR`.
9. Emit link directives:
   - `cargo:rustc-link-search=native=<OUT_DIR>/native/<artifact_id>/lib`
   - `cargo:rustc-link-lib=static=meshllm_ffi`
   - Plus per-platform system framework/library directives (Accelerate,
     Metal, MetalKit, Foundation, Security, etc. on macOS; libstdc++,
     libm, libdl, libpthread on Linux; user32, ws2_32, bcrypt on Windows).

### Consumer binary shape

Identical to what Swift apps get from `.binaryTarget`:

- Everything statically linked: patched llama.cpp, ggml, ggml-metal/cuda/etc.,
  skippy, mesh-llm host runtime, all the UniFFI scaffolding.
- Consumer's binary is **one self-contained executable**. No bundled
  `.dylib` / `.so` / `.dll`. No `@rpath` rituals. Tauri/cargo-bundle
  packaging just takes the binary as-is.
- Linker DCE strips unused code, so the binary size is proportional to
  what the consumer actually uses, not to the archive size.

## Status of this branch

### Working today

- `crates/mesh-llm-native-sdk/` — the crate, with `build.rs`, src/lib.rs,
  README, Cargo.toml. Committed.
- Local trial: `scripts/package-native-sdk.sh --build --backend metal`
  produces `libmeshllm_ffi.a` (~350 MB unstripped). A small tarballing
  step packages it as
  `dist/native-sdk-static/meshllm-native-darwin-aarch64-metal.tar.gz`
  (~131 MB compressed) with sha256 sidecar.
- A trivial Rust consumer **outside the workspace** at
  `/tmp/sprout-faux/`, depending on `mesh-llm-native-sdk` by path with
  `features = ["metal"]` and `MESH_LLM_NATIVE_TARBALL_URL=file://...`,
  builds cleanly and runs. `otool -L` confirms only system frameworks
  are linked dynamically; the entire mesh runtime is statically inside
  the consumer's binary.

### Not yet wired up

These are the steps from "trial works on this laptop" to "external Rust
consumers can use this":

1. **Release pipeline ships per-platform/backend static archives.**
   Today every release-matrix cell builds `libmeshllm_ffi.a` (it's in
   `crates/mesh-llm-ffi/Cargo.toml`'s `crate-type`) but doesn't ship it.
   Add a tar + checksum + upload step per cell. Reuses the existing
   cmake step. Asset naming matches what `build.rs` expects.
2. **`mesh-llm-api-server` adds `native-*` features that pull
   `mesh-llm-native-sdk` transparently.** Consumers depend on
   `mesh-llm-api-server` (the SDK entrypoint), not directly on
   `mesh-llm-native-sdk`. Today this layer is missing — a consumer must
   depend on the native-sdk crate directly to trial.
3. **Fix the pure-Rust publish chain.** The v0.66.0 publish run failed
   at `model-artifact` with crates.io HTTP 429 (new-crate rate limit),
   leaving `mesh-llm-api-server` itself unpublished. Either add
   retry-on-429 to `scripts/publish-crates.sh` or get the limit raised.
   Until this is fixed, the consumer-facing crate isn't on crates.io
   at all.

### Out of scope for this proposal

- Rust-native API wrappers on top of the UniFFI C symbols inside the
  static archive. The trial calls a raw UniFFI symbol
  (`ffi_meshllm_ffi_uniffi_contract_version`) to prove the link works.
  Producing an ergonomic Rust API (`MeshNode::builder()`, etc.) on top
  is a separate layer; the easiest path is `uniffi-bindgen` generating
  Rust bindings from the same `.udl` Swift and Kotlin already consume.
  Not addressed here.
- Tier-2 split (Rust app joins the mesh as a real iroh peer with no
  local serving, lighter than full host-runtime). Separate work.
- Source-build path (`-sys` style) for consumers who want auditable
  builds. Not addressed; remains the workspace-internal
  `host-runtime` feature, untouched by this proposal.

## What about other consumer-app concerns

- **Sprout-style bundling:** the consumer binary is fully self-contained.
  Tauri / cargo-bundle just packages the executable. No `.dylib` to copy
  into `Sprout.app/Contents/Frameworks/`. No install_name rewriting.
- **CI:** consumer's CI needs only a Rust toolchain. No CMake, no CUDA
  SDK, no Vulkan SDK. The cached tarball survives across CI runs in
  `~/.cache/mesh-llm-native-sdk/`.
- **Cross-compile:** `build.rs` reads `CARGO_CFG_TARGET_*`, not the
  host triple. A macOS host targeting `x86_64-unknown-linux-gnu` would
  fetch the Linux x86_64 tarball.
- **Offline builds:** `MESH_LLM_NATIVE_TARBALL_URL=file:///mirror/path`
  + `MESH_LLM_NATIVE_TARBALL_SHA256=...` + `MESH_LLM_NATIVE_CACHE_DIR`
  cover air-gapped and corporate-mirror cases.
- **Reproducibility:** sha256 verified on every fetch. A `.sha256`
  sidecar lives alongside the tarball on the release.

## Risks / honest caveats

- **Static archive size.** Compressed tarball is ~130 MB for metal CPU
  cases; expect 300-500 MB for CUDA cases because the archive carries
  nvcc-compiled CUDA kernels per architecture. Downloaded once per
  (version, platform, backend) per consumer machine and cached. No
  crates.io size limits apply because the bytes live on GitHub
  releases, not on crates.io.
- **First-build network requirement.** Consumers without network access
  must use the override env vars. Documented above; would need to be
  documented loudly in `docs/SDK.md` for external consumers.
- **Symbol surface.** The static archive exports UniFFI C symbols today.
  Calling them from Rust through `extern "C"` works but is awkward
  compared to a native Rust API. A follow-up should add a thin Rust
  wrapper (likely via `uniffi-bindgen`'s Rust generator) so consumers
  call `MeshNode::builder()` rather than poking at `ffi_meshllm_ffi_*`
  symbols. Tracked separately.

## Reference: trial commands

Local end-to-end on `micn/native-sdk-cargo-publish`:

```bash
# 1. Produce libmeshllm_ffi.a for macOS arm64 metal.
scripts/package-native-sdk.sh --build --backend metal --out dist/native-sdk

# 2. Pack into static-archive tarball + sha256 (manual for the trial;
#    in CI this would be its own step).
mkdir -p dist/native-sdk-static/meshllm-native-darwin-aarch64-metal/lib
cp target/release/libmeshllm_ffi.a \
   dist/native-sdk-static/meshllm-native-darwin-aarch64-metal/lib/
# (write manifest.json, tar czf, shasum -a 256)

# 3. Build a consumer outside the workspace.
cd /tmp/sprout-faux
MESH_LLM_NATIVE_TARBALL_URL="file:///path/to/dist/native-sdk-static/meshllm-native-darwin-aarch64-metal.tar.gz" \
    cargo build

# 4. Run it.
./target/debug/sprout-faux
# -> sprout-faux: linked libmeshllm_ffi OK
# -> sprout-faux: uniffi contract version = 30
```
