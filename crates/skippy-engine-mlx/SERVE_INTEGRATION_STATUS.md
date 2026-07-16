# MLX serve-integration — WIP status & blocker

This branch (`micn/mlx-serve-wiring`) stacks the **`mesh-llm serve` integration**
on top of the standalone MLX engine crate from PR #1009 (`micn/mlx-redux`).

Goal: on a Mac, `mesh-llm serve --model <hf-safetensors>` routes to the MLX
(Metal) engine and the model appears in `/v1/models` — with non-macOS / no-feature
builds byte-for-byte unaffected.

**Status: functionally complete and verified locally, but blocked by one repo
publish invariant. Needs a maintainer decision before it can merge.** Captured
here so the work + reasoning aren't lost.

## What works (verified on Apple Silicon this session)

- `skippy-engine-mlx` is now a real workspace member (removed its private
  `[workspace]`; added to root `members`, both CI crate-list scripts, and it
  stays out of `default-members`).
- `mlx` feature added to `mesh-llm-host-runtime` and `mesh-llm`, wired as a
  **macOS-target-gated optional dep** (`[target.'cfg(target_os = "macos")'.dependencies]`).
- `crates/mesh-llm-host-runtime/src/inference/mlx.rs`: `MlxModelHandle` +
  `MlxHttpHandle` serving over the real `openai-frontend::router_for` + axum, with
  graceful shutdown.
- `LocalRuntimeBackendHandle::Mlx` variant + all match arms (cfg-gated).
- `start_runtime_mlx_model` + `is_safetensors_model_path` routing: safetensors
  models branch to MLX *before* the GGUF planning path.
- **Build gate PASSED:** `cargo build -p mesh-llm --features mlx` → exit 0.
- **Don't-break-default PASSED:** no-feature `cargo check`/clippy clean;
  `cargo tree` confirms safemlx is absent unless `--features mlx`.
- fmt clean; clippy clean both with and without `--features mlx`
  (boxed the `Skippy` enum variant to satisfy `large_enum_variant`).
- `xtask repo-consistency ci-crate-lists` PASSES.

## Key finding: the two-native-stack link collision (RESOLVED)

Linking MLX statically alongside the patched llama.cpp fails with:

```
ld64.lld: error: duplicate symbol: gguf_get_key
  >>> defined in .../gguflib-src/gguflib.c              (MLX's vendored GGUF parser)
  >>> defined in libskippy_ffi...(gguf.cpp.o)           (patched llama.cpp)
```

MLX statically links antirez's `gguflib` (via `MLX_BUILD_GGUF=ON`, and
`safemlx-sys` link-directs `gguflib` unconditionally); the patched llama.cpp
exports the same C symbols. Two independent GGUF parsers → duplicate symbol.

**Fix:** the `mlx` feature now implies `dynamic-native-runtime`. That loads the
llama.cpp runtime as a dylib (as release builds already do), so its GGUF symbols
live in a separate link unit and don't collide with MLX's static `gguflib`.
`cargo build -p mesh-llm --features mlx` alone links clean after this.

## BLOCKER: publish invariant (needs a maintainer call)

```
cargo run -p xtask -- repo-consistency release-targets
  error: mesh-llm-host-runtime: publishable crate depends on
         non-publishable workspace crate `skippy-engine-mlx`
```

Root-cause chain:
- `skippy-engine-mlx` git-pins `safemlx` / `safemlx-lm` to a public commit of
  `github.com/jbg/safemlx` (crates.io's published safemlx produces gibberish for
  dense models — verified; only the git commit serves correctly).
- crates.io forbids git dependencies → the crate must be `publish = false`.
- `mesh-llm-host-runtime` is a **published** SDK crate, and the repo invariant
  (enforced by `xtask release-targets`, reflecting a real `cargo publish`
  constraint) forbids a publishable crate from depending on a `publish = false`
  one — even an optional, target-gated dep.

This is **not solved elsewhere in the repo**: the other `publish = false` crates
(`skippy-quantize`, `mesh-llm-commands`) are only consumed by the non-published
binary crate `mesh-llm`, never by a published library crate.

### Options

- **A. `[patch.crates-io]` redirect + make the crate publishable.** Follows the
  existing `hf-hub` precedent (a published-crate dep already redirected to a fork
  via `[patch.crates-io]` in the root manifest). Add `skippy-engine-mlx` to
  `scripts/publish-crates.sh`. Caveat that differs from hf-hub: safemlx's
  *published* version is known-broken, so a hypothetically-published
  `skippy-engine-mlx` with `mlx` on would reference broken upstream — acceptable
  only because the feature is off by default and explicitly a stopgap until
  safemlx cuts a working release.
- **B. Hold the host-runtime wiring.** Ship the standalone crate (PR #1009) as
  is; keep this branch as the ready-to-go integration until safemlx releases a
  working crates.io version, then flip git-pin → version-pin and merge. Most
  honest; doesn't deliver "serve just works" yet.
- **C. Loosen the xtask invariant** to exempt optional/target deps. Not
  recommended: it would allow a manifest that genuinely cannot `cargo publish`.

**Recommendation: B for now** (this branch is the parked, working integration),
moving to **A** if/when we want it live before safemlx releases. The real
unlock is a working safemlx crates.io release, after which this is a trivial
git-pin → version-pin swap and the invariant is satisfied automatically.

## How to reproduce / verify

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # Metal toolchain
cargo build -p mesh-llm --features mlx          # exit 0 (links clean)
cargo check -p mesh-llm-host-runtime            # no-feature: clean, no safemlx
cargo run -p xtask -- repo-consistency ci-crate-lists     # PASS
cargo run -p xtask -- repo-consistency release-targets    # FAILS (the blocker)
```

## Remaining once unblocked

- End-to-end serve test: `mesh-llm serve --model <mlx-safetensors-repo>` →
  confirm `/v1/models` + `/v1/chat/completions`.
- Fold this status into `WIRING.md` (note the `dynamic-native-runtime`
  requirement and the publish resolution chosen).
- Rebase onto `main` and open/land the real PR.
