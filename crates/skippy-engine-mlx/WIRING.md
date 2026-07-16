# Wiring `skippy-engine-mlx` into mesh-llm

This crate is a **working, self-contained MLX (Metal) serving engine** that
already serves HF safetensors models over mesh-llm's real OpenAI frontend
(`openai_frontend::router_for`). It is intentionally standalone right now — its
own cargo workspace — so the heavy MLX native build never runs in unrelated
builds/CI. This document is the concrete plan to promote it into the shipped
binary so that **on a Mac, `mesh-llm serve` can run an MLX tensor model and
users can pick one from `/v1/models`.**

## What already works (this crate, today)

- `MlxEngine` — a dedicated OS worker thread owns the non-`Send` MLX objects
  (model, streams, arrays); the outside world talks to it with `Send` channels.
- `MlxBackend: openai_frontend::OpenAiBackend` — `models`, `chat_completion`,
  `chat_completion_stream` (SSE), with usage accounting and incremental
  detokenization.
- `mlx-serve` bin — `router_for(Arc<MlxBackend>)` + `axum::serve`.
- Verified on Apple Silicon (Metal), source precision, over the real frontend:
  Qwen3-0.6B (non-stream + streaming) and SmolLM2 (Llama arch) both generate
  **coherent** output. Source precision ~321 tok/s (see
  `../../spikes/mlx-solo/FINDINGS.md`).

## Dependency: why a git-rev pin (not crates.io)

`Cargo.toml` pins `safemlx` / `safemlx-lm` to a specific **public** commit of
`github.com/jbg/safemlx` (`rev = "4e53c5e"`), not a crates.io version. This is a
deliberate, reproducible git pin — **not a private fork and no local patches**:

- safemlx's **published** crates (both `0.1.5` and `0.4.1`) produce **garbage
  output for dense models** (Qwen3 *and* Llama both emit repeated-token gibberish
  with this exact same crate code). This was verified to be a **library bug**,
  not a prompting/template problem — the Qwen3 chat template is confirmed applied
  correctly, and greedy sampling (`temp=0` → argmax) is used.
- The pinned upstream commit serves those families correctly. Swapping the same
  crate between published and the git pin flips the output coherent↔gibberish,
  which isolates the cause to the safemlx version.
- **Action item:** swap the git-rev pin for a normal version pin once safemlx
  cuts a crates.io release that serves dense models correctly.
- No JIT quantization is used (plain `LoadedModel::load`), so none of the
  quant-path loader quirks apply; this path needs zero source patches.

## Promotion plan (the actual PR)

### 1. Make it a real workspace member

- Add `crates/skippy-engine-mlx` to root `Cargo.toml` `members` and remove its
  local `[workspace]` table. The safemlx deps are already git-rev pinned to a
  public commit (see "Dependency" above), so nothing else changes about them —
  a workspace member with a git dependency is fine.
- Keep the crate's `mlx` feature; gate all MLX code with
  `#[cfg(all(feature = "mlx", target_os = "macos"))]` (already done).
- Update `scripts/affected-crates.sh`, `scripts/plan-clippy-batches.sh`,
  `scripts/publish-crates.sh` `WORKSPACE_MEMBERS`, and run
  `cargo run -p xtask -- repo-consistency ci-crate-lists`. Because MLX is a heavy
  native lane, add it to the CI backend-gating like the other native features
  (only build/test the `mlx` feature on macOS runners).

### 2. Depend on it from host-runtime, macOS-gated

In `crates/mesh-llm-host-runtime/Cargo.toml`:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
skippy-engine-mlx = { path = "../skippy-engine-mlx", features = ["mlx"], optional = true }

[features]
mlx = ["dep:skippy-engine-mlx"]
```

(The `path` here is the in-repo crate path once it is a workspace member; its own
safemlx deps stay git-rev pinned.) Propagate a `mlx` feature up through
`crates/mesh-llm/Cargo.toml`, and enable it by default only on macOS builds in
the release packaging.

### 3. Add an `Mlx` variant to the launch enum

`crates/mesh-llm-host-runtime/src/runtime/local.rs`:

- `LocalRuntimeBackendHandle` currently has one variant (`Skippy { .. }`). Add
  (macOS+feature gated):

  ```rust
  #[cfg(all(feature = "mlx", target_os = "macos"))]
  Mlx { backend: Arc<skippy_engine_mlx::MlxBackend>, http: MlxHttpHandle, _death_tx: ... },
  ```

- Every `match &self.inner { LocalRuntimeBackendHandle::Skippy { .. } => ... }`
  in `local.rs` (pid, ctx_used_tokens, openai_guardrails, llama_slots_snapshot,
  set_openai_guardrail_mode, shutdown/http accessor) needs an `Mlx` arm. Most map
  to simple/None-ish values since MLX has no llama slots or GGUF guardrail state.

### 4. Route safetensors models to the MLX branch at launch

`start_runtime_local_model` currently branches:

```
if is_layer_package_ref(..) { layer_package } else { skippy (direct GGUF) }
```

Add a first branch: if the resolved model is `ModelFormat::Safetensors` (the
model layer already classifies this — `crates/model-artifact` `ModelFormat`, and
`models/resolve` already detects `is_primary_mlx_weight_file`), and we're on
macOS with the `mlx` feature, start an `MlxEngine` instead:

```rust
#[cfg(all(feature = "mlx", target_os = "macos"))]
if resolved_format == ModelFormat::Safetensors {
    return start_runtime_mlx_model(spec, model_name, plan).await;
}
```

`start_runtime_mlx_model` mirrors `start_runtime_skippy_model`: build an
`MlxEngineConfig` from the resolved model dir + planned ctx/limits, `spawn` the
engine on a blocking task, wrap `MlxBackend` in the embedded HTTP handle
(`openai_frontend::router_for`), and return a `LocalRuntimeModelHandle` whose
`backend` string is `"mlx"`.

### 5. Model discovery / listing already works

The model layer already discovers, downloads, catalogs, and lists MLX
safetensors repos (`crates/mesh-llm-host-runtime/src/models/catalog.rs`,
`.../models/resolve`, `crates/model-resolver`). No change needed for a user to
*see* MLX models; the missing piece was purely the serving engine, which this
crate provides. Auto-behavior: on a Mac, a resolved safetensors model simply
routes to the MLX engine.

### 6. Auto on Mac + user selection

- With the `mlx` feature enabled by default on macOS builds, serving a
  safetensors model "just works" with no extra flags.
- Users pick a model the same way as today: `mesh-llm serve --model <hf-repo>`
  or from `~/.mesh-llm/config.toml`; safetensors → MLX, GGUF → llama.cpp.
- Optionally add `--serving-backend mlx|llama` to force the engine when a model
  is available in both formats (parallels the existing skippy backend selector
  noted in `docs/SKIPPY.md`).

## Out of scope for the first PR

- **Splits / staged execution.** This is single-stage, whole-model serving. The
  staged `StageEngine` trait, partial-load, and activation-frame work (plan §6–§8)
  are separate and remain gated on the go/no-go spikes.
- **Tool calling / reasoning parsing.** goose's `mlx.rs` has native + emulated
  tool parsing and thinking-output filtering worth porting later; this crate
  streams raw model text (including `<think>` blocks) for now.
- **Draft/speculative decoding** (goose's `gemma4_mtp`).
- **JIT quantization on load.** safemlx can affine-quantize dense weights at load
  time (~604 tok/s 4-bit in earlier spikes), but it is deliberately excluded here
  to keep the first PR to the goose-style plain-load path.

## Future: Linux + NVIDIA (CUDA)

MLX is **not Apple-only** — `jbg/safemlx` supports **Linux + NVIDIA GPUs via
CUDA** (a `cuda` cargo feature, plus `nccl` for multi-GPU), gated in
`safemlx-sys/build.rs` with a hard `panic!` to Linux targets, for both
`x86_64-linux` and `sbsa-linux` (ARM). The generation code
(`LoadedModel::load` + `generate_with_cache`) is backend-independent — only the
native build backend differs (Metal vs CUDA). So a later change could serve MLX
on Linux/NVIDIA mesh nodes by widening the gate from `target_os = "macos"` to
also allow `cfg(all(target_os = "linux", feature = "cuda"))` and enabling
`safemlx/cuda`. This is **out of scope for this PR** (Mac/Metal first) and is
tracked as future research; ROCm/Vulkan/Windows are not supported upstream.

## Testing the promoted path

- `just build` on macOS with the `mlx` feature.
- `mesh-llm serve --model <mlx-safetensors-repo>` → confirm the model appears in
  `/v1/models` and `/v1/chat/completions` returns a generation.
- Confirm non-macOS / no-feature builds are byte-for-byte unaffected (the crate
  and its deps compile out entirely).
