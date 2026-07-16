# Spike: goose-style solo MLX serving — findings

Validates the Phase-2 claim in `docs/design/MLX_STAGE_ENGINE_PLAN.md`: load an HF
**safetensors** model in Rust via `safemlx-lm`, optionally **JIT-quantize on
load**, and generate tokens — with no GGUF and no ahead-of-time quant step.

Throwaway crate with its own `[workspace]`, deliberately **not** part of the
mesh-llm cargo workspace, so MLX/CMake native deps never touch other builds.

## TL;DR

- **Solo serving from raw safetensors works, in Rust, end to end.** ✅
  Qwen3-0.6B → coherent output at **18.1 tok/s** decode (CPU), no GGUF, no
  pre-quant.
- **JIT quant is functional but has no fast CPU kernel.** 4-bit and 8-bit both
  produce coherent output but collapse to **~0.4 tok/s** (~45× slower than
  source precision). MLX quantized matmul is Metal-optimized; CPU is not a
  viable quant path. **Metal is required to judge JIT-quant performance.**
- **safemlx-lm is young — hit (and fixed) a real loader bug** on the quant path
  for tied-embedding checkpoints.
- **Feature wiring needs a fork/upstream change**: the published `safemlx-lm`
  hard-enables the `metal` feature, so a Metal-less (CPU/CI) build is impossible
  without a workspace-level `default-features = false`.

## Environment

| | |
| --- | --- |
| Machine | Apple Silicon (arm64), macOS 26.5 |
| Toolchain | rustc 1.97, cmake 4.4, Apple clang 21 |
| Metal shader compiler | **absent** (CommandLineTools only, no full Xcode) → CPU-only build |
| MLX backend | `accelerate` (CPU). MLX core `v0.32.0` via safemlx-sys FetchContent |
| safemlx fork | `jbg/safemlx` @ `0502a19` + 3 local edits (below) |
| Model | `Qwen/Qwen3-0.6B` (dense, safetensors, `tie_word_embeddings: true`) |

## Results

Command shape:
```
mlx-solo --model <Qwen3-0.6B dir> -n 64 [--quantize 4|8] "<prompt>"
```

| Mode | load | ttft | decode tok/s | output |
| --- | --- | --- | --- | --- |
| source precision | 0.30s | 0.174s | **18.1** | coherent ✅ |
| JIT 4-bit affine | 1.03s | 47.6s | **0.4** | coherent ✅ |
| JIT 8-bit affine | 1.13s | 48.4s | **0.4** | coherent ✅ |

The quant slowdown is uniform across 4/8-bit and dominated by per-token decode
(ttft is essentially the first decode step), which is the signature of a missing
fast CPU quant-matmul kernel rather than a load-time cost.

## Build notes

- **MLX C++ builds cleanly under cmake 4.4, CPU-only** — this was the #1 feared
  risk (cmake 4.x deprecates old-CMake compatibility). `libmlx.a` + `libmlxc.a`
  produced, no policy errors. Cold build ~1m19s; warm rebuild after a Rust-only
  change ~6.7s (MLX archives cached in the shared target dir).
- Only harmless linker warnings (`object file … has version 26.5.0, which is
  newer than target minimum of 11.0.0`).

## Issues found

### 1. Published safemlx-lm hard-enables `metal` (build blocker off-Metal)

`safemlx-lm`'s dependency on `safemlx` did not disable default features, and a
member-level `default-features = false` is **ignored** by Cargo (it warns and
requires the setting at the workspace root). Effect: any consumer inherits the
`metal` feature and the build `panic!`s when the Metal compiler is absent
(CI, Metal-less macOS, or a CPU-only lane).

Fix in this spike (in the fork):
- root `Cargo.toml`: `safemlx = { …, default-features = false }`
- `safemlx-lm/Cargo.toml`: `safemlx = { workspace = true, default-features = false, features = ["accelerate", "safetensors"] }`

**Plan implication:** the `MlxStageEngine` will need MLX backend selection as an
explicit cargo feature (`metal` / `accelerate` / `cuda`), which means either an
upstream PR to `safemlx-lm` or carrying a small fork. Real integration cost, not
a blocker.

### 2. Tied-embedding `lm_head.weight` fails strict load on the quant path (qwen3)

`Qwen3-0.6B` sets `tie_word_embeddings: true` but the checkpoint still ships a
redundant `lm_head.weight`. The **dense** load uses a lenient loader and
tolerates it; the **quantized** load uses `load_safetensors_dir_quantized_strict`
with a bare `StrictLoadConfig::default()` and rejects it as an unused tensor:

```
strict weight-load validation failed: 0 missing, 1 unused
  unused: lm_head.weight
```

Fix in this spike (in the fork, `safemlx-lm/src/models/qwen3.rs`
`load_qwen3_model_quantized`):
```rust
let config = StrictLoadConfig::default().allow_unused_prefix("lm_head.");
```
Harmless when untied (then `lm_head.weight` is a loaded param, not unused).

**Plan implication:** confirms the "safemlx-lm is young; each model family is
bespoke and separately certified" risk. Expect per-family loader papercuts.

## Reproduce

1. Sibling checkout of the fork at `../safemlx` (repo-relative:
   `/Users/<you>/Documents/code/safemlx`), base `0502a19`, with the two Cargo
   edits and the qwen3 one-liner above.
2. Model dir with `config.json`, `tokenizer*.json`, `model.safetensors`
   (e.g. `Qwen/Qwen3-0.6B`).
3. Build/run (own workspace, so it will trigger a one-time MLX C++ build):
   ```
   cd spikes/mlx-solo
   cargo build --release
   ./target/release/mlx-solo --model <MODEL_DIR> -n 64 "..."
   ./target/release/mlx-solo --model <MODEL_DIR> -n 64 --quantize 4 "..."
   ```
   Using a shared `CARGO_TARGET_DIR` avoids rebuilding MLX across crates.

## What this de-risks / what it does not

De-risked:
- safemlx-lm as a Rust solo engine driving load + generate.
- The "no GGUF, no pre-quant, serve raw safetensors" workflow claim.
- MLX native build under this toolchain (cmake 4.4).

Still open (needs Metal):
- Any performance judgement of JIT quant (CPU quant path is not representative).
- Metal throughput vs the llama.cpp backend.
- Everything staged/split (this spike is single-stage, whole-model) — the
  partial-load and boundary-fence go/no-go spikes are unchanged.
