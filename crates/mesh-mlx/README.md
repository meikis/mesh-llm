# mesh-mlx

Native **Rust** MLX runtime for Apple Silicon — **no Python, no Swift**. Links
the MLX C++/Metal engine through its C API (`mlx-c`) and implements LLM
inference in Rust: model forward passes, safetensors loading, tokenization,
generation, and distributed (pipeline / tensor) primitives.

The MLX C++ engine does all compute and all networking (ring/TCP, JACCL/RDMA
over Thunderbolt). Rust is the orchestration layer — the same role Python
`mlx-lm` plays — but compiled, single-language, and embeddable in mesh-llm.

## Why (research summary)

The MLX distribution machinery (collectives, send/recv, ring/JACCL transports)
and all kernels live in the C++ core and are exposed via `mlx-c`. Python is not
in the hot path — a sharded layer's forward is literally `x @ Wᵀ;
all_sum(x)`, three C dispatches. Model forward passes are mechanical
transcriptions shared between Python `mlx-lm` and Swift `mlx-swift-lm`. So we do
it all in Rust over `mlx-c`: reuse the engine + collectives, transcribe a short
list of forward passes, write the small distributed wiring once. Full evidence:
`docs/design/MLX_PARALLELISM_RESEARCH.md`; architecture: `docs/design/MESH_MLX.md`.

## Layout

- `array`, `ops`, `nn` — safe RAII wrappers + transformer building blocks.
- `distributed` — process `Group` + collectives; `Pipeline` layer assignment.
- `models` — config + forward passes (Llama / Mistral / Qwen2 / Qwen3).
- `loader`, `download` — selective safetensors download + load.
- `runtime` — tokenizer, generate, high-level `Engine`.
- `mesh` — latency-aware parallelism planner + transport plan (local-only;
  MLX cannot use mesh QUIC).

## Features

- `link-mlx` — build and link the native MLX engine (Apple Silicon) for real
  inference. Without it, `mesh-mlx-sys` provides panicking stubs so the crate
  links and pure-logic unit tests run on any platform in CI (no Metal build).

## Try it (Apple Silicon)

```bash
xcodebuild -downloadComponent MetalToolchain   # one-time
cargo test -p mesh-mlx --features link-mlx --test live_single_node -- --nocapture
```

Downloads a small bf16 model from Hugging Face and generates tokens entirely in
Rust + MLX on Metal.

## Serve it (mesh-facing)

`mlx-serve` loads a model and exposes the OpenAI API mesh routes to:

```bash
cargo run -p mesh-mlx --features link-mlx --bin mlx-serve -- \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --addr 127.0.0.1:9999
# POST http://127.0.0.1:9999/v1/chat/completions
```

Mesh uses `MlxOrchestrator` to gate eligibility (`mlx_supported()`), plan
tensor-vs-pipeline from measured RTT, and render the MLX hostfile; then it spawns
`mlx-serve` per node and routes OpenAI traffic to it.

## Status

Code-complete; single-node verified end-to-end (bf16 **and** quantized 4-bit
produce correct output). Pipeline + tensor parallel paths, the OpenAI server,
and the mesh orchestrator are implemented and unit-tested. Multi-node execution
awaits a hardware test rig. See `docs/design/MESH_MLX.md`.
