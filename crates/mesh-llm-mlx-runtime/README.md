# mesh-llm-mlx-runtime

A Mesh LLM backend that orchestrates an **MLX sidecar** for inference on Apple
Silicon. It is the "all-Mac" alternative to the patched llama.cpp/Skippy runtime.

## Why a sidecar, not a Skippy ABI backend

MLX already ships a batteries-included distributed stack in `mlx-lm`: both
**pipeline** and **tensor** parallelism, over **Ethernet/Wi-Fi (Ring/TCP)** or
**Thunderbolt RDMA (JACCL)**. Reusing it is far cheaper and lower risk than
re-implementing the model zoo behind Skippy's engine-private activation-frame
ABI. Mesh keeps the product surface (routing, demand, OpenAI `/v1`, membership)
and treats MLX as a managed, OpenAI-compatible local backend.

```text
mesh router ──orchestrates──> MlxBackend (this crate)
                                 │  spawns + supervises
                                 ▼
                      mlx_lm.server  (single node)
                      mlx.launch     (multi node: pipeline | tensor)
                                 │  OpenAI /v1 on 127.0.0.1
                                 ▼
                      MLX engine (Metal, unified memory)
```

This is the analog of a **native-runtime lane**, not a revival of the retired
external `llama-server`/`rpc-server` lane: MLX is a distinct engine that cannot
be embedded behind the Skippy ABI, and the sidecar is owned and supervised by
mesh and gated to Apple Silicon.

## The Rusty abstraction

`MlxBackend` is the trait mesh code talks to:

- `supported()` — Apple-Silicon capability gate.
- `plan_parallelism(&[LatencySample]) -> ParallelismPlan` — **latency-aware**
  mode selection (pure/previewable).
- `start(&ParallelismPlan) -> Backend` — spawn + wait for readiness.
- `health()` / `stop()` — lifecycle.

`Backend.endpoint` is the `…/v1` URL mesh routes OpenAI traffic to.

## Latency-aware parallelism

`ParallelismPlanner` picks the MLX split mode from measured inter-node RTT:

| Worst-case inter-node RTT | Mode | Why |
|---|---|---|
| `<=` threshold (default **2ms**) | **Tensor** | ~2 all-reduces/layer; only pays off on a low-latency fabric (Thunderbolt/JACCL or tight LAN) |
| `>` threshold | **Pipeline** | one activation send/recv per stage; latency tolerant |
| no samples | **Pipeline** | conservative default |
| single node | **Single** | no split |

Tune with `ParallelismPlanner::with_threshold(...)`.

## Confirmed MLX behaviours

1. **MLX runs from safetensors, not GGUF.** GGUF models are rejected here and
   routed to the Skippy/llama.cpp lane (`ModelRef::ensure_mlx_compatible`).
2. **MLX downloads only what's needed.** `mlx_lm.sharded_load` passes
   `allow_patterns=local_files`, so a pipeline node fetches only the safetensors
   for its layers (`DownloadPolicy::StageShardOnly`); tensor/single fetch the
   full repo (`FullRepo`).
3. **MLX can't speak mesh QUIC natively.** `mx.distributed.init` only accepts
   `{any, mpi, ring, nccl, jaccl}` and Ring opens its own TCP sockets, so the
   `transport` module models two options: a direct **LAN ring/JACCL** group, or
   tunnelling MLX TCP through mesh QUIC via local port-forwards
   (`MeshTransport::QuicTunnel`, pairs with `network/tunnel.rs`).

## Status

Experimental. Not yet wired into the host runtime; the trait + planner +
launch-spec builder are unit-tested without requiring an MLX install. The
`live-sidecar` feature gates tests that exercise a real `mlx_lm.server`.
