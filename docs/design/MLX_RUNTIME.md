# MLX Runtime (Apple Silicon sidecar)

Status: experimental crate landed (`mesh-llm-mlx-runtime`); not yet wired into
the host runtime.

## Goal

Let an all-Mac mesh run inference on **MLX** instead of the patched
llama.cpp/Skippy runtime, while mesh keeps the product surface (routing, demand,
OpenAI `/v1`, membership). Mesh **orchestrates** MLX; it does not reimplement it.

## Why a sidecar (decision)

MLX ships a complete distributed inference stack in `mlx-lm`:

- pipeline **and** tensor parallelism,
- over Ethernet/Wi-Fi (Ring/TCP) **and** Thunderbolt RDMA (JACCL),
- with safetensors models and selective per-stage downloads.

The alternative — implementing an MLX engine behind the Skippy `skippy.h` stage
ABI — was rejected for the first cut because:

- the Skippy activation frame is engine-private hidden state (can be `OPAQUE`,
  carries arch flags, leaks `llama_model`/`llama_context`), so MLX and llama.cpp
  stages cannot exchange frames; and
- it would require re-implementing and parity-gating the entire model zoo at the
  ABI boundary.

See `MLX-mesh-integration-analysis.md` (research notes) for the full comparison.

This is **not** a revival of the retired external `llama-server`/`rpc-server`
lane. MLX is a distinct engine; the supervised sidecar is the analog of a
native-runtime lane, gated to Apple Silicon and owned by mesh.

## Shape

```text
mesh router ──orchestrates──> MlxBackend (mesh-llm-mlx-runtime)
                                 │  spawns + supervises
                                 ▼
                      mlx_lm.server  (single node)
                      mlx.launch     (multi node: pipeline | tensor)
                                 │  OpenAI /v1 on 127.0.0.1
                                 ▼
                      MLX engine (Metal, unified memory)
```

`Backend.endpoint` (`http://127.0.0.1:<port>/v1`) is what mesh routes OpenAI
traffic to — reuse the existing OpenAI routing rather than re-proxying.

## The abstraction (`MlxBackend` trait)

- `supported()` — Apple-Silicon capability gate (macOS aarch64).
- `plan_parallelism(&[LatencySample]) -> ParallelismPlan` — pure, previewable.
- `start(&ParallelismPlan) -> Backend` — render hostfile, spawn, wait ready.
- `health()` / `stop()` — lifecycle.

## Latency-aware parallelism

Mesh already measures inter-node latency for routing/affinity. `ParallelismPlanner`
reuses it:

| Worst-case inter-node RTT | Mode | Rationale |
|---|---|---|
| `<=` threshold (default **2ms**) | Tensor | ~2 all-reduces/layer; needs low-latency fabric |
| `>` threshold | Pipeline | one activation send/recv per stage; latency tolerant |
| no samples | Pipeline | conservative |
| single node | Single | no split |

Threshold is configurable (`ParallelismPlanner::with_threshold`).

## Networking

MLX **cannot** use mesh's QUIC/iroh transport natively (`mx.distributed.init`
only accepts `{any, mpi, ring, nccl, jaccl}`; Ring opens its own TCP sockets).
Two supported options, modelled in `transport.rs`:

1. **LAN ring / JACCL** — MLX forms its own group directly over the LAN or
   Thunderbolt mesh. Mesh supplies the hostfile (`{ssh, ips, rdma?}`). Lowest
   overhead. Tensor parallel prefers JACCL when rdma maps are present.
2. **QUIC tunnel** — mesh terminates QUIC and exposes a per-node local TCP
   port-forward that maps onto the neighbour's Ring listen port; the hostfile
   uses `127.0.0.1:<forwarded>`. Reuses mesh connectivity (NAT traversal,
   relays, auth) via the existing `network/tunnel.rs` relay, at the cost of an
   extra hop. Pair with pipeline, not tensor.

## Model artifacts

MLX consumes **safetensors**, not GGUF. `ModelRef::ensure_mlx_compatible`
rejects GGUF and the host should route those to the Skippy lane. MLX downloads
only what's needed: pipeline stages fetch only their layers' safetensors
(`DownloadPolicy::StageShardOnly`); tensor/single fetch the full repo.

## Open follow-ups (host-runtime wiring)

1. Capability-gated backend selection (Apple Silicon + `mlx-lm` importable → MLX
   lane; else Skippy lane) in `inference/`.
2. Feed mesh's measured RTT into `plan_parallelism`.
3. Build `NodeEndpoint`s from mesh membership; choose LAN-ring vs QUIC-tunnel by
   routability and wire the tunnel via `network/tunnel.rs`.
4. Route OpenAI traffic to `Backend.endpoint` through existing routing.
5. Surface MLX sidecar logs into the instance runtime dir like skippy native
   logs.
6. Decide how much of the MLX pipeline mesh should *see* (placement/telemetry).
