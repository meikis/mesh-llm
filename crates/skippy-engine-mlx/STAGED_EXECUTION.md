# MLX partial-layer staged execution

## Status

Dense Llama-family MLX stages now run as separate OS processes from partial
SafeTensors artifacts and communicate over Skippy's existing binary stage wire.
The first proof uses `HuggingFaceTB/SmolLM2-135M-Instruct` split at layer 15.

This is a production-shaped bridge, not yet the default mesh launch path:

- `skippy-engine` owns the engine-neutral `StageEngine` contract and residual
  buffer descriptors.
- `skippy-server::engine_transport` serves that contract using the existing
  `StageWireMessage`, ready handshake, activation codec, and reply codec.
- `MlxStageEngine` loads one materialized partial SafeTensors file, owns
  per-session KV caches on a dedicated MLX worker thread, and executes only its
  configured layer range.
- `mlx-stage` starts a stage process or drives a chain as a proof client.

No process in the proof has access to the complete checkpoint. The tokenizer
and config files are small shared metadata; tensor data comes only from that
process's `model.safetensors`.

## Verified result

On Apple Silicon Metal, using two materialized 155.28 MiB partial files:

| Process | Layers | Tensor file available | RSS after the proof |
| --- | ---: | ---: | ---: |
| stage 0 | `0..15` | 155.28 MiB | 188,784 KiB |
| stage 1 | `15..30` | 155.28 MiB | 189,168 KiB |

The processes exchanged F16 residual activations and generated:

```text
[284, 260, 2240, 314, 1343, 327, 624, 8685]
```

That exactly matches the whole-model and in-process split reference for the
same prompt across prompt prefill and seven subsequent decode calls. Each stage
kept an independent per-layer KV cache, and `Stop` cleared the session in both
processes.

The two partial files are the exact-range artifacts described in
`../../spikes/mlx-safetensors-stages/FINDINGS.md`. Tied input/output embeddings
are intentionally duplicated across the stages; that is why the sum of the two
files is larger than the full checkpoint even though neither process downloads
the full checkpoint.

## Reproduce

Build once:

```bash
just mlx-stage-build
```

Start the final stage:

```bash
just mlx-stage serve \
  --model /tmp/mlx-split-smol/stage1 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --stage-index 1 --layer-start 15 --layer-end 30 \
  --bind 127.0.0.1:19091 --wire-dtype f16 --compute-dtype bf16
```

Start the first stage in another terminal:

```bash
just mlx-stage serve \
  --model /tmp/mlx-split-smol/stage0 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --stage-index 0 --layer-start 0 --layer-end 15 \
  --bind 127.0.0.1:19090 --downstream 127.0.0.1:19091 \
  --wire-dtype f16 --compute-dtype bf16
```

Drive the chain:

```bash
just mlx-stage prove --connect 127.0.0.1:19090 --wire-dtype f16
```

## Deliberate limitations of this checkpoint

- Dense Llama-family checkpoints only. The engine boundary is family-neutral,
  but the current MLX adapter is the smallest implementation that proves it.
- Greedy sampling only; sampling metadata is preserved in the contract and
  rejected explicitly when enabled.
- No KV page import/export, cache trim/checkpoint, MTP, speculative verify,
  multimodal projection, or transport batching yet.
- `engine_transport` is the reduced compatibility lane. The mature llama.cpp
  binary server remains unchanged and still owns telemetry, exact-prefix cache,
  batching, and OpenAI orchestration.
- Mesh topology planning does not yet launch `MlxStageEngine`; the next product
  step is selecting this engine from stage config and advertising it as an
  additive capability. There is no mesh protocol or Skippy ABI break here.
