# Next Goal: GLM-5.1 Native SPD Qualification On HF

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Train and qualify an SPD sidecar against the exact GLM-5.1 Skippy layer package
we would serve, using native package taps/logits on Hugging Face, then decide
from held-out acceptance and pipeline economics whether to run a real split.

## Immediate Target

- Runner: Hugging Face Jobs in the `meshllm` org.
- No heavy local M4 training/capture loops for this target.
- Base/checkpoint family: `zai-org/GLM-5.1`.
- Exact first package: `meshllm/GLM-5.1-UD-Q3_K_XL-layers`.
- Package size from HF metadata: about `341GB`.
- Model shape from HF config: `78` layers, hidden size `6144`, MoE with `256`
  routed experts and `8` active experts per token.
- Logical SPD topology: S6, `stage_layer_boundaries=13,26,39,52,65,78`.
- Required SPD taps: `[0,13,26,39,52,65,78]`.
- Physical fit for limited nodes: colocate contiguous logical stages while
  still returning every required tap, e.g. `[[0,1],[2,3],[4,5]]`.

## Skippy Layers vs SPD Topology

SPD must reuse the normal Skippy layer package. It does not get a separate copy
of model layers.

- Skippy physical stages own actual layer ranges and package materialization.
- The SPD sidecar owns logical tap requirements and proposal weights.
- Workers need only the manifest-derived tap-return allowlist.
- The coordinator owns the SPD bundle and runs the predictor.
- If Mesh packs multiple logical SPD stages onto one physical node, that node
  must still return the internal logical-boundary taps required by the manifest.

This lets us pretrain/predigest sidecars for canonical logical topologies
instead of training every possible physical host grouping.

## Why This Quant

Start with `meshllm/GLM-5.1-UD-Q3_K_XL-layers` because it already exists as a
Skippy package and is the smallest GLM-5.1 layer package checked so far:

- `meshllm/GLM-5.1-UD-Q3_K_XL-layers`: about `341GB`.
- `meshllm/GLM-5.1-Q3_K_M-plus-layers`: about `362GB`.
- `meshllm/GLM-5.1-UD-Q3_K_S-layers`: about `362GB`.

An outside public GGUF quant can work, but only after this extra phase:

`outside GGUF repo -> Skippy layer package -> staged baseline smoke -> SPD capture/train/qualify`

For the first real SPD proof, avoid that extra moving part.

## HF Machine And Cost

HF Jobs rates checked on 2026-06-19:

- `h200`: `141GB` VRAM, `256GB` RAM, `3TB` disk, about `$5/hr`.
- `h200x2`: `282GB` VRAM, `512GB` RAM, `6TB` disk, about `$10/hr`.
- `h200x4`: `564GB` VRAM, `1TB` RAM, `12TB` disk, about `$20/hr`.
- `h200x8`: `1128GB` VRAM, `2TB` RAM, `24TB` disk, about `$40/hr`.

Realistic lanes:

- Metadata/planner smoke: `h200` or CPU-only, under `$20`, no quality claim.
- Feasibility native package smoke: `h200x2`, cap `4-6h`, about `$40-$60`.
  This may need CPU offload because the Q3 package is larger than total VRAM.
- Serious native package qualification: `h200x4`, cap `4-8h`, about
  `$80-$160`. This is the first realistic lane for a quality decision.
- Reference bootstrap from full GLM-5.1 is expensive and should not be first:
  BF16 is about `1.5TB`; FP8 is about `756GB`. If needed, expect `h200x8`,
  about `$240-$480` for `6-12h`.

Do not submit spend until the job prints the exact model/package, topology,
dataset shard, prompt counts, hardware, timeout, output repo, and max cost.

## Concrete HF Job Plan

The first useful job should be native-package-first, not BF16-reference-first:

1. Build fixed train and held-out prompt-token shards.
2. Download `meshllm/GLM-5.1-UD-Q3_K_XL-layers`.
3. Run a staged baseline smoke to prove the package loads for the chosen S6
   topology.
4. Capture raw tap-concat rows at `[0,13,26,39,52,65,78]`.
5. Capture native quant verifier logits/top-k over the SPD draft vocabulary.
6. Train the SPD head from native rows/logits.
7. Score held-out top-1/top-4 against native labels.
8. Export `skippy-spd-head.json`, `spd-head.safetensors`, and
   `spd-parity-fixture.safetensors`.
9. Run Rust fixture parity.
10. Run package-backed rolling `spd-openai-smoke` on broad held-out prompts.
11. Emit latency simulation from `evals/spd/simulate_latency.py`.

If fresh native training is unstable, then add a reference/bootstrap phase using
FP8/NVFP4/AWQ, not full BF16 by default.

## Pass/Fail Gate

Pass requires:

- train and held-out prompt-token shards have zero overlap;
- native teacher argmax matches the quantized verifier target on in-scope rows;
- exported serving bundle passes Rust fixture parity;
- package-backed rolling smoke matches baseline content;
- tap failures are zero;
- saved candidate-token round trips exceed unsaved candidate-token round trips
  with margin on broad held-out prompts;
- latency simulation stays positive with realistic physical stage costs, link
  latency, and measured or explicitly budgeted sidecar latency.

Fail means record the report and adjust the HF recipe once, or change package
choice. Do not fall back to local 8B iteration.

## After A Pass

Run one real paired split test using the same package, same sidecar, same prompt
shard, and same physical placement:

- baseline split serving;
- SPD split serving;
- report acceptance, saved/unsaved candidate round trips, content equality,
  tap failures, sidecar timing, stage timing, link latency, baseline wall time,
  SPD wall time, and latency-simulation estimate.

If only M4 + mini are available, use physical clumping but still require all
manifest taps.

## Role Of Qwen3-8B

`Qwen3-8B` is harness evidence only. It proved package-backed mechanics, tap
return, Rust sidecar loading, rolling verification, and M4/mini split wiring.
It did not prove predictor quality and is no longer the immediate SPD target.
