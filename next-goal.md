# Next Goal: Train Qwen3-8B SPD Sidecar for 2-Node Split

This file is disposable and should be deleted when this immediate gate is done.
Durable status, evidence, caveats, and follow-up notes belong in
`evals/spd/README.md` and `docs/skippy/speculative_decoding.md`.

## Current Checkpoint

- Target model: `Qwen/Qwen3-8B`.
- Target Skippy package: `meshllm/Qwen3-8B-Q4_K_M-layers`.
- First real Mesh layout: two nodes, one Skippy stage per node.
- Layer split: node 0 `0..23`, node 1 `23..36`.
- SPD topology: `num_stages=2`, `stage_layer_boundaries=23,36`.
- Required sidecar tap rows: `0,23,36;0,23`.
- Worker tap-return indices: `[23,36]`.
- Product-row micro-finetunes proved the Rust/Skippy wiring but are not good
  enough for speed claims. The train56 mixed product-row head scores
  `788 / 1096` on train rows but only `28 / 128` held-out, while the HF teacher
  matches native Q4 on `120 / 128`. The blocker is sidecar generalization, not
  tap plumbing or HF/Q4 drift.
- Existing real two-node direct-cable smoke with the weak product-row head
  matched content and had `0` tap failures, but accepted only `78 / 156`
  (`paper_pipeline_estimate=1.0x`) and measured `0.321x` of baseline because
  sidecar/native overhead dominated. Treat this as mechanics evidence only.

## Immediate Objective

Train and validate a real topology-matched Qwen3-8B SPD sidecar:

- HF spend below `$50`.
- Use the paper/reference KD path, not more tiny product-row finetuning.
- Use `HuggingFaceH4/ultrachat_200k` `train_sft` for the first real run.
- First run target: 16k rows, max length `2048`, LR `1e-4`, BF16,
  `num_spec_layers=4`, draft top-k `4`.
- Upload artifacts to `meshllm/skippy-spd-qwen3-8b-s2-23` with explicit
  topology metadata.
- Export `spd-head.safetensors` and `spd-parity-fixture.safetensors`.
- Validate Rust fixture parity and live Skippy tap parity for
  `--splits 23 --layer-end 36`.
- Only after held-out acceptance improves with margin, run the real M4+mini
  split smoke and report content match, tap failures, accepted/proposed,
  saved/unsaved token round trips, and timings.

## No-Physical-Split Quality Gate

Do not use the M4+mini split as the first test of whether the sidecar is good.
The predictor can be trained and scored before any physical split:

1. Reference/HF held-out eval: report acceptance rate, equivalent accepted
   length, top-k target coverage, and theoretical saved decoder steps.
2. Rust fixture parity: prove the exported safetensors sidecar makes the same
   proposals as Python for fixed hidden-tap rows.
3. Local live-tap parity: run the logical Skippy split on one machine and verify
   returned taps at `23` and `36` feed the sidecar correctly.
4. Local package-backed SPD smoke: compare baseline versus SPD and require
   nonzero accepted proposals plus nonzero saved candidate token round trips.

Only after those pass should the real two-node split be used. The physical run
then validates QUIC/LAN behavior, per-stage KV cleanup, endpoint placement, and
timing under actual node latency; it should not be treated as the first
sidecar-quality measurement.

## Logical Topology Rule

Train sidecars for logical layer-boundary topologies, not hostnames. With one
Skippy stage per node, the first two-node Qwen3-8B run uses the `23,36`
sidecar. If a future deployment packs adjacent logical stages onto one larger
node, it may reuse the same sidecar only if the runtime still exposes every
logical boundary tap the manifest requires.

This avoids a full combinatorial explosion: precompute a small set of canonical
logical topologies, then clump contiguous logical stages during placement. The
tradeoff is that clumped logical stages may lose some physical overlap, so they
must be benchmarked honestly, but they do not require retraining if the tap
topology is unchanged.

## Next Actions

1. Monitor HF job `meshllm/6a33e49bef9220ea67d991c2`. It is an
   `a100-large` job at `$2.50/hr` with an `18h` timeout, so the hard cap is
   `$45`.
2. Cancel if it is clearly stuck, switches away from CUDA, or trends beyond the
   spend cap.
3. If the 16k checkpoint completes, inspect the reference held-out acceptance
   summary before doing any physical split work.
4. If acceptance is promising, download/export the bundle and run Rust fixture
   parity, local live-tap parity, and local package-backed baseline-versus-SPD.
5. If 16k acceptance is promising but thin, consider a 64k follow-up only if it
   still fits the remaining spend cap.
