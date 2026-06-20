# Next Goal: Qwen480 SPD Overfit Alignment Proof

This file is disposable. Durable evidence belongs in `evals/spd/README.md` and
`docs/skippy/speculative_decoding.md`.

## One-Line Goal

Prove the Qwen3-Coder-480B S8 native SPD serving path can accept an
intentionally overfit sidecar on the exact package-backed prompts before
spending on larger `16k` / `64k` / paper-scale training.

## Why This Comes First

The Qwen480 predictor has not been trained with enough data to judge final
quality. The paper used about `1.2M` filtered samples; our Qwen480 lanes have
used `2048` and `8192` native-Q4 train samples.

But the current contradiction is structural: the `2048`-sample sidecar had
nonzero offline signal and still served `0 / 256` accepted proposals. The
`8192`-sample lane improved offline signal, then failed before package smoke on
artifact/serving contract issues. More data is not useful until the trained
head, exported manifest, Rust fixture path, live Skippy taps, and package smoke
all agree on the same row/projection/draft-vocab meaning.

## Paper And Quant Grounding

The SPD paper can be downloaded at `https://arxiv.org/pdf/2605.30852`; the
reference implementation is `https://github.com/yuyijiong/speculative_pipeline_decoding`.
Do not rely on a local `~/Downloads/spd.pdf` copy for reproduction.

The paper constrains the training target and hidden-state geometry, not the
sidecar storage dtype. It freezes the target model, gathers multi-depth hidden
states from the target pipeline, and trains only the Speculation Module with KD
against the target logits. For a Skippy product proof, those target logits and
taps must come from the exact native quantized package being served, such as
Q4/Q8 GGUF layer packages. The sidecar itself may still be BF16/F32 while it
learns that quantized target distribution.

Do not quantize the sidecar as the first proof step. Sidecar quantization is an
optimization after a BF16/F32 sidecar shows package-backed served acceptance.
Starting with a quantized sidecar would add another approximation, another Rust
loader/kernel requirement, and another failure mode before proving the core
question: whether the topology-matched predictor accepts tokens against the
native quantized target.

## Immediate Acceptance Test

Run a tiny diagnostic lane for the same Qwen480 S8 topology:

- package: `meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers`
- logical boundaries: `8,16,24,32,40,48,55,62`
- required taps: `[0,8,16,24,32,40,48,55,62]`
- train on the same held-out product rows used by package smoke
- use native Q4 verifier logits and the corpus-frequency draft vocab order
- require fixed-row Rust/Python parity
- require live-row reconstruction parity
- require package-backed smoke on those same prompts

Pass means: content matches, tap failures are zero, proposals are produced, and
an intentionally overfit head gets nonzero served acceptance. That proves the
serving path is aligned and data scale is the next lever.

Fail means: do not buy more rows. Fix the reported row/projection/live-tap or
Rust/Python forward mismatch first.

## Heavy-Train Gate

Do not make `16k`, `64k`, or paper-scale training the main path until the
overfit proof, fixed-row parity, live-row reconstruction, and package-backed
acceptance all pass. If a sunk-cost GPU window is available, a heavy train may
run only as background speculation with strict caps and checkpointing; do not
interpret its result or let it steer the plan until the alignment gates are
green.

## Current Running Job

No SPD HF job is currently running. `hf jobs ps --namespace meshllm` returned
`No jobs found` after canceling `meshllm/6a36251f3093dba73ce2ab39`
(`spd-qwen480-quality-8k-draft-vocab-fix`). Do not launch another
spend-bearing job until the overfit alignment plan is reviewed and dry-run
checked again.

## Next Command Shape

Use the existing planner shape before submitting anything spend-bearing:

```bash
python3 evals/spd/plan_hf_spd_qualification.py \
  --qualification-mode native-package-fresh \
  --overfit-serving-prompts \
  --base-model Qwen/Qwen3-Coder-480B-A35B-Instruct \
  --package-ref meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers \
  --num-stages 8 \
  --stage-layer-boundaries 8,16,24,32,40,48,55,62 \
  --num-spec-layers 4 \
  --draft-top-k 4 \
  --draft-vocab-size 32000 \
  --vocab-size 151936 \
  --heldout-prompts 8 \
  --verify-steps 4 \
  --flavor rtx-pro-6000x4 \
  --timeout 2h \
  --max-cost-usd 25 \
  --smoke-stage-backend-devices CPU,CUDA0,CPU,CUDA1,CPU,CUDA2,CPU,CUDA3 \
  --json
```

Dry-run expectations: command graph must avoid `AutoModelForCausalLM`,
`hf_train_eval_qwen06.py`, and full-base `from_pretrained(`. It should train
from native package-captured product rows and native teacher logits. In overfit
mode, the prompt build must use `--draft-vocab-source heldout` so the draft
vocabulary is built from the serving rows being overfit.

## After This

- If the overfit proof accepts served proposals, scale the same native-Q4
  mixed-data recipe to `16k`, then `64k`, then paper-like data if acceptance
  improves. At that point a sunk-cost compute window can be used aggressively,
  because the remaining variable is data/recipe quality rather than serving
  alignment.
- If it fails, keep the work on artifact contract and live-row alignment. Do
  not run larger training until the overfit proof passes.
