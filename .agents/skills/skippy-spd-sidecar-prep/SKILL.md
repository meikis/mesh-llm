---
name: skippy-spd-sidecar-prep
description: Use this skill when preparing, training, exporting, packaging, or validating SPD sidecar heads for Skippy, including topology/tap selection, local MPS/CUDA proof runs, Hugging Face job planning, safetensors export, parity fixtures, and request-path SPD smoke validation.
metadata:
  short-description: Prepare Skippy SPD sidecar heads
---

# skippy-spd-sidecar-prep

Use this skill for SPD sidecar preparation for Skippy. This covers deciding the
sidecar topology, running the reference trainer/evaluator, exporting a Skippy
serving artifact, and validating that the head is usable from Rust and live
Skippy taps.

## Critical Rules

- Treat SPD sidecars as topology-bound artifacts, not generic draft models. A
  head is tied to the base model/tokenizer, chat template, hidden size, logical
  SPD stage count, selected hidden-state taps, projection layout, draft vocab,
  and spec-layer count.
- Choose the target Skippy split topology before serious training. Physical
  stage placement can differ only if it exposes the same logical hidden-state
  taps required by the sidecar manifest.
- Do not claim real distributed speedup from single-host smokes. Separate model
  quality, Rust/Python parity, live tap correctness, request-path correctness,
  and distributed performance evidence.
- Do not replace real training/eval with unit-test-only evidence. Unit tests are
  useful gates, but SPD sidecar preparation needs real checkpoints, real hidden
  taps, parity fixtures, and live Skippy smoke runs.
- Hugging Face Jobs are spend-bearing. Default to dry-run planning, print the
  model, dataset, topology, hardware flavor, timeout, output repo, and maximum
  cost, and require explicit confirmation before submitting.

## Related Skills

- Use `hf-layer-package-jobs` when turning SPD training into a Hugging Face job
  flow or any other spend-bearing HF automation.
- Use `skippy-model-package` when preparing GGUF stage artifacts or validating
  physical split boundaries.
- Use `skippy-correctness` for full-model versus staged-execution parity.
- Use `skippy-bench` for SPD request-path smoke reports and benchmark summaries.

## Repo Entry Points

- `evals/spd/hf_train_eval_qwen06.py` trains/evaluates a real SPD speculation
  head by cloning and patching the reference SPD repository. It can also prepare
  an existing checkpoint from a local path or Hugging Face model repo.
- `evals/spd/export_spd_head.py` converts `speculation_head_final.pt` plus
  `skippy-spd-head.json` into Rust-readable `spd-head.safetensors`.
- `evals/spd/export_parity_fixture.py` exports real Python/reference hidden-tap
  rows, logits, top-k proposals, and cache fixtures for Rust parity checks.
- `evals/spd/README.md` is the live progress log and command cookbook for the
  current SPD proof.

## Topology Checklist

Before training or evaluating a sidecar, write down:

- Base model repo/ref and revision.
- Target GGUF artifact, quant, and layer count if this is meant for Skippy.
- Tokenizer and chat template settings; for Qwen, explicitly decide thinking
  versus no-thinking template behavior.
- Logical SPD stage count and `stage_layer_boundaries`.
- Explicit `shallow_hidden_layer_indices` if the reference trainer needs taps
  that do not match simple stage boundaries.
- `num_spec_layers`, draft vocab choice, and `draft_top_k` for evaluation.
- Physical Skippy split boundaries that expose every required hidden tap.

For the current pretrained Qwen3.5-4B S4/L4 proof, the tap-aligned physical
split is `8,10,16,20,24,31`, which exposes ranges
`0..8, 8..10, 10..16, 16..20, 20..24, 24..31, 31..32`.

## First Real-Node Target

Use the pretrained `Qwen/Qwen3.5-4B` S4/L4 sidecar before training a new head.
It is the first target because it already has strong reference acceptance,
Rust/Python parity, live Skippy tap parity, and a known tap-aligned split.

Expected local artifact paths in the current proof workspace:

- GGUF:
  `.artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf`
- Sidecar manifest:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json`
- Serving checkpoint:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-head.safetensors`
- Parity fixture:
  `/private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors`

Keep stage 0, the OpenAI frontend, and the SPD sidecar on the coordinator. Put
downstream physical stages on worker nodes or devices. With one worker node,
start with:

```bash
target/release/skippy-bench spd-openai-smoke \
  --stage-server-bin target/release/skippy-server \
  --manifest /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/skippy-spd-head.json \
  --fixture /private/tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/20260616-152346/train/spd-parity-fixture.safetensors \
  --model-path .artifacts/spd/qwen35-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --model-id unsloth/Qwen3.5-4B-GGUF:Q4_K_M \
  --splits 8,10,16,20,24,31 \
  --layer-end 32 \
  --ctx-size 128 \
  --n-gpu-layers -1 \
  --stage-hosts local,<worker>,<worker>,<worker>,<worker>,<worker>,<worker> \
  --endpoint-host-map <worker>=<worker-lan-ip-or-name> \
  --remote-model-path-map <worker>=/path/on/worker/Qwen3.5-4B-Q4_K_M.gguf \
  --max-tokens 1 \
  --repeat-count 1 \
  --output /tmp/spd-qwen35-first-remote-openai.json
```

Do not report speedup from this first remote smoke unless the report has enough
tokens/repeats and the hardware placement actually lets stages overlap. The
first purpose is to prove stage launch, hidden-tap return, sidecar proposal,
target verification, and cleanup across a real node boundary.

## Local Proof Flow

Use the M4/MPS local path for a small proof or overfit/debug run. Do not treat
it as the final 4B-quality training path.

```bash
python3 evals/spd/hf_train_eval_qwen06.py \
  --work-dir /tmp/skippy-spd-qwen06-proof \
  --model-name Qwen/Qwen3-0.6B \
  --dataset HuggingFaceH4/ultrachat_200k \
  --dataset-split train_sft \
  --train-rows 1024 \
  --eval-rows-per-set 8 \
  --num-stages 2 \
  --num-spec-layers 4 \
  --max-length 256 \
  --max-new-tokens 64 \
  --draft-top-k 4 \
  --device mps \
  --upload-repo ''
```

Use `--device cuda` on a GPU host. Keep `--upload-repo ''` for local dry runs
unless artifact upload is explicitly wanted.

## Export Flow

After training or downloading a reference checkpoint, export it to Skippy
serving format:

```bash
python3 evals/spd/export_spd_head.py \
  --checkpoint /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/speculation_head_final.pt \
  --manifest /tmp/skippy-spd-qwen35-4b-pretrained-s4l4/artifacts/<run-id>/train/skippy-spd-head.json \
  --base-model-path Qwen/Qwen3.5-4B
```

The expected serving bundle is:

- `speculation_head_final.pt`
- `skippy-spd-head.json`
- `spd-head.safetensors`
- `spd-parity-fixture.safetensors`
- eval summaries and raw per-sample acceptance traces

## Validation Flow

Validate in increasing order of realism:

```bash
SKIPPY_SPD_MANIFEST=/path/to/skippy-spd-head.json \
  cargo test -p skippy-runtime validates_external_manifest_when_skippy_spd_manifest_is_set

SKIPPY_SPD_PARITY_FIXTURE=/path/to/spd-parity-fixture.safetensors \
  cargo test -p skippy-runtime validates_external_parity_fixture_when_skippy_spd_parity_fixture_is_set

SKIPPY_SPD_MANIFEST=/path/to/skippy-spd-head.json \
SKIPPY_SPD_PARITY_FIXTURE=/path/to/spd-parity-fixture.safetensors \
  cargo test --release -p skippy-runtime qwen3_fixture_forward_matches_python_topk_when_env_is_set

cargo run -p skippy-bench -- spd-fixture-parity \
  --manifest /path/to/skippy-spd-head.json \
  --fixture /path/to/spd-parity-fixture.safetensors \
  --top-k 8
```

Then validate live Skippy taps with `skippy-bench spd-live-tap-parity`, followed
by `skippy-bench spd-openai-smoke` on the physical split topology that exposes
the manifest-required taps. Use release binaries for request-path timing.

## Evidence To Report

When reporting SPD sidecar status, include:

- Base model, revision, tokenizer/template mode, and GGUF/quant if applicable.
- Logical topology and physical Skippy split boundaries.
- Training dataset, row count, max length, epochs, batch/accumulation, learning
  rate, and draft vocab.
- Eval acceptance, equivalent accept length, theoretical gain, and generated
  token count.
- Rust/Python fixture parity, live tap parity, accepted/proposed counts, tap
  failures, and content-match status.
- Timing broken down into baseline decode, SPD decode, downstream wait, sidecar
  cache prefill, decoder layers, final norm, lm head/top-k, and head total when
  those fields are available.
