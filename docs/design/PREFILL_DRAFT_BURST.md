# Prefill Draft Burst

Status: experimental evaluation branch.

This revisits the PR #567 idea on top of the current Skippy split-serving path.
The goal is to reduce mesh decode round trips when a draft model can cheaply
propose a short span before normal target decode starts.

## Shape

After target prefill completes, stage 0 can ask the configured draft model for a
front-loaded token burst, verify that burst through the existing distributed
`VerifySpan` path, and commit the accepted prefix before entering the normal
decode loop.

```text
target split prefill
draft proposes N tokens
target split verifies N tokens in one VerifySpan hop chain
commit accepted draft prefix
normal decode/speculative decode handles the tail
```

This is different from ordinary speculative decode. It is aimed at mesh
topologies where each target decode token pays a downstream stage round trip,
while a verification span can cover many candidate tokens at once.

## Current Assessment

This likely has legs for split serving, even if it is not obviously useful for
single-process local serving. The bet is not that draft inference is always
cheaper than target inference; it is that a good draft can replace many
per-token stage hop chains with one span verification pass.

The idea gets more interesting as draft, MTP, and speculative-tuned model pairs
improve. Higher draft acceptance means more decoded tokens can be committed
after one distributed verification, and mesh topologies amplify that win because
every accepted token avoids another serialized downstream wait.

The default should stay exact-prefix verification. Approximate mode may be
useful for conversational output when the model returns to agreement after an
isolated mismatch, but it is a quality tradeoff and should be measured
separately. It should remain off for tools, JSON, structured output, and strict
code edits unless there is explicit evidence that drift is acceptable.

Main risks:

- local-only benchmarks can understate the value because they miss hop latency
- approximate commits can drift the response even when later tokens realign
- target and draft tokenization or prompt handling must remain exactly aligned
- partial accepts require stage rollback or repair so downstream KV state matches
  the emitted prefix

## Knobs

In `config.toml`:

```toml
[models.speculative]
mode = "draft"
draft_model_path = "/path/to/draft.gguf"
draft_max_tokens = 8
prefill_draft_burst_tokens = 16
prefill_draft_max_consecutive_mismatches = 0
```

`prefill_draft_burst_tokens = 0` disables the burst. A mismatch tolerance of
`0` keeps exact-prefix behavior. Higher values allow approximate drift by
committing isolated draft mismatches when the span later returns to agreement.

For direct `skippy-server serve-binary` experiments, use:

```bash
--openai-draft-model-path /path/to/draft.gguf \
--openai-speculative-window 8 \
--openai-prefill-draft-burst-tokens 16 \
--openai-prefill-draft-max-consecutive-mismatches 0
```

## Evaluation

The first useful benchmark is a 2-node Skippy split with artificial RTT or real
Wi-Fi latency:

- baseline: no draft
- current speculative decode only: `draft_max_tokens > 0`, burst disabled
- prefill burst exact: burst enabled, mismatch tolerance `0`
- prefill burst approximate: burst enabled, mismatch tolerance `1`

Track accepted burst tokens, raw matches, tolerated mismatches, downstream wait,
time to first emitted text, and total response wall time. Approximate mode should
stay off for tool calls, structured output, and strict code edits until drift is
measured.
