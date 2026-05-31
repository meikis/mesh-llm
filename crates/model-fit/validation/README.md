# model-fit validation corpora

These manifests define repeatable GGUF sets for `model-fit-validate`.

The lists are stratified by estimator behavior rather than popularity:

- tiny dense models check fixed decode overhead and low-active-byte behavior
- small dense models check the transition into memory-bandwidth-bound decode
- 7B/8B and coder models check common local serving shapes
- quant pairs check Q4/Q8 slope changes without changing architecture
- MoE models check active expert bytes instead of total expert bytes
- embedding/reranker models check workload suitability in metadata reports

Use the smoke set for self-hosted PR validation:

```bash
target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/smoke-models.txt \
  --output-json /tmp/model-fit-validation.json

target/release/model-fit-check-validation \
  --min-models 8 \
  /tmp/model-fit-validation.json
```

Use the deep set for manual or nightly validation on high-memory runners:

```bash
target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/deep-models.txt \
  --output-json /tmp/model-fit-validation-deep.json
```

`model-fit-validate --models-file` ignores blank lines and `#` comments.
