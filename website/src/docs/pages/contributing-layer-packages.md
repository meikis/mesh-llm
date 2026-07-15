# Contributing Layer Packages

Layer packages let Mesh place a model across multiple machines without every node downloading the full model. A package records the source model, quantization, layer artifacts, and validation metadata. See the [model package specification](/docs/pages/model-package-spec/) for the `model-package.json` contract.

## Local contribution flow

Create or validate a package locally, then publish the package repository to Hugging Face.

```sh
mesh-llm models show unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M
```

After publishing, open a pull request against the catalog dataset entry. The PR should include:

- Source model repo and revision.
- Source GGUF filename and quantization.
- Layer package repo.
- Package manifest metadata.
- Validation result.

## Hugging Face contribution flow

If the package is generated on Hugging Face infrastructure, publish the package repository first. Then submit the catalog change as a dataset PR to:

[meshllm/catalog](https://huggingface.co/datasets/meshllm/catalog)

On the next website deployment, the build reads the same canonical `entries/**/*.json` tree as the CLI and publishes a same-origin catalog snapshot. No separately maintained row projection needs to catch up before the model appears. A maintainer can also run the Public Website Deploy workflow manually when a catalog change should be published immediately.

## Catalog PR behavior

Catalog PRs should be reviewable as metadata changes. The source of truth for both the website and CLI is the catalog entry tree and its referenced package repositories.
