# mesh-llm crate

`mesh-llm` is the binary/app assembly crate.

Its Rust source is intentionally tiny:

```text
src/
├── lib.rs     compatibility re-export of mesh-llm-host-runtime
└── main.rs    binary entrypoint
```

Host runtime implementation lives in
[`../mesh-llm-host-runtime`](../mesh-llm-host-runtime). Shared subsystem crates
own the reusable pieces:

- [`../mesh-llm-ui`](../mesh-llm-ui) for the web console
- [`../mesh-llm-types`](../mesh-llm-types) for pure shared model and mesh types
- [`../mesh-llm-identity`](../mesh-llm-identity) for owner identity/envelope crypto
- [`../mesh-llm-protocol`](../mesh-llm-protocol) for node protobuf schema and frame helpers
- [`../mesh-llm-routing`](../mesh-llm-routing) for shared routing target primitives
- [`../mesh-llm-system`](../mesh-llm-system) for local hardware/backend/update helpers
- [`../mesh-llm-plugin`](../mesh-llm-plugin) for plugin author API and plugin protobuf schema
- existing `model-*`, `openai-frontend`, and `skippy-*` crates for their domains

The compatibility re-export includes the shared config authoring API under
`mesh_llm::sdk::config`. Embedders that need to update `config.toml`, including
model-specific speculative decoding settings, should use `ConfigStore::update`
and the typed editors from [`../mesh-llm-config`](../mesh-llm-config) instead
of writing TOML directly.

For install and end-user usage, see the [project README](../../README.md).
