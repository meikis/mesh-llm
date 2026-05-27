# mesh-llm-host-runtime

`mesh-llm-host-runtime` composes the host-side mesh node runtime. It wires
model resolution, local serving, discovery, networking, runtime state, plugins,
the management API, and the shipped CLI entrypoint used by the `mesh-llm`
binary.

This crate is being split so reusable CLI, TUI, SDK, and embeddable runtime
surfaces can be published and consumed independently.

## Config API

The host runtime re-exports the shared config authoring surface through
`mesh_llm_host_runtime::sdk::config` and `mesh_llm_host_runtime::plugin`. Use
that API when host-runtime consumers need to read or write `config.toml`; avoid
hand-writing TOML or duplicating config structs in runtime callers.

Speculative model settings use the same high-level editor API as other model
settings:

```rust
use mesh_llm_host_runtime::sdk::config::ConfigStore;

let store = ConfigStore::default_path()?;
store.update(|config| {
    config
        .upsert_model("meta-llama/Llama-3.3-70B-Instruct-GGUF:Q3_K_M")?
        .speculative()
        .mode("draft")
        .draft_hf_source(
            "unsloth/Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        )
        .draft_selection_policy("manual")
        .draft_max_tokens(16);
    Ok(())
})?;
```

Schema ownership, validation, and persistence live in
[`../mesh-llm-config`](../mesh-llm-config).
