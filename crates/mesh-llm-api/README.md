# mesh-llm-api

`mesh-llm-api` is currently the public Rust SDK crate for embedding Mesh in
applications. It is being reworked from a client-first surface into the
node-oriented SDK described in
`docs/design/EMBEDDED_CLIENT_ADR.md`.

The target public concept is `MeshNode`: one embedded node that can:

- join a mesh and consume inference
- search, inspect, download, install, delete, and clean up local models
- load and unload local models for serving
- observe connection, model, serving, and request lifecycle events

Layering:

- `crates/mesh-client/` implements the low-level client behavior
- `crates/mesh-llm-api/` exposes the Rust SDK surface
- `crates/mesh-llm-node/` owns embeddable model management and serving
  orchestration as it is extracted from the host runtime. Serving is an
  in-process SDK boundary; REST management remains an external-daemon adapter.
- `crates/mesh-llm-host-runtime/` provides the reference `ServingController`
  implementation by attaching `MeshApi` to `MeshNodeBuilder` and forwarding
  load/unload to the runtime-control loop.
- `crates/mesh-llm-ffi/` wraps `crates/mesh-llm-api/` for Swift, Kotlin, and other native
  bindings

Serving APIs use model refs for load, explicit model-or-instance unload
targets, drain/force unload options, rich served-model status, and typed
high-level serving errors.

The current `MeshClient` API remains an extraction artifact until it is replaced
or wrapped by `MeshNode` and the `node.inference()`, `node.models()`, and
`node.serving()` namespaces.

If an API is meant for app integration, it should live here rather than in
`crates/mesh-client/`.
