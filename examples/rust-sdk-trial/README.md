# rust-sdk-trial

End-to-end proof that a Rust app can depend on the mesh-llm SDK as a
normal cargo dep and run a real in-process mesh node.

This example lives outside the workspace (it declares its own
`[workspace]` table) on purpose: it depends on mesh-llm crates the way
any external Rust app would.

## What it does

Mirrors `mesh-llm client --auto`:

1. Generates an owner keypair via `OwnerKeypair::generate()`.
2. Discovers public meshes through Nostr via `create_auto_node`.
3. Picks the best mesh and starts a node against it.
4. Lists the models the mesh exposes.
5. Cleanly stops the node.

## Run it

You need to be inside a mesh-llm checkout that has the patched llama.cpp
static archives built locally and packaged as a tarball.

```bash
# 1. Build the patched llama.cpp static archives (once per backend).
just llama-prepare
just llama-build

# 2. Package the static archives into a tarball + sha256.
mkdir -p dist/llama-stage-static/aarch64-apple-darwin-metal
cd .deps/llama-build/build-stage-abi-metal
for f in CMakeCache.txt src/libllama.a tools/mtmd/libmtmd.a \
         common/libllama-common.a common/libllama-common-base.a \
         ggml/src/libggml.a ggml/src/libggml-base.a \
         ggml/src/libggml-cpu.a \
         ggml/src/ggml-metal/libggml-metal.a; do
    [ -f "$f" ] && mkdir -p "../../dist/llama-stage-static/aarch64-apple-darwin-metal/$(dirname "$f")" \
      && cp "$f" "../../dist/llama-stage-static/aarch64-apple-darwin-metal/$f"
done
cd ../../dist/llama-stage-static
tar czf llama-stage-aarch64-apple-darwin-metal.tar.gz aarch64-apple-darwin-metal/
shasum -a 256 llama-stage-aarch64-apple-darwin-metal.tar.gz > llama-stage-aarch64-apple-darwin-metal.tar.gz.sha256

# 3. Build and run the example. SKIPPY_LLAMA_TARBALL_URL tells
# skippy-ffi's build.rs where to find the prebuilt static archives.
cd examples/rust-sdk-trial
SKIPPY_LLAMA_TARBALL_URL="file://$(pwd)/../../dist/llama-stage-static/llama-stage-aarch64-apple-darwin-metal.tar.gz" \
  cargo build
./target/debug/rust-sdk-trial
```

## Expected output

A real run against the live public mesh looks like:

```
rust-sdk-trial: starting
rust-sdk-trial: owner keypair generated (first 16 hex = bfa666ac84d93700)
rust-sdk-trial: discovering and joining a public mesh...
rust-sdk-trial: selected mesh = (unnamed) (nodes=5, vram=880.4 GB, region=None)
rust-sdk-trial: mesh serving models = ["unsloth/MiniMax-M2.5-GGUF:Q4_K_M", "unsloth/Qwen3-8B-GGUF@main:Q4_K_M", "unsloth/Qwen3.5-9B-GGUF:Q4_K_M"]
rust-sdk-trial: starting in-process node...
rust-sdk-trial: node started
rust-sdk-trial: ...
rust-sdk-trial: node stopped
```

(The exact mesh and models depend on what's published when you run it.)

## What this proves

- The mesh-llm public Rust API (`MeshNode::builder()`, `OwnerKeypair`,
  `create_auto_node`, `PublicMeshQuery`) is callable from a Rust app
  outside the workspace.
- `skippy-ffi/build.rs` successfully fetches prebuilt patched-llama.cpp
  static archives from a tarball URL at consumer build time.
- The resulting consumer binary statically links the entire mesh-llm
  runtime; no `mesh-llm` daemon, no `.dylib` to bundle, no CMake on
  the consumer's machine.
- Real public-mesh discovery via Nostr works end-to-end.

See `docs/design/RUST_NATIVE_SDK.md` for the full design.
