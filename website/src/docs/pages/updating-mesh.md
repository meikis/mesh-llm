# Update

Update Mesh with one command:

```sh
mesh-llm update
```

Restart the local node after updating:

```sh
mesh-llm stop
mesh-llm serve --auto
```

For a multi-node private mesh, restart each joining node with its invite token;
a mesh name alone does not reconnect it:

```sh
mesh-llm serve --join <invite-token> --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

A standalone private node can start a new mesh with `--mesh-name` and will print
a new invite token for machines added later.

## Switch flavors

Use `--flavor` when you want to switch the installed release bundle:

```sh
mesh-llm update --flavor cuda
mesh-llm update --flavor rocm
mesh-llm update --flavor vulkan
mesh-llm update --flavor cpu
```

Apple Silicon:

```sh
mesh-llm update --flavor metal
```

To re-detect the best flavor for the current machine:

```sh
mesh-llm update --detect-flavor
```

`--detect-flavor` cannot be combined with `--flavor`.

## Install a specific version

```sh
mesh-llm update --version v0.X.Y
mesh-llm update --version v0.X.Y --flavor vulkan
```

Update serving nodes one at a time when a mesh is actively handling traffic. Mixed-version meshes are expected to keep operating during rolling updates.
