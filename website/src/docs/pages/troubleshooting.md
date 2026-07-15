# Troubleshooting

Start with these checks before changing configuration.

## Report a problem

Use `mesh-llm doctor` when you need local status, runtime diagnostics, or logs for a bug report. It is for troubleshooting, not the normal install flow.

Capture a doctor archive:

```sh
mesh-llm doctor
```

Open a [new GitHub issue](https://github.com/Mesh-LLM/mesh-llm/issues/new) and attach the archive created by `mesh-llm doctor`. Include the command you ran, your OS, GPU/backend flavor, model ref, whether you used a private mesh or `--auto`, and what you expected to happen.

## Is Mesh running?

```sh
curl -s http://localhost:3131/api/status | jq .
```

If this fails, start a node:

```sh
mesh-llm serve --mesh-name my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

## Is a model available?

```sh
curl -s http://localhost:9337/v1/models | jq '.data[].id'
```

If no models are listed, the model did not load or no serving peer is available. Try a smaller model:

```sh
mesh-llm stop
mesh-llm serve --mesh-name my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

## Is the console reachable?

Open:

```text
http://localhost:3131
```

If the console is not reachable, another process may be using the port or the node may not be running.

## Stop stale local processes

```sh
mesh-llm stop
```

If you are developing from source, use the repository cleanup commands in the testing docs.

## Agent fails but console works

List models and pass one explicitly:

```sh
mesh-llm goose
```

## Public mesh connection issues

For first-run testing, prefer a private mesh:

```sh
mesh-llm serve --mesh-name my-private-mesh --model unsloth/gemma-4-E2B-it-GGUF:UD-Q4_K_XL
```

Then move back to `mesh-llm serve --auto` once the local install and model path work.

## Private machines do not become peers

Do not try to connect private nodes by reusing a mesh name. A name is only a
label. Start the first node, copy its invite token, and pass that token to each
additional machine:

```sh
# First machine
mesh-llm serve --mesh-name my-private-mesh --model <model-ref>

# Additional machine
mesh-llm serve --join <invite-token> --model <model-ref>
```

If you selected `--mesh-discovery-mode mdns` for LAN-only operation, use it on
both machines. Mesh implements mDNS in-process and does not require Avahi, but
the host firewall must allow mDNS multicast on UDP port `5353`. On NixOS you
can allow UDP `5353` directly; `services.avahi.openFirewall` is another way to
open it when Avahi is enabled. For the most reliable relay-less direct path,
also allow Mesh's LAN dial-back beacon on UDP `47654`; that port is Mesh
traffic, not mDNS or Avahi. If inbound UDP is otherwise blocked, also choose a
fixed `--bind-port` on each node and allow that UDP port for the actual QUIC
connection.
