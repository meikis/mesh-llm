# Private Meshes

A private mesh connects your own machines with an invite token. A mesh name is
only a label: using the same name on two commands does not connect them, and it
does not publish a private mesh for discovery.

## Start the first serving node

```sh
mesh-llm serve --mesh-name my-private-mesh --model unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
```

Keep this terminal open and copy the invite token it prints.

## Add another serving machine

Install Mesh on the second machine, then use the invite token from the first
machine:

```sh
mesh-llm serve --join <invite-token> --model <model-ref>
```

## Join as an API-only client

Use this for a laptop that should send requests but not serve a model:

```sh
mesh-llm client --join <invite-token>
```

## Keep discovery and transport on the LAN

The default private-mesh commands above use direct connections when available
and can fall back to managed relays. If every machine is on the same LAN and
you explicitly want LAN-only discovery and transport, select mDNS mode on both
machines:

```sh
# First machine
mesh-llm serve --mesh-discovery-mode mdns --mesh-name my-private-mesh --model <model-ref>

# Additional machine
mesh-llm serve --mesh-discovery-mode mdns --join <invite-token> --model <model-ref>
```

The invite token is still required. mDNS advertisements intentionally do not
contain reusable invite tokens.

Mesh includes its own mDNS implementation; Avahi is not a Mesh dependency.
The LAN firewall must allow mDNS multicast on UDP port `5353`. On NixOS, opening
UDP `5353` directly is sufficient for mDNS discovery; enabling Avahi with its
firewall option is another way to open that port. For the most reliable
relay-less direct path, also allow Mesh's LAN dial-back beacon on UDP `47654`;
that port is Mesh traffic, not mDNS or Avahi. If inbound UDP is otherwise
blocked, set `--bind-port <port>` on each node and allow that UDP port for the
actual QUIC connection.

## Check that peers are visible

Open the console:

```text
http://localhost:3131
```

Or check status:

```sh
curl -s http://localhost:3131/api/status | jq .
```

Private meshes are useful for lab machines, office workstations, or a home
cluster where only invited machines should join.
