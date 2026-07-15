# Public Mesh

Use a public mesh when you want Mesh to find a published listing through the
Nostr directory. This is different from a private mesh joined with a shared
invite token and from LAN-only mDNS mode.

Join as a serving node:

```sh
mesh-llm serve --auto
```

Join as an API-only client:

```sh
mesh-llm client --auto
```

Open the console:

```text
http://localhost:3131
```

Point OpenAI-compatible tools at:

```text
http://localhost:9337/v1
```

For predictable first-run behavior on your own hardware, use the [Quickstart](/docs/pages/quickstart/) private mesh flow instead.
