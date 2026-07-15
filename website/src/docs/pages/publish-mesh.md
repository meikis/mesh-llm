# Publish Mesh

A published mesh advertises an invite through the public Nostr directory. Use
it when people should be able to find the mesh without receiving its invite
token out of band.

Start a serving node:

```sh
mesh-llm serve --publish --mesh-name my-public-mesh --model <model-ref>
```

Other serving machines can discover it by name:

```sh
mesh-llm serve --discover my-public-mesh --model <model-ref>
```

Join as an API-only client:

```sh
mesh-llm client --discover my-public-mesh
```

Publishing is separate from admission policy. Apply owner, trust, or release
attestation requirements when a publicly listed mesh must restrict which nodes
can join.
