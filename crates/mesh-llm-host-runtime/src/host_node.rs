//! In-process mesh node entry point for the published SDK.
//!
//! This is the bridge between [`mesh-llm-api-server`][api-server] (the
//! public Rust SDK) and the real iroh-backed mesh node implementation in
//! `crate::mesh::Node`. Without this module the SDK can only run an
//! HTTP-shim "client" that flips `connected = true` and emits an event;
//! with it, a Rust application can `cargo add mesh-llm-api-server --features
//! host-runtime` and run an actual mesh peer that does gossip, relay
//! registration (including [`--relay-auth`][relay-auth]), invite tokens,
//! and QUIC peer connections — the same things the `mesh-llm` binary
//! does.
//!
//! [api-server]: https://docs.rs/mesh-llm-api-server
//! [relay-auth]: https://github.com/Mesh-LLM/mesh-llm/pull/641
//!
//! ## Scope
//!
//! This module deliberately exposes only what the SDK needs:
//!
//! - [`HostNodeSpec`] — what the SDK passes in (role, relays, relay auths,
//!   QUIC bind, VRAM cap, enumerate-host flag).
//! - [`HostNode`] — the handle the SDK gets back (`invite_token`,
//!   `start_accepting`, `id`, `shutdown`).
//! - [`start_host_node`] — the entry point.
//!
//! It does not expose the full `mesh::Node` API. Internals stay
//! `pub(crate)` so we can keep refactoring without breaking SDK
//! consumers.
//!
//! ## What this does not do (yet)
//!
//! - It does not start the OpenAI HTTP proxy. That's the
//!   `mesh-llm serve` runtime's responsibility and lives in
//!   `crate::runtime`. An SDK consumer who wants the proxy should call
//!   into [`run_with_args`][crate::run_with_args] with the relevant
//!   flags; that path drives a full runtime including proxy + console.
//! - It does not start local model serving. That requires plugging an
//!   `EmbeddedServingController` from [`crate::sdk`] into the SDK's
//!   `MeshNodeBuilder`.

use crate::mesh::{self, NodeRole, QuicBindSelection, RelayConfig};
use anyhow::Result;
use std::collections::HashMap;

/// Configuration for [`start_host_node`].
///
/// Field shape mirrors the slice of `mesh-llm`'s CLI flags that
/// `crate::mesh::Node::start` consumes. New fields here track new CLI
/// flags as they get added.
#[derive(Clone, Debug, Default)]
pub struct HostNodeSpec {
    /// Mesh role.
    pub role: NodeRole,
    /// iroh relay URLs (empty = use bundled defaults).
    pub relays: Vec<String>,
    /// Per-relay bearer tokens for gated iroh relays. Sent as
    /// `Authorization: Bearer <TOKEN>` on the WebSocket upgrade to the
    /// matching relay URL.
    pub relay_auths: HashMap<String, String>,
    /// Local QUIC bind selection (IP and/or port).
    pub quic_bind: QuicBindSelection,
    /// VRAM cap in GB. `Some(0.0)` for client-only nodes that should not
    /// advertise any VRAM.
    pub max_vram_gb: Option<f64>,
    /// Whether to publish a hardware survey to gossip.
    pub enumerate_host: bool,
}

/// A running mesh node started by [`start_host_node`].
///
/// Call [`HostNode::shutdown`] to stop the iroh endpoint and tear down
/// background tasks. Dropping this handle alone is not the shutdown
/// contract because spawned mesh tasks hold their own node clones.
#[derive(Clone)]
pub struct HostNode {
    inner: mesh::Node,
}

impl HostNode {
    /// Start accepting incoming mesh connections.
    ///
    /// The iroh endpoint binds in [`start_host_node`], but the accept
    /// loop waits for this call so the embedder can finish wiring (set a
    /// display name, advertise models) before the node is reachable.
    pub fn start_accepting(&self) {
        self.inner.start_accepting();
    }

    /// Hex-formatted endpoint ID, suitable for logging.
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// An invite token that other nodes can use to join this one.
    pub fn invite_token(&self) -> String {
        self.inner.invite_token()
    }

    /// Join an existing mesh via an invite token produced elsewhere.
    pub async fn join(&self, invite_token: &str) -> Result<()> {
        self.inner.join(invite_token).await
    }

    /// Set a human-readable display name advertised to peers.
    pub async fn set_display_name(&self, name: String) {
        self.inner.set_display_name(name).await;
    }

    /// Replace the set of models this node advertises.
    pub async fn set_models(&self, models: Vec<String>) {
        self.inner.set_models(models).await;
    }

    /// Current set of advertised models.
    pub async fn models(&self) -> Vec<String> {
        self.inner.models().await
    }

    /// Shut the node down (best-effort).
    pub async fn shutdown(&self) {
        self.inner.shutdown().await;
    }
}

/// Bring an in-process mesh node online with the given spec.
///
/// Equivalent to the iroh-endpoint slice of `mesh-llm serve` / `mesh-llm
/// client`: binds the iroh endpoint, attaches relay-auth tokens, waits
/// briefly for the home relay to come online, and returns a handle. The
/// caller is responsible for any further wiring (calling
/// [`HostNode::start_accepting`], setting models / display name, joining
/// other meshes via [`HostNode::join`]).
pub async fn start_host_node(spec: HostNodeSpec) -> Result<HostNode> {
    let relay = RelayConfig {
        urls: &spec.relays,
        auths: &spec.relay_auths,
    };
    let (node, _channels) = mesh::Node::start(
        spec.role,
        relay,
        spec.quic_bind,
        spec.max_vram_gb,
        spec.enumerate_host,
        None, // owner control config — not currently exposed to SDK
        None, // config file — not relevant to SDK consumers
    )
    .await?;
    Ok(HostNode { inner: node })
}

// Re-export the types embedders need to express a spec. Hidden inside
// the curated `host_node` namespace, NOT at the crate root, so we can
// keep refactoring the underlying `mesh` module.
pub use mesh::{NodeRole as MeshNodeRole, QuicBindSelection as MeshQuicBindSelection};

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::endpoint::{presets, Endpoint, RelayMode};
    use iroh::SecretKey;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr, UdpSocket};
    use std::time::Duration;

    fn free_local_udp_port() -> u16 {
        let socket = UdpSocket::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .expect("allocate local UDP port");
        socket.local_addr().expect("read local UDP port").port()
    }

    async fn probe_quic_port_released(port: u16) -> anyhow::Result<()> {
        let bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        let mut last_error = None;

        for _ in 0..20 {
            match Endpoint::builder(presets::Minimal)
                .secret_key(SecretKey::generate())
                .relay_mode(RelayMode::Disabled)
                .bind_addr(bind_addr)?
                .bind()
                .await
            {
                Ok(endpoint) => {
                    endpoint.close().await;
                    return Ok(());
                }
                Err(err) => {
                    last_error = Some(err);
                    tokio::time::sleep(Duration::from_millis(25)).await;
                }
            }
        }

        Err(anyhow::anyhow!(
            "host-node shutdown should release UDP port {port}: {:?}",
            last_error
        ))
    }

    #[tokio::test]
    async fn id_returns_bare_hex_endpoint_id() -> anyhow::Result<()> {
        let inner = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let expected = inner.id().to_string();
        let node = HostNode { inner };

        assert_eq!(node.id(), expected);
        assert!(!node.id().contains("PublicKey"));

        node.shutdown().await;
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_closes_the_mesh_endpoint() -> anyhow::Result<()> {
        let inner = mesh::Node::new_for_tests(mesh::NodeRole::Client).await?;
        let node = HostNode { inner };

        node.shutdown().await;

        assert!(node.inner.endpoint_is_closed_for_tests());
        Ok(())
    }

    #[tokio::test]
    async fn shutdown_releases_fixed_quic_bind() -> anyhow::Result<()> {
        let quic_port = free_local_udp_port();
        let node = start_host_node(HostNodeSpec {
            role: MeshNodeRole::Client,
            quic_bind: MeshQuicBindSelection {
                ip: Some(IpAddr::V4(Ipv4Addr::LOCALHOST)),
                port: Some(quic_port),
            },
            max_vram_gb: Some(0.0),
            enumerate_host: false,
            ..HostNodeSpec::default()
        })
        .await?;

        node.start_accepting();
        node.shutdown().await;
        drop(node);

        probe_quic_port_released(quic_port).await
    }
}
