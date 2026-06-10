//! How MLX nodes talk to each other under mesh orchestration.
//!
//! ## Can MLX use mesh's QUIC/iroh transport? No — not natively.
//!
//! `mx.distributed.init(backend=...)` only accepts `{any, mpi, ring, nccl,
//! jaccl}`. The Ring backend opens its **own** TCP sockets (`socket`/`bind`/
//! `listen`/`accept`/`connect`) to IP:port pairs from a hostfile; JACCL uses
//! RDMA verbs over Thunderbolt. There is no hook to inject a custom byte
//! transport, so MLX cannot speak mesh's QUIC tunnel directly.
//!
//! That leaves two ways for mesh to wire an MLX cluster:
//!
//! 1. [`MeshTransport::LanRing`] — let MLX form its own Ring (TCP) or JACCL
//!    (Thunderbolt RDMA) group directly over the LAN/Thunderbolt fabric. Mesh
//!    supplies the hostfile (IPs + optional rdma device map) and otherwise stays
//!    out of the activation path. Lowest overhead; requires the nodes to be on a
//!    routable L2/L3 network or Thunderbolt mesh.
//! 2. [`MeshTransport::QuicTunnel`] — mesh terminates QUIC/iroh and exposes a
//!    **local TCP port-forward** on each node that maps onto the neighbour's
//!    Ring listen port. MLX connects to `127.0.0.1:<forwarded>` believing it is a
//!    direct peer; mesh relays the bytes over its existing tunnel. This reuses
//!    mesh connectivity (NAT traversal, relays, auth) at the cost of an extra
//!    userspace hop and TCP-over-QUIC overhead — acceptable for pipeline, poor
//!    for tensor.
//!
//! This module produces the hostfile/endpoint plan the [`crate::process`] layer
//! feeds to `mlx.launch`.

use crate::parallelism::ParallelismMode;
use serde::{Deserialize, Serialize};

/// Which fabric MLX should use between nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum MeshTransport {
    /// MLX forms its own group directly over the LAN / Thunderbolt fabric.
    LanRing {
        /// `ring` (TCP, Ethernet/Wi-Fi) or `jaccl` (RDMA over Thunderbolt).
        backend: MlxBackendKind,
    },
    /// MLX TCP is tunnelled through mesh's QUIC via per-node local port-forwards.
    QuicTunnel {
        /// The local 127.0.0.1 port on each node that forwards to its ring
        /// neighbour through the mesh tunnel.
        local_forward_base_port: u16,
    },
}

/// The MLX distributed backend to request from `mlx.launch --backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxBackendKind {
    /// TCP ring — works over Ethernet/Wi-Fi.
    Ring,
    /// RDMA over Thunderbolt 5 — lowest latency, needed for good tensor parallel.
    Jaccl,
    /// MPI — mature TCP collectives.
    Mpi,
}

impl MlxBackendKind {
    pub fn launch_arg(self) -> &'static str {
        match self {
            MlxBackendKind::Ring => "ring",
            MlxBackendKind::Jaccl => "jaccl",
            MlxBackendKind::Mpi => "mpi",
        }
    }
}

/// A resolved per-node endpoint MLX should bind/connect to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeEndpoint {
    /// ssh host (used by `mlx.launch` to start the remote process).
    pub ssh: String,
    /// IPs MLX listens on for this node (Ring backend).
    pub ips: Vec<String>,
    /// Optional rdma device map row for JACCL (per-peer device names; `null`
    /// for self). Mirrors the hostfile `rdma` array.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rdma: Vec<Option<String>>,
}

/// The concrete networking plan mesh hands to `mlx.launch`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportPlan {
    pub transport: MeshTransport,
    pub backend: MlxBackendKind,
    pub nodes: Vec<NodeEndpoint>,
}

impl TransportPlan {
    /// Recommend a transport plan given the chosen parallelism mode and the
    /// available node endpoints.
    ///
    /// Heuristic:
    /// - Tensor parallel ⇒ prefer JACCL (Thunderbolt RDMA) if rdma maps are
    ///   present, since tensor parallel is latency-bound; otherwise fall back to
    ///   a direct LAN ring.
    /// - Pipeline ⇒ a LAN ring (TCP) is plenty.
    /// - If nodes are not directly routable, callers should switch to a
    ///   [`MeshTransport::QuicTunnel`] plan via [`TransportPlan::quic_tunnel`].
    pub fn recommend(mode: ParallelismMode, nodes: Vec<NodeEndpoint>) -> Self {
        let has_rdma = nodes.iter().any(|n| !n.rdma.is_empty());
        let backend = match mode {
            ParallelismMode::Tensor if has_rdma => MlxBackendKind::Jaccl,
            ParallelismMode::Tensor => MlxBackendKind::Ring,
            _ => MlxBackendKind::Ring,
        };
        TransportPlan {
            transport: MeshTransport::LanRing { backend },
            backend,
            nodes,
        }
    }

    /// Build a QUIC-tunnelled plan that routes MLX TCP through mesh.
    ///
    /// Used when nodes are not directly routable (NAT, separate networks) and
    /// mesh must relay the activation bytes. Tensor parallel over a tunnel is
    /// discouraged; callers should usually pair this with pipeline.
    pub fn quic_tunnel(nodes: Vec<NodeEndpoint>, local_forward_base_port: u16) -> Self {
        TransportPlan {
            transport: MeshTransport::QuicTunnel {
                local_forward_base_port,
            },
            backend: MlxBackendKind::Ring,
            nodes,
        }
    }

    /// True if this plan relays MLX traffic through mesh's QUIC tunnel.
    pub fn is_tunnelled(&self) -> bool {
        matches!(self.transport, MeshTransport::QuicTunnel { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes_with_rdma() -> Vec<NodeEndpoint> {
        vec![
            NodeEndpoint {
                ssh: "m3-ultra-1".into(),
                ips: vec!["10.0.0.1".into()],
                rdma: vec![None, Some("rdma_en5".into())],
            },
            NodeEndpoint {
                ssh: "m3-ultra-2".into(),
                ips: vec!["10.0.0.2".into()],
                rdma: vec![Some("rdma_en5".into()), None],
            },
        ]
    }

    fn nodes_plain() -> Vec<NodeEndpoint> {
        vec![
            NodeEndpoint {
                ssh: "mac-1".into(),
                ips: vec!["10.0.0.1".into()],
                rdma: vec![],
            },
            NodeEndpoint {
                ssh: "mac-2".into(),
                ips: vec!["10.0.0.2".into()],
                rdma: vec![],
            },
        ]
    }

    #[test]
    fn tensor_with_thunderbolt_picks_jaccl() {
        let plan = TransportPlan::recommend(ParallelismMode::Tensor, nodes_with_rdma());
        assert_eq!(plan.backend, MlxBackendKind::Jaccl);
    }

    #[test]
    fn tensor_without_rdma_falls_back_to_ring() {
        let plan = TransportPlan::recommend(ParallelismMode::Tensor, nodes_plain());
        assert_eq!(plan.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn pipeline_uses_ring() {
        let plan = TransportPlan::recommend(ParallelismMode::Pipeline, nodes_plain());
        assert_eq!(plan.backend, MlxBackendKind::Ring);
        assert!(!plan.is_tunnelled());
    }

    #[test]
    fn quic_tunnel_plan_is_tunnelled() {
        let plan = TransportPlan::quic_tunnel(nodes_plain(), 41000);
        assert!(plan.is_tunnelled());
    }
}
