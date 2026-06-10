//! Typed configuration for an MLX sidecar deployment.

use crate::download::ModelRef;
use crate::transport::TransportPlan;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Role of a node within an MLX distributed group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    /// Rank 0: the node mesh routes OpenAI requests to. It owns the HTTP
    /// endpoint and drives the pipeline/tensor group.
    Coordinator,
    /// Rank > 0: a worker stage. No public HTTP endpoint.
    Worker,
}

/// One node in the MLX group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SidecarNode {
    /// 0-based rank within the MLX distributed group.
    pub rank: usize,
    pub role: NodeRole,
    /// ssh target `mlx.launch` uses to start the remote process.
    pub ssh: String,
}

/// Everything needed to start and supervise an MLX sidecar (single or multi node).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MlxRuntimeConfig {
    /// Model to serve (must be safetensors — see [`ModelRef::ensure_mlx_compatible`]).
    pub model: ModelRef,

    /// Loopback host the coordinator's OpenAI server binds to. Mesh routes here.
    #[serde(default = "default_api_host")]
    pub api_host: String,
    /// Port for the coordinator's OpenAI-compatible server.
    #[serde(default = "default_api_port")]
    pub api_port: u16,

    /// Nodes in the group. A single-element list means single-node serving.
    pub nodes: Vec<SidecarNode>,

    /// Networking/parallelism plan (filled in by the planner + transport layer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<TransportPlan>,

    /// Absolute path to the python interpreter that has `mlx-lm` installed.
    /// `mlx.launch` requires the same path on every node.
    #[serde(default = "default_python")]
    pub python: String,

    /// Max context length to request from the server (`--max-tokens` etc. are
    /// per-request; this is the model context budget).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_size: Option<u32>,

    /// How long to wait for the sidecar to become healthy before failing.
    #[serde(default = "default_readiness", with = "duration_secs")]
    pub readiness_timeout: Duration,

    /// Extra environment variables to pass to the sidecar (e.g.
    /// `MLX_METAL_FAST_SYNCH=1`, which the docs flag as important for JACCL).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_env: Vec<(String, String)>,
}

impl MlxRuntimeConfig {
    /// Convenience constructor for single-node local serving.
    pub fn single_node(model: ModelRef, api_port: u16) -> Self {
        Self {
            model,
            api_host: default_api_host(),
            api_port,
            nodes: vec![SidecarNode {
                rank: 0,
                role: NodeRole::Coordinator,
                ssh: "localhost".into(),
            }],
            transport: None,
            python: default_python(),
            context_size: None,
            readiness_timeout: default_readiness(),
            extra_env: Vec::new(),
        }
    }

    /// True if this is a multi-node (distributed) deployment.
    pub fn is_distributed(&self) -> bool {
        self.nodes.len() > 1
    }

    /// The coordinator node (rank 0), if present.
    pub fn coordinator(&self) -> Option<&SidecarNode> {
        self.nodes.iter().find(|n| n.role == NodeRole::Coordinator)
    }

    /// Validate the configuration before attempting to launch.
    pub fn validate(&self) -> crate::Result<()> {
        self.model.ensure_mlx_compatible()?;
        if self.nodes.is_empty() {
            return Err(crate::MlxError::Config("no nodes configured".into()));
        }
        if self.coordinator().is_none() {
            return Err(crate::MlxError::Config(
                "no coordinator (rank 0) node configured".into(),
            ));
        }
        let coordinators = self
            .nodes
            .iter()
            .filter(|n| n.role == NodeRole::Coordinator)
            .count();
        if coordinators != 1 {
            return Err(crate::MlxError::Config(format!(
                "expected exactly one coordinator, found {coordinators}"
            )));
        }
        if self.is_distributed() && self.transport.is_none() {
            return Err(crate::MlxError::Config(
                "distributed deployment requires a transport plan".into(),
            ));
        }
        Ok(())
    }

    /// The base URL mesh should route requests to.
    pub fn endpoint(&self) -> String {
        format!("http://{}:{}/v1", self.api_host, self.api_port)
    }
}

fn default_api_host() -> String {
    "127.0.0.1".into()
}
fn default_api_port() -> u16 {
    9410
}
fn default_python() -> String {
    "python3".into()
}
fn default_readiness() -> Duration {
    Duration::from_secs(180)
}

/// Serde helper to (de)serialise a `Duration` as whole seconds.
mod duration_secs {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_secs(u64::deserialize(d)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::ModelRef;

    #[test]
    fn single_node_validates() {
        let cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/x-4bit"), 9410);
        assert!(cfg.validate().is_ok());
        assert!(!cfg.is_distributed());
        assert_eq!(cfg.endpoint(), "http://127.0.0.1:9410/v1");
    }

    #[test]
    fn distributed_without_transport_fails() {
        let mut cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/x-4bit"), 9410);
        cfg.nodes.push(SidecarNode {
            rank: 1,
            role: NodeRole::Worker,
            ssh: "mac-2".into(),
        });
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn two_coordinators_fail() {
        let mut cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/x-4bit"), 9410);
        cfg.nodes.push(SidecarNode {
            rank: 1,
            role: NodeRole::Coordinator,
            ssh: "mac-2".into(),
        });
        assert!(cfg.validate().is_err());
    }
}
