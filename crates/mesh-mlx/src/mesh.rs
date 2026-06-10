//! The mesh-facing surface: latency-aware parallelism planning and the
//! transport/hostfile plan. Pure logic — no engine calls — so mesh can preview
//! and unit-test placement decisions.
//!
//! MLX mode is **local-only**: MLX opens its own TCP (ring) / RDMA (jaccl)
//! sockets and cannot use mesh's QUIC transport, and tunnelling would defeat the
//! latency MLX distributed exists for. Mesh forms an MLX group only from
//! Apple-Silicon, MLX-capable, directly-routable peers.

use crate::distributed::Backend;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// How MLX should split the model across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismMode {
    /// Single node, no split.
    Single,
    /// Layer pipeline (one activation per stage hop). Latency tolerant — the
    /// default over Ethernet/Wi-Fi.
    Pipeline,
    /// Tensor (Megatron) sharding — all-reduce per layer. Needs a low-latency
    /// fabric (Thunderbolt/JACCL or tight LAN).
    Tensor,
}

/// One inter-node round-trip-time sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencySample {
    pub from_rank: usize,
    pub to_rank: usize,
    pub rtt: Duration,
}

impl LatencySample {
    pub fn new(from_rank: usize, to_rank: usize, rtt: Duration) -> Self {
        Self {
            from_rank,
            to_rank,
            rtt,
        }
    }
}

/// Chooses [`ParallelismMode`] from measured inter-node latency.
#[derive(Debug, Clone, Copy)]
pub struct ParallelismPlanner {
    /// Worst-case inter-node RTT at or below which tensor parallelism is chosen.
    pub tensor_rtt_threshold: Duration,
}

impl Default for ParallelismPlanner {
    fn default() -> Self {
        Self {
            tensor_rtt_threshold: Duration::from_millis(2),
        }
    }
}

/// Planning outcome plus reasoning for telemetry/console.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelismPlan {
    pub mode: ParallelismMode,
    pub node_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_rtt: Option<Duration>,
    pub reason: String,
}

impl ParallelismPlanner {
    pub fn with_threshold(tensor_rtt_threshold: Duration) -> Self {
        Self {
            tensor_rtt_threshold,
        }
    }

    pub fn plan(&self, node_count: usize, samples: &[LatencySample]) -> ParallelismPlan {
        if node_count <= 1 {
            return ParallelismPlan {
                mode: ParallelismMode::Single,
                node_count,
                worst_rtt: None,
                reason: "single node: no model split".into(),
            };
        }
        let worst = samples.iter().map(|s| s.rtt).max();
        let (mode, reason) = match worst {
            None => (
                ParallelismMode::Pipeline,
                "no latency samples yet: defaulting to latency-tolerant pipeline".into(),
            ),
            Some(rtt) if rtt <= self.tensor_rtt_threshold => (
                ParallelismMode::Tensor,
                format!(
                    "worst RTT {rtt:?} <= {:?}: low-latency fabric, using tensor parallel",
                    self.tensor_rtt_threshold
                ),
            ),
            Some(rtt) => (
                ParallelismMode::Pipeline,
                format!(
                    "worst RTT {rtt:?} > {:?}: using latency-tolerant pipeline",
                    self.tensor_rtt_threshold
                ),
            ),
        };
        ParallelismPlan {
            mode,
            node_count,
            worst_rtt: worst,
            reason,
        }
    }
}

/// A directly-routable node in the local MLX group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeEndpoint {
    /// ssh target `mlx.launch`-equivalent orchestration uses.
    pub ssh: String,
    /// Directly-routable IPs MLX binds/connects to (ring backend).
    pub ips: Vec<String>,
    /// Optional per-peer rdma device names for JACCL (None for self).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rdma: Vec<Option<String>>,
}

/// The networking plan: which MLX backend + the node hostfile entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportPlan {
    pub backend: MlxBackendKind,
    pub nodes: Vec<NodeEndpoint>,
}

/// Mirror of [`Backend`] for serialisation in plans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MlxBackendKind {
    Ring,
    Jaccl,
    Mpi,
}

impl MlxBackendKind {
    pub fn to_backend(self) -> Backend {
        match self {
            MlxBackendKind::Ring => Backend::Ring,
            MlxBackendKind::Jaccl => Backend::Jaccl,
            MlxBackendKind::Mpi => Backend::Mpi,
        }
    }
}

/// What transport the operator wants for MLX inter-node traffic.
///
/// JACCL (RDMA over Thunderbolt) can't be silently auto-enabled — it needs
/// macOS 26.2+, `rdma_ctl enable` in recovery mode, and a Thunderbolt-5 mesh —
/// so it's opt-in. But it also shouldn't require hand-editing JSON, so `Auto`
/// uses JACCL **only when RDMA devices are actually detected** and otherwise
/// falls back to the TCP ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportPreference {
    /// Use JACCL if RDMA is detected on all nodes; otherwise ring. (default)
    #[default]
    Auto,
    /// Force the TCP ring even if RDMA is available.
    Ring,
    /// Require JACCL. If RDMA is not present this is an error (no silent
    /// downgrade — the operator explicitly asked for it).
    Jaccl,
}

impl TransportPreference {
    /// Parse from the `MESH_LLM_MLX_TRANSPORT` env var (auto|ring|jaccl).
    /// Unknown/empty values fall back to `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("MESH_LLM_MLX_TRANSPORT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "ring" | "tcp" => TransportPreference::Ring,
            "jaccl" | "rdma" | "thunderbolt" => TransportPreference::Jaccl,
            _ => TransportPreference::Auto,
        }
    }
}

impl TransportPlan {
    /// Recommend a backend for the chosen mode and the available nodes, using
    /// the default `Auto` preference. Tensor prefers JACCL when RDMA maps are
    /// present; otherwise ring (TCP over the LAN).
    pub fn recommend(mode: ParallelismMode, nodes: Vec<NodeEndpoint>) -> Self {
        Self::recommend_with(mode, nodes, TransportPreference::Auto)
            .expect("Auto preference never errors")
    }

    /// Recommend a backend honouring an explicit [`TransportPreference`].
    ///
    /// - `Ring` → always ring.
    /// - `Jaccl` → JACCL, but **errors** if no node advertises RDMA maps.
    /// - `Auto` → JACCL when every node has an RDMA map (a complete Thunderbolt
    ///   mesh); otherwise ring.
    pub fn recommend_with(
        mode: ParallelismMode,
        nodes: Vec<NodeEndpoint>,
        pref: TransportPreference,
    ) -> Result<Self, String> {
        // A usable JACCL mesh needs every node to expose its RDMA device map.
        let all_have_rdma = !nodes.is_empty() && nodes.iter().all(|n| !n.rdma.is_empty());
        let any_rdma = nodes.iter().any(|n| !n.rdma.is_empty());

        let backend = match pref {
            TransportPreference::Ring => MlxBackendKind::Ring,
            TransportPreference::Jaccl => {
                if !all_have_rdma {
                    return Err(format!(
                        "MESH_LLM_MLX_TRANSPORT=jaccl requested but RDMA is not available on all \
                         {} node(s) (any_rdma={any_rdma}). JACCL needs macOS 26.2+, \
                         `rdma_ctl enable` in recovery mode, and a Thunderbolt-5 mesh. \
                         Set MESH_LLM_MLX_TRANSPORT=auto to fall back to the TCP ring.",
                        nodes.len()
                    ));
                }
                // JACCL is only beneficial for tensor parallelism; honour the
                // operator's request regardless of mode but it pairs with tensor.
                MlxBackendKind::Jaccl
            }
            TransportPreference::Auto => match mode {
                ParallelismMode::Tensor if all_have_rdma => MlxBackendKind::Jaccl,
                _ => MlxBackendKind::Ring,
            },
        };
        Ok(TransportPlan { backend, nodes })
    }

    /// Render the JSON hostfile MLX launch consumes (`[{ssh, ips, rdma?}]`).
    pub fn render_hostfile(&self) -> String {
        let entries: Vec<serde_json::Value> = self
            .nodes
            .iter()
            .map(|n| {
                let mut obj = serde_json::Map::new();
                obj.insert("ssh".into(), n.ssh.clone().into());
                obj.insert("ips".into(), n.ips.clone().into());
                if !n.rdma.is_empty() {
                    let rdma: Vec<serde_json::Value> = n
                        .rdma
                        .iter()
                        .map(|d| match d {
                            Some(s) => serde_json::Value::String(s.clone()),
                            None => serde_json::Value::Null,
                        })
                        .collect();
                    obj.insert("rdma".into(), rdma.into());
                }
                serde_json::Value::Object(obj)
            })
            .collect();
        serde_json::to_string_pretty(&serde_json::Value::Array(entries))
            .unwrap_or_else(|_| "[]".into())
    }
}

/// Whether this host can run the MLX backend (Apple Silicon + macOS).
///
/// Mesh calls this to decide if a peer is MLX-eligible before forming a group.
/// MLX mode is local-only and Metal-only, so the gate is macOS on aarch64.
pub fn mlx_supported() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// Detect this host's RDMA (Thunderbolt) devices by running `ibv_devices`.
///
/// Returns the device names (e.g. `["rdma_en2", "rdma_en3", …]`) — these are
/// what JACCL needs to map the Thunderbolt mesh. Empty when RDMA isn't enabled
/// (no macOS 26.2 / `rdma_ctl enable`), no Thunderbolt fabric, or the tool is
/// missing — in which case the planner falls back to the TCP ring.
///
/// This is the auto-detection that makes JACCL opt-in-but-zero-config: a node
/// gossips this list, and a complete mesh (every node has devices) unlocks
/// JACCL under `Auto`.
pub fn detect_rdma_devices() -> Vec<String> {
    let output = match std::process::Command::new("ibv_devices").output() {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };
    let text = String::from_utf8_lossy(&output.stdout);
    // `ibv_devices` prints a header then `<device>\t<node_guid>` rows.
    text.lines()
        .skip_while(|l| l.contains("device") && l.contains("node GUID"))
        .filter_map(|l| l.split_whitespace().next())
        .filter(|tok| tok.starts_with("rdma_") || tok.starts_with("rdma"))
        .map(|s| s.to_string())
        .collect()
}

/// Build the JACCL environment for a node, given the per-peer RDMA device map
/// (the row of the mesh matrix for this rank) and the coordinator `ip:port`.
///
/// Returns the env vars MLX's JACCL backend reads at init:
/// `MLX_IBV_DEVICES` (device file/list), `MLX_JACCL_COORDINATOR`, `MLX_RANK`.
pub fn jaccl_env(
    rank: usize,
    rdma_row: &[Option<String>],
    coordinator: &str,
) -> Vec<(String, String)> {
    // MLX_IBV_DEVICES is a comma-separated list of this node's devices used to
    // reach each peer (null entries — self — are skipped).
    let devices: Vec<&str> = rdma_row.iter().filter_map(|d| d.as_deref()).collect();
    vec![
        ("MLX_RANK".to_string(), rank.to_string()),
        ("MLX_IBV_DEVICES".to_string(), devices.join(",")),
        ("MLX_JACCL_COORDINATOR".to_string(), coordinator.to_string()),
    ]
}

/// The mesh-facing orchestration surface for an MLX node.
///
/// Mesh uses this to (1) check eligibility, (2) plan parallelism from measured
/// latency, and (3) obtain the backend address it should bind the OpenAI server
/// to and then route OpenAI traffic to. The actual serving (loading the model
/// and running the OpenAI server) is driven by [`crate::runtime`]; this type is
/// the thin, testable decision layer mesh owns.
#[derive(Debug, Clone, Default)]
pub struct MlxOrchestrator {
    pub planner: ParallelismPlanner,
}

impl MlxOrchestrator {
    pub fn new(planner: ParallelismPlanner) -> Self {
        Self { planner }
    }

    /// Whether this host can serve MLX.
    pub fn supported(&self) -> bool {
        mlx_supported()
    }

    /// Plan the parallelism mode + transport for a candidate MLX group.
    ///
    /// `nodes` are the directly-routable, MLX-eligible peers (mesh must have
    /// already filtered to Apple-Silicon, MLX-capable, same-LAN/Thunderbolt
    /// peers — MLX is local-only and cannot use mesh QUIC). `samples` is the
    /// measured inter-node RTT.
    pub fn plan(
        &self,
        nodes: Vec<NodeEndpoint>,
        samples: &[LatencySample],
    ) -> (ParallelismPlan, TransportPlan) {
        let plan = self.planner.plan(nodes.len().max(1), samples);
        let transport = TransportPlan::recommend(plan.mode, nodes);
        (plan, transport)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(n: usize) -> Vec<NodeEndpoint> {
        (0..n)
            .map(|i| NodeEndpoint {
                ssh: format!("mac-{i}"),
                ips: vec![format!("10.0.0.{}", i + 1)],
                rdma: vec![],
            })
            .collect()
    }

    #[test]
    fn single_node_is_single() {
        assert_eq!(
            ParallelismPlanner::default().plan(1, &[]).mode,
            ParallelismMode::Single
        );
    }

    #[test]
    fn sub_threshold_is_tensor_above_is_pipeline() {
        let p = ParallelismPlanner::default();
        assert_eq!(
            p.plan(2, &[LatencySample::new(0, 1, Duration::from_micros(800))])
                .mode,
            ParallelismMode::Tensor
        );
        assert_eq!(
            p.plan(2, &[LatencySample::new(0, 1, Duration::from_millis(5))])
                .mode,
            ParallelismMode::Pipeline
        );
    }

    #[test]
    fn no_samples_defaults_pipeline() {
        assert_eq!(
            ParallelismPlanner::default().plan(2, &[]).mode,
            ParallelismMode::Pipeline
        );
    }

    #[test]
    fn worst_link_drives_decision() {
        let plan = ParallelismPlanner::default().plan(
            3,
            &[
                LatencySample::new(0, 1, Duration::from_micros(500)),
                LatencySample::new(1, 2, Duration::from_millis(8)),
            ],
        );
        assert_eq!(plan.mode, ParallelismMode::Pipeline);
        assert_eq!(plan.worst_rtt, Some(Duration::from_millis(8)));
    }

    #[test]
    fn tensor_with_rdma_picks_jaccl() {
        let mut nodes = plain(2);
        nodes[0].rdma = vec![None, Some("rdma_en5".into())];
        nodes[1].rdma = vec![Some("rdma_en5".into()), None];
        let plan = TransportPlan::recommend(ParallelismMode::Tensor, nodes);
        assert_eq!(plan.backend, MlxBackendKind::Jaccl);
    }

    #[test]
    fn orchestrator_plans_mode_and_transport_together() {
        let orch = MlxOrchestrator::default();
        // Low-latency 2-node → tensor + (ring, since no rdma maps here).
        let (plan, transport) = orch.plan(
            plain(2),
            &[LatencySample::new(0, 1, Duration::from_micros(700))],
        );
        assert_eq!(plan.mode, ParallelismMode::Tensor);
        assert_eq!(transport.backend, MlxBackendKind::Ring);

        // High-latency → pipeline + ring.
        let (plan, transport) = orch.plan(
            plain(3),
            &[LatencySample::new(0, 1, Duration::from_millis(7))],
        );
        assert_eq!(plan.mode, ParallelismMode::Pipeline);
        assert_eq!(transport.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn pipeline_uses_ring_and_renders_hostfile() {
        let plan = TransportPlan::recommend(ParallelismMode::Pipeline, plain(2));
        assert_eq!(plan.backend, MlxBackendKind::Ring);
        let hf = plan.render_hostfile();
        assert!(hf.contains("10.0.0.1"));
        assert!(hf.contains("mac-0"));
        assert!(!hf.contains("rdma"));
    }

    /// Two nodes each advertising an RDMA device map (a complete Thunderbolt
    /// mesh), used to exercise the JACCL paths.
    fn rdma_pair() -> Vec<NodeEndpoint> {
        vec![
            NodeEndpoint {
                ssh: "mac-0".into(),
                ips: vec!["10.0.0.1:5680".into()],
                rdma: vec![None, Some("rdma_en5".into())],
            },
            NodeEndpoint {
                ssh: "mac-1".into(),
                ips: vec!["10.0.0.2:5680".into()],
                rdma: vec![Some("rdma_en5".into()), None],
            },
        ]
    }

    #[test]
    fn auto_uses_jaccl_for_tensor_only_when_full_rdma_mesh() {
        // Tensor + complete RDMA mesh → JACCL.
        let p = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Auto,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Jaccl);

        // Pipeline + RDMA → still ring (JACCL only benefits tensor under Auto).
        let p = TransportPlan::recommend_with(
            ParallelismMode::Pipeline,
            rdma_pair(),
            TransportPreference::Auto,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Ring);

        // Tensor but no RDMA maps → ring.
        let p = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            plain(2),
            TransportPreference::Auto,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn explicit_ring_forces_tcp_even_with_rdma() {
        let p = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Ring,
        )
        .unwrap();
        assert_eq!(p.backend, MlxBackendKind::Ring);
    }

    #[test]
    fn explicit_jaccl_errors_without_rdma() {
        let err = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            plain(2),
            TransportPreference::Jaccl,
        );
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("jaccl"));

        // With a full mesh it succeeds.
        let ok = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Jaccl,
        );
        assert_eq!(ok.unwrap().backend, MlxBackendKind::Jaccl);
    }

    #[test]
    fn jaccl_renders_rdma_field_and_env() {
        let plan = TransportPlan::recommend_with(
            ParallelismMode::Tensor,
            rdma_pair(),
            TransportPreference::Jaccl,
        )
        .unwrap();
        let hf = plan.render_hostfile();
        assert!(hf.contains("rdma_en5"));
        assert!(hf.contains("rdma")); // the rdma field is present

        let env = jaccl_env(0, &plan.nodes[0].rdma, "10.0.0.1:6000");
        let map: std::collections::HashMap<_, _> = env.into_iter().collect();
        assert_eq!(map["MLX_RANK"], "0");
        assert_eq!(map["MLX_IBV_DEVICES"], "rdma_en5");
        assert_eq!(map["MLX_JACCL_COORDINATOR"], "10.0.0.1:6000");
    }

    #[test]
    fn transport_preference_parses_from_env_values() {
        // Default when unset/unknown.
        assert_eq!(TransportPreference::default(), TransportPreference::Auto);
    }
}
