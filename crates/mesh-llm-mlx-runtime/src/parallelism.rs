//! Latency-aware selection of MLX parallelism mode.
//!
//! Mesh measures inter-node latency anyway (for routing/affinity). We reuse that
//! signal to pick how MLX should split the model:
//!
//! - **Tensor parallelism** does ~2 all-reduce collectives *per transformer
//!   layer*. Throughput is dominated by network round-trip time, so it only pays
//!   off on a very low-latency fabric (Thunderbolt RDMA / JACCL, or a tight LAN).
//! - **Pipeline parallelism** sends one activation per stage boundary
//!   (point-to-point). It tolerates higher latency and is the right default over
//!   Ethernet/Wi-Fi.
//!
//! The planner therefore selects **tensor** when the worst-case inter-node RTT is
//! below a threshold (default 2ms, i.e. roughly Thunderbolt/tight-LAN territory)
//! and **pipeline** otherwise. The threshold is configurable so operators can
//! tune it per deployment.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// How MLX should split the model across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismMode {
    /// Single node, no model split.
    Single,
    /// Layer-pipeline across nodes (one activation per stage hop). Latency
    /// tolerant — the default over Ethernet/Wi-Fi.
    Pipeline,
    /// Tensor (Megatron-style) sharding — all-reduce per layer. Needs a
    /// low-latency fabric (Thunderbolt/JACCL or tight LAN).
    Tensor,
}

impl ParallelismMode {
    /// The CLI flag `mlx_lm.server` expects. Tensor parallel is the default
    /// (no flag); pipeline needs `--pipeline`.
    pub fn server_flag(self) -> Option<&'static str> {
        match self {
            ParallelismMode::Pipeline => Some("--pipeline"),
            // Tensor parallel is the default distributed mode (no flag).
            ParallelismMode::Tensor => None,
            ParallelismMode::Single => None,
        }
    }
}

/// A single inter-node round-trip-time sample between two sidecar nodes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatencySample {
    /// 0-based rank of the source node.
    pub from_rank: usize,
    /// 0-based rank of the destination node.
    pub to_rank: usize,
    /// Measured round-trip time.
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

/// Decides [`ParallelismMode`] from measured inter-node latency.
///
/// ```
/// use std::time::Duration;
/// use mesh_llm_mlx_runtime::{ParallelismPlanner, ParallelismMode, LatencySample};
///
/// // Default: tensor when worst-case RTT < 2ms, else pipeline.
/// let planner = ParallelismPlanner::default();
///
/// // Tight Thunderbolt-class fabric -> tensor parallel.
/// let plan = planner.plan(2, &[LatencySample::new(0, 1, Duration::from_micros(800))]);
/// assert_eq!(plan.mode, ParallelismMode::Tensor);
///
/// // Ethernet-class latency -> pipeline.
/// let plan = planner.plan(2, &[LatencySample::new(0, 1, Duration::from_millis(5))]);
/// assert_eq!(plan.mode, ParallelismMode::Pipeline);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ParallelismPlanner {
    /// Worst-case inter-node RTT at or below which tensor parallelism is chosen.
    pub tensor_rtt_threshold: Duration,
}

impl Default for ParallelismPlanner {
    fn default() -> Self {
        Self {
            // ~Thunderbolt / tight-LAN territory. Above this, prefer pipeline.
            tensor_rtt_threshold: Duration::from_millis(2),
        }
    }
}

/// The outcome of planning: a mode plus the reasoning, for telemetry/console.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParallelismPlan {
    pub mode: ParallelismMode,
    pub node_count: usize,
    /// Worst-case RTT observed across the supplied samples, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_rtt: Option<Duration>,
    /// Human-readable explanation for logs/console.
    pub reason: String,
}

impl ParallelismPlanner {
    /// Create a planner with a custom tensor-parallel RTT threshold.
    pub fn with_threshold(tensor_rtt_threshold: Duration) -> Self {
        Self {
            tensor_rtt_threshold,
        }
    }

    /// Choose a parallelism mode for `node_count` nodes given pairwise latency
    /// samples (which may be empty if not yet measured).
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
                // No measurements yet: be conservative and pick the latency
                // tolerant mode so a bad fabric can't tank throughput.
                ParallelismMode::Pipeline,
                "no latency samples yet: defaulting to latency-tolerant pipeline".into(),
            ),
            Some(rtt) if rtt <= self.tensor_rtt_threshold => (
                ParallelismMode::Tensor,
                format!(
                    "worst RTT {:?} <= {:?} threshold: low-latency fabric, using tensor parallel",
                    rtt, self.tensor_rtt_threshold
                ),
            ),
            Some(rtt) => (
                ParallelismMode::Pipeline,
                format!(
                    "worst RTT {:?} > {:?} threshold: using latency-tolerant pipeline",
                    rtt, self.tensor_rtt_threshold
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_is_single() {
        let plan = ParallelismPlanner::default().plan(1, &[]);
        assert_eq!(plan.mode, ParallelismMode::Single);
    }

    #[test]
    fn sub_ms_is_tensor() {
        let plan = ParallelismPlanner::default()
            .plan(4, &[LatencySample::new(0, 1, Duration::from_micros(900))]);
        assert_eq!(plan.mode, ParallelismMode::Tensor);
    }

    #[test]
    fn at_threshold_is_tensor_above_is_pipeline() {
        let p = ParallelismPlanner::with_threshold(Duration::from_millis(1));
        assert_eq!(
            p.plan(2, &[LatencySample::new(0, 1, Duration::from_millis(1))])
                .mode,
            ParallelismMode::Tensor
        );
        assert_eq!(
            p.plan(2, &[LatencySample::new(0, 1, Duration::from_micros(1100))])
                .mode,
            ParallelismMode::Pipeline
        );
    }

    #[test]
    fn worst_case_pair_drives_decision() {
        // One slow link forces pipeline even if others are fast.
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
    fn no_samples_defaults_pipeline() {
        let plan = ParallelismPlanner::default().plan(2, &[]);
        assert_eq!(plan.mode, ParallelismMode::Pipeline);
    }

    #[test]
    fn server_flag_mapping() {
        assert_eq!(ParallelismMode::Pipeline.server_flag(), Some("--pipeline"));
        assert_eq!(ParallelismMode::Tensor.server_flag(), None);
    }
}
