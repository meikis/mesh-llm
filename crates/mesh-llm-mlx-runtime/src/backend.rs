//! The backend abstraction mesh code talks to, plus the MLX sidecar impl.

use crate::client::{Health, SidecarClient};
use crate::config::MlxRuntimeConfig;
use crate::parallelism::{LatencySample, ParallelismPlan, ParallelismPlanner};
use crate::process::{self, LaunchSpec};
use crate::transport::{NodeEndpoint, TransportPlan};
use crate::{MlxError, Result};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::Mutex;

/// A handle mesh routing uses to reach a running backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backend {
    /// OpenAI-compatible base URL (`http://host:port/v1`) mesh routes to.
    pub endpoint: String,
    /// Model ids the backend currently serves.
    pub model_ids: Vec<String>,
}

/// The contract mesh orchestrates an inference engine through.
///
/// Kept minimal and engine-agnostic so the host runtime can treat MLX as just
/// another lane: capability-gate it, plan parallelism from measured latency,
/// start/health/stop it, and route OpenAI traffic to its endpoint.
#[async_trait]
pub trait MlxBackend: Send + Sync {
    /// Whether this host can run the backend at all (MLX ⇒ Apple Silicon).
    fn supported(&self) -> Result<()>;

    /// Decide how to parallelise given measured inter-node latency. Pure and
    /// side-effect free so mesh can preview a plan before committing.
    fn plan_parallelism(&self, samples: &[LatencySample]) -> ParallelismPlan;

    /// Start the backend (idempotent: returns the live handle if already up).
    async fn start(&self, plan: &ParallelismPlan) -> Result<Backend>;

    /// Current health of the backend.
    async fn health(&self) -> Health;

    /// Stop the backend and release resources.
    async fn stop(&self) -> Result<()>;
}

/// MLX implementation backed by a supervised `mlx-lm` / `mlx.launch` sidecar.
pub struct MlxSidecar {
    cfg: MlxRuntimeConfig,
    planner: ParallelismPlanner,
    client: SidecarClient,
    /// Node endpoints used to build the transport/hostfile for distributed runs.
    endpoints: Vec<NodeEndpoint>,
    state: Arc<Mutex<SidecarState>>,
}

#[derive(Default)]
struct SidecarState {
    child: Option<Child>,
    hostfile: Option<tempfile_path::TempPath>,
    backend: Option<Backend>,
}

impl MlxSidecar {
    /// Build a sidecar from config plus the node endpoints mesh discovered.
    /// `endpoints` may be empty for single-node serving.
    pub fn new(
        cfg: MlxRuntimeConfig,
        planner: ParallelismPlanner,
        endpoints: Vec<NodeEndpoint>,
    ) -> Self {
        let client = SidecarClient::new(cfg.endpoint());
        Self {
            cfg,
            planner,
            client,
            endpoints,
            state: Arc::new(Mutex::new(SidecarState::default())),
        }
    }

    /// Detect whether the current host is a supported MLX target.
    ///
    /// MLX runs on Apple Silicon (Metal). We gate on macOS + aarch64; the host
    /// runtime can layer a stronger probe (e.g. confirming `mlx` imports).
    fn host_supported() -> Result<()> {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Ok(())
        } else {
            Err(MlxError::Unsupported(
                "MLX requires Apple Silicon (macOS aarch64)".into(),
            ))
        }
    }
}

#[async_trait]
impl MlxBackend for MlxSidecar {
    fn supported(&self) -> Result<()> {
        Self::host_supported()
    }

    fn plan_parallelism(&self, samples: &[LatencySample]) -> ParallelismPlan {
        self.planner.plan(self.cfg.nodes.len(), samples)
    }

    async fn start(&self, plan: &ParallelismPlan) -> Result<Backend> {
        self.supported()?;
        self.cfg.validate()?;

        let mut state = self.state.lock().await;
        if let Some(backend) = &state.backend {
            return Ok(backend.clone());
        }

        // Build the transport plan + hostfile for distributed runs.
        let (spec, hostfile_guard) = if self.cfg.is_distributed() {
            let transport = self
                .cfg
                .transport
                .clone()
                .unwrap_or_else(|| TransportPlan::recommend(plan.mode, self.endpoints.clone()));
            let hostfile_contents = process::render_hostfile(&transport)?;
            let path = tempfile_path::write_temp("mlx-hostfile", &hostfile_contents)
                .map_err(|e| MlxError::Transport(format!("hostfile write failed: {e}")))?;
            let path_str = path.as_path_str();
            // Use a config clone carrying the resolved transport for arg building.
            let mut cfg = self.cfg.clone();
            cfg.transport = Some(transport);
            let spec = LaunchSpec::build(&cfg, plan.mode, Some(&path_str))?;
            (spec, Some(path))
        } else {
            (LaunchSpec::build(&self.cfg, plan.mode, None)?, None)
        };

        tracing::info!(
            mode = ?plan.mode,
            reason = %plan.reason,
            program = %spec.program,
            "starting MLX sidecar"
        );

        let child = process::spawn(&spec)?;
        state.child = Some(child);
        state.hostfile = hostfile_guard;

        // Poll readiness on the coordinator's endpoint.
        let model_ids = self.client.wait_ready(self.cfg.readiness_timeout).await?;
        let backend = Backend {
            endpoint: self.cfg.endpoint(),
            model_ids,
        };
        state.backend = Some(backend.clone());
        Ok(backend)
    }

    async fn health(&self) -> Health {
        self.client.health().await
    }

    async fn stop(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        if let Some(mut child) = state.child.take() {
            // kill_on_drop is set, but request a clean kill explicitly.
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        state.hostfile = None;
        state.backend = None;
        Ok(())
    }
}

/// Tiny temp-file helper kept local to avoid a `tempfile` dependency in a crate
/// that otherwise has none. Writes a uniquely-named file under the OS temp dir
/// and removes it on drop.
mod tempfile_path {
    use std::io::Write;
    use std::path::PathBuf;

    /// An owned temp path that deletes its file on drop.
    #[derive(Debug)]
    pub struct TempPath(PathBuf);

    impl TempPath {
        pub fn as_path_str(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    pub fn write_temp(prefix: &str, contents: &str) -> std::io::Result<TempPath> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let pid = std::process::id();
        let mut path = std::env::temp_dir();
        path.push(format!("{prefix}-{pid}-{nanos}.json"));
        let mut f = std::fs::File::create(&path)?;
        f.write_all(contents.as_bytes())?;
        f.flush()?;
        Ok(TempPath(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::download::ModelRef;
    use std::time::Duration;

    fn sidecar_single() -> MlxSidecar {
        let cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/m-4bit"), 9410);
        MlxSidecar::new(cfg, ParallelismPlanner::default(), vec![])
    }

    #[test]
    fn single_node_plan_is_single() {
        let s = sidecar_single();
        let plan = s.plan_parallelism(&[]);
        assert_eq!(plan.mode, crate::ParallelismMode::Single);
    }

    #[test]
    fn multi_node_low_latency_plans_tensor() {
        let mut cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/m-4bit"), 9410);
        cfg.nodes.push(crate::config::SidecarNode {
            rank: 1,
            role: crate::config::NodeRole::Worker,
            ssh: "mac-2".into(),
        });
        let s = MlxSidecar::new(cfg, ParallelismPlanner::default(), vec![]);
        let plan = s.plan_parallelism(&[LatencySample::new(0, 1, Duration::from_micros(700))]);
        assert_eq!(plan.mode, crate::ParallelismMode::Tensor);
    }

    #[test]
    fn temp_hostfile_roundtrips_and_cleans_up() {
        let p = tempfile_path::write_temp("mlx-test", "[]").unwrap();
        let path = p.as_path_str();
        assert!(std::path::Path::new(&path).exists());
        drop(p);
        assert!(!std::path::Path::new(&path).exists());
    }
}
