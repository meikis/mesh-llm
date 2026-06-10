//! Builds and supervises the MLX sidecar process.
//!
//! Two launch shapes, both driven by the same [`crate::config::MlxRuntimeConfig`]:
//!
//! - **Single node** → `python -m mlx_lm server --model <id> --host <h> --port <p>`.
//! - **Multi node** → `mlx.launch --backend <ring|jaccl|mpi> --hostfile <file>
//!   [--env MLX_METAL_FAST_SYNCH=1] -- python -m mlx_lm server [--pipeline] …`.
//!
//! This module only *constructs* the command and owns the child handle. The
//! decision of pipeline-vs-tensor comes from [`crate::parallelism`] and the
//! networking from [`crate::transport`]; we never hardcode those here.
//!
//! Note on project norms: AGENTS.md forbids resurrecting the external
//! `llama-server`/`rpc-server` lane. MLX is a *distinct engine* that cannot be
//! embedded behind the Skippy ABI without re-implementing the model zoo, so a
//! supervised MLX sidecar is the analog of a native-runtime lane, not a revival
//! of the old llama proxy lane. It is gated to Apple Silicon and owned by mesh.

use crate::config::MlxRuntimeConfig;
use crate::parallelism::ParallelismMode;
use crate::transport::{MeshTransport, TransportPlan};
use crate::{MlxError, Result};
use std::process::Stdio;
use tokio::process::{Child, Command};

/// A fully-resolved launch specification (argv + env), independent of execution.
/// Kept separate from spawning so it can be unit-tested without MLX installed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchSpec {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

impl LaunchSpec {
    /// Build the launch spec for the given config and parallelism mode.
    ///
    /// `hostfile_path` is required for distributed launches (it is produced by
    /// [`write_hostfile`] from the transport plan).
    pub fn build(
        cfg: &MlxRuntimeConfig,
        mode: ParallelismMode,
        hostfile_path: Option<&str>,
    ) -> Result<Self> {
        cfg.validate()?;

        if !cfg.is_distributed() {
            return Ok(Self::single_node(cfg));
        }

        let plan = cfg
            .transport
            .as_ref()
            .ok_or_else(|| MlxError::Config("distributed launch requires a transport".into()))?;
        let hostfile = hostfile_path.ok_or_else(|| {
            MlxError::Config("distributed launch requires a hostfile path".into())
        })?;
        Ok(Self::distributed(cfg, mode, plan, hostfile))
    }

    fn server_argv(cfg: &MlxRuntimeConfig, mode: ParallelismMode) -> Vec<String> {
        let mut argv = vec![
            "-m".into(),
            "mlx_lm".into(),
            "server".into(),
            "--model".into(),
            cfg.model.id.clone(),
            "--host".into(),
            cfg.api_host.clone(),
            "--port".into(),
            cfg.api_port.to_string(),
        ];
        if let Some(rev) = &cfg.model.revision {
            argv.push("--revision".into());
            argv.push(rev.clone());
        }
        // Pipeline needs the explicit flag; tensor parallel is the default.
        if let Some(flag) = mode.server_flag() {
            argv.push(flag.into());
        }
        argv
    }

    fn single_node(cfg: &MlxRuntimeConfig) -> Self {
        Self {
            program: cfg.python.clone(),
            args: Self::server_argv(cfg, ParallelismMode::Single),
            env: cfg.extra_env.clone(),
        }
    }

    fn distributed(
        cfg: &MlxRuntimeConfig,
        mode: ParallelismMode,
        plan: &TransportPlan,
        hostfile_path: &str,
    ) -> Self {
        let mut args = vec![
            "--backend".into(),
            plan.backend.launch_arg().into(),
            "--hostfile".into(),
            hostfile_path.into(),
        ];

        // JACCL benefits strongly from fast GPU/CPU sync; surface it explicitly.
        let mut env = cfg.extra_env.clone();
        if matches!(plan.backend, crate::transport::MlxBackendKind::Jaccl)
            && !env.iter().any(|(k, _)| k == "MLX_METAL_FAST_SYNCH")
        {
            args.push("--env".into());
            args.push("MLX_METAL_FAST_SYNCH=1".into());
            env.push(("MLX_METAL_FAST_SYNCH".into(), "1".into()));
        }

        // Separator, then the actual program mlx.launch should run on each node.
        args.push("--".into());
        args.push(cfg.python.clone());
        args.extend(Self::server_argv(cfg, mode));

        Self {
            program: "mlx.launch".into(),
            args,
            env,
        }
    }
}

/// Render the JSON hostfile `mlx.launch` consumes from a transport plan.
///
/// For a direct LAN ring/JACCL group this is the standard `[{ssh, ips, rdma?}]`
/// schema. For a QUIC tunnel, the IPs are rewritten to `127.0.0.1` because each
/// node connects to a local forwarded port that mesh relays through QUIC (see
/// `network/tunnel.rs` for the relay primitive).
pub fn render_hostfile(plan: &TransportPlan) -> Result<String> {
    let tunnelled = matches!(plan.transport, MeshTransport::QuicTunnel { .. });
    let base_port = match plan.transport {
        MeshTransport::QuicTunnel {
            local_forward_base_port,
        } => Some(local_forward_base_port),
        MeshTransport::LanRing { .. } => None,
    };

    let entries: Vec<serde_json::Value> = plan
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let ips: Vec<String> = if tunnelled {
                let port = base_port.unwrap_or(0).saturating_add(i as u16);
                vec![format!("127.0.0.1:{port}")]
            } else {
                n.ips.clone()
            };
            let mut obj = serde_json::Map::new();
            obj.insert("ssh".into(), serde_json::Value::String(n.ssh.clone()));
            obj.insert(
                "ips".into(),
                serde_json::Value::Array(ips.into_iter().map(serde_json::Value::String).collect()),
            );
            if !n.rdma.is_empty() {
                obj.insert(
                    "rdma".into(),
                    serde_json::Value::Array(
                        n.rdma
                            .iter()
                            .map(|d| match d {
                                Some(s) => serde_json::Value::String(s.clone()),
                                None => serde_json::Value::Null,
                            })
                            .collect(),
                    ),
                );
            }
            serde_json::Value::Object(obj)
        })
        .collect();

    serde_json::to_string_pretty(&serde_json::Value::Array(entries))
        .map_err(|e| MlxError::Transport(format!("failed to render hostfile: {e}")))
}

/// Spawn the sidecar from a [`LaunchSpec`], returning the child handle.
///
/// stdout/stderr are inherited by default so native MLX logs land alongside the
/// host runtime's log redirection (mirrors how skippy native logs are handled).
pub fn spawn(spec: &LaunchSpec) -> Result<Child> {
    let mut cmd = Command::new(&spec.program);
    cmd.args(&spec.args)
        .envs(spec.env.iter().cloned())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    cmd.spawn().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            MlxError::ToolingUnavailable(format!(
                "could not execute '{}': {e}. Is mlx-lm installed for this interpreter?",
                spec.program
            ))
        } else {
            MlxError::Process(format!("failed to spawn '{}': {e}", spec.program))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NodeRole, SidecarNode};
    use crate::download::ModelRef;
    use crate::transport::{MlxBackendKind, NodeEndpoint};

    fn two_node_cfg(transport: TransportPlan) -> MlxRuntimeConfig {
        let mut cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/m-4bit"), 9410);
        cfg.nodes.push(SidecarNode {
            rank: 1,
            role: NodeRole::Worker,
            ssh: "mac-2".into(),
        });
        cfg.transport = Some(transport);
        cfg
    }

    #[test]
    fn single_node_argv_has_no_pipeline_flag() {
        let cfg = MlxRuntimeConfig::single_node(ModelRef::hf("mlx-community/m-4bit"), 9410);
        let spec = LaunchSpec::build(&cfg, ParallelismMode::Single, None).unwrap();
        assert!(spec.program.ends_with("python3"));
        assert!(spec.args.contains(&"server".to_string()));
        assert!(!spec.args.contains(&"--pipeline".to_string()));
        assert!(!spec.args.contains(&"--backend".to_string()));
    }

    #[test]
    fn pipeline_distributed_uses_launch_and_flag() {
        let plan = TransportPlan::recommend(
            ParallelismMode::Pipeline,
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
            ],
        );
        let cfg = two_node_cfg(plan);
        let spec =
            LaunchSpec::build(&cfg, ParallelismMode::Pipeline, Some("/tmp/hosts.json")).unwrap();
        assert_eq!(spec.program, "mlx.launch");
        assert!(spec.args.contains(&"--pipeline".to_string()));
        assert!(spec.args.windows(2).any(|w| w == ["--backend", "ring"]));
    }

    #[test]
    fn tensor_jaccl_injects_fast_synch_env() {
        let plan = TransportPlan::recommend(
            ParallelismMode::Tensor,
            vec![
                NodeEndpoint {
                    ssh: "m1".into(),
                    ips: vec!["10.0.0.1".into()],
                    rdma: vec![None, Some("rdma_en5".into())],
                },
                NodeEndpoint {
                    ssh: "m2".into(),
                    ips: vec!["10.0.0.2".into()],
                    rdma: vec![Some("rdma_en5".into()), None],
                },
            ],
        );
        assert_eq!(plan.backend, MlxBackendKind::Jaccl);
        let cfg = two_node_cfg(plan);
        let spec =
            LaunchSpec::build(&cfg, ParallelismMode::Tensor, Some("/tmp/hosts.json")).unwrap();
        assert!(spec.args.windows(2).any(|w| w == ["--backend", "jaccl"]));
        assert!(spec.args.iter().any(|a| a == "MLX_METAL_FAST_SYNCH=1"));
        assert!(!spec.args.contains(&"--pipeline".to_string()));
    }

    #[test]
    fn quic_tunnel_hostfile_uses_loopback_ports() {
        let plan = TransportPlan::quic_tunnel(
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
            ],
            41000,
        );
        let hostfile = render_hostfile(&plan).unwrap();
        assert!(hostfile.contains("127.0.0.1:41000"));
        assert!(hostfile.contains("127.0.0.1:41001"));
    }
}
