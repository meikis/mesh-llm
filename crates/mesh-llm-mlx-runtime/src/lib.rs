//! # mesh-llm-mlx-runtime
//!
//! A Mesh LLM backend that orchestrates an **MLX sidecar** for inference on
//! Apple Silicon. It is the "all-Mac" alternative to the patched
//! llama.cpp/Skippy runtime.
//!
//! ## Why a sidecar (and not a Skippy ABI backend)
//!
//! MLX ships its own batteries-included distributed stack in `mlx-lm`:
//! both **pipeline** and **tensor** parallelism, over **Ethernet/Wi-Fi (Ring/TCP)**
//! or **Thunderbolt RDMA (JACCL)**. Reusing that is dramatically cheaper and
//! lower risk than re-implementing the model zoo behind Skippy's engine-private
//! activation-frame ABI. So mesh keeps ownership of the product surface (routing,
//! demand, OpenAI `/v1`, mesh membership) and treats MLX as a managed,
//! OpenAI-compatible local backend.
//!
//! ```text
//! mesh router ──orchestrates──> MlxBackend (this crate)
//!                                  │  spawns + supervises
//!                                  ▼
//!                       mlx_lm.server  (single node)
//!                       mlx.launch     (multi node: pipeline | tensor)
//!                                  │  OpenAI /v1 on 127.0.0.1
//!                                  ▼
//!                       MLX engine (Metal, unified memory)
//! ```
//!
//! ## What mesh gets (the Rusty abstraction)
//!
//! [`MlxBackend`] is the trait mesh code talks to. It is deliberately small and
//! transport/process agnostic so the host runtime can:
//!
//! - decide *when* to use MLX (Apple Silicon capability gate),
//! - decide *how* to parallelise based on measured inter-node latency
//!   ([`ParallelismPlanner`]: low latency ⇒ tensor, otherwise ⇒ pipeline),
//! - start/stop/health-check the sidecar ([`MlxBackend::start`] / [`stop`] /
//!   [`health`]),
//! - reach the OpenAI-compatible endpoint it exposes ([`Backend::endpoint`]).
//!
//! ## Confirmed MLX behaviours this crate relies on
//!
//! 1. **MLX runs from safetensors, not GGUF.** `mlx-lm` loads `*.safetensors`
//!    via `huggingface_hub.snapshot_download`.
//! 2. **MLX downloads only what is needed.** `mlx_lm.utils.sharded_load` resolves
//!    the local stage from `model.safetensors.index.json` and passes
//!    `allow_patterns=local_files`, so a **pipeline** node downloads only the
//!    weight files for the layers it will run. See [`download`] for the typed
//!    model reference and the policy we forward.
//! 3. **MLX opens its own TCP/RDMA sockets.** `mx.distributed.init()` only accepts
//!    `{any, mpi, ring, nccl, jaccl}` and the Ring backend binds/connects its own
//!    sockets to IPs from a hostfile. MLX **cannot** natively speak mesh's
//!    QUIC/iroh transport, so [`transport`] models the two supported options:
//!    a LAN ring/JACCL mesh, or tunnelling MLX's TCP through mesh QUIC via local
//!    port-forwards.

#![forbid(unsafe_code)]

pub mod backend;
pub mod client;
pub mod config;
pub mod download;
pub mod parallelism;
pub mod process;
pub mod transport;

pub use backend::{Backend, MlxBackend, MlxSidecar};
pub use config::{MlxRuntimeConfig, NodeRole, SidecarNode};
pub use download::{ModelRef, ModelWeightFormat};
pub use parallelism::{LatencySample, ParallelismMode, ParallelismPlan, ParallelismPlanner};
pub use transport::{MeshTransport, TransportPlan};

use std::time::Duration;

/// Errors surfaced by the MLX runtime backend.
#[derive(Debug, thiserror::Error)]
pub enum MlxError {
    /// The host is not a supported MLX target (MLX requires Apple Silicon / Metal).
    #[error("host is not a supported MLX target: {0}")]
    Unsupported(String),

    /// The `mlx-lm` / `mlx.launch` tooling could not be located or executed.
    #[error("MLX tooling unavailable: {0}")]
    ToolingUnavailable(String),

    /// The sidecar process failed to spawn, exited, or never became ready.
    #[error("MLX sidecar process error: {0}")]
    Process(String),

    /// The sidecar did not become healthy within the readiness deadline.
    #[error("MLX sidecar not ready after {0:?}")]
    ReadinessTimeout(Duration),

    /// Configuration was invalid (e.g. empty host list for a distributed run).
    #[error("invalid MLX runtime configuration: {0}")]
    Config(String),

    /// Networking/transport could not be established for a distributed run.
    #[error("MLX transport error: {0}")]
    Transport(String),

    /// An HTTP error talking to the sidecar's OpenAI endpoint.
    #[error("MLX sidecar request failed: {0}")]
    Request(String),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, MlxError>;
