//! Model references and weight-download policy for MLX.
//!
//! ## Confirmed: MLX uses safetensors and downloads only what is needed
//!
//! From `mlx-lm`'s `utils.py`:
//!
//! - `_download(...)` calls `huggingface_hub.snapshot_download` with an
//!   `allow_patterns` filter. The default pattern set is `["*.safetensors",
//!   "*.json", ...]` — i.e. **safetensors weights, not GGUF**.
//! - `sharded_load(...)` does a two-pass download:
//!   1. metadata only (`*.json`, tokenizer, `model.safetensors.index.json`),
//!   2. for a **pipeline** stage it computes the set of `*.safetensors` files
//!      that hold *this rank's* layers and passes
//!      `allow_patterns=local_files`, so each node downloads only its slice.
//!      For **tensor** parallelism it currently downloads the full repo and
//!      slices in memory at load time.
//!
//! mesh-llm does not download MLX weights itself — that stays inside the sidecar
//! so we inherit `mlx-lm`'s selective behaviour for free. This module just gives
//! mesh a typed way to *describe* the model and to *assert* the download policy
//! we expect, so routing/telemetry can reason about footprint per node.

use serde::{Deserialize, Serialize};

/// How a model's weights are stored. MLX consumes safetensors; GGUF is the
/// llama.cpp/Skippy lane and is intentionally rejected here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelWeightFormat {
    /// MLX-native safetensors (possibly MLX-quantised).
    Safetensors,
    /// GGUF — not consumable by MLX; present only so we can reject it clearly.
    Gguf,
}

/// A reference to a model the MLX sidecar should serve.
///
/// This mirrors how `mlx-lm` accepts either a Hugging Face repo id or a local
/// path. We keep it transport-agnostic; the sidecar performs the actual
/// `snapshot_download`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    /// Hugging Face repo id (e.g. `mlx-community/Llama-3.3-70B-Instruct-4bit`)
    /// or an absolute local path.
    pub id: String,
    /// Optional immutable revision (commit sha / tag) for reproducible serving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Declared weight format. MLX requires [`ModelWeightFormat::Safetensors`].
    pub format: ModelWeightFormat,
}

impl ModelRef {
    /// Construct a safetensors HF model reference.
    pub fn hf(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            revision: None,
            format: ModelWeightFormat::Safetensors,
        }
    }

    /// Attach an immutable revision for reproducible serving.
    pub fn at_revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Returns an error if this model cannot be served by MLX.
    pub fn ensure_mlx_compatible(&self) -> crate::Result<()> {
        match self.format {
            ModelWeightFormat::Safetensors => Ok(()),
            ModelWeightFormat::Gguf => Err(crate::MlxError::Config(format!(
                "model '{}' is GGUF; MLX serves safetensors. Route GGUF models to \
                 the Skippy/llama.cpp lane instead.",
                self.id
            ))),
        }
    }
}

/// The selective-download policy mesh expects the sidecar to apply.
///
/// This is descriptive metadata for routing/telemetry: it lets mesh estimate
/// per-node disk/network footprint *before* the sidecar starts, matching the
/// behaviour observed in `mlx-lm.sharded_load`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadPolicy {
    /// Whole repo lands on disk (tensor parallel today, or single node).
    FullRepo,
    /// Only the safetensors files for this stage's layers are fetched
    /// (pipeline parallel — `allow_patterns=local_files`).
    StageShardOnly,
}

impl DownloadPolicy {
    /// The download policy MLX actually applies for a given parallelism mode,
    /// as confirmed in `mlx-lm.sharded_load`.
    pub fn for_mode(mode: crate::ParallelismMode, distributed: bool) -> Self {
        match (mode, distributed) {
            // Pipeline shards downloads per node.
            (crate::ParallelismMode::Pipeline, true) => DownloadPolicy::StageShardOnly,
            // Tensor parallel reads the full repo on each node (slices in RAM),
            // and single-node always needs the whole model.
            _ => DownloadPolicy::FullRepo,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParallelismMode;

    #[test]
    fn gguf_is_rejected_for_mlx() {
        let m = ModelRef {
            id: "some/model-gguf".into(),
            revision: None,
            format: ModelWeightFormat::Gguf,
        };
        assert!(m.ensure_mlx_compatible().is_err());
    }

    #[test]
    fn safetensors_is_accepted() {
        let m = ModelRef::hf("mlx-community/Llama-3.3-70B-Instruct-4bit").at_revision("abc123");
        assert!(m.ensure_mlx_compatible().is_ok());
        assert_eq!(m.revision.as_deref(), Some("abc123"));
    }

    #[test]
    fn pipeline_downloads_only_the_stage_shard() {
        assert_eq!(
            DownloadPolicy::for_mode(ParallelismMode::Pipeline, true),
            DownloadPolicy::StageShardOnly
        );
        assert_eq!(
            DownloadPolicy::for_mode(ParallelismMode::Tensor, true),
            DownloadPolicy::FullRepo
        );
        assert_eq!(
            DownloadPolicy::for_mode(ParallelismMode::Pipeline, false),
            DownloadPolicy::FullRepo
        );
    }
}
