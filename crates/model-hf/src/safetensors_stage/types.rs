use std::path::PathBuf;

use anyhow::{Result, ensure};
use model_artifact::safetensors::TensorHeader;
use serde::{Deserialize, Serialize};

pub(crate) const MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsStageRequest {
    pub repo: String,
    /// Immutable Hugging Face commit SHA, not a branch or tag.
    pub revision: String,
    pub layer_start: u32,
    pub layer_end: u32,
    #[serde(default)]
    pub include_prefixes: Vec<String>,
}

impl SafetensorsStageRequest {
    pub(crate) fn normalized(mut self) -> Result<Self> {
        ensure!(
            !self.repo.trim().is_empty(),
            "Hugging Face repo is required"
        );
        ensure!(
            self.repo.split('/').count() == 2 && self.repo.split('/').all(|part| !part.is_empty()),
            "Hugging Face repo must be owner/name"
        );
        ensure!(
            self.revision.len() == 40 && self.revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "SafeTensors stage revision must be an immutable 40-character commit SHA"
        );
        ensure!(
            self.layer_start < self.layer_end,
            "SafeTensors stage layer range must be non-empty"
        );
        self.include_prefixes = self
            .include_prefixes
            .into_iter()
            .map(|prefix| prefix.trim().to_string())
            .filter(|prefix| !prefix.is_empty())
            .collect();
        self.include_prefixes.sort();
        self.include_prefixes.dedup();
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsStagePlan {
    pub repo: String,
    pub revision: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_prefixes: Vec<String>,
    pub total_model_tensor_bytes: Option<u64>,
    pub config_bytes: u64,
    pub index_bytes: u64,
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub largest_selected_tensor_bytes: u64,
    pub source_shard_count: usize,
    pub source_shard_bytes: u64,
    pub range_request_count: usize,
    pub range_payload_bytes: u64,
    pub header_probe_bytes: u64,
    pub planned_download_bytes: u64,
    pub source_shard_bytes_avoided: u64,
    pub full_model_tensor_bytes_avoided: Option<u64>,
    pub shards: Vec<SafetensorsShardPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsShardPlan {
    pub file: String,
    pub file_bytes: u64,
    pub header_probe_bytes: u64,
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub largest_selected_tensor_bytes: u64,
    pub ranges: Vec<ByteRange>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteRange {
    pub start: u64,
    pub end_exclusive: u64,
}

impl ByteRange {
    pub fn len(&self) -> u64 {
        self.end_exclusive - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start >= self.end_exclusive
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsStageManifest {
    pub schema_version: u32,
    pub cache_key: String,
    pub source_endpoint: String,
    pub request: SafetensorsStageRequest,
    pub selected_tensor_count: usize,
    pub selected_tensor_bytes: u64,
    pub output_file_bytes: u64,
    pub output_sha256: String,
    pub config_sha256: String,
    pub config_etag: Option<String>,
    pub index_sha256: Option<String>,
    pub index_etag: Option<String>,
    pub source_shards: Vec<SafetensorsSourceShard>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsSourceShard {
    pub file: String,
    pub file_bytes: u64,
    pub etag: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SafetensorsStageArtifact {
    pub path: PathBuf,
    pub manifest: SafetensorsStageManifest,
    pub plan: SafetensorsStagePlan,
    pub cache_hit: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedTensor {
    pub name: String,
    pub source_file: String,
    pub source_range: ByteRange,
    pub header: TensorHeader,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedStage {
    pub plan: SafetensorsStagePlan,
    pub tensors: Vec<SelectedTensor>,
    pub config: Vec<u8>,
    pub config_sha256: String,
    pub config_etag: Option<String>,
    pub index_sha256: Option<String>,
    pub index_etag: Option<String>,
    pub source_shards: Vec<SafetensorsSourceShard>,
}
