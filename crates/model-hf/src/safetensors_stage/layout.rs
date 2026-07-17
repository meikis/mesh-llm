use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, anyhow, ensure};
use model_artifact::safetensors::{
    IndexMetadata, LlamaConfig, SafetensorsIndex, TensorHeader, parse_header, parse_index,
    parse_llama_config,
};
use sha2::{Digest, Sha256};

use super::{
    http::RemoteSource,
    types::{
        ByteRange, PreparedStage, SafetensorsShardPlan, SafetensorsSourceShard,
        SafetensorsStagePlan, SafetensorsStageRequest, SelectedTensor,
    },
};

pub(crate) const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug)]
struct RemoteHeader {
    header_len: u64,
    header_sha256: String,
    file_bytes: u64,
    etag: String,
    tensors: BTreeMap<String, TensorHeader>,
}

struct CheckpointLayout {
    index: SafetensorsIndex,
    layout_sha256: String,
    index_bytes: u64,
    index_sha256: Option<String>,
    index_etag: Option<String>,
    headers: BTreeMap<String, RemoteHeader>,
}

pub(crate) fn prepare(
    remote: &RemoteSource,
    request: &SafetensorsStageRequest,
) -> Result<PreparedStage> {
    let config_url = remote.url(&request.repo, &request.revision, "config.json")?;
    let config = remote
        .small_file(config_url, MAX_INDEX_BYTES)
        .context("download SafeTensors model config")?;
    let model_config = parse_llama_config(&config.bytes)?;
    validate_layer_range(request, &model_config)?;
    let config_sha256 = sha256_hex(&config.bytes);
    let mut selection_request = request.clone();
    add_required_prefixes(&mut selection_request, &model_config);

    let mut layout = load_checkpoint_layout(remote, request)?;
    let checkpoint_sha256 = checkpoint_sha256(request, &config_sha256, &layout.layout_sha256)?;
    validate_layer_coverage(&layout.index, request)?;
    let selected = select_tensors(&layout.index.weight_map, &selection_request);
    ensure!(
        !selected.is_empty(),
        "no tensors matched layers {}..{} or requested prefixes",
        request.layer_start,
        request.layer_end
    );
    validate_required_tensors(&selected, request, &model_config)?;

    let mut by_shard: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for name in &selected {
        let shard = layout
            .index
            .weight_map
            .get(*name)
            .with_context(|| format!("selected tensor {name} is absent from weight map"))?;
        by_shard.entry(shard).or_default().insert(*name);
    }

    let mut shards = Vec::with_capacity(by_shard.len());
    let mut tensors = Vec::with_capacity(selected.len());
    for (file, names) in by_shard {
        if !layout.headers.contains_key(file) {
            let header = fetch_safetensor_header(remote, request, file)
                .with_context(|| format!("inspect {file}"))?;
            layout.headers.insert(file.to_string(), header);
        }
        let header = layout
            .headers
            .get(file)
            .with_context(|| format!("missing inspected header for {file}"))?;
        shards.push(plan_shard(file, header, &names)?);
        tensors.extend(selected_tensors(file, header, &names)?);
    }
    let source_shards = shards
        .iter()
        .map(|shard| {
            let header = layout
                .headers
                .get(&shard.file)
                .expect("planned shard has inspected header");
            SafetensorsSourceShard {
                file: shard.file.clone(),
                file_bytes: header.file_bytes,
                etag: Some(header.etag.clone()),
            }
        })
        .collect();
    let plan = summarize_plan(
        &selection_request,
        &checkpoint_sha256,
        &layout.index,
        config.bytes.len() as u64,
        layout.index_bytes,
        shards,
    )?;
    Ok(PreparedStage {
        checkpoint_sha256,
        plan,
        tensors,
        config: config.bytes,
        config_sha256,
        config_etag: config.etag,
        index_sha256: layout.index_sha256,
        index_etag: layout.index_etag,
        source_shards,
    })
}

fn validate_layer_range(request: &SafetensorsStageRequest, config: &LlamaConfig) -> Result<()> {
    ensure!(
        request.layer_end <= config.num_hidden_layers,
        "stage layer end {} exceeds model layer count {}",
        request.layer_end,
        config.num_hidden_layers
    );
    Ok(())
}

fn add_required_prefixes(request: &mut SafetensorsStageRequest, config: &LlamaConfig) {
    if request.layer_start == 0 || request.layer_end == config.num_hidden_layers {
        request
            .include_prefixes
            .push("model.embed_tokens.".to_string());
    }
    if request.layer_end == config.num_hidden_layers {
        request.include_prefixes.push("model.norm.".to_string());
        request.include_prefixes.push("lm_head.".to_string());
    }
    request.include_prefixes.sort();
    request.include_prefixes.dedup();
}

fn validate_layer_coverage(
    index: &SafetensorsIndex,
    request: &SafetensorsStageRequest,
) -> Result<()> {
    for layer in request.layer_start..request.layer_end {
        ensure!(
            index
                .weight_map
                .keys()
                .any(|name| layer_index(name) == Some(layer)),
            "SafeTensors checkpoint has no tensors for requested layer {layer}"
        );
    }
    Ok(())
}

fn validate_required_tensors(
    selected: &BTreeSet<&str>,
    request: &SafetensorsStageRequest,
    config: &LlamaConfig,
) -> Result<()> {
    let has_prefix = |prefix: &str| selected.iter().any(|name| name.starts_with(prefix));
    if request.layer_start == 0 {
        ensure!(
            has_prefix("model.embed_tokens."),
            "first MLX stage requires model.embed_tokens tensors"
        );
    }
    if request.layer_end == config.num_hidden_layers {
        ensure!(
            has_prefix("model.norm."),
            "final MLX stage requires model.norm tensors"
        );
        ensure!(
            has_prefix("lm_head.") || has_prefix("model.embed_tokens."),
            "final MLX stage requires lm_head or tied embedding tensors"
        );
    }
    Ok(())
}

fn load_checkpoint_layout(
    remote: &RemoteSource,
    request: &SafetensorsStageRequest,
) -> Result<CheckpointLayout> {
    let index_url = remote.url(
        &request.repo,
        &request.revision,
        "model.safetensors.index.json",
    )?;
    if let Some(index_file) = remote.optional_small_file(index_url, MAX_INDEX_BYTES)? {
        let index = parse_index(&index_file.bytes)?;
        let index_sha256 = sha256_hex(&index_file.bytes);
        return Ok(CheckpointLayout {
            index,
            layout_sha256: index_sha256.clone(),
            index_bytes: index_file.bytes.len() as u64,
            index_sha256: Some(index_sha256),
            index_etag: index_file.etag,
            headers: BTreeMap::new(),
        });
    }

    let file = "model.safetensors";
    let header = fetch_safetensor_header(remote, request, file)
        .context("inspect unsharded SafeTensors checkpoint")?;
    let total_size = header
        .tensors
        .values()
        .map(|tensor| tensor.data_offsets[1])
        .max();
    let weight_map = header
        .tensors
        .keys()
        .map(|name| (name.clone(), file.to_string()))
        .collect();
    Ok(CheckpointLayout {
        index: SafetensorsIndex {
            metadata: IndexMetadata { total_size },
            weight_map,
        },
        layout_sha256: header.header_sha256.clone(),
        index_bytes: 0,
        index_sha256: None,
        index_etag: None,
        headers: BTreeMap::from([(file.to_string(), header)]),
    })
}

fn fetch_safetensor_header(
    remote: &RemoteSource,
    request: &SafetensorsStageRequest,
    file: &str,
) -> Result<RemoteHeader> {
    let url = remote.url(&request.repo, &request.revision, file)?;
    let len_response = remote.exact_range(url.clone(), 0..8)?;
    let file_bytes = len_response.total_file_bytes;
    let len_etag = len_response
        .etag()
        .context("SafeTensors header-length response omitted ETag")?
        .to_string();
    let len_bytes = len_response.into_bytes()?;
    let header_len = u64::from_le_bytes(
        len_bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("invalid 8-byte SafeTensors header length"))?,
    );
    ensure!(
        header_len <= MAX_HEADER_BYTES,
        "SafeTensors header is unexpectedly large: {header_len} bytes"
    );
    let header_end = 8_u64
        .checked_add(header_len)
        .context("SafeTensors header range overflow")?;
    ensure!(
        header_end <= file_bytes,
        "SafeTensors header exceeds source file length"
    );
    let header_response = remote.exact_range_if_range(url, 8..header_end, &len_etag)?;
    ensure!(
        header_response.total_file_bytes == file_bytes,
        "source file size changed while reading SafeTensors header"
    );
    let header_etag = header_response
        .etag()
        .context("SafeTensors header response omitted ETag")?;
    ensure_matching_etag(&len_etag, header_etag, file)?;
    let etag = header_etag.to_string();
    let header_bytes = header_response.into_bytes()?;
    let header_sha256 = sha256_hex(&header_bytes);
    let data_bytes = file_bytes - header_end;
    let tensors = parse_header(&header_bytes, data_bytes)?;
    Ok(RemoteHeader {
        header_len,
        header_sha256,
        file_bytes,
        etag,
        tensors,
    })
}

fn ensure_matching_etag(left: &str, right: &str, file: &str) -> Result<()> {
    ensure!(
        left == right,
        "source identity changed while reading SafeTensors shard {file}"
    );
    Ok(())
}

fn select_tensors<'a>(
    weight_map: &'a BTreeMap<String, String>,
    request: &SafetensorsStageRequest,
) -> BTreeSet<&'a str> {
    weight_map
        .keys()
        .filter(|name| {
            layer_index(name)
                .is_some_and(|layer| layer >= request.layer_start && layer < request.layer_end)
                || request
                    .include_prefixes
                    .iter()
                    .any(|prefix| name.starts_with(prefix))
        })
        .map(String::as_str)
        .collect()
}

fn layer_index(name: &str) -> Option<u32> {
    name.strip_prefix("model.layers.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

fn plan_shard(
    file: &str,
    header: &RemoteHeader,
    selected: &BTreeSet<&str>,
) -> Result<SafetensorsShardPlan> {
    let data_start = 8_u64
        .checked_add(header.header_len)
        .context("SafeTensors data offset overflow")?;
    let mut ranges = Vec::with_capacity(selected.len());
    let mut selected_tensor_bytes = 0_u64;
    let mut largest_selected_tensor_bytes = 0_u64;
    for name in selected {
        let tensor = header
            .tensors
            .get(*name)
            .with_context(|| format!("weight-map tensor {name} is absent from {file}"))?;
        let start = data_start
            .checked_add(tensor.data_offsets[0])
            .with_context(|| format!("absolute offset overflow for {name}"))?;
        let end_exclusive = data_start
            .checked_add(tensor.data_offsets[1])
            .with_context(|| format!("absolute offset overflow for {name}"))?;
        let tensor_bytes = end_exclusive - start;
        selected_tensor_bytes = selected_tensor_bytes
            .checked_add(tensor_bytes)
            .context("selected tensor byte count overflow")?;
        largest_selected_tensor_bytes = largest_selected_tensor_bytes.max(tensor_bytes);
        ranges.push(ByteRange {
            start,
            end_exclusive,
        });
    }
    let ranges = coalesce_contiguous_ranges(ranges);
    Ok(SafetensorsShardPlan {
        file: file.to_string(),
        file_bytes: header.file_bytes,
        header_probe_bytes: data_start,
        selected_tensor_count: selected.len(),
        selected_tensor_bytes,
        largest_selected_tensor_bytes,
        ranges,
    })
}

fn selected_tensors(
    file: &str,
    header: &RemoteHeader,
    selected: &BTreeSet<&str>,
) -> Result<Vec<SelectedTensor>> {
    let data_start = 8_u64
        .checked_add(header.header_len)
        .context("SafeTensors data offset overflow")?;
    selected
        .iter()
        .map(|name| {
            let tensor = header
                .tensors
                .get(*name)
                .with_context(|| format!("weight-map tensor {name} is absent from {file}"))?;
            Ok(SelectedTensor {
                name: (*name).to_string(),
                source_file: file.to_string(),
                source_range: ByteRange {
                    start: data_start
                        .checked_add(tensor.data_offsets[0])
                        .with_context(|| format!("absolute offset overflow for {name}"))?,
                    end_exclusive: data_start
                        .checked_add(tensor.data_offsets[1])
                        .with_context(|| format!("absolute offset overflow for {name}"))?,
                },
                header: tensor.clone(),
            })
        })
        .collect()
}

fn coalesce_contiguous_ranges(mut ranges: Vec<ByteRange>) -> Vec<ByteRange> {
    ranges.sort_by_key(|range| range.start);
    let mut merged: Vec<ByteRange> = Vec::with_capacity(ranges.len());
    for range in ranges {
        if let Some(previous) = merged.last_mut()
            && range.start == previous.end_exclusive
        {
            previous.end_exclusive = range.end_exclusive;
        } else {
            merged.push(range);
        }
    }
    merged
}

fn summarize_plan(
    request: &SafetensorsStageRequest,
    checkpoint_sha256: &str,
    index: &SafetensorsIndex,
    config_bytes: u64,
    index_bytes: u64,
    shards: Vec<SafetensorsShardPlan>,
) -> Result<SafetensorsStagePlan> {
    let selected_tensor_count = shards.iter().map(|shard| shard.selected_tensor_count).sum();
    let selected_tensor_bytes =
        checked_sum(shards.iter().map(|shard| shard.selected_tensor_bytes))?;
    let largest_selected_tensor_bytes = shards
        .iter()
        .map(|shard| shard.largest_selected_tensor_bytes)
        .max()
        .unwrap_or(0);
    let source_shard_bytes = checked_sum(shards.iter().map(|shard| shard.file_bytes))?;
    let range_request_count = shards.iter().map(|shard| shard.ranges.len()).sum();
    let range_payload_bytes = checked_sum(
        shards
            .iter()
            .flat_map(|shard| shard.ranges.iter())
            .map(ByteRange::len),
    )?;
    let header_probe_bytes = checked_sum(shards.iter().map(|shard| shard.header_probe_bytes))?;
    let planned_download_bytes = config_bytes
        .checked_add(index_bytes)
        .and_then(|bytes| bytes.checked_add(header_probe_bytes))
        .and_then(|bytes| bytes.checked_add(range_payload_bytes))
        .context("planned download byte count overflow")?;
    Ok(SafetensorsStagePlan {
        checkpoint_sha256: checkpoint_sha256.to_string(),
        repo: request.repo.clone(),
        revision: request.revision.clone(),
        layer_start: request.layer_start,
        layer_end: request.layer_end,
        include_prefixes: request.include_prefixes.clone(),
        total_model_tensor_bytes: index.metadata.total_size,
        config_bytes,
        index_bytes,
        selected_tensor_count,
        selected_tensor_bytes,
        largest_selected_tensor_bytes,
        source_shard_count: shards.len(),
        source_shard_bytes,
        range_request_count,
        range_payload_bytes,
        header_probe_bytes,
        planned_download_bytes,
        source_shard_bytes_avoided: source_shard_bytes
            .saturating_sub(header_probe_bytes + range_payload_bytes),
        full_model_tensor_bytes_avoided: index
            .metadata
            .total_size
            .map(|total| total.saturating_sub(selected_tensor_bytes)),
        shards,
    })
}

fn checkpoint_sha256(
    request: &SafetensorsStageRequest,
    config_sha256: &str,
    layout_sha256: &str,
) -> Result<String> {
    let identity = serde_json::to_vec(&(
        "mesh-mlx-safetensors-checkpoint-v1",
        &request.repo,
        &request.revision,
        config_sha256,
        layout_sha256,
    ))?;
    Ok(sha256_hex(&identity))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).context("byte count overflow")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_only_llama_layer_paths() {
        assert_eq!(layer_index("model.layers.42.mlp.up_proj.weight"), Some(42));
        assert_eq!(layer_index("transformer.h.7.attn.weight"), None);
        assert_eq!(layer_index("model.embed_tokens.weight"), None);
    }

    #[test]
    fn coalesces_only_contiguous_ranges() {
        assert_eq!(
            coalesce_contiguous_ranges(vec![
                ByteRange {
                    start: 20,
                    end_exclusive: 30,
                },
                ByteRange {
                    start: 0,
                    end_exclusive: 10,
                },
                ByteRange {
                    start: 10,
                    end_exclusive: 20,
                },
                ByteRange {
                    start: 32,
                    end_exclusive: 40,
                },
            ]),
            vec![
                ByteRange {
                    start: 0,
                    end_exclusive: 30,
                },
                ByteRange {
                    start: 32,
                    end_exclusive: 40,
                },
            ]
        );
    }

    #[test]
    fn assigns_embedding_and_readout_tensors_to_final_llama_stage() {
        let mut request = SafetensorsStageRequest {
            repo: "org/model".to_string(),
            revision: "a".repeat(40),
            layer_start: 1,
            layer_end: 2,
            include_prefixes: Vec::new(),
        };
        let config = LlamaConfig {
            model_type: "llama".to_string(),
            hidden_size: 2,
            num_hidden_layers: 2,
            tie_word_embeddings: true,
        };

        add_required_prefixes(&mut request, &config);

        assert_eq!(
            request.include_prefixes,
            vec![
                "lm_head.".to_string(),
                "model.embed_tokens.".to_string(),
                "model.norm.".to_string(),
            ]
        );
    }
}
