//! Model loading: resolve a HF repo, download **only the safetensors a stage
//! needs**, parse config, and load tensors into [`Weights`].
//!
//! Selective download mirrors `mlx-lm.sharded_load`: for pipeline parallelism we
//! read `model.safetensors.index.json` (the weight→file map), compute which
//! files hold this rank's layers, and fetch only those. Single-node / tensor
//! parallel fetch the whole repo. MLX consumes safetensors only — never GGUF.

use crate::array::check;
use crate::array::{Array, Stream};
use crate::distributed::Pipeline;
use crate::models::ModelConfig;
use crate::nn::Weights;
use crate::{MlxError, Result};
use mesh_mlx_sys as sys;
use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

/// What to download for a given parallelism setup.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum DownloadScope {
    /// Whole repo (single node or tensor parallel).
    FullRepo,
    /// Only the safetensors files holding this stage's layers (pipeline).
    StageShard,
}

impl DownloadScope {
    /// The scope MLX uses, matching `sharded_load`.
    pub fn for_pipeline(pipeline_size: i32) -> Self {
        if pipeline_size > 1 {
            DownloadScope::StageShard
        } else {
            DownloadScope::FullRepo
        }
    }
}

/// A resolved local model directory plus its parsed config.
pub struct LoadedModel {
    pub dir: PathBuf,
    pub config: ModelConfig,
    /// Safetensors files (absolute paths) to load for this stage.
    pub shard_files: Vec<PathBuf>,
}

/// Parse `config.json` from a local model directory.
pub fn read_config(dir: &Path) -> Result<ModelConfig> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| MlxError::Load(format!("read {}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| MlxError::Load(format!("parse config.json: {e}")))
}

/// The safetensors index `weight_map` (tensor name → file name), if present.
fn read_weight_map(dir: &Path) -> Result<Option<HashMap<String, String>>> {
    let idx = dir.join("model.safetensors.index.json");
    if !idx.exists() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&idx).map_err(|e| MlxError::Load(format!("read index: {e}")))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| MlxError::Load(format!("parse index: {e}")))?;
    let map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| MlxError::Load("index missing weight_map".into()))?;
    let out = map
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect();
    Ok(Some(out))
}

/// Decide which safetensors files this stage needs from the weight map.
///
/// A tensor belongs to this stage if it is a global (embeddings / final norm /
/// lm_head) or a layer in this rank's owned range.
pub fn shard_files_for_stage(
    dir: &Path,
    pipeline: &Pipeline,
    scope: DownloadScope,
) -> Result<Vec<PathBuf>> {
    let weight_map = read_weight_map(dir)?;

    // No index (single-file model): just load model.safetensors.
    let Some(weight_map) = weight_map else {
        let single = dir.join("model.safetensors");
        return Ok(vec![single]);
    };

    if scope == DownloadScope::FullRepo {
        let files: HashSet<String> = weight_map.values().cloned().collect();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }

    let mut files = HashSet::new();
    for (tensor, file) in &weight_map {
        if tensor_belongs_to_stage(tensor, pipeline) {
            files.insert(file.clone());
        }
    }
    Ok(files.into_iter().map(|f| dir.join(f)).collect())
}

/// Whether a tensor name is owned by this pipeline stage.
fn tensor_belongs_to_stage(tensor: &str, pipeline: &Pipeline) -> bool {
    if let Some(idx) = layer_index(tensor) {
        idx >= pipeline.range.start && idx < pipeline.range.end
    } else {
        // Globals (embed_tokens / norm / lm_head): the output stage (rank 0)
        // and the first-forward stage own embeddings; load globals everywhere
        // they may be referenced. Keeping them on every stage is cheap and
        // avoids missing-weight errors at stage boundaries.
        true
    }
}

/// Extract the layer index from a tensor name like `model.layers.12.…`.
fn layer_index(tensor: &str) -> Option<usize> {
    let rest = tensor.strip_prefix("model.layers.")?;
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// Load tensors from the given safetensors files into a [`Weights`] store.
pub fn load_weights(files: &[PathBuf], s: &Stream) -> Result<Weights> {
    let mut weights = Weights::new();
    for file in files {
        load_safetensors_into(file, &mut weights, s)?;
    }
    if weights.is_empty() {
        return Err(MlxError::Load("no tensors loaded".into()));
    }
    // Materialise every weight now (on the load/CPU stream) so the Load graph
    // nodes are resolved before GPU inference pulls them — MLX cannot evaluate a
    // `Load` op on the GPU stream.
    for name in weights.names().cloned().collect::<Vec<_>>() {
        weights.get(&name)?.eval()?;
    }
    Ok(weights)
}

fn load_safetensors_into(file: &Path, weights: &mut Weights, s: &Stream) -> Result<()> {
    let path = CString::new(file.to_string_lossy().as_bytes())
        .map_err(|_| MlxError::Load("path had NUL".into()))?;

    let mut arrays = unsafe { sys::mlx_map_string_to_array_new() };
    let mut meta = unsafe { sys::mlx_map_string_to_string_new() };
    let rc = unsafe { sys::mlx_load_safetensors(&mut arrays, &mut meta, path.as_ptr(), s.raw) };
    // Free metadata regardless.
    let res = (|| {
        check(rc, "load_safetensors")?;
        let it = unsafe { sys::mlx_map_string_to_array_iterator_new(arrays) };
        loop {
            let mut key: *const std::os::raw::c_char = std::ptr::null();
            let mut value = unsafe { sys::mlx_array_new() };
            let done =
                unsafe { sys::mlx_map_string_to_array_iterator_next(&mut key, &mut value, it) };
            if done != 0 || key.is_null() {
                break;
            }
            let name = unsafe { CStr::from_ptr(key) }
                .to_string_lossy()
                .into_owned();
            weights.insert(name, Array::from_raw(value));
        }
        unsafe { sys::mlx_map_string_to_array_iterator_free(it) };
        Ok(())
    })();
    unsafe {
        sys::mlx_map_string_to_array_free(arrays);
        sys::mlx_map_string_to_string_free(meta);
    }
    res
}

/// Shard the per-layer projection weights for tensor parallelism, in place.
///
/// Following the Megatron / mlx-lm pattern across a group of size `n`, rank `r`:
/// - **column-parallel** (split output dim, axis 0): `q/k/v/gate/up` proj — each
///   rank keeps `[r*out/n .. (r+1)*out/n, :]`.
/// - **row-parallel** (split input dim, axis 1): `o/down` proj — each rank keeps
///   `[:, r*in/n .. (r+1)*in/n]`; the model `all_sum`s their partial outputs.
///
/// Quantized weights are packed along the input dim; only the **output-dim**
/// (column-parallel) split is safe to do by plain slicing without dequantizing,
/// so for quantized models we shard only the column-parallel projections and
/// leave row-parallel ones replicated (still correct, just less memory saving).
/// Dense models shard both.
pub fn shard_tensor_parallel(
    weights: &mut Weights,
    config: &ModelConfig,
    rank: i32,
    size: i32,
    s: &crate::array::Stream,
) -> Result<()> {
    if size <= 1 {
        return Ok(());
    }
    let quantized = config.quantization.is_some();
    for l in 0..config.num_hidden_layers {
        let p = format!("model.layers.{l}");
        // Column-parallel (axis 0): always safe.
        for proj in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
        ] {
            shard_axis(weights, &format!("{p}.{proj}"), 0, rank, size, quantized, s)?;
        }
        // Row-parallel (axis 1): dense only (quantized packing prevents naive
        // slicing along the input dim).
        if !quantized {
            for proj in ["self_attn.o_proj", "mlp.down_proj"] {
                shard_axis(weights, &format!("{p}.{proj}"), 1, rank, size, false, s)?;
            }
        }
    }
    Ok(())
}

/// Slice `{prefix}.weight` (and `.scales`/`.biases` for quantized column shards)
/// to this rank's contiguous chunk along `axis`.
fn shard_axis(
    weights: &mut Weights,
    prefix: &str,
    axis: usize,
    rank: i32,
    size: i32,
    quantized: bool,
    s: &crate::array::Stream,
) -> Result<()> {
    let names: Vec<String> = if quantized && axis == 0 {
        // Column-parallel quantized: weight, scales, biases all split on rows.
        ["weight", "scales", "biases"]
            .iter()
            .map(|k| format!("{prefix}.{k}"))
            .filter(|n| weights.contains(n))
            .collect()
    } else {
        vec![format!("{prefix}.weight")]
    };

    for name in names {
        let arr = weights.get(&name)?;
        let shape = arr.shape();
        if axis >= shape.len() {
            continue;
        }
        let dim = shape[axis];
        if dim % size != 0 {
            return Err(MlxError::Load(format!(
                "cannot shard '{name}' dim {dim} across {size} ranks"
            )));
        }
        let chunk = dim / size;
        let mut start = vec![0i32; shape.len()];
        let mut stop = shape.clone();
        start[axis] = rank * chunk;
        stop[axis] = (rank + 1) * chunk;
        let sliced = crate::ops::slice(arr, &start, &stop, s)?;
        sliced.eval()?;
        weights.replace(&name, sliced);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_index_parsing() {
        assert_eq!(
            layer_index("model.layers.12.self_attn.q_proj.weight"),
            Some(12)
        );
        assert_eq!(layer_index("model.embed_tokens.weight"), None);
        assert_eq!(layer_index("lm_head.weight"), None);
    }

    #[test]
    fn stage_ownership_by_layer_range() {
        // rank 0 of 2, 8 layers -> owns 4..8.
        let pipe = Pipeline::plan(0, 2, 8);
        assert!(tensor_belongs_to_stage(
            "model.layers.5.mlp.up_proj.weight",
            &pipe
        ));
        assert!(!tensor_belongs_to_stage(
            "model.layers.1.mlp.up_proj.weight",
            &pipe
        ));
        // globals are owned everywhere.
        assert!(tensor_belongs_to_stage("model.embed_tokens.weight", &pipe));
    }

    #[test]
    fn download_scope_follows_pipeline_size() {
        assert_eq!(DownloadScope::for_pipeline(1), DownloadScope::FullRepo);
        assert_eq!(DownloadScope::for_pipeline(4), DownloadScope::StageShard);
    }
}
