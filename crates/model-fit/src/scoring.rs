use crate::{
    AcceleratorKind, AcceleratorProfile, BackendKind, CapabilityEvidence, DecodeCostBreakdown,
    DecodeCostGroupBreakdown, DecodeEstimateRange, EstimateConfidence, FirstTokenEstimateRange,
    FitStatus, HardwareProfile, KvCacheKind, MatmulShapeProfile, MeasurementSource,
    ModelArchitectureClass, ModelProfile, ModelRecommendation, Requirement, ScoreWeights,
    SelectionConfig, TensorGroupBytes, TensorMatmulGroupProfile, TensorTypeBytes, WeightCoverage,
    WorkloadTask,
};
use mesh_llm_gpu_bench::DecodeKernelProbe;
use std::cmp::Ordering;

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const LLAMA_DEFAULT_UBATCH_TOKENS: u32 = 512;
const MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE: f64 = 0.75;

#[derive(Clone, Debug)]
struct ExecutionBudget {
    backend: BackendKind,
    accelerator_name: Option<String>,
    accelerator_kind: AcceleratorKind,
    usable_memory_bytes: u64,
    memory_bandwidth_bytes_per_sec: Option<u64>,
    decode_effective_bandwidth_bytes_per_sec: Option<u64>,
    decode_fixed_overhead_ms: Option<f32>,
    decode_runtime_overhead_ms: Option<f32>,
    post_prefill_decode_overhead_ms: Option<f32>,
    bandwidth_source: MeasurementSource,
    benchmark_noise_pct: Option<f32>,
    compute_tflops_fp16: Option<f32>,
    prefill_matmul_tflops_fp16: Option<f32>,
    prefill_ubatch_matmul_tflops_fp16: Option<f32>,
    prefill_moe_matmul_tflops_fp16: Option<f32>,
    sampler_history_us_per_token: Option<f32>,
    sampler_vocab_us_per_token: Option<f32>,
    decode_kernel_probes: Vec<DecodeKernelProbe>,
    unified_memory: bool,
}

#[derive(Clone, Copy, Debug)]
struct DecodeProbeTarget {
    tensor_type: &'static str,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeGroupKind {
    TransformerBlock,
    AttentionMatmul,
    FeedForwardMatmul,
    OutputMatmul,
    RoutedExpert,
}

#[derive(Clone, Copy, Debug)]
struct DecodeTrafficGroup {
    kind: DecodeGroupKind,
    type_bytes: TensorTypeBytes,
    shape: MatmulShapeProfile,
    expert_scaled: bool,
}

#[derive(Clone, Debug, Default)]
struct GroupedDecodeCost {
    bandwidth_ms: f32,
    probed_bytes: u64,
    fallback_bytes: u64,
    groups: Vec<DecodeCostGroupBreakdown>,
}

#[derive(Clone, Copy, Debug)]
struct DecodeProbeSelection<'a> {
    probe: &'a DecodeKernelProbe,
    shape_distance: f64,
}

#[derive(Clone, Copy, Debug)]
struct DenseBlockProbeSelection<'a> {
    probe: &'a DecodeKernelProbe,
    tensor_type: &'static str,
    shape_distance: f64,
    bandwidth_bytes_per_sec: u64,
}

#[derive(Clone, Copy, Debug)]
struct DenseBlockProbeTime {
    variable_ms: f32,
    effective_bandwidth_bytes_per_sec: u64,
    source: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeMemoryEstimate {
    pub runtime_bytes: u64,
    pub kv_cache_bytes: u64,
    pub resident_weight_bytes: u64,
    pub scratch_bytes: u64,
    pub backend_overhead_bytes: u64,
}

pub fn estimate_kv_cache_bytes(model: &ModelProfile, config: &SelectionConfig) -> u64 {
    // Public KV estimate for the configured target context. See
    // `kv_cache_bytes_for_context` for the shape assumptions. This is exposed
    // because callers often want to show users how much of the memory budget is
    // model weights vs context.
    let context_tokens = target_context_tokens(model, config);
    kv_cache_bytes_for_context(model, config, context_tokens)
}

pub fn estimate_runtime_memory_bytes(model: &ModelProfile, config: &SelectionConfig) -> u64 {
    runtime_memory_estimate(model, config).runtime_bytes
}

pub fn score_model(
    hardware: &HardwareProfile,
    model: &ModelProfile,
    config: &SelectionConfig,
) -> ModelRecommendation {
    let mut recommendations = execution_budgets(hardware, config)
        .into_iter()
        .map(|budget| score_for_budget(model, config, &budget))
        .collect::<Vec<_>>();

    if recommendations.is_empty() {
        let budget = ExecutionBudget {
            backend: BackendKind::Unknown,
            accelerator_name: None,
            accelerator_kind: AcceleratorKind::Unknown,
            usable_memory_bytes: 0,
            memory_bandwidth_bytes_per_sec: None,
            decode_effective_bandwidth_bytes_per_sec: None,
            decode_fixed_overhead_ms: None,
            decode_runtime_overhead_ms: None,
            post_prefill_decode_overhead_ms: None,
            bandwidth_source: MeasurementSource::Unknown,
            benchmark_noise_pct: None,
            compute_tflops_fp16: None,
            prefill_matmul_tflops_fp16: None,
            prefill_ubatch_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            sampler_history_us_per_token: None,
            sampler_vocab_us_per_token: None,
            decode_kernel_probes: Vec::new(),
            unified_memory: false,
        };
        return score_for_budget(model, config, &budget);
    }

    recommendations.sort_by(compare_execution_budget_recommendations);
    recommendations.remove(0)
}

pub fn rank_models(
    hardware: &HardwareProfile,
    models: &[ModelProfile],
    config: &SelectionConfig,
) -> Vec<ModelRecommendation> {
    let mut recommendations = models
        .iter()
        .map(|model| score_model(hardware, model, config))
        .collect::<Vec<_>>();
    recommendations.sort_by(compare_recommendations);
    recommendations
}

fn score_for_budget(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
) -> ModelRecommendation {
    let memory = runtime_memory_estimate(model, config);
    let active_decode_bytes = active_decode_bytes_per_token(model, config);
    let active_decode_flops = active_decode_flops_per_token(model);
    let estimated_decode_tps = decode_tokens_per_sec(
        active_decode_bytes,
        active_decode_flops,
        budget,
        config,
        model,
    );
    let decode_cost_breakdown = decode_cost_breakdown(active_decode_flops, budget, config, model);
    let estimated_decode_range =
        decode_tokens_per_sec_range(estimated_decode_tps, active_decode_bytes, model, budget);
    let estimated_prefill_tps = prefill_tokens_per_sec(model, config, budget, estimated_decode_tps);
    let first_token = first_token_estimate(
        model,
        estimated_prefill_tps,
        estimated_decode_tps,
        config,
        budget,
    );
    let estimated_first_token_range = first_token_ms_range(first_token.total_ms, model, budget);
    let memory_limit = memory_limit_with_margin(budget.usable_memory_bytes, config.safety_margin);
    let mut warnings = Vec::new();
    let mut reasons = Vec::new();

    let (workload_score, workload_reject) =
        workload_score(model, config, &mut reasons, &mut warnings);
    let fit_status = fit_status(
        model,
        &memory,
        memory_limit,
        workload_reject,
        &mut reasons,
        &mut warnings,
    );
    let memory_score = memory_score(memory.runtime_bytes, memory_limit);
    let context_score = context_score(model, config, &mut reasons, &mut warnings);
    let decode_score = decode_score(estimated_decode_tps, config);
    let prefill_score = prefill_score(model, config, budget);
    let total_score = total_score(
        config.weights,
        memory_score,
        context_score,
        decode_score,
        prefill_score,
        workload_score,
        fit_status,
    );

    if budget.memory_bandwidth_bytes_per_sec.is_none() {
        warnings
            .push("memory bandwidth is missing; decode score uses a conservative fallback".into());
    }
    if measured_gpu_budget(budget) && budget.decode_effective_bandwidth_bytes_per_sec.is_none() {
        warnings.push(
            "measured GPU profile is missing decode-shaped bandwidth; decode estimate falls back to raw bandwidth"
                .into(),
        );
    }
    if measured_gpu_budget(budget) && budget.decode_fixed_overhead_ms.is_none() {
        warnings.push(
            "measured GPU profile is missing fixed decode overhead; no measured fixed overhead is applied"
                .into(),
        );
    }
    if measured_gpu_budget(budget) && budget.decode_kernel_probes.is_empty() {
        warnings.push(
            "measured GPU profile is missing llama-shaped decode kernel probes; tok/s confidence cannot be high"
                .into(),
        );
    } else if let Some(tensor_type) = missing_exact_decode_kernel_probe_tensor_type(model, budget) {
        warnings.push(format!(
            "measured GPU profile does not include a shape-representative decode kernel probe for dominant tensor type {tensor_type}; tok/s confidence cannot be high"
        ));
    } else if measured_gpu_budget(budget) && !has_high_confidence_decode_probe(model, budget) {
        warnings.push(high_confidence_decode_probe_warning(model));
    } else if measured_gpu_budget(budget) && has_high_confidence_decode_probe(model, budget) {
        warnings.push(
            "tok/s confidence is medium; composite decode probes are hardware evidence, but metadata-only estimates are not yet validated to +/-10% observed tok/s across model shapes"
                .into(),
        );
    }
    if let Some(warning) = shallow_q8_graph_probe_warning(model, budget) {
        warnings.push(warning);
    }
    if budget.unified_memory {
        reasons.push("using unified-memory budget for model weights, KV cache, and scratch".into());
    }
    reasons.push(format!(
        "runtime estimate includes {:.1} GiB resident weights, {:.1} GiB scratch, and {:.1} GiB backend overhead",
        gib(memory.resident_weight_bytes),
        gib(memory.scratch_bytes),
        gib(memory.backend_overhead_bytes)
    ));
    add_decode_estimate_reason(model, budget, config, &mut reasons);
    add_prefill_estimate_reason(first_token.total_ms, config, &mut reasons);
    add_architecture_warnings(model, &mut warnings);

    ModelRecommendation {
        source: model.source.clone(),
        selected_backend: budget.backend,
        selected_accelerator: budget.accelerator_name.clone(),
        architecture_class: model.architecture_class,
        estimate_confidence: estimate_confidence(model, budget, fit_status),
        fit_status,
        total_score,
        memory_score,
        context_score,
        decode_score,
        prefill_score,
        workload_score,
        estimated_runtime_memory_bytes: memory.runtime_bytes,
        estimated_kv_cache_bytes: memory.kv_cache_bytes,
        estimated_active_decode_bytes_per_token: active_decode_bytes,
        estimated_decode_tokens_per_sec: local_fit_value(fit_status, estimated_decode_tps),
        estimated_decode_tokens_per_sec_range: local_fit_value(fit_status, estimated_decode_range),
        decode_cost_breakdown: local_fit_value(fit_status, decode_cost_breakdown),
        estimated_prefill_tokens_per_sec: local_fit_value(fit_status, estimated_prefill_tps),
        estimated_first_token_prefill_ms: local_fit_value(fit_status, first_token.prefill_ms),
        estimated_first_token_decode_ms: local_fit_value(fit_status, first_token.decode_ms),
        estimated_first_token_overhead_ms: local_fit_value(fit_status, first_token.overhead_ms),
        estimated_first_token_sampler_ms: local_fit_value(fit_status, first_token.sampler_ms),
        estimated_first_token_ms: local_fit_value(fit_status, first_token.total_ms),
        estimated_first_token_ms_range: local_fit_value(fit_status, estimated_first_token_range),
        capability_evidence: model.capability_evidence.clone(),
        reasons,
        warnings,
    }
}

fn local_fit_value<T>(fit_status: FitStatus, value: Option<T>) -> Option<T> {
    if matches!(
        fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    ) {
        value
    } else {
        None
    }
}

fn execution_budgets(hardware: &HardwareProfile, config: &SelectionConfig) -> Vec<ExecutionBudget> {
    let mut budgets = hardware
        .accelerators
        .iter()
        .map(|accelerator| accelerator_budget(hardware, accelerator))
        .collect::<Vec<_>>();
    if let (true, Some(memory)) = (
        include_cpu_budget(hardware, config),
        hardware.memory.available_system_bytes,
    ) {
        budgets.push(ExecutionBudget {
            backend: BackendKind::Cpu,
            accelerator_name: Some("CPU".into()),
            accelerator_kind: AcceleratorKind::Cpu,
            usable_memory_bytes: memory,
            memory_bandwidth_bytes_per_sec: hardware.cpu.memory_bandwidth_bytes_per_sec,
            decode_effective_bandwidth_bytes_per_sec: None,
            decode_fixed_overhead_ms: None,
            decode_runtime_overhead_ms: None,
            bandwidth_source: hardware
                .cpu
                .memory_bandwidth_bytes_per_sec
                .map(|_| MeasurementSource::Measured)
                .unwrap_or(MeasurementSource::Unknown),
            benchmark_noise_pct: None,
            post_prefill_decode_overhead_ms: hardware.cpu.post_prefill_decode_overhead_ms,
            compute_tflops_fp16: hardware.cpu.compute_tflops_fp16,
            prefill_matmul_tflops_fp16: hardware.cpu.prefill_matmul_tflops_fp16,
            prefill_ubatch_matmul_tflops_fp16: hardware.cpu.prefill_ubatch_matmul_tflops_fp16,
            prefill_moe_matmul_tflops_fp16: hardware.cpu.prefill_moe_matmul_tflops_fp16,
            sampler_history_us_per_token: hardware.cpu.sampler_history_us_per_token,
            sampler_vocab_us_per_token: hardware.cpu.sampler_vocab_us_per_token,
            decode_kernel_probes: Vec::new(),
            unified_memory: false,
        });
    }
    budgets
}

fn include_cpu_budget(hardware: &HardwareProfile, config: &SelectionConfig) -> bool {
    // A CPU budget is real when the host is CPU-only, and it is still useful for
    // non-generative workloads such as embeddings and reranking where CPU
    // serving can be an acceptable local path. It must not be a silent fallback
    // for transformer generation on a discrete-GPU host, though.
    //
    // llama.cpp full-offload loads resident model tensors into the selected
    // backend buffer. If a 30B MoE GGUF requires ~17 GiB of CUDA buffer and the
    // device has ~15 GiB free, that model does not fit the accelerated local
    // serving path even if system RAM can hold the file. Reporting `FitsLocal`
    // through CPU RAM would conflate "can be mmap'd somewhere" with "fits the
    // machine profile that mesh-llm will actually serve with." For those
    // generation workloads, keep the accelerator budget in charge so the result
    // becomes `Rejected` instead.
    !has_non_cpu_accelerator(hardware) || cpu_is_valid_workload_budget(config.workload.task)
}

fn has_non_cpu_accelerator(hardware: &HardwareProfile) -> bool {
    hardware.accelerators.iter().any(|accelerator| {
        accelerator.backend != BackendKind::Cpu && accelerator.kind != AcceleratorKind::Cpu
    })
}

fn cpu_is_valid_workload_budget(task: WorkloadTask) -> bool {
    matches!(
        task,
        WorkloadTask::Embedding | WorkloadTask::Reranking | WorkloadTask::Classification
    )
}

fn accelerator_budget(
    hardware: &HardwareProfile,
    accelerator: &AcceleratorProfile,
) -> ExecutionBudget {
    let usable_memory_bytes = if accelerator.unified_memory {
        accelerator
            .available_memory_bytes
            .or(hardware.memory.available_unified_bytes)
            .or(hardware.memory.available_system_bytes)
            .or(accelerator.total_memory_bytes)
            .or(hardware.memory.total_unified_bytes)
            .unwrap_or(0)
    } else {
        accelerator
            .available_memory_bytes
            .or(accelerator.total_memory_bytes)
            .unwrap_or(0)
    };

    ExecutionBudget {
        backend: accelerator.backend,
        accelerator_name: accelerator.name.clone().or_else(|| {
            (accelerator.kind != AcceleratorKind::Unknown)
                .then(|| format!("{:?}", accelerator.kind))
        }),
        accelerator_kind: accelerator.kind,
        usable_memory_bytes,
        memory_bandwidth_bytes_per_sec: accelerator.memory_bandwidth_bytes_per_sec,
        decode_effective_bandwidth_bytes_per_sec: accelerator
            .decode_effective_bandwidth_bytes_per_sec,
        decode_fixed_overhead_ms: accelerator.decode_fixed_overhead_ms,
        decode_runtime_overhead_ms: accelerator.decode_runtime_overhead_ms,
        post_prefill_decode_overhead_ms: accelerator.post_prefill_decode_overhead_ms,
        bandwidth_source: accelerator.bandwidth_source,
        benchmark_noise_pct: accelerator.benchmark_noise_pct,
        compute_tflops_fp16: accelerator.compute_tflops_fp16,
        prefill_matmul_tflops_fp16: accelerator.prefill_matmul_tflops_fp16,
        prefill_ubatch_matmul_tflops_fp16: accelerator.prefill_ubatch_matmul_tflops_fp16,
        prefill_moe_matmul_tflops_fp16: accelerator.prefill_moe_matmul_tflops_fp16,
        sampler_history_us_per_token: accelerator.sampler_history_us_per_token,
        sampler_vocab_us_per_token: accelerator.sampler_vocab_us_per_token,
        decode_kernel_probes: accelerator.decode_kernel_probes.clone(),
        unified_memory: accelerator.unified_memory,
    }
}

fn runtime_memory_estimate(
    model: &ModelProfile,
    config: &SelectionConfig,
) -> RuntimeMemoryEstimate {
    // Runtime memory is more than the GGUF file. The resident weights are the
    // largest term, but context-heavy workloads can add a large KV cache and
    // every backend needs scratch/activation space. We keep this estimate
    // explainable instead of trying to reproduce llama.cpp allocation internals:
    //
    //   resident weights + KV cache + scratch + backend overhead
    //
    // The selector applies a configurable safety margin to this value before it
    // ranks a model as a local fit. That margin absorbs allocator
    // fragmentation, OS pressure, backend work buffers, and imperfect metadata.
    let resident_weight_bytes = resident_weight_bytes(model);
    let kv_cache_bytes = estimate_kv_cache_bytes(model, config);
    let scratch_bytes = scratch_bytes(model, resident_weight_bytes);
    let backend_overhead_bytes = 256 * MIB + resident_weight_bytes / 100;
    let runtime_bytes = resident_weight_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(scratch_bytes)
        .saturating_add(backend_overhead_bytes);
    RuntimeMemoryEstimate {
        runtime_bytes,
        kv_cache_bytes,
        resident_weight_bytes,
        scratch_bytes,
        backend_overhead_bytes,
    }
}

fn resident_weight_bytes(model: &ModelProfile) -> u64 {
    model
        .tensor_bytes
        .or_else(|| {
            Some(
                model
                    .base_resident_bytes?
                    .saturating_add(model.expert_tensor_bytes.unwrap_or(0)),
            )
        })
        .unwrap_or(model.file_size_bytes)
}

fn scratch_bytes(model: &ModelProfile, resident_weight_bytes: u64) -> u64 {
    let minimum = match model.architecture_class {
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            256 * MIB
        }
        _ => 512 * MIB,
    };
    minimum.max(resident_weight_bytes / 20)
}

fn kv_cache_bytes_for_context(
    model: &ModelProfile,
    config: &SelectionConfig,
    context_tokens: u32,
) -> u64 {
    // KV cache is the memory term that scales with requested context rather
    // than file size. The exact layout is backend/model dependent, but GGUF
    // normally gives enough architecture metadata to estimate the dominant
    // shape:
    //
    //   (K row + V row) * layer_count * context_tokens
    //
    // If grouped-query attention metadata is present, `kv_width` uses
    // kv_heads * key/value_length. Otherwise it falls back to hidden_size, which
    // is conservative for arbitrary GGUFs. Cache quantization is modeled with
    // ggml block row sizes so Q8/Q4 KV settings reduce memory in roughly the
    // same proportions llama.cpp will use.
    if !uses_transformer_kv_cache(model.architecture_class) {
        return 0;
    }
    let Some(layers) = model.layer_count else {
        return fallback_kv_cache_bytes(model, config, context_tokens);
    };
    let k_width = kv_width(model, model.key_length);
    let v_width = kv_width(model, model.value_length);
    let k_bytes = row_size(config.kv_cache_type.k, k_width).saturating_mul(u64::from(layers));
    let v_bytes = row_size(config.kv_cache_type.v, v_width).saturating_mul(u64::from(layers));
    k_bytes
        .saturating_add(v_bytes)
        .saturating_mul(u64::from(context_tokens))
}

fn fallback_kv_cache_bytes(
    model: &ModelProfile,
    config: &SelectionConfig,
    context_tokens: u32,
) -> u64 {
    let Some(hidden) = model.hidden_size else {
        return 0;
    };
    let layers = model.layer_count.unwrap_or(1);
    let k = row_size(config.kv_cache_type.k, u64::from(hidden));
    let v = row_size(config.kv_cache_type.v, u64::from(hidden));
    k.saturating_add(v)
        .saturating_mul(u64::from(layers))
        .saturating_mul(u64::from(context_tokens))
}

fn kv_width(model: &ModelProfile, vector_length: Option<u32>) -> u64 {
    match (model.kv_heads, vector_length) {
        (Some(kv_heads), Some(length)) => u64::from(kv_heads).saturating_mul(u64::from(length)),
        _ => u64::from(model.hidden_size.unwrap_or_default()),
    }
}

fn row_size(kind: KvCacheKind, elements: u64) -> u64 {
    let (block_elements, block_bytes) = match kind {
        KvCacheKind::F16 => (1, 2),
        KvCacheKind::Q8_0 => (32, 34),
        KvCacheKind::Q4_0 => (32, 18),
    };
    elements.div_ceil(block_elements) * block_bytes
}

fn active_decode_bytes_per_token(model: &ModelProfile, config: &SelectionConfig) -> Option<u64> {
    // Decode is usually the user-visible bottleneck for local chat/agent loops:
    // it runs once per generated token, and on common GGUF backends it is often
    // closer to memory-bandwidth-bound than compute-bound. The core predictor is
    // therefore "how many bytes must be touched for one token?"
    //
    // For dense models that is mostly resident transformer weights. For MoE
    // models it is base/resident weights plus only the routed experts, because a
    // token does not execute every expert. When tensor group sizes are available
    // from GGUF inspection we use them directly; otherwise we fall back to the
    // coarser resident/active byte estimates.
    //
    // KV reads and a small activation proxy are included so that context length
    // and layer/hidden width can still affect decode ranking. For the default
    // config, `kv_read_scale` is 1.0 because llama.cpp's decode attention path
    // builds KQ from cached keys and KQV from cached values for the active
    // attention window. A backend may tile, fuse, or cache pieces internally,
    // but the metadata-visible source graph still has to consume those K/V
    // rows. This is not a full llama.cpp execution trace; it is an explainable
    // active-byte pressure estimate that can be produced from GGUF metadata
    // alone.
    let active_weights = match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => active_moe_decode_weight_traffic(model),
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            return None;
        }
        _ => active_dense_decode_weight_traffic(model),
    };
    let context = config
        .workload
        .interaction
        .expected_prompt_tokens
        .unwrap_or_else(|| target_context_tokens(model, config) / 2);
    let kv_read_bytes = kv_cache_bytes_for_context(model, config, context)
        .saturating_mul((config.kv_read_scale * 1000.0).round() as u64)
        / 1000;
    let activation_overhead = activation_overhead_bytes(model);
    Some(
        active_weights
            .saturating_add(kv_read_bytes)
            .saturating_add(activation_overhead),
    )
}

fn active_dense_decode_weight_traffic(model: &ModelProfile) -> u64 {
    if let Some(bytes) = dense_matmul_traffic_bytes(model) {
        return bytes;
    }
    let groups = model.tensor_group_bytes;
    let storage_bytes = if tensor_groups_available(groups) {
        groups
            .attention_bytes
            .saturating_add(groups.feed_forward_bytes)
            .saturating_add(groups.output_bytes)
            .saturating_add(groups.normalization_bytes)
            .saturating_add(groups.other_bytes)
    } else {
        resident_weight_bytes(model)
    };
    decode_weight_traffic_bytes(model, storage_bytes)
}

fn active_moe_decode_weight_traffic(model: &ModelProfile) -> u64 {
    if let Some(bytes) = moe_matmul_traffic_bytes(model) {
        return bytes;
    }
    let groups = model.tensor_group_bytes;
    let storage_bytes = if tensor_groups_available(groups) {
        let active_expert_bytes = active_expert_bytes(
            groups.expert_feed_forward_bytes,
            model.expert_count,
            model.expert_used_count,
        );
        groups
            .attention_bytes
            .saturating_add(groups.feed_forward_bytes)
            .saturating_add(active_expert_bytes)
            .saturating_add(groups.output_bytes)
            .saturating_add(groups.normalization_bytes)
            .saturating_add(groups.other_bytes)
    } else {
        active_moe_weight_bytes(model)
    };
    decode_weight_traffic_bytes(model, storage_bytes)
}

fn decode_weight_traffic_bytes(model: &ModelProfile, storage_bytes: u64) -> u64 {
    // GGUF tensor bytes are a resident-storage fact. For a decode-token
    // GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID path, those bytes are the only
    // source-grounded value we currently have for arbitrary GGUFs unless the
    // profile also carries per-tensor type/shape data and a llama.cpp matmul
    // kernel cost model.
    //
    // Do not apply validation-derived quantization multipliers here. Q8_0,
    // Q4_K, IQ, and MoE tensors do take different llama.cpp kernels, but the
    // right representation is an explicit matmul model derived from GGML tensor
    // block layout and backend kernel traits, not a storage-to-traffic scale
    // tuned until one local validation set looks good. Until that model exists,
    // decode prediction should remain conservative and the validator should
    // report misses honestly.
    let _ = model;
    storage_bytes
}

fn dense_matmul_traffic_bytes(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped = matmul
        .attention
        .kernel_traffic_bytes()
        .saturating_add(matmul.feed_forward.kernel_traffic_bytes())
        .saturating_add(matmul.output.kernel_traffic_bytes())
        .saturating_add(tied_output_projection_bytes(model));
    if grouped > 0 {
        Some(grouped)
    } else {
        let base = tensor_type_kernel_traffic_bytes(matmul.base_type_bytes);
        if base > 0 {
            Some(base)
        } else {
            (matmul.base_bytes > 0).then_some(matmul.base_bytes)
        }
    }
}

fn moe_matmul_traffic_bytes(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped_base = matmul
        .attention
        .kernel_traffic_bytes()
        .saturating_add(matmul.feed_forward.kernel_traffic_bytes())
        .saturating_add(matmul.output.kernel_traffic_bytes())
        .saturating_add(tied_output_projection_bytes(model));
    let grouped_expert = matmul.expert_feed_forward.kernel_traffic_bytes();
    if grouped_base > 0 || grouped_expert > 0 {
        return Some(grouped_base.saturating_add(active_expert_bytes(
            grouped_expert,
            model.expert_count,
            model.expert_used_count,
        )));
    }
    if matmul.base_bytes > 0 || matmul.expert_bytes > 0 {
        let base = tensor_type_kernel_traffic_bytes(matmul.base_type_bytes);
        let base = if base > 0 { base } else { matmul.base_bytes };
        let expert = tensor_type_kernel_traffic_bytes(matmul.expert_type_bytes);
        let expert = if expert > 0 {
            expert
        } else {
            matmul.expert_bytes
        };
        return Some(base.saturating_add(active_expert_bytes(
            expert,
            model.expert_count,
            model.expert_used_count,
        )));
    }
    None
}

trait MatmulKernelTraffic {
    fn kernel_traffic_bytes(&self) -> u64;
}

impl MatmulKernelTraffic for TensorMatmulGroupProfile {
    fn kernel_traffic_bytes(&self) -> u64 {
        let traffic = tensor_type_kernel_traffic_bytes(self.type_bytes);
        if traffic > 0 { traffic } else { self.bytes }
    }
}

fn tensor_type_kernel_traffic_bytes(bytes: TensorTypeBytes) -> u64 {
    // This is not a filename/model-family boost. It is the first source-derived
    // bridge between GGUF resident storage bytes and the decode loop that
    // llama.cpp actually executes.
    //
    // `scan_gguf_tensor_byte_profile()` tells us, per matmul tensor group, how
    // many bytes are stored as each GGML tensor type. llama.cpp then dispatches
    // those tensors into different GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID
    // kernels. The stored byte count is therefore a real memory-capacity fact,
    // but it is not always the best decode-time cost unit:
    //
    // - Q8_0 stores about 34 bytes per 32 weights (`block_q8_0`). Its Metal
    //   mat-vec kernel reads that block directly (`qs` plus `d`), and CUDA's
    //   MMVQ/MMQ path uses q8-specific `vec_dot_q8_0_q8_1` helpers. The source
    //   does not justify counting less than the resident Q8_0 block bytes as
    //   decode traffic, so Q8_0 is charged at its stored bytes.
    // - Q4_K stores far fewer bytes, but its `block_q4_K` super-block carries
    //   packed nibbles, scales/mins, and a q4x4 dequant path. The physical
    //   bytes are lower, while the useful bandwidth-equivalent work is not
    //   proportionally lower.
    //
    // The multipliers below produce "bandwidth-equivalent traffic": bytes of
    // resident matmul storage adjusted by GGML tensor-type kernel structure.
    // They deliberately apply to every GGUF with the same tensor type mix. They
    // are not backend labels, not architecture labels, and not model names.
    //
    // Source anchors in the pinned llama.cpp tree:
    // - ggml.c defines block size/type size for Q8_0 and K-quants.
    // - ggml-metal.metal instantiates separate matmul kernels for Q8_0 and
    //   Q4_K/Q5_K/Q6_K instead of a single generic bytes-only path.
    // - ggml-metal-ops.cpp dispatches Q8_0 with the f32/f16/bf16 row-grouping
    //   path for generic mat-vec, not with the K-quant grouping branch.
    // - ggml-cuda uses q8-specific MMVQ/MMQ vec-dot helpers and q8-specific
    //   MUL_MAT_ID templates.
    //
    // Earlier versions discounted Q8_0 storage bytes because Q8 has a simpler
    // block layout than K-quants. Broader validation falsified that as a
    // portable traffic model: Q8_0 was over-predicted on both Metal and CUDA.
    // The source-grounded rule is now simpler and stricter: count stored bytes
    // for Q4_K/Q5_K/Q6_K/Q8_0 and let measured hardware bandwidth, graph shape,
    // and compute floors decide the final rate.
    scaled_type_bytes(bytes.f32_bytes, 1.00)
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.f16_bytes,
            "f16",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.bf16_bytes,
            "bf16",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.q4_0_bytes,
            "q4_0",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.q4_k_bytes,
            "q4_k",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.q5_k_bytes,
            "q5_k",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.q6_k_bytes,
            "q6_k",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.q8_0_bytes,
            "q8_0",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.iq_bytes,
            "iq",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.other_quantized_bytes,
            "other_quantized",
        ))
        .saturating_add(tensor_type_kernel_traffic_bytes_for_kind(
            bytes.unknown_bytes,
            "unknown",
        ))
}

fn tensor_type_byte_entries(bytes: TensorTypeBytes) -> [(&'static str, u64); 11] {
    [
        ("f32", bytes.f32_bytes),
        ("f16", bytes.f16_bytes),
        ("bf16", bytes.bf16_bytes),
        ("q4_0", bytes.q4_0_bytes),
        ("q4_k", bytes.q4_k_bytes),
        ("q5_k", bytes.q5_k_bytes),
        ("q6_k", bytes.q6_k_bytes),
        ("q8_0", bytes.q8_0_bytes),
        ("iq", bytes.iq_bytes),
        ("other_quantized", bytes.other_quantized_bytes),
        ("unknown", bytes.unknown_bytes),
    ]
}

fn tensor_type_total_bytes(bytes: TensorTypeBytes) -> u64 {
    tensor_type_byte_entries(bytes)
        .into_iter()
        .map(|(_, bytes)| bytes)
        .sum()
}

fn dominant_tensor_type_for_bytes(bytes: TensorTypeBytes) -> Option<&'static str> {
    tensor_type_byte_entries(bytes)
        .into_iter()
        .max_by_key(|(_, bytes)| *bytes)
        .and_then(|(kind, bytes)| (bytes > 0).then_some(kind))
}

fn tensor_type_bytes_except(bytes: TensorTypeBytes, excluded: &str) -> TensorTypeBytes {
    let mut filtered = bytes;
    match excluded {
        "f32" => filtered.f32_bytes = 0,
        "f16" => filtered.f16_bytes = 0,
        "bf16" => filtered.bf16_bytes = 0,
        "q4_0" => filtered.q4_0_bytes = 0,
        "q4_k" => filtered.q4_k_bytes = 0,
        "q5_k" => filtered.q5_k_bytes = 0,
        "q6_k" => filtered.q6_k_bytes = 0,
        "q8_0" => filtered.q8_0_bytes = 0,
        "iq" => filtered.iq_bytes = 0,
        "other_quantized" => filtered.other_quantized_bytes = 0,
        "unknown" => filtered.unknown_bytes = 0,
        _ => {}
    }
    filtered
}

fn tensor_type_kernel_traffic_bytes_for_kind(bytes: u64, tensor_type: &str) -> u64 {
    match tensor_type {
        "q4_0" | "iq" => scaled_type_bytes(bytes, 1.18),
        "other_quantized" => scaled_type_bytes(bytes, 1.05),
        _ => bytes,
    }
}

fn scaled_type_bytes(bytes: u64, factor: f32) -> u64 {
    ((bytes as f64) * f64::from(factor)).round() as u64
}

fn active_decode_flops_per_token(model: &ModelProfile) -> Option<u64> {
    match model.architecture_class {
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown => {
            dense_matmul_flops(model)
        }
        ModelArchitectureClass::SparseMoeTransformer => moe_matmul_flops(model),
        _ => None,
    }
}

fn dense_matmul_flops(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped = matmul
        .attention
        .flops_per_token
        .saturating_add(matmul.feed_forward.flops_per_token)
        .saturating_add(matmul.output.flops_per_token)
        .saturating_add(tied_output_projection_flops(model));
    if grouped > 0 {
        Some(grouped)
    } else {
        (matmul.base_flops_per_token > 0).then_some(matmul.base_flops_per_token)
    }
}

fn moe_matmul_flops(model: &ModelProfile) -> Option<u64> {
    let matmul = &model.tensor_matmul;
    let grouped_base = matmul
        .attention
        .flops_per_token
        .saturating_add(matmul.feed_forward.flops_per_token)
        .saturating_add(matmul.output.flops_per_token)
        .saturating_add(tied_output_projection_flops(model));
    let grouped_expert = matmul.expert_feed_forward.flops_per_token;
    if grouped_base > 0 || grouped_expert > 0 {
        return Some(grouped_base.saturating_add(active_expert_bytes(
            grouped_expert,
            model.expert_count,
            model.expert_used_count,
        )));
    }
    if matmul.base_flops_per_token > 0 || matmul.expert_flops_per_token > 0 {
        return Some(
            matmul
                .base_flops_per_token
                .saturating_add(active_expert_bytes(
                    matmul.expert_flops_per_token,
                    model.expert_count,
                    model.expert_used_count,
                )),
        );
    }
    None
}

fn tensor_groups_available(groups: TensorGroupBytes) -> bool {
    groups.attention_bytes > 0
        || groups.feed_forward_bytes > 0
        || groups.expert_feed_forward_bytes > 0
        || groups.output_bytes > 0
}

fn active_moe_weight_bytes(model: &ModelProfile) -> u64 {
    let base = model.base_resident_bytes.unwrap_or(0);
    let expert = model.expert_tensor_bytes.unwrap_or(0);
    base.saturating_add(active_expert_bytes(
        expert,
        model.expert_count,
        model.expert_used_count,
    ))
}

fn active_expert_bytes(
    expert_bytes: u64,
    expert_count: Option<u32>,
    expert_used_count: Option<u32>,
) -> u64 {
    // llama.cpp treats MoE expert FFN tensors as routed matmuls, not as a dense
    // FFN that executes every expert on every token. The pinned source reads
    // GGUF `*.expert_count` and `*.expert_used_count` into
    // `hparams.n_expert` / `hparams.n_expert_used` in `llama-model.cpp`, then
    // registers `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, and fused
    // `ffn_gate_up_exps` as `GGML_OP_MUL_MAT_ID` in `llama-arch.cpp`.
    // `llama-model-loader.cpp` builds representative `mul_mat_id` tensors with
    // `n_expert_used`, and `llama-graph.h` passes both values into
    // `build_moe_ffn()`.
    //
    // This is why model-fit scales the expert pool by
    // `expert_used_count / expert_count` for per-token decode and prefill
    // compute/traffic. Resident memory still includes the full expert pool, so
    // memory fit and token-time cost intentionally use different accounting.
    // Do not replace this with a validation-fitted MoE multiplier. If the
    // validator shows a MoE miss, inspect the graph/dispatch overhead model or
    // missing GGUF metadata such as expert groups/shared experts before changing
    // this source-derived active expert rule.
    let Some(expert_count) = expert_count.filter(|count| *count > 0) else {
        return expert_bytes;
    };
    let active = expert_used_count.unwrap_or(expert_count).min(expert_count);
    expert_bytes.saturating_mul(u64::from(active)) / u64::from(expert_count)
}

fn activation_overhead_bytes(model: &ModelProfile) -> u64 {
    let layer_width = u64::from(model.layer_count.unwrap_or(1))
        .saturating_mul(u64::from(model.hidden_size.unwrap_or_default()));
    layer_width.saturating_mul(16)
}

fn decode_tokens_per_sec(
    active_decode_bytes: Option<u64>,
    active_decode_flops: Option<u64>,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    model: &ModelProfile,
) -> Option<f32> {
    // Decode estimate shape:
    //
    //   tok/s = 1000 / (active_bytes / effective_bandwidth + overhead_ms)
    //
    // `active_bytes / effective_bandwidth` gives the slope for models that are
    // mostly streaming weights and KV state. The overhead terms then account for
    // costs that do not scale linearly with model bytes: backend/runtime fixed
    // cost, MoE routing/dispatch, and small-model inefficiency. Keeping those
    // pieces separate is important for ranking: a 70B model should mostly be
    // limited by memory traffic, while a 135M model can be dominated by launch
    // and scheduling costs even though it barely touches memory.
    //
    // This function must not use filename/catalog reputation. It only consumes
    // metadata, measured hardware facts, and explicit config knobs so validation
    // can run against any GGUF and explain misses.
    let bytes = active_decode_bytes?;
    if bytes == 0 {
        return None;
    }
    let base_bandwidth = decode_base_bandwidth_bytes_per_sec(model, budget, config);
    let architecture_factor = match model.architecture_class {
        ModelArchitectureClass::Unknown => 0.75,
        ModelArchitectureClass::RecurrentOrStateSpace => 0.85,
        _ => 1.0,
    };
    let quantization_factor = quantization_efficiency_factor(model.quantization.as_deref());
    let shape_factor = decode_shape_bandwidth_factor(model, bytes);
    let effective_bandwidth =
        base_bandwidth as f32 * architecture_factor * quantization_factor * shape_factor;
    let grouped_cost = grouped_decode_cost(model, budget, config);
    let bandwidth_ms = grouped_cost
        .map(|cost| cost.bandwidth_ms)
        .unwrap_or_else(|| bytes as f32 / effective_bandwidth.max(1.0) * 1000.0);
    let compute_ms = decode_compute_ms(active_decode_flops, budget).unwrap_or(0.0);
    let overhead_ms = fixed_decode_overhead_ms(budget, config)
        + measured_decode_runtime_overhead_ms(budget)
        + measured_decode_graph_overhead_ms(model, budget)
        + architecture_decode_overhead_ms(model, budget, config)
        + sampled_decode_sampler_ms(model, budget);
    Some(1000.0 / (bandwidth_ms.max(compute_ms) + overhead_ms).max(0.001))
}

fn decode_cost_breakdown(
    active_decode_flops: Option<u64>,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    model: &ModelProfile,
) -> Option<DecodeCostBreakdown> {
    let grouped_cost = grouped_decode_cost(model, budget, config)?;
    let compute_ms = decode_compute_ms(active_decode_flops, budget).unwrap_or(0.0);
    let fixed_overhead_ms = fixed_decode_overhead_ms(budget, config);
    let runtime_overhead_ms = measured_decode_runtime_overhead_ms(budget);
    let measured_graph_overhead_ms = measured_decode_graph_overhead_ms(model, budget);
    let architecture_overhead_ms = architecture_decode_overhead_ms(model, budget, config);
    let sampled_decode_sampler_ms = sampled_decode_sampler_ms(model, budget);
    let selected_time_ms = grouped_cost.bandwidth_ms.max(compute_ms)
        + fixed_overhead_ms
        + runtime_overhead_ms
        + measured_graph_overhead_ms
        + architecture_overhead_ms
        + sampled_decode_sampler_ms;
    Some(DecodeCostBreakdown {
        bandwidth_ms: grouped_cost.bandwidth_ms,
        compute_ms,
        fixed_overhead_ms,
        runtime_overhead_ms,
        measured_graph_overhead_ms,
        architecture_overhead_ms,
        sampled_decode_sampler_ms,
        selected_time_ms,
        estimated_tokens_per_sec: 1000.0 / selected_time_ms.max(0.001),
        probed_bytes: grouped_cost.probed_bytes,
        fallback_bytes: grouped_cost.fallback_bytes,
        groups: grouped_cost.groups,
    })
}

fn grouped_decode_cost(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
) -> Option<GroupedDecodeCost> {
    // llama.cpp does not execute decode as one anonymous byte stream. The model
    // graph is built from source-visible GGML operations: attention projections,
    // FFN projections, optional `GGML_OP_MUL_MAT_ID` expert projections, output
    // projection, KV reads, and activation-side elementwise work. GGUF tensor
    // metadata gives us the same portable group boundaries through
    // `TensorMatmulProfile`.
    //
    // The estimator therefore sums group times instead of selecting one
    // "representative" probe for the entire model. Each group uses a measured
    // GGML decode probe only when the probe matches tensor type, op family, and
    // matrix shape closely enough. Missing groups fall back to the measured
    // decode-shaped bandwidth, keeping the miss visible in warnings and
    // confidence rather than inventing a backend constant.
    if !measured_gpu_budget(budget) || budget.decode_kernel_probes.is_empty() {
        return None;
    }
    let fallback_bandwidth = budget
        .decode_effective_bandwidth_bytes_per_sec
        .or(budget.memory_bandwidth_bytes_per_sec)?;
    if let Some(cost) = sparse_moe_block_decode_cost(model, budget, config, fallback_bandwidth) {
        return Some(cost);
    }
    if let Some(cost) =
        linear_attention_block_decode_cost(model, budget, config, fallback_bandwidth)
    {
        return Some(cost);
    }
    if let Some(cost) = dense_block_decode_cost(model, budget, config, fallback_bandwidth) {
        return Some(cost);
    }
    let mut cost = GroupedDecodeCost::default();
    for group in decode_traffic_groups(model) {
        add_group_decode_cost(group, model, budget, fallback_bandwidth, &mut cost);
    }
    add_non_weight_decode_cost(model, config, fallback_bandwidth, &mut cost);
    (cost.probed_bytes > 0 || cost.fallback_bytes > 0).then_some(cost)
}

fn sparse_moe_block_decode_cost(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    fallback_bandwidth: u64,
) -> Option<GroupedDecodeCost> {
    // Sparse MoE decode has two very different sources of work in one llama.cpp
    // layer: normal attention projections around the residual stream, and a
    // routed FFN path built with `GGML_OP_MUL_MAT_ID`, router softmax/top-k,
    // selected expert up/gate/down matmuls, activation, and expert reduction.
    //
    // Earlier model-fit revisions had evidence for the routed FFN subgraph but
    // composed the rest of the sparse layer from generic dense/matvec rows. That
    // was source-plausible but too optimistic for OLMoE validation because the
    // backend schedules the whole token graph, not an isolated expert FFN. A
    // `moe_block_graph` row is still metadata-only: its shape comes from GGUF
    // dimensions, and its operations come from llama.cpp source. It does not use
    // observed model tok/s, model names, or backend-specific constants.
    if model.architecture_class != ModelArchitectureClass::SparseMoeTransformer {
        return None;
    }
    let selection = select_sparse_moe_block_probe(model, budget)?;
    let (resident_bytes, traffic_bytes) =
        sparse_moe_block_bytes(model, selection.tensor_type, true)?;
    let all_block_traffic = sparse_moe_block_bytes(model, selection.tensor_type, false)
        .map(|(_, traffic)| traffic)
        .unwrap_or(traffic_bytes);
    let mut cost = GroupedDecodeCost::default();
    let probe_time =
        exact_sparse_moe_block_probe_time_ms(selection, model, budget, all_block_traffic);
    let block_graph_covers_residual_types = probe_time.is_some();
    let charged_traffic_bytes = if block_graph_covers_residual_types {
        all_block_traffic
    } else {
        traffic_bytes
    };
    let (bandwidth_ms, bandwidth_bytes_per_sec, source) = if let Some(probe_time) = probe_time {
        (
            probe_time.variable_ms,
            probe_time.effective_bandwidth_bytes_per_sec,
            probe_time.source,
        )
    } else {
        let bandwidth = decode_probe_bandwidth_bytes_per_sec(selection.probe, budget);
        (
            charged_traffic_bytes as f32 / bandwidth.max(1) as f32 * 1000.0,
            bandwidth,
            "probe_sparse_block",
        )
    };
    cost.probed_bytes = cost.probed_bytes.saturating_add(charged_traffic_bytes);
    cost.bandwidth_ms += bandwidth_ms;
    cost.groups.push(DecodeCostGroupBreakdown {
        group: "sparse_transformer_block".into(),
        tensor_type: selection.tensor_type.into(),
        resident_bytes,
        traffic_bytes: charged_traffic_bytes,
        expert_scaled: true,
        shape_input_width: u64::from(selection.probe.cols),
        shape_output_width: u64::from(selection.probe.rows),
        source: source.into(),
        bandwidth_bytes_per_sec,
        bandwidth_ms,
        probe_name: Some(selection.probe.name.clone()),
        probe_rows: Some(selection.probe.rows),
        probe_cols: Some(selection.probe.cols),
        probe_batch_tokens: Some(selection.probe.batch_tokens),
        probe_effective_gbps: Some(selection.probe.effective_gbps),
        probe_shape_distance: Some(selection.shape_distance),
    });

    if !block_graph_covers_residual_types {
        for group in sparse_moe_block_residual_groups(model, selection.tensor_type) {
            add_group_decode_cost(group, model, budget, fallback_bandwidth, &mut cost);
        }
    }
    add_group_decode_cost(
        DecodeTrafficGroup {
            kind: DecodeGroupKind::OutputMatmul,
            type_bytes: model.tensor_matmul.output.type_bytes,
            shape: model.tensor_matmul.output.shape,
            expert_scaled: false,
        },
        model,
        budget,
        fallback_bandwidth,
        &mut cost,
    );
    if let Some(group) = tied_output_projection_group(model) {
        add_group_decode_cost(group, model, budget, fallback_bandwidth, &mut cost);
    }
    add_non_weight_decode_cost(model, config, fallback_bandwidth, &mut cost);
    (cost.probed_bytes > 0 || cost.fallback_bytes > 0).then_some(cost)
}

fn select_sparse_moe_block_probe<'a>(
    model: &ModelProfile,
    budget: &'a ExecutionBudget,
) -> Option<DenseBlockProbeSelection<'a>> {
    let block_type_bytes = add_type_bytes(
        model.tensor_matmul.attention.type_bytes,
        model.tensor_matmul.expert_feed_forward.type_bytes,
    );
    let tensor_type = dominant_tensor_type_for_bytes(block_type_bytes)?;
    let target = moe_layer_graph_probe_target(model, tensor_type)?;
    budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && probe.tensor_type.eq_ignore_ascii_case(tensor_type)
                && is_composite_moe_block_graph_decode_probe(probe)
                && moe_block_graph_probe_kv_matches_model(probe, model)
        })
        .map(|probe| (probe, decode_probe_shape_distance(probe, target)))
        .filter(|(_, shape_distance)| {
            *shape_distance <= MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE
        })
        .min_by(
            |(left_probe, left_distance), (right_probe, right_distance)| {
                moe_block_graph_probe_rank(left_probe, model)
                    .cmp(&moe_block_graph_probe_rank(right_probe, model))
                    .then_with(|| left_distance.total_cmp(right_distance))
            },
        )
        .map(|(probe, shape_distance)| DenseBlockProbeSelection {
            probe,
            tensor_type,
            shape_distance,
            bandwidth_bytes_per_sec: decode_probe_bandwidth_bytes_per_sec(probe, budget),
        })
}

fn sparse_moe_block_bytes(
    model: &ModelProfile,
    tensor_type: &'static str,
    selected_type_only: bool,
) -> Option<(u64, u64)> {
    let attention = model.tensor_matmul.attention.type_bytes;
    let experts = model.tensor_matmul.expert_feed_forward.type_bytes;
    let attention_resident = if selected_type_only {
        tensor_type_bytes_for_kind(attention, tensor_type)
    } else {
        tensor_type_total_bytes(attention)
    };
    let expert_resident = if selected_type_only {
        tensor_type_bytes_for_kind(experts, tensor_type)
    } else {
        tensor_type_total_bytes(experts)
    };
    let attention_traffic = if selected_type_only {
        tensor_type_kernel_traffic_bytes_for_kind(attention_resident, tensor_type)
    } else {
        tensor_type_kernel_traffic_bytes(attention)
    };
    let expert_traffic = if selected_type_only {
        tensor_type_kernel_traffic_bytes_for_kind(expert_resident, tensor_type)
    } else {
        tensor_type_kernel_traffic_bytes(experts)
    };
    let active_expert_traffic =
        active_expert_bytes(expert_traffic, model.expert_count, model.expert_used_count);
    let resident_bytes = attention_resident.saturating_add(active_expert_bytes(
        expert_resident,
        model.expert_count,
        model.expert_used_count,
    ));
    let traffic_bytes = attention_traffic.saturating_add(active_expert_traffic);
    (traffic_bytes > 0).then_some((resident_bytes, traffic_bytes))
}

fn sparse_moe_block_residual_groups(
    model: &ModelProfile,
    block_tensor_type: &'static str,
) -> Vec<DecodeTrafficGroup> {
    [
        (
            DecodeGroupKind::AttentionMatmul,
            model.tensor_matmul.attention.clone(),
            false,
        ),
        (
            DecodeGroupKind::RoutedExpert,
            model.tensor_matmul.expert_feed_forward.clone(),
            true,
        ),
    ]
    .into_iter()
    .filter_map(|(kind, group, expert_scaled)| {
        let type_bytes = tensor_type_bytes_except(group.type_bytes, block_tensor_type);
        (tensor_type_kernel_traffic_bytes(type_bytes) > 0).then_some(DecodeTrafficGroup {
            kind,
            type_bytes,
            shape: group.shape,
            expert_scaled,
        })
    })
    .collect()
}

fn exact_sparse_moe_block_probe_time_ms(
    selection: DenseBlockProbeSelection<'_>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    traffic_bytes: u64,
) -> Option<DenseBlockProbeTime> {
    // Same principle as the dense block elapsed path, but for sparse blocks:
    // when a model-shaped `moe_block_graph` row exactly matches expert width,
    // hidden width, GQA width, and tensor type, the measured graph elapsed time
    // is a better source-grounded unit than converting the row to a generic
    // bandwidth and re-composing attention plus experts separately.
    if selection.shape_distance.abs() > f64::EPSILON {
        return None;
    }
    let elapsed_ms = selection
        .probe
        .elapsed_ms
        .filter(|elapsed| *elapsed > 0.0)?;
    let model_layers = f64::from(model.layer_count.filter(|layers| *layers > 0)?);
    let probe_layers = f64::from(moe_block_graph_probe_layers(selection.probe).max(1));
    let fixed_ms = budget
        .decode_fixed_overhead_ms
        .filter(|fixed| *fixed > 0.0)
        .map(f64::from)
        .unwrap_or(0.0);
    let variable_probe_ms = (elapsed_ms - fixed_ms).max(0.0);
    let depth_variable_ms =
        moe_block_graph_depth_extrapolated_elapsed_ms(selection, model, budget, fixed_ms);
    let (variable_ms, source) = depth_variable_ms
        .map(|variable_ms| (variable_ms, "probe_sparse_block_depth_elapsed"))
        .unwrap_or_else(|| {
            (
                variable_probe_ms * (model_layers / probe_layers),
                "probe_sparse_block_elapsed",
            )
        });
    if variable_ms <= 0.0 || traffic_bytes == 0 {
        return None;
    }
    let effective_bandwidth_bytes_per_sec =
        ((traffic_bytes as f64) / (variable_ms / 1000.0)).round() as u64;
    Some(DenseBlockProbeTime {
        variable_ms: variable_ms as f32,
        effective_bandwidth_bytes_per_sec,
        source,
    })
}

fn moe_block_graph_depth_extrapolated_elapsed_ms(
    selection: DenseBlockProbeSelection<'_>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    fixed_ms: f64,
) -> Option<f64> {
    let target_layers = f64::from(model.layer_count.filter(|layers| *layers > 0)?);
    let selected_layers = f64::from(moe_block_graph_probe_layers(selection.probe).max(1));
    if target_layers <= selected_layers {
        return None;
    }
    let mut points = budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe
                    .tensor_type
                    .eq_ignore_ascii_case(selection.tensor_type)
                && is_composite_moe_block_graph_decode_probe(probe)
                && moe_block_graph_probe_kv_matches_model(probe, model)
                && probe.rows == selection.probe.rows
                && probe.cols == selection.probe.cols
        })
        .filter_map(|probe| {
            let elapsed_ms = probe.elapsed_ms.filter(|elapsed| *elapsed > 0.0)?;
            let layers = f64::from(moe_block_graph_probe_layers(probe).max(1));
            let variable_ms = (elapsed_ms - fixed_ms).max(0.0);
            (variable_ms > 0.0).then_some((layers, variable_ms))
        })
        .collect::<Vec<_>>();
    extrapolate_elapsed_points(&mut points, target_layers)
}

fn dense_block_decode_cost(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    fallback_bandwidth: u64,
) -> Option<GroupedDecodeCost> {
    // Dense decode is the place where "bytes per token / measured bandwidth" is
    // most tempting and most wrong. In llama.cpp, a generated token is not a
    // sequence of isolated matvec microbenchmarks. `llama_decode()` builds a
    // full GGML graph for the token, the backend scheduler sees the graph as a
    // unit, and backends such as Metal/CUDA can pick graph-level kernels,
    // command-buffer layouts, fusion opportunities, and residency behavior that
    // never appear in a single-op probe.
    //
    // GGUF tensor metadata still gives us portable accounting: attention and
    // FFN tensor bytes tell us how much transformer-block resident weight is
    // touched, and hidden/FFN/KV dimensions tell us which source-shaped probe is
    // representative. The block path therefore charges the dominant tensor type
    // with a composite llama decode graph probe, not with a backend-specific
    // multiplier. Mixed residual tensor types are charged separately so Q6/F32
    // leftovers stay visible instead of being smuggled into the Q4 block.
    //
    // This rule is deliberately limited to dense transformers. Sparse MoE uses
    // `GGML_OP_MUL_MAT_ID` and routing/top-k graph work; recurrent and
    // state-space models need their own source-shaped probes before they can
    // borrow this block treatment.
    if has_recurrent_attention_graph(model)
        || !matches!(
            model.architecture_class,
            ModelArchitectureClass::DenseTransformer
        )
    {
        return None;
    }
    let block = dense_transformer_block_group(model)?;
    let selection = select_dense_block_probe(block, model, budget)?;
    let mut cost = GroupedDecodeCost::default();
    let block_graph_covers_residual_types =
        add_dense_block_cost(block, selection, model, budget, &mut cost);

    if !block_graph_covers_residual_types {
        for group in dense_block_residual_groups(model, selection.tensor_type) {
            add_group_decode_cost(group, model, budget, fallback_bandwidth, &mut cost);
        }
    }
    add_group_decode_cost(
        DecodeTrafficGroup {
            kind: DecodeGroupKind::OutputMatmul,
            type_bytes: model.tensor_matmul.output.type_bytes,
            shape: model.tensor_matmul.output.shape,
            expert_scaled: false,
        },
        model,
        budget,
        fallback_bandwidth,
        &mut cost,
    );
    if let Some(group) = tied_output_projection_group(model) {
        add_group_decode_cost(group, model, budget, fallback_bandwidth, &mut cost);
    }
    add_non_weight_decode_cost(model, config, fallback_bandwidth, &mut cost);
    (cost.probed_bytes > 0 || cost.fallback_bytes > 0).then_some(cost)
}

fn linear_attention_block_decode_cost(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    fallback_bandwidth: u64,
) -> Option<GroupedDecodeCost> {
    // llama.cpp recurrent/linear attention blocks are not well represented by
    // the normal dense llama graph probe. The projection inventory looks
    // transformer-like at the byte level, but the submitted GGML graph has
    // different source-visible matmul roles (`attn_qkv`, `attn_gate`, beta,
    // alpha, recurrent state work, and `ssm_out`) before the FFN tail.
    //
    // This path is selected only from GGUF tensor-role evidence. It does not
    // ask for a family name, filename, catalog entry, or observed model tok/s.
    // If the benchmark profile lacks a matching linear-attention graph probe,
    // the estimator falls back to per-group measured bandwidth rather than
    // borrowing the dense transformer block shortcut.
    if !has_recurrent_attention_graph(model) {
        return None;
    }
    let block = dense_transformer_block_group(model)?;
    let selection = select_linear_attention_block_probe(block, model, budget)?;
    let mut cost = GroupedDecodeCost::default();
    add_linear_attention_block_cost(block, selection, budget, &mut cost);
    add_group_decode_cost(
        DecodeTrafficGroup {
            kind: DecodeGroupKind::OutputMatmul,
            type_bytes: model.tensor_matmul.output.type_bytes,
            shape: model.tensor_matmul.output.shape,
            expert_scaled: false,
        },
        model,
        budget,
        fallback_bandwidth,
        &mut cost,
    );
    if let Some(group) = tied_output_projection_group(model) {
        add_group_decode_cost(group, model, budget, fallback_bandwidth, &mut cost);
    }
    add_non_weight_decode_cost(model, config, fallback_bandwidth, &mut cost);
    (cost.probed_bytes > 0 || cost.fallback_bytes > 0).then_some(cost)
}

fn select_linear_attention_block_probe<'a>(
    group: DecodeTrafficGroup,
    model: &ModelProfile,
    budget: &'a ExecutionBudget,
) -> Option<DenseBlockProbeSelection<'a>> {
    let tensor_type = dominant_tensor_type_for_bytes(group.type_bytes)?;
    let target = linear_attention_graph_probe_target(model, tensor_type)?;
    budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && probe.tensor_type.eq_ignore_ascii_case(tensor_type)
                && is_composite_linear_attention_graph_decode_probe(probe)
                && linear_attention_graph_probe_features_match_model(probe, model)
                && linear_attention_graph_probe_shape_matches_model(probe, model)
        })
        .map(|probe| (probe, decode_probe_shape_distance(probe, target)))
        .filter(|(_, shape_distance)| {
            *shape_distance <= MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE
        })
        .min_by(
            |(left_probe, left_distance), (right_probe, right_distance)| {
                linear_attention_graph_probe_rank(left_probe, model)
                    .cmp(&linear_attention_graph_probe_rank(right_probe, model))
                    .then_with(|| left_distance.total_cmp(right_distance))
            },
        )
        .map(|(probe, shape_distance)| DenseBlockProbeSelection {
            probe,
            tensor_type,
            shape_distance,
            bandwidth_bytes_per_sec: decode_probe_bandwidth_bytes_per_sec(probe, budget),
        })
}

fn add_linear_attention_block_cost(
    group: DecodeTrafficGroup,
    selection: DenseBlockProbeSelection<'_>,
    budget: &ExecutionBudget,
    cost: &mut GroupedDecodeCost,
) {
    let traffic_bytes = tensor_type_kernel_traffic_bytes(group.type_bytes);
    if traffic_bytes == 0 {
        return;
    }
    let elapsed_ms = selection.probe.elapsed_ms.unwrap_or(0.0);
    let fixed_ms = budget
        .decode_fixed_overhead_ms
        .filter(|fixed| *fixed > 0.0)
        .unwrap_or_default();
    let bandwidth_ms = (elapsed_ms as f32 - fixed_ms).max(0.0);
    let bandwidth_bytes_per_sec = if bandwidth_ms > 0.0 {
        ((traffic_bytes as f32) / (bandwidth_ms / 1000.0)).round() as u64
    } else {
        selection.bandwidth_bytes_per_sec
    };
    let bandwidth_ms = if bandwidth_ms > 0.0 {
        bandwidth_ms
    } else {
        traffic_bytes as f32 / selection.bandwidth_bytes_per_sec.max(1) as f32 * 1000.0
    };
    cost.probed_bytes = cost.probed_bytes.saturating_add(traffic_bytes);
    cost.bandwidth_ms += bandwidth_ms;
    cost.groups.push(DecodeCostGroupBreakdown {
        group: "linear_attention_block".into(),
        tensor_type: selection.tensor_type.into(),
        resident_bytes: tensor_type_total_bytes(group.type_bytes),
        traffic_bytes,
        expert_scaled: false,
        shape_input_width: group.shape.weighted_avg_input_width,
        shape_output_width: group.shape.weighted_avg_output_width,
        source: "probe_linear_attention_block_elapsed".into(),
        bandwidth_bytes_per_sec,
        bandwidth_ms,
        probe_name: Some(selection.probe.name.clone()),
        probe_rows: Some(selection.probe.rows),
        probe_cols: Some(selection.probe.cols),
        probe_batch_tokens: Some(selection.probe.batch_tokens),
        probe_effective_gbps: Some(selection.probe.effective_gbps),
        probe_shape_distance: Some(selection.shape_distance),
    });
}

fn dense_transformer_block_group(model: &ModelProfile) -> Option<DecodeTrafficGroup> {
    let type_bytes = add_type_bytes(
        model.tensor_matmul.attention.type_bytes,
        model.tensor_matmul.feed_forward.type_bytes,
    );
    let shape = dense_transformer_block_shape(model)?;
    Some(DecodeTrafficGroup {
        kind: DecodeGroupKind::TransformerBlock,
        type_bytes,
        shape,
        expert_scaled: false,
    })
}

fn dense_transformer_block_shape(model: &ModelProfile) -> Option<MatmulShapeProfile> {
    let target = dense_layer_graph_probe_target(model, dominant_decode_tensor_type(model)?)?;
    Some(MatmulShapeProfile {
        tensor_count: model
            .tensor_matmul
            .attention
            .shape
            .tensor_count
            .saturating_add(model.tensor_matmul.feed_forward.shape.tensor_count),
        logical_matrix_count: model
            .tensor_matmul
            .attention
            .shape
            .logical_matrix_count
            .saturating_add(model.tensor_matmul.feed_forward.shape.logical_matrix_count),
        total_elements: model
            .tensor_matmul
            .attention
            .shape
            .total_elements
            .saturating_add(model.tensor_matmul.feed_forward.shape.total_elements),
        min_input_width: u64::from(target.cols),
        max_input_width: u64::from(target.cols),
        min_output_width: u64::from(target.rows),
        max_output_width: u64::from(target.rows),
        weighted_avg_input_width: u64::from(target.cols),
        weighted_avg_output_width: u64::from(target.rows),
    })
}

fn select_dense_block_probe<'a>(
    group: DecodeTrafficGroup,
    model: &ModelProfile,
    budget: &'a ExecutionBudget,
) -> Option<DenseBlockProbeSelection<'a>> {
    let tensor_type = dominant_tensor_type_for_bytes(group.type_bytes)?;
    let target = dense_layer_graph_probe_target(model, tensor_type)?;
    budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && probe.tensor_type.eq_ignore_ascii_case(tensor_type)
                && is_composite_llama_graph_decode_probe(probe)
                && is_supported_dense_layer_graph_probe(probe)
                && dense_layer_graph_probe_features_match_model(probe, model)
                && dense_layer_graph_probe_depth_matches_model(probe, model)
                && dense_graph_probe_kv_matches_model(probe, model)
        })
        .map(|probe| {
            let shape_distance = decode_probe_shape_distance(probe, target);
            (
                probe,
                shape_distance,
                dense_block_graph_bandwidth_bytes_per_sec(probe, model, budget),
            )
        })
        .filter(|(_, shape_distance, _)| {
            *shape_distance <= MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE
        })
        .min_by(
            |(left_probe, left_distance, _), (right_probe, right_distance, _)| {
                dense_layer_graph_probe_rank(left_probe, model)
                    .cmp(&dense_layer_graph_probe_rank(right_probe, model))
                    .then_with(|| left_distance.total_cmp(right_distance))
            },
        )
        .map(
            |(probe, shape_distance, bandwidth_bytes_per_sec)| DenseBlockProbeSelection {
                probe,
                tensor_type,
                shape_distance,
                bandwidth_bytes_per_sec,
            },
        )
}

fn add_dense_block_cost(
    group: DecodeTrafficGroup,
    selection: DenseBlockProbeSelection<'_>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    cost: &mut GroupedDecodeCost,
) -> bool {
    let selected_resident_bytes =
        tensor_type_bytes_for_kind(group.type_bytes, selection.tensor_type);
    let selected_traffic_bytes =
        tensor_type_kernel_traffic_bytes_for_kind(selected_resident_bytes, selection.tensor_type);
    if selected_traffic_bytes == 0 {
        return false;
    }
    let all_resident_bytes = tensor_type_total_bytes(group.type_bytes);
    let all_traffic_bytes = tensor_type_kernel_traffic_bytes(group.type_bytes);
    let probe_time = exact_dense_block_probe_time_ms(selection, model, budget, all_traffic_bytes);
    let block_graph_covers_residual_types = probe_time.is_some();
    let resident_bytes = if block_graph_covers_residual_types {
        all_resident_bytes
    } else {
        selected_resident_bytes
    };
    let traffic_bytes = if block_graph_covers_residual_types {
        tensor_type_kernel_traffic_bytes(group.type_bytes)
    } else {
        selected_traffic_bytes
    };
    cost.probed_bytes = cost.probed_bytes.saturating_add(traffic_bytes);
    let (bandwidth_ms, bandwidth_bytes_per_sec, source) = if let Some(probe_time) = probe_time {
        (
            probe_time.variable_ms,
            probe_time.effective_bandwidth_bytes_per_sec,
            probe_time.source,
        )
    } else {
        (
            traffic_bytes as f32 / selection.bandwidth_bytes_per_sec.max(1) as f32 * 1000.0,
            selection.bandwidth_bytes_per_sec,
            "probe_block",
        )
    };
    cost.bandwidth_ms += bandwidth_ms;
    cost.groups.push(DecodeCostGroupBreakdown {
        group: "transformer_block".into(),
        tensor_type: selection.tensor_type.into(),
        resident_bytes,
        traffic_bytes,
        expert_scaled: false,
        shape_input_width: group.shape.weighted_avg_input_width,
        shape_output_width: group.shape.weighted_avg_output_width,
        source: source.into(),
        bandwidth_bytes_per_sec,
        bandwidth_ms,
        probe_name: Some(selection.probe.name.clone()),
        probe_rows: Some(selection.probe.rows),
        probe_cols: Some(selection.probe.cols),
        probe_batch_tokens: Some(selection.probe.batch_tokens),
        probe_effective_gbps: Some(selection.probe.effective_gbps),
        probe_shape_distance: Some(selection.shape_distance),
    });
    block_graph_covers_residual_types
}

fn exact_dense_block_probe_time_ms(
    selection: DenseBlockProbeSelection<'_>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    traffic_bytes: u64,
) -> Option<DenseBlockProbeTime> {
    // A source-shaped dense graph probe is different from an isolated matmul
    // probe. The elapsed time includes graph-level work that llama.cpp actually
    // submits during decode: Q/K/V projections, RoPE-adjacent scheduling
    // boundaries, attention/output projections, SWIGLU gate/up/down, residual
    // adds, and backend scheduler behavior for the whole block. If the probe
    // exactly matches the GGUF-derived block shape and GQA width, reusing its
    // elapsed time is more faithful than converting it to an "effective GB/s"
    // and then pretending the whole graph is just a single resident-byte pass.
    //
    // Keep this intentionally narrow. Approximate shape matches still go
    // through bytes/bandwidth interpolation so we do not smuggle a graph-time
    // constant from one model shape into another.
    //
    // When the hardware profile contains several same-shape graph depths
    // (`l1`, `l4`, `l8`, ...), prefer that measured depth curve over naive
    // `elapsed * model_layers / probe_layers` scaling. This is not a
    // backend-specific correction: it follows directly from llama.cpp's decode
    // graph behavior. `llama_context::process_ubatch()` submits a whole token
    // graph whose command encoding, graph optimization, residency tracking,
    // elementwise tails, and backend scheduling costs do not necessarily scale
    // one-for-one with layer count. CUDA can be nearly linear for the measured
    // shapes, while Metal can amortize the l4->l8 graph heavily. Both are valid
    // hardware facts, and the estimator should consume the measured curve
    // rather than assuming either backend's shape.
    //
    // The fixed empty-submit cost is removed from measured points before fitting
    // the curve and added back once per generated token by
    // `fixed_decode_overhead_ms()`, matching the separation used in
    // `decode_probe_bandwidth_bytes_per_sec()`. Any remaining intercept in the
    // depth curve is real graph work from the source-shaped probe, so we keep it
    // as part of the extrapolated variable term instead of multiplying it by
    // every layer.
    if selection.shape_distance.abs() > f64::EPSILON {
        return None;
    }
    let elapsed_ms = selection
        .probe
        .elapsed_ms
        .filter(|elapsed| *elapsed > 0.0)?;
    let model_layers = f64::from(model.layer_count.filter(|layers| *layers > 0)?);
    let probe_layers = f64::from(dense_layer_graph_probe_layers(selection.probe).max(1));
    let fixed_ms = budget
        .decode_fixed_overhead_ms
        .filter(|fixed| *fixed > 0.0)
        .map(f64::from)
        .unwrap_or(0.0);
    let variable_probe_ms = (elapsed_ms - fixed_ms).max(0.0);
    let depth_variable_ms =
        dense_graph_depth_extrapolated_elapsed_ms(selection, model, budget, fixed_ms);
    let (variable_ms, source) = depth_variable_ms
        .map(|variable_ms| (variable_ms, "probe_block_depth_elapsed"))
        .unwrap_or_else(|| {
            (
                variable_probe_ms * (model_layers / probe_layers),
                "probe_block_elapsed",
            )
        });
    if variable_ms <= 0.0 || traffic_bytes == 0 {
        return None;
    }
    let effective_bandwidth_bytes_per_sec =
        ((traffic_bytes as f64) / (variable_ms / 1000.0)).round() as u64;
    Some(DenseBlockProbeTime {
        variable_ms: variable_ms as f32,
        effective_bandwidth_bytes_per_sec,
        source,
    })
}

fn dense_graph_depth_extrapolated_elapsed_ms(
    selection: DenseBlockProbeSelection<'_>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    fixed_ms: f64,
) -> Option<f64> {
    let target_layers = f64::from(model.layer_count.filter(|layers| *layers > 0)?);
    let selected_layers = f64::from(dense_layer_graph_probe_layers(selection.probe).max(1));
    if target_layers <= selected_layers {
        return None;
    }
    let mut points = budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe
                    .tensor_type
                    .eq_ignore_ascii_case(selection.tensor_type)
                && is_composite_llama_graph_decode_probe(probe)
                && is_supported_dense_layer_graph_probe(probe)
                && dense_layer_graph_probe_features_match_model(probe, model)
                && dense_layer_graph_probe_depth_matches_model(probe, model)
                && dense_graph_probe_kv_matches_model(probe, model)
                && probe.rows == selection.probe.rows
                && probe.cols == selection.probe.cols
        })
        .filter_map(|probe| {
            let elapsed_ms = probe.elapsed_ms.filter(|elapsed| *elapsed > 0.0)?;
            let layers = f64::from(dense_layer_graph_probe_layers(probe).max(1));
            let variable_ms = (elapsed_ms - fixed_ms).max(0.0);
            (variable_ms > 0.0).then_some((layers, variable_ms))
        })
        .collect::<Vec<_>>();
    extrapolate_elapsed_points(&mut points, target_layers)
}

fn extrapolate_elapsed_points(points: &mut Vec<(f64, f64)>, target_layers: f64) -> Option<f64> {
    if !target_layers.is_finite() || points.len() < 2 {
        return None;
    }
    points.sort_by(|left, right| left.0.total_cmp(&right.0));
    points.dedup_by(|left, right| {
        if (left.0 - right.0).abs() <= f64::EPSILON {
            left.1 = left.1.min(right.1);
            true
        } else {
            false
        }
    });
    if points.len() < 2 {
        return None;
    }
    if let Some(window) = points
        .windows(2)
        .find(|pair| target_layers >= pair[0].0 && target_layers <= pair[1].0)
    {
        return interpolate_elapsed_segment(window[0], window[1], target_layers);
    }
    let first = *points.first()?;
    let last = *points.last()?;
    if target_layers <= last.0 {
        return None;
    }
    // Extrapolation beyond the deepest measured graph point is deliberately
    // more conservative than interpolation inside the measured range. Dense
    // llama.cpp decode graphs can hit backend thresholds where one local
    // segment, for example l8->l16, amortizes graph/scheduler work much more
    // aggressively than the rest of the curve. Letting only that final segment
    // define l28/l32 behavior turns a threshold artifact into a throughput
    // forecast. Using the first-to-deepest measured envelope still consumes the
    // deeper probe evidence, but it asks the whole measured curve to justify
    // the extrapolated slope.
    interpolate_elapsed_segment(first, last, target_layers)
}

fn interpolate_elapsed_segment(
    left: (f64, f64),
    right: (f64, f64),
    target_layers: f64,
) -> Option<f64> {
    let layer_delta = right.0 - left.0;
    if layer_delta <= 0.0 {
        return None;
    }
    let slope_ms_per_layer = (right.1 - left.1) / layer_delta;
    if !slope_ms_per_layer.is_finite() || slope_ms_per_layer <= 0.0 {
        return None;
    }
    Some(left.1 + slope_ms_per_layer * (target_layers - left.0))
}

fn dense_block_residual_groups(
    model: &ModelProfile,
    block_tensor_type: &'static str,
) -> Vec<DecodeTrafficGroup> {
    [
        (
            DecodeGroupKind::AttentionMatmul,
            model.tensor_matmul.attention.clone(),
        ),
        (
            DecodeGroupKind::FeedForwardMatmul,
            model.tensor_matmul.feed_forward.clone(),
        ),
    ]
    .into_iter()
    .filter_map(|(kind, group)| {
        let type_bytes = tensor_type_bytes_except(group.type_bytes, block_tensor_type);
        (tensor_type_kernel_traffic_bytes(type_bytes) > 0).then_some(DecodeTrafficGroup {
            kind,
            type_bytes,
            shape: group.shape,
            expert_scaled: false,
        })
    })
    .collect()
}

fn dense_block_graph_bandwidth_bytes_per_sec(
    selected_probe: &DecodeKernelProbe,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> u64 {
    // The selected composite probe may be close to the model but not identical:
    // a benchmark profile might include 2560x9728 and 4096x12288 graph rows,
    // while the GGUF in hand is 3584x18944; it might include l4/l8 repeated
    // block rows while the model has 28 or 36 layers. We can use those rows
    // only as measured hardware evidence, never as a fitted correction from the
    // validation target.
    //
    // Shape interpolation is done in log(rows * cols), because matrix dimensions
    // affect both kernel occupancy and memory movement over orders of magnitude.
    // Depth extrapolation is done in log(layer count), using repeated
    // source-shaped graph probes when present. That mirrors the way llama.cpp
    // submits a whole repeated-layer token graph while remaining falsifiable:
    // if another model shape or backend breaks the curve, the validation table
    // will show the miss without us hiding it behind a name/backend constant.
    //
    // Finally, keep the result bounded by measured hardware facts. A graph probe
    // can report an effective byte rate above raw memory bandwidth because our
    // byte accounting is an approximation of source traffic, not a hardware bus
    // counter. We allow the selected probe's own measured rate as a floor for
    // consistency, but do not let interpolation invent unbounded throughput.
    let base = decode_probe_bandwidth_bytes_per_sec(selected_probe, budget);
    let shape_adjusted =
        dense_graph_shape_interpolated_bandwidth(selected_probe, model, budget).unwrap_or(base);
    let depth_adjusted =
        dense_graph_depth_extrapolated_bandwidth(selected_probe, model, budget, shape_adjusted);
    let measured_ceiling = budget
        .memory_bandwidth_bytes_per_sec
        .or(budget.decode_effective_bandwidth_bytes_per_sec)
        .unwrap_or(depth_adjusted);
    depth_adjusted.min(measured_ceiling.max(base))
}

fn dense_graph_shape_interpolated_bandwidth(
    selected_probe: &DecodeKernelProbe,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<u64> {
    let (target_rows, target_cols) = dense_layer_graph_shape_target(model)?;
    let target_x = dense_graph_shape_log_size(target_rows, target_cols);
    let selected_layers = dense_layer_graph_probe_layers(selected_probe);
    let mut points = budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && probe
                    .tensor_type
                    .eq_ignore_ascii_case(&selected_probe.tensor_type)
                && is_composite_llama_graph_decode_probe(probe)
                && is_supported_dense_layer_graph_probe(probe)
                && dense_layer_graph_probe_features_match_model(probe, model)
                && dense_layer_graph_probe_depth_matches_model(probe, model)
                && dense_layer_graph_probe_layers(probe) == selected_layers
                && dense_graph_probe_kv_matches_model(probe, model)
        })
        .map(|probe| {
            (
                dense_graph_shape_log_size(probe.rows, probe.cols),
                decode_probe_bandwidth_bytes_per_sec(probe, budget) as f64,
            )
        })
        .filter(|(_, bandwidth)| bandwidth.is_finite() && *bandwidth > 0.0)
        .collect::<Vec<_>>();
    interpolate_bandwidth_points(&mut points, target_x)
}

fn dense_graph_depth_extrapolated_bandwidth(
    selected_probe: &DecodeKernelProbe,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    shape_adjusted_bandwidth: u64,
) -> u64 {
    let Some(model_layers) = model.layer_count.filter(|layers| *layers > 0) else {
        return shape_adjusted_bandwidth;
    };
    let selected_layers = dense_layer_graph_probe_layers(selected_probe);
    if model_layers <= selected_layers {
        return shape_adjusted_bandwidth;
    }
    let mut points = budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && probe
                    .tensor_type
                    .eq_ignore_ascii_case(&selected_probe.tensor_type)
                && is_composite_llama_graph_decode_probe(probe)
                && is_supported_dense_layer_graph_probe(probe)
                && dense_layer_graph_probe_features_match_model(probe, model)
                && dense_layer_graph_probe_depth_matches_model(probe, model)
                && dense_graph_probe_kv_matches_model(probe, model)
                && probe.rows == selected_probe.rows
                && probe.cols == selected_probe.cols
        })
        .map(|probe| {
            (
                f64::from(dense_layer_graph_probe_layers(probe)).ln(),
                decode_probe_bandwidth_bytes_per_sec(probe, budget) as f64,
            )
        })
        .filter(|(_, bandwidth)| bandwidth.is_finite() && *bandwidth > 0.0)
        .collect::<Vec<_>>();
    let Some(depth_bandwidth) =
        interpolate_bandwidth_points(&mut points, f64::from(model_layers).ln())
    else {
        return shape_adjusted_bandwidth;
    };
    let selected_bandwidth = decode_probe_bandwidth_bytes_per_sec(selected_probe, budget);
    if selected_bandwidth == 0 {
        return shape_adjusted_bandwidth;
    }
    let ratio = depth_bandwidth as f64 / selected_bandwidth as f64;
    ((shape_adjusted_bandwidth as f64) * ratio).round().max(1.0) as u64
}

fn interpolate_bandwidth_points(points: &mut Vec<(f64, f64)>, target_x: f64) -> Option<u64> {
    if !target_x.is_finite() || points.is_empty() {
        return None;
    }
    points.sort_by(|(left_x, _), (right_x, _)| left_x.total_cmp(right_x));
    points.dedup_by(|(left_x, _), (right_x, _)| (*left_x - *right_x).abs() < f64::EPSILON);
    if points.len() == 1 {
        return Some(points[0].1.round() as u64);
    }
    let pair = interpolation_pair(points, target_x)?;
    let ((left_x, left_y), (right_x, right_y)) = pair;
    if (right_x - left_x).abs() < f64::EPSILON {
        return Some(left_y.round() as u64);
    }
    let slope = (right_y - left_y) / (right_x - left_x);
    let interpolated = left_y + slope * (target_x - left_x);
    interpolated
        .is_finite()
        .then(|| interpolated.max(1.0).round() as u64)
}

fn interpolation_pair(points: &[(f64, f64)], target_x: f64) -> Option<((f64, f64), (f64, f64))> {
    for window in points.windows(2) {
        let left = window[0];
        let right = window[1];
        if target_x >= left.0 && target_x <= right.0 {
            return Some((left, right));
        }
    }
    if target_x < points.first()?.0 {
        return Some((points[0], points[1]));
    }
    let len = points.len();
    Some((points[len - 2], points[len - 1]))
}

fn dense_graph_shape_log_size(rows: u32, cols: u32) -> f64 {
    let rows = f64::from(rows.max(1));
    let cols = f64::from(cols.max(1));
    (rows * cols).ln()
}

fn dense_graph_probe_kv_matches_model(probe: &DecodeKernelProbe, model: &ModelProfile) -> bool {
    let hidden = u64::from(model.hidden_size.unwrap_or_default());
    if hidden == 0 {
        return true;
    }
    let model_kv = model_attention_kv_width(model);
    let probe_kv = dense_layer_graph_probe_kv_width(probe).unwrap_or(u64::from(probe.cols));
    if model_kv < hidden {
        probe_kv == model_kv
    } else {
        probe_kv == u64::from(probe.cols)
    }
}

fn decode_traffic_groups(model: &ModelProfile) -> Vec<DecodeTrafficGroup> {
    let matmul = &model.tensor_matmul;
    let mut groups = vec![
        DecodeTrafficGroup {
            kind: DecodeGroupKind::AttentionMatmul,
            type_bytes: matmul.attention.type_bytes,
            shape: matmul.attention.shape,
            expert_scaled: false,
        },
        DecodeTrafficGroup {
            kind: DecodeGroupKind::FeedForwardMatmul,
            type_bytes: matmul.feed_forward.type_bytes,
            shape: matmul.feed_forward.shape,
            expert_scaled: false,
        },
        DecodeTrafficGroup {
            kind: DecodeGroupKind::OutputMatmul,
            type_bytes: matmul.output.type_bytes,
            shape: matmul.output.shape,
            expert_scaled: false,
        },
        DecodeTrafficGroup {
            kind: DecodeGroupKind::RoutedExpert,
            type_bytes: matmul.expert_feed_forward.type_bytes,
            shape: matmul.expert_feed_forward.shape,
            expert_scaled: true,
        },
    ];
    if let Some(group) = tied_output_projection_group(model) {
        groups.push(group);
    }
    groups
}

fn tied_output_projection_group(model: &ModelProfile) -> Option<DecodeTrafficGroup> {
    let bytes = tied_output_projection_bytes(model);
    (bytes > 0).then_some(DecodeTrafficGroup {
        kind: DecodeGroupKind::OutputMatmul,
        type_bytes: tied_output_projection_type_bytes(model, bytes),
        shape: tied_output_projection_shape(model),
        expert_scaled: false,
    })
}

fn tied_output_projection_bytes(model: &ModelProfile) -> u64 {
    // Llama.cpp explicitly ties the output projection to token embeddings when
    // GGUF has no standalone `output.weight` tensor:
    //
    //   if (output == NULL) {
    //       output = create_tensor(... TOKEN_EMBD ..., TENSOR_DUPLICATED);
    //   }
    //
    // The GGUF scanner reports that as embedding resident storage, not as an
    // output matmul group. Decode still multiplies the final hidden state by
    // that matrix to produce logits, so treating tied-output embeddings as free
    // makes small Llama-style models look too fast. Charge the embedding bytes
    // as an output projection only when the explicit output group is absent.
    // This is source behavior plus GGUF tensor grouping, not a model-name or
    // backend-specific correction.
    if model.tensor_matmul.output.bytes > 0 || model.tensor_group_bytes.output_bytes > 0 {
        return 0;
    }
    match model.architecture_class {
        ModelArchitectureClass::DenseTransformer
        | ModelArchitectureClass::SparseMoeTransformer
        | ModelArchitectureClass::Unknown => model.tensor_group_bytes.embedding_bytes,
        _ => 0,
    }
}

fn tied_output_projection_type_bytes(model: &ModelProfile, bytes: u64) -> TensorTypeBytes {
    // Tied output projections reuse the token embedding tensor as the logits
    // matmul source in llama.cpp. That means the projection has the same GGUF
    // tensor type distribution as the embedding tensor, even though GGUF does
    // not report an `output.weight` group. Preserve that type information when
    // the scanner saw a complete embedding type profile so benchmark-derived
    // decode probes can match the real matmul class. If older serialized
    // profiles lack the field, or the byte totals drift, keep the conservative
    // `unknown` bucket rather than inventing a quantization.
    let embedding_types = model.tensor_group_bytes.embedding_type_bytes;
    if tensor_type_total_bytes(embedding_types) == bytes {
        return embedding_types;
    }

    TensorTypeBytes {
        unknown_bytes: bytes,
        ..TensorTypeBytes::default()
    }
}

fn tied_output_projection_shape(model: &ModelProfile) -> MatmulShapeProfile {
    let hidden = u64::from(model.hidden_size.unwrap_or_default());
    let vocab = u64::from(model.tokenizer.vocab_size.unwrap_or_default());
    let total_elements = hidden.saturating_mul(vocab);
    MatmulShapeProfile {
        tensor_count: 1,
        logical_matrix_count: 1,
        total_elements,
        min_input_width: hidden,
        max_input_width: hidden,
        min_output_width: vocab,
        max_output_width: vocab,
        weighted_avg_input_width: hidden,
        weighted_avg_output_width: vocab,
    }
}

fn tied_output_projection_flops(model: &ModelProfile) -> u64 {
    let bytes = tied_output_projection_bytes(model);
    if bytes == 0 {
        return 0;
    }
    let hidden = u64::from(model.hidden_size.unwrap_or_default());
    let vocab = u64::from(model.tokenizer.vocab_size.unwrap_or_default());
    hidden.saturating_mul(vocab).saturating_mul(2)
}

fn add_group_decode_cost(
    group: DecodeTrafficGroup,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    fallback_bandwidth: u64,
    cost: &mut GroupedDecodeCost,
) {
    for (tensor_type, resident_bytes) in tensor_type_byte_entries(group.type_bytes) {
        let mut traffic_bytes =
            tensor_type_kernel_traffic_bytes_for_kind(resident_bytes, tensor_type);
        if group.expert_scaled {
            traffic_bytes =
                active_expert_bytes(traffic_bytes, model.expert_count, model.expert_used_count);
        }
        if traffic_bytes == 0 {
            continue;
        }
        let selection = select_decode_group_probe(group, tensor_type, model, budget);
        let bandwidth = selection
            .as_ref()
            .map(|selection| {
                decode_group_probe_bandwidth_bytes_per_sec(
                    selection.probe,
                    group.kind,
                    model,
                    budget,
                )
            })
            .unwrap_or(fallback_bandwidth);
        if selection.is_some() {
            cost.probed_bytes = cost.probed_bytes.saturating_add(traffic_bytes);
        } else {
            cost.fallback_bytes = cost.fallback_bytes.saturating_add(traffic_bytes);
        }
        let bandwidth_ms = traffic_bytes as f32 / bandwidth.max(1) as f32 * 1000.0;
        cost.bandwidth_ms += bandwidth_ms;
        cost.groups.push(decode_cost_group_breakdown(
            group,
            tensor_type,
            resident_bytes,
            traffic_bytes,
            bandwidth,
            bandwidth_ms,
            selection,
        ));
    }
}

fn add_non_weight_decode_cost(
    model: &ModelProfile,
    config: &SelectionConfig,
    fallback_bandwidth: u64,
    cost: &mut GroupedDecodeCost,
) {
    let context = config
        .workload
        .interaction
        .expected_prompt_tokens
        .unwrap_or_else(|| target_context_tokens(model, config) / 2);
    let kv_read_bytes = kv_cache_bytes_for_context(model, config, context)
        .saturating_mul((config.kv_read_scale * 1000.0).round() as u64)
        / 1000;
    let non_weight_bytes = kv_read_bytes.saturating_add(activation_overhead_bytes(model));
    if non_weight_bytes == 0 {
        return;
    }
    cost.fallback_bytes = cost.fallback_bytes.saturating_add(non_weight_bytes);
    let bandwidth_ms = non_weight_bytes as f32 / fallback_bandwidth.max(1) as f32 * 1000.0;
    cost.bandwidth_ms += bandwidth_ms;
    cost.groups.push(DecodeCostGroupBreakdown {
        group: "kv_and_activation".into(),
        tensor_type: "runtime".into(),
        resident_bytes: non_weight_bytes,
        traffic_bytes: non_weight_bytes,
        expert_scaled: false,
        shape_input_width: u64::from(model.hidden_size.unwrap_or_default()),
        shape_output_width: u64::from(model.hidden_size.unwrap_or_default()),
        source: "fallback".into(),
        bandwidth_bytes_per_sec: fallback_bandwidth,
        bandwidth_ms,
        probe_name: None,
        probe_rows: None,
        probe_cols: None,
        probe_batch_tokens: None,
        probe_effective_gbps: None,
        probe_shape_distance: None,
    });
}

fn decode_cost_group_breakdown(
    group: DecodeTrafficGroup,
    tensor_type: &'static str,
    resident_bytes: u64,
    traffic_bytes: u64,
    bandwidth: u64,
    bandwidth_ms: f32,
    selection: Option<DecodeProbeSelection<'_>>,
) -> DecodeCostGroupBreakdown {
    DecodeCostGroupBreakdown {
        group: decode_group_kind_name(group.kind).into(),
        tensor_type: tensor_type.into(),
        resident_bytes,
        traffic_bytes,
        expert_scaled: group.expert_scaled,
        shape_input_width: group.shape.weighted_avg_input_width,
        shape_output_width: group.shape.weighted_avg_output_width,
        source: if selection.is_some() {
            "probe".into()
        } else {
            "fallback".into()
        },
        bandwidth_bytes_per_sec: bandwidth,
        bandwidth_ms,
        probe_name: selection.map(|selection| selection.probe.name.clone()),
        probe_rows: selection.map(|selection| selection.probe.rows),
        probe_cols: selection.map(|selection| selection.probe.cols),
        probe_batch_tokens: selection.map(|selection| selection.probe.batch_tokens),
        probe_effective_gbps: selection.map(|selection| selection.probe.effective_gbps),
        probe_shape_distance: selection.map(|selection| selection.shape_distance),
    }
}

fn select_decode_group_probe<'a>(
    group: DecodeTrafficGroup,
    tensor_type: &'static str,
    model: &ModelProfile,
    budget: &'a ExecutionBudget,
) -> Option<DecodeProbeSelection<'a>> {
    let target = matmul_shape_target(group.shape)?;
    budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && is_llama_decode_kernel_probe(probe)
                && probe.tensor_type.eq_ignore_ascii_case(tensor_type)
                && decode_probe_matches_group_kind(probe, group.kind)
                && (!has_recurrent_attention_graph(model)
                    || !is_composite_llama_graph_decode_probe(probe))
                && (!is_composite_llama_graph_decode_probe(probe)
                    || (dense_layer_graph_probe_features_match_model(probe, model)
                        && dense_layer_graph_probe_depth_matches_model(probe, model)))
        })
        .map(|probe| {
            (
                probe,
                decode_group_probe_shape_distance(probe, group.kind, model, tensor_type, target),
            )
        })
        .filter(|(probe, distance)| decode_group_probe_is_usable(probe, group.kind, *distance))
        .min_by(|(left, left_distance), (right, right_distance)| {
            decode_group_probe_rank(left, group.kind, model)
                .cmp(&decode_group_probe_rank(right, group.kind, model))
                .then_with(|| left_distance.total_cmp(right_distance))
        })
        .map(|(probe, shape_distance)| DecodeProbeSelection {
            probe,
            shape_distance,
        })
}

fn decode_group_kind_name(kind: DecodeGroupKind) -> &'static str {
    match kind {
        DecodeGroupKind::TransformerBlock => "transformer_block",
        DecodeGroupKind::AttentionMatmul => "attention_matmul",
        DecodeGroupKind::FeedForwardMatmul => "feed_forward_matmul",
        DecodeGroupKind::OutputMatmul => "output_matmul",
        DecodeGroupKind::RoutedExpert => "routed_expert",
    }
}

fn decode_probe_bandwidth_bytes_per_sec(
    probe: &DecodeKernelProbe,
    budget: &ExecutionBudget,
) -> u64 {
    // `mesh-llm gpus benchmark` runs GGML probes through the same scheduler
    // boundary llama.cpp uses for decode:
    //
    //   ggml_backend_sched_graph_compute_async(...)
    //   ggml_backend_sched_synchronize(...)
    //
    // The probe row reports `effective_gbps = measured_probe_bytes /
    // elapsed_ms`. That is a useful whole-probe fact, but using it directly as
    // a reusable bandwidth for every GGUF tensor group smears the fixed graph
    // submission/synchronization time into the bandwidth slope. The grouped
    // decode estimator then adds `fixed_decode_overhead_ms()` once per token,
    // which means fixed submission cost is charged twice: once inside each
    // selected probe bandwidth and once explicitly.
    //
    // Keep the two source-visible terms separate:
    //
    // - probe elapsed minus measured fixed overhead -> reusable kernel/graph
    //   bandwidth for bytes that scale with model metadata;
    // - measured fixed overhead -> one per generated token, handled by
    //   `fixed_decode_overhead_ms()`.
    //
    // This is not calibrated from model validation results. It only uses facts
    // already emitted by the hardware benchmark. If the benchmark did not
    // measure fixed overhead, or if a very small probe is shorter than that
    // overhead, we leave the row unchanged rather than inventing a cap. We also
    // leave it unchanged when the inferred variable portion is no larger than
    // the measured fixed portion. In that regime the benchmark has not
    // separated "bytes that scale with model metadata" from "one scheduler
    // submission" with enough signal to reuse the adjusted slope for arbitrary
    // GGUFs. Using the adjusted value would divide by a tiny residual and turn
    // a small probe into an accidental optimistic constant. Larger source-shaped
    // probes still get the fixed-cost separation because their elapsed time has
    // enough variable work to identify a reusable bandwidth term.
    let reported = (probe.effective_gbps * 1_000_000_000.0).round() as u64;
    let Some(elapsed_ms) = probe.elapsed_ms.filter(|elapsed| *elapsed > 0.0) else {
        return reported;
    };
    let Some(fixed_ms) = budget
        .decode_fixed_overhead_ms
        .filter(|fixed| *fixed > 0.0)
        .map(f64::from)
    else {
        return reported;
    };
    if fixed_ms >= elapsed_ms {
        return reported;
    }

    let measured_bytes = probe.effective_gbps * 1_000_000_000.0 * (elapsed_ms / 1000.0);
    let variable_seconds = (elapsed_ms - fixed_ms) / 1000.0;
    if variable_seconds <= fixed_ms / 1000.0 {
        return reported;
    }
    if measured_bytes <= 0.0 || variable_seconds <= 0.0 {
        return reported;
    }
    (measured_bytes / variable_seconds).round() as u64
}

fn decode_group_probe_bandwidth_bytes_per_sec(
    probe: &DecodeKernelProbe,
    kind: DecodeGroupKind,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> u64 {
    if kind == DecodeGroupKind::RoutedExpert && is_composite_moe_graph_decode_probe(probe) {
        return moe_graph_bandwidth_bytes_per_sec(probe, model, budget);
    }
    decode_probe_bandwidth_bytes_per_sec(probe, budget)
}

fn moe_graph_bandwidth_bytes_per_sec(
    selected_probe: &DecodeKernelProbe,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> u64 {
    let base = decode_probe_bandwidth_bytes_per_sec(selected_probe, budget);
    let depth_adjusted =
        moe_graph_depth_extrapolated_bandwidth(selected_probe, model, budget, base);
    let measured_ceiling = budget
        .memory_bandwidth_bytes_per_sec
        .or(budget.decode_effective_bandwidth_bytes_per_sec)
        .unwrap_or(depth_adjusted);
    depth_adjusted.min(measured_ceiling.max(base))
}

fn moe_graph_depth_extrapolated_bandwidth(
    selected_probe: &DecodeKernelProbe,
    model: &ModelProfile,
    budget: &ExecutionBudget,
    selected_bandwidth: u64,
) -> u64 {
    let Some(model_layers) = model.layer_count.filter(|layers| *layers > 0) else {
        return selected_bandwidth;
    };
    let selected_layers = moe_graph_probe_layers(selected_probe);
    if model_layers <= selected_layers {
        return selected_bandwidth;
    }
    let mut points = budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && probe
                    .tensor_type
                    .eq_ignore_ascii_case(&selected_probe.tensor_type)
                && is_composite_moe_graph_decode_probe(probe)
                && probe.rows == selected_probe.rows
                && probe.cols == selected_probe.cols
        })
        .map(|probe| {
            (
                f64::from(moe_graph_probe_layers(probe)).ln(),
                decode_probe_bandwidth_bytes_per_sec(probe, budget) as f64,
            )
        })
        .filter(|(_, bandwidth)| bandwidth.is_finite() && *bandwidth > 0.0)
        .collect::<Vec<_>>();
    let Some(depth_bandwidth) =
        interpolate_bandwidth_points(&mut points, f64::from(model_layers).ln())
    else {
        return selected_bandwidth;
    };
    let selected_bandwidth = decode_probe_bandwidth_bytes_per_sec(selected_probe, budget);
    if selected_bandwidth == 0 {
        return selected_bandwidth;
    }
    let ratio = depth_bandwidth as f64 / selected_bandwidth as f64;
    ((selected_bandwidth as f64) * ratio).round().max(1.0) as u64
}

fn decode_group_probe_is_usable(
    probe: &DecodeKernelProbe,
    kind: DecodeGroupKind,
    distance: f64,
) -> bool {
    if distance <= MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE {
        return true;
    }

    // Earlier validation tried to let FFN traffic use isolated FFN matvec
    // probes when the full llama graph shape was not representative. That was
    // source-plausible, because llama.cpp does build FFN projections from
    // GGML_OP_MUL_MAT nodes around `GGML_SWIGLU_SPLIT`, but it was falsified by
    // Metal/CUDA sweeps: isolated matvec probes can run much faster than the
    // same matrices inside the scheduled full-token graph. The estimator is for
    // llama.cpp decode, not for an individual GGML microkernel, so attention and
    // FFN transformer-block bytes stay tied to composite llama graph probes.
    //
    // Do not apply a non-representative matvec escape hatch to output projection
    // either. The
    // attention path includes q/k/v/o projection shape plus KV-cache work and is
    // better represented by a composite llama graph row. Output projection can
    // be vocab-sized and very different from FFN matrices, so using an unrelated
    // FFN probe would be a shape mismatch disguised as precision.
    //
    // We do not apply this escape hatch to routed MoE traffic. llama.cpp sparse
    // expert decode uses indexed expert matmul (`GGML_OP_MUL_MAT_ID`) and graph
    // scheduling behavior that a dense matvec does not represent.
    let _ = (probe, kind);
    false
}

fn decode_group_probe_shape_distance(
    probe: &DecodeKernelProbe,
    kind: DecodeGroupKind,
    model: &ModelProfile,
    tensor_type: &'static str,
    group_target: (u32, u32),
) -> f64 {
    // The composite `*_llama_graph_*` probes are not one GGML_OP_MUL_MAT row.
    // They are source-shaped decode graphs: q/k/v/o attention projections plus
    // gate/up/down FFN projections for one token. llama.cpp builds the full
    // token graph, calls `set_inputs()`, and submits it through
    // `ggml_backend_sched_graph_compute_async()`; it does not benchmark every
    // transformer projection as a separate isolated operation. This matters on
    // backends that schedule, reorder, or fuse work over the graph. The stored
    // `rows`/`cols` are therefore FFN and hidden dimensions of the graph, not an
    // individual attention matrix shape.
    //
    // Deep validation may emit `*_llama_graph_lN_*` probes that repeat
    // source-shaped decoder blocks in one scheduled graph. That is useful
    // source-grounded evidence for Q4 and Q8 dense models because llama.cpp
    // submits a whole repeated-layer decode graph, not one isolated matvec. We
    // still require the probe to match tensor type, composite graph shape, GQA
    // KV width, and model layer depth before it can affect scoring; old or
    // unrelated diagnostic rows should remain diagnostics, not hidden
    // corrections.
    //
    // For attention and FFN groups we compare composite probes to the layer
    // graph shape implied by GGUF metadata. The probe is a scheduled llama.cpp
    // token graph, not a single isolated matrix, so this deliberately uses the
    // transformer-block shape for both halves of the block. For output
    // projection and plain matvec probes we keep the stricter per-group matrix
    // shape because output logits are not part of the repeated transformer block
    // graph and can be vocabulary-sized.
    let target =
        if kind == DecodeGroupKind::RoutedExpert && is_composite_moe_graph_decode_probe(probe) {
            moe_layer_graph_probe_target(model, tensor_type).unwrap_or(DecodeProbeTarget {
                tensor_type,
                rows: group_target.0,
                cols: group_target.1,
            })
        } else if matches!(
            kind,
            DecodeGroupKind::AttentionMatmul | DecodeGroupKind::FeedForwardMatmul
        ) && is_composite_llama_graph_decode_probe(probe)
        {
            dense_layer_graph_probe_target(model, tensor_type).unwrap_or(DecodeProbeTarget {
                tensor_type,
                rows: group_target.0,
                cols: group_target.1,
            })
        } else {
            DecodeProbeTarget {
                tensor_type,
                rows: group_target.0,
                cols: group_target.1,
            }
        };
    decode_probe_shape_distance(probe, target)
}

fn moe_layer_graph_probe_target(
    model: &ModelProfile,
    tensor_type: &'static str,
) -> Option<DecodeProbeTarget> {
    // A composite MoE probe represents llama.cpp's routed FFN subgraph rather
    // than an average expert matrix. Its stored rows/cols are therefore the
    // GGUF expert intermediate width and residual hidden width used to build
    // `GGML_OP_MUL_MAT_ID` up/gate/down nodes. Matching against the aggregate
    // expert tensor summary can pick an accidental weighted-average shape, so
    // prefer the architecture fields that llama.cpp itself consumes when it
    // creates the graph.
    let hidden = model
        .hidden_size
        .map(u64::from)
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            let shape = model.tensor_matmul.expert_feed_forward.shape;
            (shape.max_input_width > 0).then_some(shape.max_input_width)
        })?;
    let expert_width = model
        .ffn_size
        .map(u64::from)
        .filter(|ffn| *ffn > 0)
        .or_else(|| {
            let shape = model.tensor_matmul.expert_feed_forward.shape;
            (shape.min_input_width > 0).then_some(shape.min_input_width)
        })?;
    Some(DecodeProbeTarget {
        tensor_type,
        rows: u32::try_from(expert_width).ok()?,
        cols: u32::try_from(hidden).ok()?,
    })
}

fn dense_layer_graph_probe_target(
    model: &ModelProfile,
    tensor_type: &'static str,
) -> Option<DecodeProbeTarget> {
    let (rows, cols) = dense_layer_graph_shape_target(model)?;
    Some(DecodeProbeTarget {
        tensor_type,
        rows,
        cols,
    })
}

fn linear_attention_graph_probe_target(
    model: &ModelProfile,
    tensor_type: &'static str,
) -> Option<DecodeProbeTarget> {
    let (rows, cols) = dense_layer_graph_shape_target(model)?;
    Some(DecodeProbeTarget {
        tensor_type,
        rows,
        cols,
    })
}

fn dense_layer_graph_shape_target(model: &ModelProfile) -> Option<(u32, u32)> {
    let hidden = model
        .hidden_size
        .map(u64::from)
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            let shape = model.tensor_matmul.attention.shape;
            (shape.max_input_width > 0).then_some(shape.max_input_width)
        })
        .or_else(|| {
            let shape = model.tensor_matmul.feed_forward.shape;
            (shape.max_input_width > 0).then_some(shape.max_input_width)
        })?;
    let ffn = model
        .ffn_size
        .map(u64::from)
        .filter(|ffn| *ffn > 0)
        .or_else(|| {
            let shape = model.tensor_matmul.feed_forward.shape;
            (shape.weighted_avg_output_width > 0).then_some(shape.weighted_avg_output_width)
        })
        .or_else(|| {
            let shape = model.tensor_matmul.feed_forward.shape;
            (shape.max_output_width > 0).then_some(shape.max_output_width)
        })?;
    Some((u32::try_from(ffn).ok()?, u32::try_from(hidden).ok()?))
}

fn decode_probe_matches_group_kind(probe: &DecodeKernelProbe, kind: DecodeGroupKind) -> bool {
    match kind {
        DecodeGroupKind::TransformerBlock => false,
        DecodeGroupKind::AttentionMatmul => {
            is_composite_llama_graph_decode_probe(probe)
                && !is_routed_moe_decode_probe(probe)
                && is_supported_dense_layer_graph_probe(probe)
        }
        DecodeGroupKind::FeedForwardMatmul => {
            is_composite_llama_graph_decode_probe(probe)
                && !is_routed_moe_decode_probe(probe)
                && is_supported_dense_layer_graph_probe(probe)
        }
        DecodeGroupKind::OutputMatmul => {
            !is_routed_moe_decode_probe(probe)
                && !is_composite_llama_graph_decode_probe(probe)
                && !is_composite_moe_block_graph_decode_probe(probe)
                && !is_synthetic_expert_matvec_probe(probe)
        }
        DecodeGroupKind::RoutedExpert => is_routed_moe_decode_probe(probe),
    }
}

fn decode_group_probe_rank(
    probe: &DecodeKernelProbe,
    kind: DecodeGroupKind,
    model: &ModelProfile,
) -> u16 {
    match kind {
        DecodeGroupKind::TransformerBlock if is_composite_llama_graph_decode_probe(probe) => {
            dense_layer_graph_probe_rank(probe, model)
        }
        DecodeGroupKind::AttentionMatmul if is_composite_llama_graph_decode_probe(probe) => {
            dense_layer_graph_probe_rank(probe, model)
        }
        DecodeGroupKind::FeedForwardMatmul if is_composite_llama_graph_decode_probe(probe) => {
            dense_layer_graph_probe_rank(probe, model)
        }
        DecodeGroupKind::OutputMatmul if is_matvec_decode_probe(probe) => 1,
        DecodeGroupKind::RoutedExpert if is_composite_moe_graph_decode_probe(probe) => {
            moe_graph_probe_rank(probe, model)
        }
        DecodeGroupKind::RoutedExpert if is_routed_moe_decode_probe(probe) => {
            // A plain `GGML_OP_MUL_MAT_ID` row is still useful sparse-expert
            // evidence, but it is only the inner expert matmul. The
            // model-shaped MoE graph rows include the surrounding gate,
            // softmax/top-k, indexed up/gate/down expert matmuls, activation,
            // and graph-scheduler boundary that llama.cpp executes for routed
            // FFN decode. Prefer those composite rows whenever present; keep
            // isolated MUL_MAT_ID as a fallback for older benchmark JSON or
            // backends where the composite graph could not be measured.
            1_000
        }
        _ => 5,
    }
}

fn moe_block_graph_probe_rank(probe: &DecodeKernelProbe, model: &ModelProfile) -> u16 {
    let hidden = u64::from(model.hidden_size.unwrap_or_default());
    let model_kv = model_attention_kv_width(model);
    let probe_kv = moe_block_graph_probe_kv_width(probe).unwrap_or(hidden);
    let kv_rank: u16 = if hidden > 0 && model_kv < hidden {
        if probe_kv == model_kv { 0 } else { 1 }
    } else if probe_kv == hidden {
        0
    } else {
        1
    };
    let probe_layers = moe_block_graph_probe_layers(probe);
    let layer_rank = match model.layer_count.filter(|layers| *layers > 0) {
        Some(model_layers) => {
            if probe_layers <= model_layers && probe_layers > 1 {
                u16::try_from(model_layers.saturating_sub(probe_layers)).unwrap_or(u16::MAX)
            } else if probe_layers == 1 {
                900
            } else {
                950
            }
        }
        None => {
            if probe_layers == 1 {
                0
            } else {
                900
            }
        }
    };
    kv_rank * 1000 + layer_rank
}

fn moe_graph_probe_rank(probe: &DecodeKernelProbe, model: &ModelProfile) -> u16 {
    let probe_layers = moe_graph_probe_layers(probe);
    match model.layer_count.filter(|layers| *layers > 0) {
        Some(model_layers) => {
            if probe_layers <= model_layers && probe_layers > 1 {
                u16::try_from(model_layers.saturating_sub(probe_layers)).unwrap_or(u16::MAX)
            } else if probe_layers == 1 {
                900
            } else {
                950
            }
        }
        None => {
            if probe_layers == 1 {
                0
            } else {
                900
            }
        }
    }
}

fn dense_layer_graph_probe_rank(probe: &DecodeKernelProbe, model: &ModelProfile) -> u16 {
    // GQA/MQA changes the actual llama.cpp decode graph: K/V projection tensors
    // are hidden x kv_width, not hidden x hidden. A full-width synthetic graph
    // is still useful fallback evidence, but when the benchmark provides a GQA
    // graph whose KV width matches GGUF metadata it is the better source-shaped
    // probe. This is a metadata shape match, not a backend or model-name rule.
    let hidden = u64::from(model.hidden_size.unwrap_or_default());
    let model_kv = model_attention_kv_width(model);
    let probe_kv = dense_layer_graph_probe_kv_width(probe).unwrap_or(hidden);
    let kv_rank: u16 = if hidden > 0 && model_kv < hidden {
        if probe_kv == model_kv { 0 } else { 1 }
    } else if probe_kv == hidden {
        0
    } else {
        1
    };

    // llama.cpp builds decode as one graph containing the repeated transformer
    // blocks and submits that graph through `ggml_backend_sched_graph_compute_async`.
    // A one-layer synthetic graph is source-shaped at the operation level, but
    // it still under-represents the scheduler/allocator/kernel-launch behavior
    // of real decode where dozens of layer blocks are present in the same graph.
    //
    // `mesh-llm gpus benchmark --probe-depth deep` and `model-fit-validate`
    // can therefore emit stacked graph probes such as `*_llama_graph_l4_*`,
    // `*_llama_graph_l8_*`, or a model-shaped full-depth `lN` row. This is not
    // a backend-specific multiplier: it is a measured GGML graph row whose only
    // extra dimension is how many repeated source-shaped layer blocks it
    // contains. When GGUF exposes a layer count, prefer the deepest measured
    // stack that does not exceed the model's layer count, because that is the
    // closest graph unit to what llama.cpp actually executes. If the layer
    // count is missing, keep the one-layer graph first; guessing a stack depth
    // would make missing metadata look more certain than it is.
    let layer_rank = match model.layer_count.filter(|layers| *layers > 0) {
        Some(model_layers) => {
            let probe_layers = dense_layer_graph_probe_layers(probe);
            if probe_layers <= model_layers && probe_layers > 1 {
                u16::try_from(model_layers.saturating_sub(probe_layers)).unwrap_or(u16::MAX)
            } else if probe_layers == 1 {
                900
            } else {
                950
            }
        }
        None => {
            if dense_layer_graph_probe_layers(probe) == 1 {
                0
            } else {
                900
            }
        }
    };
    kv_rank * 1000 + layer_rank
}

fn decode_compute_ms(active_decode_flops: Option<u64>, budget: &ExecutionBudget) -> Option<f32> {
    let flops = active_decode_flops? as f32;
    let tflops = budget.compute_tflops_fp16.filter(|value| *value > 0.0)?;
    Some(flops / (tflops * 1_000_000_000_000.0) * 1000.0)
}

fn decode_shape_bandwidth_factor(model: &ModelProfile, active_decode_bytes: u64) -> f32 {
    // Wide dense models can achieve slightly better effective streaming
    // bandwidth than tiny/narrow models because there is enough per-token work
    // to keep the backend occupied. This is intentionally a small factor and
    // only applies when both the hidden width and active byte footprint look
    // like a normal medium/large dense transformer. It should improve the slope
    // for 8B-ish dense models without letting architecture labels dominate the
    // selector.
    if !matches!(
        model.architecture_class,
        ModelArchitectureClass::DenseTransformer
    ) {
        return 1.0;
    }
    let Some(hidden) = model.hidden_size else {
        return 1.0;
    };
    let active_gib = active_decode_bytes as f32 / GIB as f32;
    let width = hidden as f32;
    if active_gib < 4.0 || width < 3072.0 {
        return 1.0;
    }
    let width_term = ((width - 3072.0) / 1024.0).clamp(0.0, 1.0);
    let active_term = ((active_gib - 4.0) / 1.0).clamp(0.0, 1.0);
    let ffn_term = ffn_expansion_occupancy_term(model);
    1.0 + 0.12 * width_term.max(active_term * 0.70) + ffn_term
}

fn ffn_expansion_occupancy_term(model: &ModelProfile) -> f32 {
    // Large FFN matrices mean the decode graph spends more of its time in a few
    // large feed-forward matmuls rather than in many smaller surrounding
    // operations. That tends to improve occupancy for active-byte-heavy dense
    // models because each FFN kernel has more useful multiply-add work to
    // amortize dispatch, dequant, and memory scheduling.
    //
    // Prefer the GGUF tensor shape summary over metadata widths. GGUF stores
    // matmul tensors in GGML `ne[]` order, so model-artifact records the
    // element-weighted input/output widths for the actual FFN matrices that
    // llama.cpp will feed to GGML_OP_MUL_MAT. That handles different-sized
    // matmuls directly: a model with a very expanded FFN, a compact FFN, or
    // mixed grouped tensors receives the term implied by its real tensor shapes.
    if let Some(term) = ffn_shape_occupancy_term(model.tensor_matmul.feed_forward.shape) {
        return term;
    }
    let Some(hidden) = model.hidden_size.filter(|hidden| *hidden > 0) else {
        return 0.0;
    };
    let Some(ffn) = model.ffn_size else {
        return 0.0;
    };
    let ratio = ffn as f32 / hidden as f32;
    if ratio <= 4.0 {
        return 0.0;
    }
    ((ratio - 4.0) / 1.5 * 0.14).clamp(0.0, 0.14)
}

fn ffn_shape_occupancy_term(shape: MatmulShapeProfile) -> Option<f32> {
    let expansion = ffn_shape_expansion_ratio(shape)?;
    if expansion <= 4.0 {
        return Some(0.0);
    }
    Some(((expansion - 4.0) / 1.5 * 0.14).clamp(0.0, 0.14))
}

fn ffn_shape_expansion_ratio(shape: MatmulShapeProfile) -> Option<f32> {
    if shape.logical_matrix_count == 0
        || shape.weighted_avg_input_width == 0
        || shape.weighted_avg_output_width == 0
    {
        return None;
    }
    // Use the range across the FFN group rather than only the aggregate average:
    // `up/gate` and `down` matrices are transposes in shape terms, so averaging
    // input/output widths together would hide the expansion that determines how
    // large each individual FFN matmul is.
    let narrow = shape
        .min_input_width
        .min(shape.min_output_width)
        .min(shape.weighted_avg_input_width)
        .min(shape.weighted_avg_output_width) as f32;
    let wide = shape
        .max_input_width
        .max(shape.max_output_width)
        .max(shape.weighted_avg_input_width)
        .max(shape.weighted_avg_output_width) as f32;
    if narrow <= 0.0 {
        return None;
    }
    Some(wide / narrow)
}

fn measured_decode_graph_overhead_ms(model: &ModelProfile, budget: &ExecutionBudget) -> f32 {
    // Older revisions multiplied `decode_fixed_overhead_ms` by an estimated
    // layer/matmul graph count. A deeper read of llama.cpp made that too coarse:
    // `llama_context::process_ubatch()` reuses the previous decode graph when
    // the topology is stable, calls `set_inputs()`, then submits the whole GGML
    // graph through `ggml_backend_sched_graph_compute_async()`. Backends such as
    // Metal also reorder/fuse node work inside that graph. Charging a measured
    // empty submission once per layer or matmul therefore double-counts graph
    // overhead and badly penalizes architectures with many source-visible
    // nodes.
    //
    // The replacement is the grouped decode-cost model above: attention, FFN,
    // output, and routed expert groups each use measured GGML op-shape probes
    // when available. This function intentionally returns zero so there is no
    // hidden per-layer constant left in the decode estimate. Residual graph
    // misses should be handled by adding source-shaped probes or graph
    // introspection, not by reintroducing a broad multiplier here.
    let _ = (model, budget);
    0.0
}

fn aggregate_matmul_type_bytes(matmul: &crate::TensorMatmulProfile) -> TensorTypeBytes {
    add_type_bytes(
        add_type_bytes(
            add_type_bytes(matmul.attention.type_bytes, matmul.feed_forward.type_bytes),
            matmul.output.type_bytes,
        ),
        matmul.expert_feed_forward.type_bytes,
    )
}

fn add_type_bytes(left: TensorTypeBytes, right: TensorTypeBytes) -> TensorTypeBytes {
    TensorTypeBytes {
        f32_bytes: left.f32_bytes.saturating_add(right.f32_bytes),
        f16_bytes: left.f16_bytes.saturating_add(right.f16_bytes),
        bf16_bytes: left.bf16_bytes.saturating_add(right.bf16_bytes),
        q4_0_bytes: left.q4_0_bytes.saturating_add(right.q4_0_bytes),
        q4_k_bytes: left.q4_k_bytes.saturating_add(right.q4_k_bytes),
        q5_k_bytes: left.q5_k_bytes.saturating_add(right.q5_k_bytes),
        q6_k_bytes: left.q6_k_bytes.saturating_add(right.q6_k_bytes),
        q8_0_bytes: left.q8_0_bytes.saturating_add(right.q8_0_bytes),
        iq_bytes: left.iq_bytes.saturating_add(right.iq_bytes),
        other_quantized_bytes: left
            .other_quantized_bytes
            .saturating_add(right.other_quantized_bytes),
        unknown_bytes: left.unknown_bytes.saturating_add(right.unknown_bytes),
    }
}

fn raw_memory_bandwidth_bytes_per_sec(budget: &ExecutionBudget) -> u64 {
    budget
        .memory_bandwidth_bytes_per_sec
        .unwrap_or(match budget.backend {
            BackendKind::Cpu => 80_000_000_000,
            _ => 200_000_000_000,
        })
}

fn decode_base_bandwidth_bytes_per_sec(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
) -> u64 {
    if measured_gpu_budget(budget) {
        if let Some(probe) = selected_representative_decode_kernel_probe(model, budget) {
            return (probe.effective_gbps * 1_000_000_000.0).round() as u64;
        }
        return budget
            .decode_effective_bandwidth_bytes_per_sec
            .or(budget.memory_bandwidth_bytes_per_sec)
            .unwrap_or(50_000_000_000);
    }
    (raw_memory_bandwidth_bytes_per_sec(budget) as f32
        * fallback_backend_efficiency(budget.backend, config)) as u64
}

fn selected_representative_decode_kernel_probe<'a>(
    model: &ModelProfile,
    budget: &'a ExecutionBudget,
) -> Option<&'a DecodeKernelProbe> {
    let target = decode_probe_target(model)?;
    selected_decode_kernel_probe_with_distance(model, budget)
        .filter(|(_, distance)| {
            target.rows == 0
                || target.cols == 0
                || *distance <= MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE
        })
        .map(|(probe, _)| probe)
}

fn selected_decode_kernel_probe_with_distance<'a>(
    model: &ModelProfile,
    budget: &'a ExecutionBudget,
) -> Option<(&'a DecodeKernelProbe, f64)> {
    let target = decode_probe_target(model)?;
    budget
        .decode_kernel_probes
        .iter()
        .filter(|probe| {
            probe.batch_tokens == 1
                && probe.effective_gbps > 0.0
                && is_llama_decode_kernel_probe(probe)
                && decode_kernel_probe_matches_model_execution(probe, model)
                && probe.tensor_type.eq_ignore_ascii_case(target.tensor_type)
        })
        .map(|probe| (probe, decode_probe_shape_distance(probe, target)))
        .min_by(
            |(left_probe, left_distance), (right_probe, right_distance)| {
                left_distance.total_cmp(right_distance).then_with(|| {
                    decode_probe_operation_rank(left_probe, model)
                        .cmp(&decode_probe_operation_rank(right_probe, model))
                })
            },
        )
}

fn decode_probe_target(model: &ModelProfile) -> Option<DecodeProbeTarget> {
    let tensor_type = dominant_decode_tensor_type(model)?;
    let shape = dominant_decode_shape_for_tensor_type(model, tensor_type)
        .or_else(|| model.hidden_size.map(|hidden| (hidden, hidden)))?;
    Some(DecodeProbeTarget {
        tensor_type,
        rows: shape.0,
        cols: shape.1,
    })
}

fn decode_probe_shape_distance(probe: &DecodeKernelProbe, target: DecodeProbeTarget) -> f64 {
    if probe.rows == 0 || probe.cols == 0 || target.rows == 0 || target.cols == 0 {
        return 0.0;
    }
    let row_distance = log_dimension_ratio(probe.rows, target.rows);
    let col_distance = log_dimension_ratio(probe.cols, target.cols);
    row_distance.max(col_distance)
}

fn log_dimension_ratio(left: u32, right: u32) -> f64 {
    let left = f64::from(left.max(1));
    let right = f64::from(right.max(1));
    (left / right).ln().abs()
}

fn model_attention_kv_width(model: &ModelProfile) -> u64 {
    let key_width = kv_width(model, model.key_length);
    let value_width = kv_width(model, model.value_length);
    key_width.max(value_width).max(1)
}

fn dense_layer_graph_probe_kv_width(probe: &DecodeKernelProbe) -> Option<u64> {
    let name = probe.name.to_ascii_lowercase();
    let (_, suffix) = name.split_once("_kv")?;
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    (!digits.is_empty())
        .then(|| digits.parse::<u64>().ok())
        .flatten()
}

fn linear_attention_graph_probe_width(probe: &DecodeKernelProbe, marker: &str) -> Option<u64> {
    let name = probe.name.to_ascii_lowercase();
    let (_, suffix) = name.split_once(marker)?;
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    (!digits.is_empty())
        .then(|| digits.parse::<u64>().ok())
        .flatten()
}

fn linear_attention_graph_recurrent_layers(probe: &DecodeKernelProbe) -> u32 {
    linear_attention_graph_probe_width(probe, "_linear_attn_graph_r")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(1)
        .max(1)
}

fn linear_attention_graph_full_attention_layers(probe: &DecodeKernelProbe) -> u32 {
    linear_attention_graph_probe_width(probe, "_f")
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_default()
}

fn linear_attention_graph_probe_features_match_model(
    probe: &DecodeKernelProbe,
    model: &ModelProfile,
) -> bool {
    probe.graph_features == dense_graph_feature_bits(model)
}

fn linear_attention_graph_probe_shape_matches_model(
    probe: &DecodeKernelProbe,
    model: &ModelProfile,
) -> bool {
    let recurrent = &model.recurrent_attention;
    if recurrent.recurrent_layer_count == 0 {
        return false;
    }
    let full_layers = model
        .layer_count
        .unwrap_or(recurrent.recurrent_layer_count)
        .saturating_sub(recurrent.recurrent_layer_count);
    linear_attention_graph_recurrent_layers(probe) == recurrent.recurrent_layer_count
        && linear_attention_graph_full_attention_layers(probe) == full_layers
        && linear_attention_graph_probe_width(probe, "_h") == model.hidden_size.map(u64::from)
        && linear_attention_graph_probe_width(probe, "_qkv")
            == nonzero_shape_width(recurrent.qkv_projection.shape.max_output_width)
        && linear_attention_graph_probe_width(probe, "_gate")
            == nonzero_shape_width(recurrent.gate_projection.shape.max_output_width)
        && linear_attention_graph_probe_width(probe, "_state") == recurrent_state_width(model)
        && linear_attention_graph_probe_width(probe, "_out")
            == nonzero_shape_width(recurrent.output_projection.shape.max_input_width)
        && linear_attention_graph_probe_width(probe, "_kv") == Some(model_attention_kv_width(model))
}

fn recurrent_state_width(model: &ModelProfile) -> Option<u64> {
    let recurrent = &model.recurrent_attention;
    nonzero_shape_width(
        recurrent
            .beta_projection
            .shape
            .max_output_width
            .max(recurrent.alpha_projection.shape.max_output_width),
    )
}

fn nonzero_shape_width(width: u64) -> Option<u64> {
    (width > 0).then_some(width)
}

fn has_recurrent_attention_graph(model: &ModelProfile) -> bool {
    let recurrent = &model.recurrent_attention;
    recurrent.recurrent_layer_count > 0
        && recurrent.qkv_projection.shape.tensor_count > 0
        && recurrent.gate_projection.shape.tensor_count > 0
        && recurrent.output_projection.shape.tensor_count > 0
}

fn moe_block_graph_probe_kv_width(probe: &DecodeKernelProbe) -> Option<u64> {
    dense_layer_graph_probe_kv_width(probe)
}

fn moe_block_graph_probe_kv_matches_model(probe: &DecodeKernelProbe, model: &ModelProfile) -> bool {
    let hidden = u64::from(model.hidden_size.unwrap_or_default());
    if hidden == 0 {
        return true;
    }
    let model_kv = model_attention_kv_width(model);
    let probe_kv = moe_block_graph_probe_kv_width(probe).unwrap_or(hidden);
    if model_kv < hidden {
        probe_kv == model_kv
    } else {
        probe_kv == hidden
    }
}

fn dense_layer_graph_probe_layers(probe: &DecodeKernelProbe) -> u32 {
    let name = probe.name.to_ascii_lowercase();
    let Some((_, suffix)) = name.split_once("_llama_graph_l") else {
        return 1;
    };
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse::<u32>().unwrap_or(1).max(1)
}

fn linear_attention_graph_probe_rank(probe: &DecodeKernelProbe, model: &ModelProfile) -> u16 {
    let recurrent = model.recurrent_attention.recurrent_layer_count;
    let recurrent_rank = if linear_attention_graph_recurrent_layers(probe) == recurrent {
        0
    } else {
        900
    };
    let model_full = model
        .layer_count
        .unwrap_or(recurrent)
        .saturating_sub(recurrent);
    let full_rank = if linear_attention_graph_full_attention_layers(probe) == model_full {
        0
    } else {
        50
    };
    recurrent_rank + full_rank
}

fn dense_layer_graph_probe_depth_matches_model(
    probe: &DecodeKernelProbe,
    model: &ModelProfile,
) -> bool {
    let probe_layers = dense_layer_graph_probe_layers(probe);
    match model.layer_count.filter(|layers| *layers > 0) {
        Some(model_layers) => probe_layers <= model_layers,
        None => probe_layers == 1,
    }
}

fn dense_layer_graph_probe_features_match_model(
    probe: &DecodeKernelProbe,
    model: &ModelProfile,
) -> bool {
    probe.graph_features == dense_graph_feature_bits(model)
}

fn dense_graph_feature_bits(model: &ModelProfile) -> u32 {
    let mut features = 0;
    if model.dense_graph_features.attention_q_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_Q_NORM;
    }
    if model.dense_graph_features.attention_k_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_K_NORM;
    }
    if model.dense_graph_features.attention_post_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_POST_NORM;
    }
    if model.dense_graph_features.feed_forward_post_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_FFN_POST_NORM;
    }
    features
}

fn moe_block_graph_probe_layers(probe: &DecodeKernelProbe) -> u32 {
    let name = probe.name.to_ascii_lowercase();
    let Some((_, suffix)) = name.split_once("_moe_block_graph_l") else {
        return 1;
    };
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse::<u32>().unwrap_or(1).max(1)
}

fn moe_graph_probe_layers(probe: &DecodeKernelProbe) -> u32 {
    let name = probe.name.to_ascii_lowercase();
    let Some((_, suffix)) = name.split_once("_moe_graph_l") else {
        return 1;
    };
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse::<u32>().unwrap_or(1).max(1)
}

fn is_supported_dense_layer_graph_probe(probe: &DecodeKernelProbe) -> bool {
    let layers = dense_layer_graph_probe_layers(probe);
    if layers == 1 {
        return true;
    }
    // Stacked dense probes exist because llama.cpp decode does not submit one
    // isolated matvec per token. `llama_context::process_ubatch()` builds a
    // GGML graph containing the repeated transformer blocks, calls `set_inputs`
    // for the token state, and submits that graph through the backend scheduler.
    // A one-layer synthetic graph is source-shaped at the operation level, but
    // it misses depth-dependent scheduler, graph optimization, residency, and
    // kernel-fusion behavior that appears when many repeated blocks are present
    // in one decode graph.
    //
    // Earlier revisions kept stacked selection to Q4_K because a narrow Q8 row
    // was not enough evidence to promote Q8 stacks. The skippy ABI graph
    // inventory now lets validation compare GGUF-derived tensor groups against
    // the actual llama.cpp decode graph. Across the Q8 held-out sweep, the
    // inventory showed the same repeated-layer risk and the lN graph rows were
    // the source-shaped evidence that moved estimates toward observed decode.
    // This is still not a backend or model-name correction: a row is usable only
    // when it is a measured llama graph probe for the same tensor type, shape,
    // KV width, and a depth no greater than the GGUF model layer count.
    probe.tensor_type.eq_ignore_ascii_case("q4_k") || probe.tensor_type.eq_ignore_ascii_case("q8_0")
}

fn is_llama_decode_kernel_probe(probe: &DecodeKernelProbe) -> bool {
    // A measured row is not automatically a model-fit decode row just because
    // it has a tensor type. `mesh-llm gpus benchmark` can also report useful
    // diagnostic probes from backend libraries such as cuBLAS or MPS. Those are
    // hardware facts, but they are not the same kernels llama.cpp uses for
    // GGML_OP_MUL_MAT decode on every backend. High-confidence tok/s prediction
    // should only consume probes that explicitly identify themselves as GGML or
    // llama decode kernels. This keeps diagnostic benchmark expansion from
    // silently becoming a fitted estimator.
    let name = probe.name.as_str();
    name.starts_with("ggml_") || name.starts_with("llama_")
}

fn decode_kernel_probe_matches_model_execution(
    probe: &DecodeKernelProbe,
    model: &ModelProfile,
) -> bool {
    if has_recurrent_attention_graph(model) {
        return is_composite_linear_attention_graph_decode_probe(probe)
            && linear_attention_graph_probe_shape_matches_model(probe, model);
    }
    if model.architecture_class != ModelArchitectureClass::SparseMoeTransformer {
        return true;
    }

    // MoE decode is not just "a smaller dense matvec." llama.cpp routes tokens
    // through selected experts using expert-indexed matmul machinery
    // (`GGML_OP_MUL_MAT_ID` in current upstream), and that changes batching,
    // memory locality, graph shape, and backend scheduling. A dense/square GGML
    // matvec probe can be useful hardware evidence for dense transformers, but
    // it should not grant high confidence for a sparse model simply because the
    // rows/cols are numerically near an averaged expert shape. Until the GPU
    // benchmark exposes an explicit routed-expert/MUL_MAT_ID or sparse-block
    // probe, sparse MoE estimates must fall back to the broader measured
    // bandwidth inputs and report lower tok/s confidence.
    is_composite_moe_block_graph_decode_probe(probe) || is_routed_moe_decode_probe(probe)
}

fn decode_probe_operation_rank(probe: &DecodeKernelProbe, model: &ModelProfile) -> u8 {
    if has_recurrent_attention_graph(model) {
        return if is_composite_linear_attention_graph_decode_probe(probe) {
            0
        } else {
            3
        };
    }
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer
            if is_composite_moe_block_graph_decode_probe(probe) =>
        {
            0
        }
        ModelArchitectureClass::SparseMoeTransformer
            if is_composite_moe_graph_decode_probe(probe) =>
        {
            1
        }
        ModelArchitectureClass::SparseMoeTransformer if is_routed_moe_decode_probe(probe) => 2,
        ModelArchitectureClass::DenseTransformer
            if is_composite_llama_graph_decode_probe(probe) =>
        {
            0
        }
        _ => 2,
    }
}

fn has_high_confidence_decode_probe(model: &ModelProfile, budget: &ExecutionBudget) -> bool {
    selected_representative_decode_kernel_probe(model, budget)
        .is_some_and(|probe| high_confidence_decode_probe_matches_model(model, probe))
}

fn high_confidence_decode_probe_matches_model(
    model: &ModelProfile,
    probe: &DecodeKernelProbe,
) -> bool {
    if has_recurrent_attention_graph(model) {
        return is_composite_linear_attention_graph_decode_probe(probe)
            && linear_attention_graph_probe_features_match_model(probe, model)
            && linear_attention_graph_probe_shape_matches_model(probe, model);
    }
    match model.architecture_class {
        ModelArchitectureClass::DenseTransformer => {
            is_composite_llama_graph_decode_probe(probe)
                && dense_layer_graph_probe_features_match_model(probe, model)
        }
        ModelArchitectureClass::SparseMoeTransformer => {
            is_composite_moe_block_graph_decode_probe(probe)
                || is_composite_moe_graph_decode_probe(probe)
        }
        _ => false,
    }
}

fn shallow_q8_graph_probe_warning(
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<String> {
    if !measured_gpu_budget(budget)
        || model.architecture_class != ModelArchitectureClass::DenseTransformer
        || model.layer_count.unwrap_or_default() < 16
    {
        return None;
    }
    let probe = selected_representative_decode_kernel_probe(model, budget)?;
    if !probe.tensor_type.eq_ignore_ascii_case("q8_0")
        || !is_composite_llama_graph_decode_probe(probe)
        || dense_layer_graph_probe_layers(probe) != 1
    {
        return None;
    }
    Some(
        "selected Q8_0 decode probe is a one-layer llama graph for a deeper dense model; tok/s estimate is reported without a depth-matched Q8 graph probe and should be validated before relying on +/-10%"
            .into(),
    )
}

fn high_confidence_decode_probe_warning(model: &ModelProfile) -> String {
    if has_recurrent_attention_graph(model) {
        return "measured GPU profile has no graph-feature-matched composite linear-attention decode graph probe; dense llama graph probes are not enough for high tok/s confidence".into();
    }
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => {
            "measured GPU profile has no composite sparse MoE block decode graph probe using GGML_OP_MUL_MAT_ID; tok/s confidence cannot be high".into()
        }
        ModelArchitectureClass::DenseTransformer => {
            "measured GPU profile has no graph-feature-matched composite llama decode graph probe; single matvec or plain llama graph probes are not enough for high tok/s confidence".into()
        }
        _ => {
            "measured GPU profile has no architecture-matched composite decode probe; tok/s confidence cannot be high".into()
        }
    }
}

fn is_composite_llama_graph_decode_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("llama_graph")
}

fn is_composite_linear_attention_graph_decode_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("linear_attn_graph")
}

fn is_matvec_decode_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("matvec") || name.contains("mul_mat")
}

fn is_routed_moe_decode_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("mul_mat_id") || is_composite_moe_graph_decode_probe(probe)
}

fn is_synthetic_expert_matvec_probe(probe: &DecodeKernelProbe) -> bool {
    // `mesh-llm gpus benchmark` emits a very narrow `*_matvec_expert_*` row to
    // expose how the backend handles expert-like matrices. That shape is useful
    // diagnostic evidence, but it is not a dense transformer block: llama.cpp
    // dense decode lowers attention/FFN tensors through normal
    // GGML_OP_MUL_MAT graph nodes, while sparse expert routing uses
    // GGML_OP_MUL_MAT_ID / MoE graph paths. Letting the expert row compete with
    // dense/output matmul rows can make a small dense model look artificially
    // slow because the narrow probe is launch/occupancy dominated. Keep it out
    // of dense selection and require routed-expert groups to use explicit MoE
    // probes instead.
    probe.name.to_ascii_lowercase().contains("_expert_")
}

fn is_composite_moe_graph_decode_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("moe_graph") && !name.contains("moe_block_graph")
}

fn is_composite_moe_block_graph_decode_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("moe_block_graph")
}

fn missing_exact_decode_kernel_probe_tensor_type(
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<&'static str> {
    if !measured_gpu_budget(budget)
        || has_recurrent_attention_graph(model)
        || selected_representative_decode_kernel_probe(model, budget).is_some()
    {
        return None;
    }
    dominant_decode_tensor_type(model)
}

fn dominant_decode_tensor_type(model: &ModelProfile) -> Option<&'static str> {
    let bytes = aggregate_matmul_type_bytes(&model.tensor_matmul);
    let candidates = [
        ("f32", bytes.f32_bytes),
        ("f16", bytes.f16_bytes),
        ("bf16", bytes.bf16_bytes),
        ("q4_0", bytes.q4_0_bytes),
        ("q4_k", bytes.q4_k_bytes),
        ("q5_k", bytes.q5_k_bytes),
        ("q6_k", bytes.q6_k_bytes),
        ("q8_0", bytes.q8_0_bytes),
        ("iq", bytes.iq_bytes),
        ("other_quantized", bytes.other_quantized_bytes),
        ("unknown", bytes.unknown_bytes),
    ];
    candidates
        .into_iter()
        .max_by_key(|(_, bytes)| *bytes)
        .and_then(|(kind, bytes)| (bytes > 0).then_some(kind))
        .or_else(|| quantization_tensor_type(model.quantization.as_deref()))
}

fn dominant_decode_shape_for_tensor_type(
    model: &ModelProfile,
    tensor_type: &str,
) -> Option<(u32, u32)> {
    let mut candidates = [
        decode_shape_candidate(&model.tensor_matmul.attention, tensor_type, model, false),
        decode_shape_candidate(&model.tensor_matmul.feed_forward, tensor_type, model, false),
        decode_shape_candidate(&model.tensor_matmul.output, tensor_type, model, false),
        decode_shape_candidate(
            &model.tensor_matmul.expert_feed_forward,
            tensor_type,
            model,
            true,
        ),
    ];
    candidates
        .iter_mut()
        .filter_map(Option::take)
        .max_by_key(|candidate| candidate.traffic_bytes)
        .map(|candidate| (candidate.rows, candidate.cols))
}

#[derive(Clone, Copy, Debug)]
struct DecodeShapeCandidate {
    traffic_bytes: u64,
    rows: u32,
    cols: u32,
}

fn decode_shape_candidate(
    group: &TensorMatmulGroupProfile,
    tensor_type: &str,
    model: &ModelProfile,
    expert: bool,
) -> Option<DecodeShapeCandidate> {
    let mut traffic_bytes = tensor_type_bytes_for_kind(group.type_bytes, tensor_type);
    if expert {
        traffic_bytes =
            active_expert_bytes(traffic_bytes, model.expert_count, model.expert_used_count);
    }
    if traffic_bytes == 0 {
        return None;
    }
    let (rows, cols) = matmul_shape_target(group.shape)?;
    Some(DecodeShapeCandidate {
        traffic_bytes,
        rows,
        cols,
    })
}

fn tensor_type_bytes_for_kind(bytes: TensorTypeBytes, tensor_type: &str) -> u64 {
    match tensor_type {
        "f32" => bytes.f32_bytes,
        "f16" => bytes.f16_bytes,
        "bf16" => bytes.bf16_bytes,
        "q4_0" => bytes.q4_0_bytes,
        "q4_k" => bytes.q4_k_bytes,
        "q5_k" => bytes.q5_k_bytes,
        "q6_k" => bytes.q6_k_bytes,
        "q8_0" => bytes.q8_0_bytes,
        "iq" => bytes.iq_bytes,
        "other_quantized" => bytes.other_quantized_bytes,
        "unknown" => bytes.unknown_bytes,
        _ => 0,
    }
}

fn matmul_shape_target(shape: MatmulShapeProfile) -> Option<(u32, u32)> {
    if shape.logical_matrix_count == 0 {
        return None;
    }
    let rows = if shape.weighted_avg_output_width > 0 {
        shape.weighted_avg_output_width
    } else {
        shape.max_output_width
    };
    let cols = if shape.weighted_avg_input_width > 0 {
        shape.weighted_avg_input_width
    } else {
        shape.max_input_width
    };
    Some((u32::try_from(rows).ok()?, u32::try_from(cols).ok()?))
}

fn quantization_tensor_type(quantization: Option<&str>) -> Option<&'static str> {
    let quantization = quantization?.to_ascii_lowercase();
    if quantization.contains("q4_k") {
        Some("q4_k")
    } else if quantization.contains("q5_k") {
        Some("q5_k")
    } else if quantization.contains("q6_k") {
        Some("q6_k")
    } else if quantization.contains("q8_0") || quantization.contains("q8") {
        Some("q8_0")
    } else if quantization.contains("q4_0") {
        Some("q4_0")
    } else if quantization.contains("f16") {
        Some("f16")
    } else if quantization.contains("bf16") {
        Some("bf16")
    } else if quantization.contains("f32") {
        Some("f32")
    } else {
        None
    }
}

fn prefill_tokens_per_sec(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
    decode_tokens_per_sec: Option<f32>,
) -> Option<f32> {
    if !uses_transformer_kv_cache(model.architecture_class) {
        return None;
    }
    let prompt_tokens = config.workload.interaction.expected_prompt_tokens?;
    if prompt_tokens == 0 {
        return None;
    }
    let roofline = prefill_roofline_tokens_per_sec(model, config, budget, prompt_tokens);
    let fallback = legacy_prefill_tokens_per_sec(model, decode_tokens_per_sec);
    if model.architecture_class == ModelArchitectureClass::SparseMoeTransformer {
        return match (roofline, fallback) {
            (Some(roofline), Some(fallback)) => Some(roofline.min(fallback)),
            (Some(roofline), None) => Some(roofline),
            (None, fallback) => fallback,
        };
    }
    roofline.or(fallback)
}

fn prefill_roofline_tokens_per_sec(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
    prompt_tokens: u32,
) -> Option<f32> {
    // llama.cpp builds a different graph for prompt processing than for
    // one-token decode. `llama-context.cpp` splits the prompt into ubatches
    // (`n_ubatch` defaults to 512 in `llama_context_default_params`), and
    // the backend receives batched `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID`
    // work instead of a long stream of single-token matvecs. That gives prefill
    // a compute-shaped roofline:
    //
    //   total_ms = max(prompt_matmul_flops / measured_compute,
    //                  ubatches * active_weight_bytes / measured_bandwidth)
    //              + ubatches * graph_overhead
    //
    // This is intentionally not a CUDA/Metal/ROCm branch. The scorer consumes
    // only the GGUF-derived matmul FLOPs/bytes and the measured hardware facts
    // that `mesh-llm gpus benchmark` already places in `HardwareProfile`.
    //
    // Very narrow models are the awkward corner. Their prompt graph is often
    // dominated by non-matmul kernels, scheduling, logits/pooling work, and
    // launch latency that our current generic hardware benchmark does not
    // measure directly. For those, keep the older decode-correlated fallback
    // until we add a measured prefill-shaped hardware probe.
    if !prefill_roofline_has_enough_matmul_shape(model) {
        return None;
    }
    let flops_per_token = active_decode_flops_per_token(model)? as f32;
    let active_weight_bytes = active_prefill_pressure_bytes(model)? as f32;
    let ubatches = prefill_ubatch_count(prompt_tokens) as f32;
    let compute_ms = prefill_compute_ms(flops_per_token, prompt_tokens, model, budget)?;
    let bandwidth_ms = prefill_weight_stream_ms(active_weight_bytes, ubatches, budget)?;
    let overhead_ms = prefill_graph_overhead_ms(model, budget, config, ubatches);
    let total_ms = compute_ms.max(bandwidth_ms) + overhead_ms;
    (total_ms > 0.0).then_some(prompt_tokens as f32 / total_ms * 1000.0)
}

fn prefill_roofline_has_enough_matmul_shape(model: &ModelProfile) -> bool {
    if model.architecture_class == ModelArchitectureClass::SparseMoeTransformer {
        return true;
    }
    let Some(hidden) = model.hidden_size else {
        return false;
    };
    hidden >= 2048
}

fn prefill_ubatch_count(prompt_tokens: u32) -> u32 {
    prompt_tokens.div_ceil(LLAMA_DEFAULT_UBATCH_TOKENS).max(1)
}

fn prefill_compute_ms(
    flops_per_token: f32,
    prompt_tokens: u32,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<f32> {
    let tflops = prefill_compute_tflops(model, budget)?;
    let total_flops = flops_per_token * prompt_tokens as f32;
    Some(total_flops / (tflops * 1_000_000_000_000.0) * 1000.0)
}

fn prefill_compute_tflops(model: &ModelProfile, budget: &ExecutionBudget) -> Option<f32> {
    // Dense and sparse-MoE prefill are intentionally separate measured hardware
    // facts. Dense prompt processing maps to batched GEMM, but llama.cpp does
    // not feed the backend one square peak-throughput GEMM. It splits prompt
    // tokens into ubatches (`n_ubatch` defaults to 512), so the common source
    // shape is a weight matrix times a skinny token batch. Prefer that measured
    // ubatch-shaped probe when present, and keep the square GEMM probe as a
    // fallback for older benchmark JSON.
    //
    // Sparse MoE routes through expert selection and `GGML_OP_MUL_MAT_ID`, so
    // it only uses the MoE-shaped probe when that probe is present. If the MoE
    // probe is absent, return `None` so the caller falls back to the older
    // MoE-aware estimate instead of overpredicting MoE with the dense matmul
    // probe.
    let measured = match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => budget.prefill_moe_matmul_tflops_fp16,
        _ => budget
            .prefill_ubatch_matmul_tflops_fp16
            .or(budget.prefill_matmul_tflops_fp16)
            .or(budget.compute_tflops_fp16),
    };
    measured.filter(|value| *value > 0.0)
}

fn prefill_weight_stream_ms(
    active_weight_bytes: f32,
    ubatches: f32,
    budget: &ExecutionBudget,
) -> Option<f32> {
    let bandwidth = raw_memory_bandwidth_bytes_per_sec(budget) as f32;
    (bandwidth > 0.0).then_some(active_weight_bytes * ubatches / bandwidth * 1000.0)
}

fn prefill_graph_overhead_ms(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    ubatches: f32,
) -> f32 {
    ubatches
        * (measured_decode_graph_overhead_ms(model, budget)
            + architecture_decode_overhead_ms(model, budget, config))
}

fn legacy_prefill_tokens_per_sec(
    model: &ModelProfile,
    decode_tokens_per_sec: Option<f32>,
) -> Option<f32> {
    // This fallback is still useful for tiny and narrow models where the
    // matmul roofline has the wrong dominant term. It is deliberately isolated
    // from the main prefill path so future work can replace it with a measured
    // prefill-shaped hardware probe instead of spreading more special cases
    // through the scorer.
    let decode_tokens_per_sec = decode_tokens_per_sec?;
    let parallelism = prefill_decode_parallelism_factor(model)?;
    let prefill_tokens_per_sec = decode_tokens_per_sec * parallelism;
    Some(prefill_tokens_per_sec.max(0.0))
}

fn prefill_decode_parallelism_factor(model: &ModelProfile) -> Option<f32> {
    // This factor estimates how much more parallelism prefill gets compared to
    // decode. Smaller hidden sizes and smaller active-byte footprints generally
    // let prefill run many prompt tokens in parallel, so their factor is higher.
    // Large active models get less of a boost because weight traffic and memory
    // pressure still dominate.
    //
    // Q8 gets a higher prefill factor for non-tiny models because validation
    // showed that its decode path can be relatively slower than its batched
    // prefill path. This is a metadata-level quantization adjustment, not a
    // model identity boost.
    let hidden = model.hidden_size.filter(|hidden| *hidden > 0)? as f32;
    let active_gib = active_prefill_pressure_bytes(model)? as f32 / GIB as f32;
    let width_ratio = (1024.0 / hidden).min(1.0);
    let width_term = 26.0 * width_ratio.powf(1.35);
    let active_pressure = 1.0 + (active_gib - 1.0).max(0.0) * 0.22;
    let mut factor = 9.0 + width_term / active_pressure;

    if matches!(
        model.architecture_class,
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown
    ) && (1536.0..=2304.0).contains(&hidden)
    {
        factor += 3.0;
    }
    if (800.0..=1536.0).contains(&hidden) {
        factor += 3.0;
    }
    if hidden < 800.0 {
        factor *= 1.80;
    }
    factor *= 1.12;
    if quantization_is_q8(model.quantization.as_deref()) && hidden >= 800.0 {
        factor *= 1.70;
    }
    Some(factor.clamp(8.0, 72.0))
}

fn active_prefill_pressure_bytes(model: &ModelProfile) -> Option<u64> {
    match model.architecture_class {
        ModelArchitectureClass::DenseTransformer | ModelArchitectureClass::Unknown => {
            Some(active_dense_decode_weight_traffic(model))
        }
        ModelArchitectureClass::SparseMoeTransformer => {
            let dense_active = active_moe_decode_weight_traffic(model);
            let groups = model.tensor_group_bytes;
            let resident = groups
                .attention_bytes
                .saturating_add(groups.feed_forward_bytes)
                .saturating_add(groups.expert_feed_forward_bytes)
                .saturating_add(groups.output_bytes)
                .saturating_add(groups.normalization_bytes)
                .saturating_add(groups.other_bytes);
            Some(dense_active.max(resident / 3))
        }
        _ => None,
    }
}

fn quantization_is_q8(quantization: Option<&str>) -> bool {
    let Some(quantization) = quantization else {
        return false;
    };
    let lower = quantization.to_ascii_lowercase();
    lower.contains("q8_0") || gguf_file_type_is(&lower, 7)
}

#[derive(Clone, Copy, Debug, Default)]
struct FirstTokenEstimate {
    prefill_ms: Option<f32>,
    decode_ms: Option<f32>,
    overhead_ms: Option<f32>,
    sampler_ms: Option<f32>,
    total_ms: Option<f32>,
}

fn first_token_estimate(
    model: &ModelProfile,
    prefill_tps: Option<f32>,
    decode_tps: Option<f32>,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
) -> FirstTokenEstimate {
    let prefill_ms = config
        .workload
        .interaction
        .expected_prompt_tokens
        .zip(prefill_tps)
        .map(|(prompt_tokens, tps)| prompt_tokens as f32 / tps.max(0.001) * 1000.0);
    let decode_ms = decode_tps.map(|tps| 1000.0 / tps.max(0.001));
    // This is a lower-bound hardware/runtime transition fact: it measures the
    // backend cost of issuing decode-shaped work immediately after
    // prefill-shaped matmul work without loading a GGUF. It deliberately does
    // not try to cover llama.cpp graph construction, KV/session bookkeeping,
    // tokenization, HTTP, or sampling. Validation keeps those residuals visible
    // so we do not smuggle model-benchmark observations into metadata-only fit.
    let overhead_value_ms = budget.post_prefill_decode_overhead_ms.unwrap_or(0.0);
    let overhead_ms = Some(overhead_value_ms);
    let sampler_ms = sampler_first_token_ms(model, config, budget);
    let total_ms = prefill_ms.map(|prefill| {
        prefill + decode_ms.unwrap_or_default() + overhead_value_ms + sampler_ms.unwrap_or_default()
    });
    FirstTokenEstimate {
        prefill_ms,
        decode_ms,
        overhead_ms,
        sampler_ms,
        total_ms,
    }
}

fn sampler_first_token_ms(
    model: &ModelProfile,
    config: &SelectionConfig,
    budget: &ExecutionBudget,
) -> Option<f32> {
    let prompt_tokens = config.workload.interaction.expected_prompt_tokens?;
    if budget.sampler_history_us_per_token.is_none() && budget.sampler_vocab_us_per_token.is_none()
    {
        return None;
    }
    let history_us = budget.sampler_history_us_per_token.unwrap_or(0.0);
    let vocab_us = budget.sampler_vocab_us_per_token.unwrap_or(0.0);
    if history_us <= 0.0 && vocab_us <= 0.0 {
        return Some(0.0);
    }
    let history_ms = prompt_tokens as f32 * history_us / 1000.0;
    let vocab_ms = model
        .tokenizer
        .vocab_size
        .map(|vocab| vocab as f32 * vocab_us / 1000.0)
        .unwrap_or_default();
    Some(history_ms + vocab_ms)
}

fn sampled_decode_sampler_ms(model: &ModelProfile, budget: &ExecutionBudget) -> f32 {
    // Steady decode in the validator and in normal chat serving is sampled
    // generation: after llama.cpp produces logits for a token, the sampler
    // builds/scans vocabulary candidates and accepts the chosen token into the
    // sampling history. The first-token estimator already charges the measured
    // prompt-history sync separately because it scales with prompt length. For
    // every subsequent generated token, source-visible sampler work is one
    // sampled vocabulary pass plus one new-token history accept. Both inputs
    // come from `mesh-llm gpus benchmark` as machine/runtime facts, while the
    // vocabulary size comes from GGUF tokenizer metadata. No model benchmark
    // observation is fed back into this estimate.
    let vocab_ms = model
        .tokenizer
        .vocab_size
        .zip(budget.sampler_vocab_us_per_token)
        .map(|(vocab, us_per_token)| vocab as f32 * us_per_token / 1000.0)
        .unwrap_or_default();
    let history_ms = budget
        .sampler_history_us_per_token
        .map(|us| us / 1000.0)
        .unwrap_or_default();
    vocab_ms + history_ms
}

fn first_token_ms_range(
    point: Option<f32>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<FirstTokenEstimateRange> {
    let point = point?;
    let mut uncertainty = 0.35f32;
    if budget.memory_bandwidth_bytes_per_sec.is_none() {
        uncertainty += 0.15;
    }
    if !tensor_groups_available(model.tensor_group_bytes) {
        uncertainty += 0.10;
    }
    if model.hidden_size.is_none() || model.layer_count.is_none() {
        uncertainty += 0.10;
    }
    let uncertainty = uncertainty.clamp(0.25, 0.70);
    Some(FirstTokenEstimateRange {
        lower_ms: point * (1.0 - uncertainty),
        upper_ms: point * (1.0 + uncertainty),
    })
}

fn quantization_efficiency_factor(quantization: Option<&str>) -> f32 {
    let Some(quantization) = quantization else {
        return 0.90;
    };
    let lower = quantization.to_ascii_lowercase();
    if lower.contains("bf16") || gguf_file_type_is(&lower, 32) {
        0.45
    } else if lower.contains("f16") || gguf_file_type_is(&lower, 1) {
        0.60
    } else if lower.contains("q8_0") || gguf_file_type_is(&lower, 7) {
        1.0
    } else if lower.contains("q6_k") || gguf_file_type_is(&lower, 18) {
        0.88
    } else if lower.contains("q5_k")
        || gguf_file_type_is(&lower, 16)
        || gguf_file_type_is(&lower, 17)
    {
        0.95
    } else if lower.contains("iq") || gguf_file_type_is(&lower, 19) {
        0.85
    } else {
        1.0
    }
}

fn gguf_file_type_is(quantization: &str, expected: u32) -> bool {
    quantization
        .strip_prefix("gguf_file_type_")
        .and_then(|value| value.parse::<u32>().ok())
        == Some(expected)
}

fn decode_tokens_per_sec_range(
    point: Option<f32>,
    active_decode_bytes: Option<u64>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> Option<DecodeEstimateRange> {
    let point = point?;
    let uncertainty = decode_uncertainty(active_decode_bytes, model, budget);
    Some(DecodeEstimateRange {
        lower: point * (1.0 - uncertainty),
        upper: point * (1.0 + uncertainty),
    })
}

fn decode_uncertainty(
    active_decode_bytes: Option<u64>,
    model: &ModelProfile,
    budget: &ExecutionBudget,
) -> f32 {
    let mut uncertainty = 0.25f32;
    if budget.memory_bandwidth_bytes_per_sec.is_none() {
        uncertainty += 0.20;
    }
    if !tensor_groups_available(model.tensor_group_bytes) {
        uncertainty += 0.15;
    }
    if matches!(
        model.architecture_class,
        ModelArchitectureClass::SparseMoeTransformer | ModelArchitectureClass::Unknown
    ) {
        uncertainty += 0.10;
    }
    if active_decode_bytes.is_some_and(|bytes| bytes < 2 * 1024 * 1024 * 1024) {
        uncertainty += 0.10;
    }
    if model
        .quantization
        .as_deref()
        .is_none_or(quantization_is_uncertain)
    {
        uncertainty += 0.05;
    }
    uncertainty.clamp(0.20, 0.60)
}

fn quantization_is_uncertain(quantization: &str) -> bool {
    let lower = quantization.to_ascii_lowercase();
    lower.contains("iq") || gguf_file_type_is(&lower, 19) || gguf_file_type_is(&lower, 20)
}

fn decode_bandwidth_efficiency(budget: &ExecutionBudget, config: &SelectionConfig) -> f32 {
    // There are two modes:
    //
    // 1. Measured bandwidth: use one portable decode-efficiency knob plus a
    //    small noise adjustment. The benchmark has already exercised the actual
    //    backend path, so backend-specific multipliers would double-count local
    //    assumptions.
    // 2. Estimated/manual bandwidth: fall back to backend defaults because a
    //    hand-authored profile needs some prior about Metal vs CUDA vs CPU.
    if budget.bandwidth_source == MeasurementSource::Measured {
        return benchmark_noise_factor(budget);
    }
    fallback_backend_efficiency(budget.backend, config)
}

fn benchmark_noise_factor(budget: &ExecutionBudget) -> f32 {
    // Noise is not used as a hard rejection. It is a mild pessimism term: if the
    // benchmark itself had meaningful spread, the selector should not rank a
    // model as if every decode token will get the best sustained bandwidth.
    let Some(noise_pct) = budget.benchmark_noise_pct else {
        return 1.0;
    };
    (1.0 - noise_pct.max(0.0) / 100.0 * 0.5).clamp(0.85, 1.0)
}

fn fallback_backend_efficiency(backend: BackendKind, config: &SelectionConfig) -> f32 {
    match backend {
        BackendKind::Metal => config.backend_efficiency.metal,
        BackendKind::Cuda => config.backend_efficiency.cuda,
        BackendKind::Rocm => config.backend_efficiency.rocm,
        BackendKind::Vulkan => config.backend_efficiency.vulkan,
        BackendKind::Cpu => config.backend_efficiency.cpu,
        BackendKind::Unknown => config.backend_efficiency.unknown,
    }
}

fn fixed_decode_overhead_ms(budget: &ExecutionBudget, config: &SelectionConfig) -> f32 {
    // Fixed overhead matters most when active bytes are small. For measured GPU
    // hardware this must come from the benchmark profile. A universal measured
    // GPU constant would encode backend/runtime behavior in model-fit instead
    // of observing it on the target machine.
    if measured_gpu_budget(budget) {
        return budget.decode_fixed_overhead_ms.unwrap_or(0.0);
    }
    let backend = budget.backend;
    match backend {
        BackendKind::Metal => config.decode_overhead.metal_fixed_ms,
        BackendKind::Cuda => config.decode_overhead.cuda_fixed_ms,
        BackendKind::Rocm => config.decode_overhead.rocm_fixed_ms,
        BackendKind::Vulkan => config.decode_overhead.vulkan_fixed_ms,
        BackendKind::Cpu => config.decode_overhead.cpu_fixed_ms,
        BackendKind::Unknown => config.decode_overhead.unknown_fixed_ms,
    }
}

fn measured_decode_runtime_overhead_ms(budget: &ExecutionBudget) -> f32 {
    // This term is different from `decode_fixed_overhead_ms`. The fixed field
    // is native backend dispatch cost measured by the GPU benchmark. This field
    // is host/runtime control-path cost measured by benchmark code around a
    // generated-token loop. It intentionally has no fallback: if a hardware
    // profile did not observe this runtime fact, model-fit leaves it out and
    // validation must continue to show the miss instead of smuggling in a
    // backend- or model-specific constant.
    if measured_gpu_budget(budget) {
        budget.decode_runtime_overhead_ms.unwrap_or(0.0)
    } else {
        0.0
    }
}

fn measured_gpu_budget(budget: &ExecutionBudget) -> bool {
    budget.bandwidth_source == MeasurementSource::Measured
        && budget.backend != BackendKind::Cpu
        && budget.accelerator_kind != AcceleratorKind::Cpu
}

fn architecture_decode_overhead_ms(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
) -> f32 {
    // Architecture is already represented in source-shaped traffic:
    //
    // - dense transformers contribute attention/FFN/output groups;
    // - sparse MoE transformers contribute router/base groups plus active
    //   expert `GGML_OP_MUL_MAT_ID` traffic;
    // - recurrent/state-space models carry lower confidence until their own
    //   source-shaped groups are modeled.
    //
    // Keeping an extra per-layer architecture latency term here would be a
    // magic number even when scaled by measured fixed overhead. Leave it at
    // zero and make missing op coverage visible through warnings/confidence.
    let _ = (model, budget, config);
    0.0
}

fn fit_status(
    model: &ModelProfile,
    memory: &RuntimeMemoryEstimate,
    memory_limit: u64,
    workload_reject: bool,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> FitStatus {
    if !standalone_weight_coverage(model, reasons, warnings) {
        return FitStatus::Rejected;
    }
    if workload_reject {
        reasons.push("model capability evidence conflicts with required workload".into());
        return FitStatus::Rejected;
    }
    if memory.runtime_bytes <= memory_limit {
        reasons.push(format!(
            "estimated runtime memory fits within safety-adjusted budget ({:.1} GiB <= {:.1} GiB)",
            gib(memory.runtime_bytes),
            gib(memory_limit)
        ));
        if memory.runtime_bytes > memory_limit.saturating_mul(9) / 10 {
            warnings.push("model fits but leaves little memory headroom".into());
            FitStatus::FitsWithWarning
        } else {
            FitStatus::FitsLocal
        }
    } else {
        reasons.push(format!(
            "estimated runtime memory exceeds safety-adjusted budget ({:.1} GiB > {:.1} GiB)",
            gib(memory.runtime_bytes),
            gib(memory_limit)
        ));
        FitStatus::Rejected
    }
}

fn standalone_weight_coverage(
    model: &ModelProfile,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> bool {
    match model.weight_coverage {
        WeightCoverage::Full | WeightCoverage::Unknown => true,
        WeightCoverage::PartialTransformer {
            present_layers,
            expected_layers,
        } => {
            reasons.push(format!(
                "GGUF tensor coverage is partial ({present_layers}/{expected_layers} transformer blocks)"
            ));
            warnings.push(
                "partial GGUF artifacts are not ranked as standalone local model files".into(),
            );
            false
        }
        WeightCoverage::MetadataOnly => {
            reasons.push(
                "GGUF has model metadata but no standalone transformer weight coverage".into(),
            );
            warnings.push(
                "metadata-only or tokenizer/package GGUF artifacts are not standalone model candidates"
                    .into(),
            );
            false
        }
    }
}

fn memory_limit_with_margin(usable_memory_bytes: u64, safety_margin: f32) -> u64 {
    let margin = safety_margin.clamp(0.0, 0.9);
    (usable_memory_bytes as f32 * (1.0 - margin)) as u64
}

fn memory_score(runtime_bytes: u64, memory_limit: u64) -> f32 {
    if memory_limit == 0 || runtime_bytes > memory_limit {
        return 0.0;
    }
    let headroom = (memory_limit - runtime_bytes) as f32 / memory_limit as f32;
    (0.35 + headroom * 1.30).clamp(0.0, 1.0)
}

fn context_score(
    model: &ModelProfile,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> f32 {
    let required = required_context_tokens(config);
    let Some(native) = model.context_length else {
        warnings.push("model native context length is unknown".into());
        return 0.50;
    };
    if native >= required {
        reasons.push(format!(
            "native context {native} meets requested {required} tokens"
        ));
        return rope_context_penalty(model, required, warnings);
    }
    warnings.push(format!(
        "native context {native} is below requested {required} tokens"
    ));
    (native as f32 / required as f32).clamp(0.0, 0.80)
}

fn rope_context_penalty(model: &ModelProfile, required: u32, warnings: &mut Vec<String>) -> f32 {
    if model
        .rope
        .original_context_length
        .is_some_and(|original| original < required)
        && model.rope.finetuned != Some(true)
    {
        warnings.push("requested context appears to rely on unconfirmed rope scaling".into());
        return 0.80;
    }
    1.0
}

fn decode_score(estimated_decode_tps: Option<f32>, config: &SelectionConfig) -> f32 {
    if config.weights.decode == 0.0 {
        return 1.0;
    }
    let Some(tps) = estimated_decode_tps else {
        return 0.0;
    };
    let minimum = config
        .workload
        .preferences
        .minimum_decode_tps
        .unwrap_or(1.0);
    let preferred = config
        .workload
        .preferences
        .preferred_decode_tps
        .unwrap_or(minimum.max(1.0));
    if tps < minimum {
        return (tps / minimum * 0.40).clamp(0.0, 0.40);
    }
    (0.40 + (tps - minimum) / (preferred - minimum).max(1.0) * 0.60).clamp(0.0, 1.0)
}

fn prefill_score(model: &ModelProfile, config: &SelectionConfig, budget: &ExecutionBudget) -> f32 {
    let active = match model.architecture_class {
        ModelArchitectureClass::Embedding | ModelArchitectureClass::RerankerOrClassifier => {
            resident_weight_bytes(model)
        }
        ModelArchitectureClass::SparseMoeTransformer => active_moe_weight_bytes(model),
        _ => resident_weight_bytes(model),
    };
    let bandwidth = budget
        .memory_bandwidth_bytes_per_sec
        .unwrap_or(120_000_000_000) as f32
        * decode_bandwidth_efficiency(budget, config);
    if active == 0 {
        return 0.50;
    }
    let pressure = active as f32 / bandwidth.max(1.0);
    (1.0 / (1.0 + pressure)).clamp(0.0, 1.0)
}

fn workload_score(
    model: &ModelProfile,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> (f32, bool) {
    let requirements = &config.workload.requirements;
    let checks = [
        (
            requirements.chat_template,
            has(model, CapabilityEvidence::ChatTemplatePresent),
            "chat template",
        ),
        (
            requirements.system_messages,
            has(model, CapabilityEvidence::SystemRoleInChatTemplate),
            "system-message template support",
        ),
        (
            requirements.tool_calling,
            has(model, CapabilityEvidence::ToolUseTemplateMarkers),
            "tool-call template markers",
        ),
        (
            requirements.fill_in_middle,
            has(model, CapabilityEvidence::FillInMiddleTokensPresent),
            "fill-in-middle tokens",
        ),
        (
            requirements.embeddings,
            has(model, CapabilityEvidence::EmbeddingModel),
            "embedding model evidence",
        ),
        (
            requirements.reranking,
            has(model, CapabilityEvidence::ClassifierOrReranker),
            "reranker/classifier evidence",
        ),
        (
            requirements.vision,
            has(model, CapabilityEvidence::MultimodalProjector),
            "vision/multimodal evidence",
        ),
    ];

    let mut total = 0.0f32;
    let mut weight = 0.0f32;
    let mut reject = false;
    for (requirement, present, label) in checks {
        let (score, check_weight, failed) = requirement_score(requirement, present);
        if check_weight > 0.0 {
            if present {
                reasons.push(format!("workload evidence matched: {label}"));
            } else if requirement == Requirement::Required {
                warnings.push(format!("required workload evidence missing: {label}"));
            }
        }
        total += score * check_weight;
        weight += check_weight;
        reject |= failed;
    }

    let tag_score = explicit_tag_score(model, config);
    total += tag_score * 0.5;
    weight += 0.5;

    let score = if weight == 0.0 { 0.70 } else { total / weight };
    (score.clamp(0.0, 1.0), reject)
}

fn requirement_score(requirement: Requirement, present: bool) -> (f32, f32, bool) {
    match requirement {
        Requirement::Required => (if present { 1.0 } else { 0.0 }, 1.0, !present),
        Requirement::Preferred => (if present { 1.0 } else { 0.45 }, 0.75, false),
        Requirement::Neutral => (0.70, 0.0, false),
        Requirement::Penalize => (if present { 0.25 } else { 0.80 }, 0.50, false),
        Requirement::Reject => (if present { 0.0 } else { 0.80 }, 1.0, present),
    }
}

fn explicit_tag_score(model: &ModelProfile, config: &SelectionConfig) -> f32 {
    let tags = model
        .capability_evidence
        .iter()
        .filter_map(|evidence| match evidence {
            CapabilityEvidence::ExplicitGeneralTag(tag) => Some(tag.to_ascii_lowercase()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if tags.is_empty() {
        return 0.55;
    }
    let wanted = match config.workload.task {
        WorkloadTask::Coding => ["code", "coding", "programming", "fim"].as_slice(),
        WorkloadTask::ToolCalling => ["tool", "function", "agent"].as_slice(),
        WorkloadTask::Embedding => ["embedding", "sentence-transformers"].as_slice(),
        WorkloadTask::Reranking => ["rerank", "reranker", "classifier"].as_slice(),
        WorkloadTask::Summarization => ["summarization", "summary"].as_slice(),
        _ => ["chat", "instruct"].as_slice(),
    };
    if tags
        .iter()
        .any(|tag| wanted.iter().any(|wanted| tag.contains(wanted)))
    {
        1.0
    } else {
        0.60
    }
}

fn has(model: &ModelProfile, evidence: CapabilityEvidence) -> bool {
    model.capability_evidence.contains(&evidence)
}

fn total_score(
    weights: ScoreWeights,
    memory_score: f32,
    context_score: f32,
    decode_score: f32,
    prefill_score: f32,
    workload_score: f32,
    fit_status: FitStatus,
) -> f32 {
    if fit_status == FitStatus::Rejected {
        return 0.0;
    }
    let weight_sum =
        weights.memory + weights.context + weights.decode + weights.prefill + weights.workload;
    if weight_sum <= 0.0 {
        return 0.0;
    }
    let score = weights.memory * memory_score
        + weights.context * context_score
        + weights.decode * decode_score
        + weights.prefill * prefill_score
        + weights.workload * workload_score;
    let status_factor = match fit_status {
        FitStatus::FitsLocal => 1.0,
        FitStatus::FitsWithWarning => 0.85,
        FitStatus::Rejected => 0.0,
    };
    (score / weight_sum * status_factor).clamp(0.0, 1.0)
}

fn target_context_tokens(model: &ModelProfile, config: &SelectionConfig) -> u32 {
    let required = required_context_tokens(config);
    let model_context = model.context_length.unwrap_or(required);
    round_up_context(required.min(model_context.max(required.min(4_096))))
}

fn required_context_tokens(config: &SelectionConfig) -> u32 {
    let from_requirements = config.workload.requirements.min_context_tokens;
    let from_interaction = config.workload.interaction.expected_prompt_tokens;
    from_requirements
        .into_iter()
        .chain(from_interaction)
        .max()
        .unwrap_or(4_096)
        .max(1)
}

fn round_up_context(tokens: u32) -> u32 {
    tokens.div_ceil(256) * 256
}

fn uses_transformer_kv_cache(class: ModelArchitectureClass) -> bool {
    matches!(
        class,
        ModelArchitectureClass::DenseTransformer
            | ModelArchitectureClass::SparseMoeTransformer
            | ModelArchitectureClass::Unknown
    )
}

fn add_architecture_warnings(model: &ModelProfile, warnings: &mut Vec<String>) {
    match model.architecture_class {
        ModelArchitectureClass::SparseMoeTransformer => warnings.push(
            "MoE decode estimate uses active experts, but resident memory may still require all experts"
                .into(),
        ),
        ModelArchitectureClass::Unknown => {
            warnings.push("unknown architecture; estimates use conservative full-tensor assumptions".into());
        }
        ModelArchitectureClass::RecurrentOrStateSpace => warnings.push(
            "recurrent/state-space architecture has approximate context and decode estimates".into(),
        ),
        _ => {}
    }
}

fn add_decode_estimate_reason(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
) {
    if !uses_transformer_kv_cache(model.architecture_class) {
        return;
    }
    let matmul = &model.tensor_matmul;
    let grouped_matmul_bytes = matmul
        .attention
        .bytes
        .saturating_add(matmul.feed_forward.bytes)
        .saturating_add(matmul.output.bytes)
        .saturating_add(matmul.expert_feed_forward.bytes);
    if grouped_matmul_bytes > 0 {
        reasons.push(format!(
            "decode estimate uses GGUF matmul groups ({:.1} GiB attention, {:.1} GiB FFN, {:.1} GiB output, {:.1} GiB expert FFN pool)",
            gib(matmul.attention.bytes),
            gib(matmul.feed_forward.bytes),
            gib(matmul.output.bytes),
            gib(matmul.expert_feed_forward.bytes)
        ));
        if let Some(flops) = active_decode_flops_per_token(model) {
            reasons.push(format!(
                "decode estimate includes {:.1} GFLOP/token matmul compute floor from GGUF tensor shapes",
                flops as f32 / 1_000_000_000.0
            ));
        }
        add_matmul_shape_reason(matmul, reasons);
    } else if model.tensor_matmul.base_bytes > 0 || model.tensor_matmul.expert_bytes > 0 {
        reasons.push(format!(
            "decode estimate uses GGUF matmul tensors ({:.1} GiB base, {:.1} GiB expert pool) and active MoE experts when present",
            gib(model.tensor_matmul.base_bytes),
            gib(model.tensor_matmul.expert_bytes)
        ));
        if let Some(flops) = active_decode_flops_per_token(model) {
            reasons.push(format!(
                "decode estimate includes {:.1} GFLOP/token matmul compute floor from GGUF tensor shapes",
                flops as f32 / 1_000_000_000.0
            ));
        }
        add_matmul_shape_reason(matmul, reasons);
    } else if tensor_groups_available(model.tensor_group_bytes) {
        reasons.push(
            "decode estimate uses GGUF tensor groups for attention, FFN, experts, output, and KV pressure"
                .into(),
        );
    }
    let fixed_ms = fixed_decode_overhead_ms(budget, config);
    if measured_gpu_budget(budget) {
        if let Some(grouped) = grouped_decode_cost(model, budget, config) {
            reasons.push(format!(
                "decode estimate sums source-shaped GGML groups: {:.1} GiB through shape-matched probes and {:.1} GiB through measured fallback bandwidth",
                gib(grouped.probed_bytes),
                gib(grouped.fallback_bytes)
            ));
        } else if let Some((probe, distance)) =
            selected_decode_kernel_probe_with_distance(model, budget)
        {
            if distance <= MAX_REPRESENTATIVE_DECODE_PROBE_LOG_DISTANCE {
                reasons.push(format!(
                    "decode estimate uses measured {} probe for {} tensors ({:.1} GB/s, shape-representative {}x{} row)",
                    probe.name, probe.tensor_type, probe.effective_gbps, probe.rows, probe.cols
                ));
            } else {
                reasons.push(format!(
                    "decode estimate found measured {} probe for {} tensors ({:.1} GB/s, closest {}x{} row) but it is not shape-representative",
                    probe.name, probe.tensor_type, probe.effective_gbps, probe.rows, probe.cols
                ));
            }
        } else if let Some(bytes_per_sec) = budget.decode_effective_bandwidth_bytes_per_sec {
            reasons.push(format!(
                "decode estimate uses measured decode-shaped bandwidth ({:.1} GB/s)",
                bytes_per_sec as f32 / 1_000_000_000.0
            ));
        }
    }
    let arch_ms = architecture_decode_overhead_ms(model, budget, config);
    let runtime_ms = measured_decode_runtime_overhead_ms(budget);
    let graph_ms = measured_decode_graph_overhead_ms(model, budget);
    let sampler_ms = sampled_decode_sampler_ms(model, budget);
    let shape_adjustment = if grouped_decode_cost(model, budget, config).is_none() {
        active_decode_bytes_per_token(model, config)
            .map(|bytes| (decode_shape_bandwidth_factor(model, bytes), graph_ms))
    } else {
        None
    };
    if arch_ms > 0.0 || runtime_ms > 0.0 || graph_ms > 0.0 || sampler_ms > 0.0 {
        reasons.push(format!(
            "decode estimate adds {:.1} ms/token backend overhead, {:.1} ms/token measured runtime overhead, {:.1} ms/token measured graph overhead, {:.1} ms/token sampled decode overhead, and {:.1} ms/token architecture overhead from GGUF metadata",
            fixed_ms, runtime_ms, graph_ms, sampler_ms, arch_ms
        ));
    } else {
        reasons.push(format!(
            "decode estimate adds {:.1} ms/token measured backend overhead once per generated token",
            fixed_ms
        ));
    }
    match shape_adjustment {
        Some((shape_factor, graph_ms)) if shape_factor != 1.0 || graph_ms > 0.0 => {
            reasons.push(format!(
                "decode estimate applies {:.2}x shape bandwidth factor and {:.1} ms/token measured graph overhead from GGUF layer/matmul shape metadata",
                shape_factor, graph_ms
            ));
        }
        _ => {}
    }
}

fn add_matmul_shape_reason(matmul: &crate::TensorMatmulProfile, reasons: &mut Vec<String>) {
    let ffn = matmul.feed_forward.shape;
    if ffn.logical_matrix_count == 0 {
        return;
    }
    reasons.push(format!(
        "decode estimate uses FFN matmul shape summary ({} logical matrices, avg {}x{}, width range {}..{} by {}..{})",
        ffn.logical_matrix_count,
        ffn.weighted_avg_output_width,
        ffn.weighted_avg_input_width,
        ffn.min_output_width,
        ffn.max_output_width,
        ffn.min_input_width,
        ffn.max_input_width
    ));
}

fn add_prefill_estimate_reason(
    estimated_first_token_ms: Option<f32>,
    config: &SelectionConfig,
    reasons: &mut Vec<String>,
) {
    let Some(estimated_first_token_ms) = estimated_first_token_ms else {
        return;
    };
    let prompt_tokens = config
        .workload
        .interaction
        .expected_prompt_tokens
        .unwrap_or_default();
    reasons.push(format!(
        "prefill estimate predicts {:.0} ms first-token latency for about {prompt_tokens} prompt tokens",
        estimated_first_token_ms
    ));
}

fn estimate_confidence(
    model: &ModelProfile,
    budget: &ExecutionBudget,
    fit_status: FitStatus,
) -> EstimateConfidence {
    if fit_status == FitStatus::Rejected {
        return EstimateConfidence::Low;
    }
    if model.weight_coverage != WeightCoverage::Full {
        return EstimateConfidence::Low;
    }
    if model.architecture_class == ModelArchitectureClass::Unknown
        || budget.memory_bandwidth_bytes_per_sec.is_none()
        || (measured_gpu_budget(budget)
            && budget.decode_effective_bandwidth_bytes_per_sec.is_none())
    {
        return EstimateConfidence::Low;
    }
    // In this crate, `High` means the predicted tok/s is expected to land
    // within about +/-10% of observed local decode throughput. A measured GPU
    // benchmark plus GGUF metadata is necessary evidence, but validation has
    // shown it is not sufficient by itself: even architecture-matched GGML
    // composite probes can under-predict real llama.cpp/Skippy decode when the
    // model graph, scheduler fusion, or runtime path differs from the probe.
    //
    // Keep these recommendations at `Medium` until the estimator has a
    // validation-backed, falsifiable rule that reaches that +/-10% bar without
    // using model names, observed throughput from the target run, or
    // backend-specific fitted constants.
    EstimateConfidence::Medium
}

fn compare_recommendations(left: &ModelRecommendation, right: &ModelRecommendation) -> Ordering {
    status_rank(left.fit_status)
        .cmp(&status_rank(right.fit_status))
        .then_with(|| compare_f32_desc(left.total_score, right.total_score))
        .then_with(|| {
            compare_option_f32_desc(
                left.estimated_decode_tokens_per_sec,
                right.estimated_decode_tokens_per_sec,
            )
        })
        .then_with(|| compare_f32_desc(left.context_score, right.context_score))
        .then_with(|| {
            left.estimated_runtime_memory_bytes
                .cmp(&right.estimated_runtime_memory_bytes)
        })
        .then_with(|| left.source.id.cmp(&right.source.id))
}

fn compare_execution_budget_recommendations(
    left: &ModelRecommendation,
    right: &ModelRecommendation,
) -> Ordering {
    // This comparator is deliberately different from cross-model ranking.
    //
    // When we are choosing *how this one model should run on this one machine*,
    // memory headroom should decide only after the execution path is viable and
    // the estimated throughput is known. Otherwise a CPU budget with lots of
    // spare RAM can beat a measured GPU budget, causing the model's published
    // fit estimate to describe the wrong execution path. The Qwen2.5-Coder 7B
    // validation run caught exactly that shape: the model fit in GPU VRAM, but
    // the old comparator selected the CPU memory budget because its memory
    // score was higher, dropping the predicted decode rate by almost an order of
    // magnitude.
    //
    // Also prefer estimates backed by measured hardware facts over estimates
    // that came from fallback bandwidth constants. A fallback CPU budget should
    // not outrank a measured Metal/CUDA/ROCm budget just because the fallback
    // has no measured graph overhead. If a future CPU has measured bandwidth
    // and genuinely predicts faster decode, it can still win on the same rule.
    //
    // This is not a CUDA/Metal/backend preference and it is not model-specific.
    // It says: for a single model, among budgets with the same fit status and
    // estimate evidence quality, prefer the budget whose metadata-only estimate
    // predicts faster decode.
    status_rank(left.fit_status)
        .cmp(&status_rank(right.fit_status))
        .then_with(|| estimate_evidence_rank(left).cmp(&estimate_evidence_rank(right)))
        .then_with(|| {
            compare_option_f32_desc(
                left.estimated_decode_tokens_per_sec,
                right.estimated_decode_tokens_per_sec,
            )
        })
        .then_with(|| compare_f32_desc(left.total_score, right.total_score))
        .then_with(|| compare_f32_desc(left.memory_score, right.memory_score))
        .then_with(|| {
            left.estimated_runtime_memory_bytes
                .cmp(&right.estimated_runtime_memory_bytes)
        })
        .then_with(|| left.source.id.cmp(&right.source.id))
}

fn estimate_evidence_rank(recommendation: &ModelRecommendation) -> u8 {
    match recommendation.estimate_confidence {
        EstimateConfidence::High => 0,
        EstimateConfidence::Medium => 1,
        EstimateConfidence::Low => 2,
    }
}

fn status_rank(status: FitStatus) -> u8 {
    match status {
        FitStatus::FitsLocal => 0,
        FitStatus::FitsWithWarning => 1,
        FitStatus::Rejected => 2,
    }
}

fn compare_f32_desc(left: f32, right: f32) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

fn compare_option_f32_desc(left: Option<f32>, right: Option<f32>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => compare_f32_desc(left, right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn gib(bytes: u64) -> f32 {
    bytes as f32 / 1024.0 / 1024.0 / 1024.0
}
