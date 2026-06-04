use crate::*;
use mesh_llm_gpu_bench::DecodeKernelProbe;

const GIB: u64 = 1024 * 1024 * 1024;

fn m1_ultra() -> HardwareProfile {
    HardwareProfile {
        memory: MemoryProfile {
            total_system_bytes: Some(128 * GIB),
            available_system_bytes: Some(110 * GIB),
            total_unified_bytes: Some(128 * GIB),
            available_unified_bytes: Some(110 * GIB),
        },
        accelerators: vec![AcceleratorProfile {
            name: Some("Apple M1 Ultra".into()),
            kind: AcceleratorKind::IntegratedGpu,
            backend: BackendKind::Metal,
            total_memory_bytes: Some(128 * GIB),
            available_memory_bytes: Some(110 * GIB),
            memory_bandwidth_bytes_per_sec: Some(800_000_000_000),
            decode_effective_bandwidth_bytes_per_sec: Some(320_000_000_000),
            decode_fixed_overhead_ms: Some(1.25),
            decode_runtime_overhead_ms: None,
            post_prefill_decode_overhead_ms: None,
            bandwidth_source: MeasurementSource::Measured,
            benchmark_noise_pct: Some(1.0),
            bandwidth_efficiency_pct: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            prefill_matmul_tflops_fp16: None,
            prefill_ubatch_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            sampler_history_us_per_token: None,
            sampler_vocab_us_per_token: None,
            decode_kernel_probes: Vec::new(),
            unified_memory: true,
        }],
        cpu: CpuProfile {
            physical_cores: Some(20),
            logical_cores: Some(20),
            memory_bandwidth_bytes_per_sec: Some(200_000_000_000),
            compute_tflops_fp16: None,
            post_prefill_decode_overhead_ms: None,
            prefill_matmul_tflops_fp16: None,
            prefill_ubatch_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            sampler_history_us_per_token: None,
            sampler_vocab_us_per_token: None,
        },
    }
}

fn discrete_cuda_16g() -> HardwareProfile {
    HardwareProfile {
        memory: MemoryProfile {
            total_system_bytes: Some(64 * GIB),
            available_system_bytes: Some(48 * GIB),
            total_unified_bytes: None,
            available_unified_bytes: None,
        },
        accelerators: vec![AcceleratorProfile {
            name: Some("Measured CUDA GPU".into()),
            kind: AcceleratorKind::DiscreteGpu,
            backend: BackendKind::Cuda,
            total_memory_bytes: Some(16 * GIB),
            available_memory_bytes: Some(15 * GIB),
            memory_bandwidth_bytes_per_sec: Some(900_000_000_000),
            decode_effective_bandwidth_bytes_per_sec: Some(850_000_000_000),
            decode_fixed_overhead_ms: Some(0.002),
            decode_runtime_overhead_ms: None,
            post_prefill_decode_overhead_ms: None,
            bandwidth_source: MeasurementSource::Measured,
            benchmark_noise_pct: Some(0.5),
            bandwidth_efficiency_pct: Some(90.0),
            compute_tflops_fp32: None,
            compute_tflops_fp16: Some(50.0),
            prefill_matmul_tflops_fp16: None,
            prefill_ubatch_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            sampler_history_us_per_token: None,
            sampler_vocab_us_per_token: None,
            decode_kernel_probes: Vec::new(),
            unified_memory: false,
        }],
        cpu: CpuProfile {
            physical_cores: Some(8),
            logical_cores: Some(16),
            memory_bandwidth_bytes_per_sec: None,
            compute_tflops_fp16: None,
            post_prefill_decode_overhead_ms: None,
            prefill_matmul_tflops_fp16: None,
            prefill_ubatch_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            sampler_history_us_per_token: None,
            sampler_vocab_us_per_token: None,
        },
    }
}

fn dense_model(id: &str, bytes: u64, layers: u32, hidden: u32, context: u32) -> ModelProfile {
    let attention_bytes = bytes / 3;
    let feed_forward_bytes = bytes / 2;
    let output_bytes = bytes / 12;
    ModelProfile {
        source: ModelSource {
            id: id.into(),
            path: None,
            metadata_name: None,
        },
        architecture: Some("llama".into()),
        architecture_class: ModelArchitectureClass::DenseTransformer,
        weight_coverage: WeightCoverage::Full,
        file_size_bytes: bytes,
        tensor_bytes: Some(bytes),
        base_resident_bytes: Some(bytes),
        expert_tensor_bytes: Some(0),
        tensor_group_bytes: TensorGroupBytes {
            attention_bytes,
            feed_forward_bytes,
            expert_feed_forward_bytes: 0,
            embedding_bytes: bytes / 12,
            embedding_type_bytes: TensorTypeBytes {
                q4_k_bytes: bytes / 12,
                ..TensorTypeBytes::default()
            },
            output_bytes,
            normalization_bytes: bytes / 100,
            other_bytes: bytes
                .saturating_sub(bytes / 3)
                .saturating_sub(bytes / 2)
                .saturating_sub(bytes / 12)
                .saturating_sub(bytes / 12)
                .saturating_sub(bytes / 100),
        },
        tensor_matmul: TensorMatmulProfile {
            base_bytes: attention_bytes + feed_forward_bytes + output_bytes,
            expert_bytes: 0,
            base_flops_per_token: 0,
            expert_flops_per_token: 0,
            base_type_bytes: TensorTypeBytes {
                q4_k_bytes: attention_bytes + feed_forward_bytes + output_bytes,
                ..TensorTypeBytes::default()
            },
            expert_type_bytes: TensorTypeBytes::default(),
            attention: synthetic_matmul_group(attention_bytes, layers * 4, hidden, hidden),
            feed_forward: synthetic_matmul_group(
                feed_forward_bytes,
                layers * 3,
                hidden,
                hidden * 4,
            ),
            output: synthetic_matmul_group(output_bytes, 1, hidden, hidden),
            expert_feed_forward: TensorMatmulGroupProfile::default(),
        },
        dense_graph_features: DenseGraphFeatures::default(),
        recurrent_attention: RecurrentAttentionProfile::default(),
        parameter_count: None,
        quantization: Some("Q4_K_M".into()),
        layer_count: Some(layers),
        hidden_size: Some(hidden),
        ffn_size: Some(hidden * 4),
        attention_heads: Some(32),
        kv_heads: Some(8),
        key_length: Some(hidden / 32),
        value_length: Some(hidden / 8),
        context_length: Some(context),
        expert_count: None,
        expert_used_count: None,
        rope: RopeProfile::default(),
        tokenizer: TokenizerProfile {
            model: Some("gpt2".into()),
            vocab_size: Some(32_000),
            chat_template_available: true,
        },
        capability_evidence: vec![
            CapabilityEvidence::ChatTemplatePresent,
            CapabilityEvidence::SystemRoleInChatTemplate,
            CapabilityEvidence::NativeContextAtLeast(context),
        ],
    }
}

fn qwen3_30b_a3b_q4_moe() -> ModelProfile {
    let file_bytes = 18_556_686_912;
    let attention_bytes = 1_700_000_000;
    let feed_forward_bytes = 900_000_000;
    let expert_bytes = file_bytes - attention_bytes - feed_forward_bytes - 800_000_000;
    let output_bytes = 400_000_000;
    ModelProfile {
        source: ModelSource {
            id: "unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M".into(),
            path: None,
            metadata_name: Some("Qwen3-30B-A3B-Q4_K_M.gguf".into()),
        },
        architecture: Some("qwen3moe".into()),
        architecture_class: ModelArchitectureClass::SparseMoeTransformer,
        weight_coverage: WeightCoverage::Full,
        file_size_bytes: file_bytes,
        tensor_bytes: Some(file_bytes),
        base_resident_bytes: Some(file_bytes.saturating_sub(expert_bytes)),
        expert_tensor_bytes: Some(expert_bytes),
        tensor_group_bytes: TensorGroupBytes {
            attention_bytes,
            feed_forward_bytes,
            expert_feed_forward_bytes: expert_bytes,
            embedding_bytes: 300_000_000,
            embedding_type_bytes: TensorTypeBytes {
                q4_k_bytes: 300_000_000,
                ..TensorTypeBytes::default()
            },
            output_bytes,
            normalization_bytes: 100_000_000,
            other_bytes: file_bytes
                .saturating_sub(attention_bytes)
                .saturating_sub(feed_forward_bytes)
                .saturating_sub(expert_bytes)
                .saturating_sub(300_000_000)
                .saturating_sub(output_bytes)
                .saturating_sub(100_000_000),
        },
        tensor_matmul: TensorMatmulProfile {
            base_bytes: attention_bytes + feed_forward_bytes + output_bytes,
            expert_bytes,
            base_flops_per_token: 0,
            expert_flops_per_token: 0,
            base_type_bytes: TensorTypeBytes {
                q4_k_bytes: attention_bytes + feed_forward_bytes + output_bytes,
                ..TensorTypeBytes::default()
            },
            expert_type_bytes: TensorTypeBytes {
                q4_k_bytes: expert_bytes,
                ..TensorTypeBytes::default()
            },
            attention: synthetic_matmul_group(attention_bytes, 48 * 4, 2048, 2048),
            feed_forward: synthetic_matmul_group(feed_forward_bytes, 48 * 3, 2048, 6144),
            output: synthetic_matmul_group(output_bytes, 1, 2048, 2048),
            expert_feed_forward: synthetic_matmul_group(expert_bytes, 48 * 128 * 3, 2048, 768),
        },
        dense_graph_features: DenseGraphFeatures::default(),
        recurrent_attention: RecurrentAttentionProfile::default(),
        parameter_count: None,
        quantization: Some("Q4_K_M".into()),
        layer_count: Some(48),
        hidden_size: Some(2048),
        ffn_size: Some(6144),
        attention_heads: Some(32),
        kv_heads: Some(4),
        key_length: Some(128),
        value_length: Some(128),
        context_length: Some(40_960),
        expert_count: Some(128),
        expert_used_count: Some(8),
        rope: RopeProfile::default(),
        tokenizer: TokenizerProfile {
            model: Some("gpt2".into()),
            vocab_size: Some(151_936),
            chat_template_available: true,
        },
        capability_evidence: vec![
            CapabilityEvidence::ChatTemplatePresent,
            CapabilityEvidence::SystemRoleInChatTemplate,
            CapabilityEvidence::NativeContextAtLeast(40_960),
        ],
    }
}

fn synthetic_matmul_group(
    bytes: u64,
    logical_matrix_count: u32,
    input_width: u32,
    output_width: u32,
) -> TensorMatmulGroupProfile {
    TensorMatmulGroupProfile {
        bytes,
        type_bytes: TensorTypeBytes {
            q4_k_bytes: bytes,
            ..TensorTypeBytes::default()
        },
        shape: MatmulShapeProfile {
            tensor_count: u64::from(logical_matrix_count),
            logical_matrix_count: u64::from(logical_matrix_count),
            total_elements: u64::from(logical_matrix_count)
                .saturating_mul(u64::from(input_width))
                .saturating_mul(u64::from(output_width)),
            min_input_width: u64::from(input_width.min(output_width)),
            max_input_width: u64::from(input_width.max(output_width)),
            min_output_width: u64::from(input_width.min(output_width)),
            max_output_width: u64::from(input_width.max(output_width)),
            weighted_avg_input_width: u64::from(input_width),
            weighted_avg_output_width: u64::from(output_width),
        },
        ..TensorMatmulGroupProfile::default()
    }
}

#[test]
fn dense_14b_beats_dense_70b_for_latency_sensitive_chat() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let models = vec![
        dense_model("dense-14b", 9 * GIB, 40, 5120, 32_768),
        dense_model("dense-70b", 42 * GIB, 80, 8192, 32_768),
    ];

    let ranked = rank_models(&hardware, &models, &config);

    assert_eq!(ranked[0].source.id, "dense-14b");
    assert!(ranked[0].estimated_decode_tokens_per_sec > ranked[1].estimated_decode_tokens_per_sec);
}

#[test]
fn coding_agent_prefers_explicit_fim_and_tool_evidence() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::coding_agent(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut coding = dense_model("opaque-id-a", 18 * GIB, 48, 6144, 65_536);
    coding.capability_evidence.extend([
        CapabilityEvidence::ToolUseTemplateMarkers,
        CapabilityEvidence::FillInMiddleTokensPresent,
    ]);
    let plain = dense_model("opaque-id-b", 18 * GIB, 48, 6144, 65_536);

    let ranked = rank_models(&hardware, &[plain, coding], &config);

    assert_eq!(ranked[0].source.id, "opaque-id-a");
    assert!(ranked[0].workload_score > ranked[1].workload_score);
}

#[test]
fn embedding_workload_accepts_embedding_model_and_rejects_chat_model() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::embedding(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let chat = dense_model("chat", 5 * GIB, 32, 4096, 8_192);
    let mut embedding = dense_model("embedding", GIB, 12, 768, 512);
    embedding.architecture_class = ModelArchitectureClass::Embedding;
    embedding.capability_evidence = vec![CapabilityEvidence::EmbeddingModel];

    let ranked = rank_models(&hardware, &[chat, embedding], &config);

    assert_eq!(ranked[0].source.id, "embedding");
    assert_eq!(ranked[1].fit_status, FitStatus::Rejected);
}

#[test]
fn chat_workload_rejects_embedding_model() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut embedding = dense_model("embedding", GIB, 12, 768, 512);
    embedding.architecture_class = ModelArchitectureClass::Embedding;
    embedding.capability_evidence = vec![CapabilityEvidence::EmbeddingModel];

    let rec = score_model(&hardware, &embedding, &config);

    assert_eq!(rec.fit_status, FitStatus::Rejected);
}

#[test]
fn moe_decode_uses_active_expert_bytes() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut moe = dense_model("moe", 60 * GIB, 48, 6144, 32_768);
    moe.architecture_class = ModelArchitectureClass::SparseMoeTransformer;
    moe.base_resident_bytes = Some(12 * GIB);
    moe.expert_tensor_bytes = Some(48 * GIB);
    moe.tensor_group_bytes.attention_bytes = 8 * GIB;
    moe.tensor_group_bytes.feed_forward_bytes = 4 * GIB;
    moe.tensor_group_bytes.expert_feed_forward_bytes = 48 * GIB;
    moe.tensor_matmul.base_bytes = 12 * GIB;
    moe.tensor_matmul.expert_bytes = 48 * GIB;
    moe.tensor_matmul.attention.bytes = 8 * GIB;
    moe.tensor_matmul.feed_forward.bytes = 4 * GIB;
    moe.tensor_matmul.expert_feed_forward.bytes = 48 * GIB;
    moe.tensor_matmul.base_type_bytes = TensorTypeBytes {
        q4_k_bytes: 12 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.expert_type_bytes = TensorTypeBytes {
        q4_k_bytes: 48 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.attention.type_bytes = TensorTypeBytes {
        q4_k_bytes: 8 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.feed_forward.type_bytes = TensorTypeBytes {
        q4_k_bytes: 4 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.tensor_matmul.expert_feed_forward.type_bytes = TensorTypeBytes {
        q4_k_bytes: 48 * GIB,
        ..TensorTypeBytes::default()
    };
    moe.expert_count = Some(16);
    moe.expert_used_count = Some(2);

    let rec = score_model(&hardware, &moe, &config);

    assert!(rec.estimated_active_decode_bytes_per_token.unwrap() < 30 * GIB);
    assert!(
        rec.warnings
            .iter()
            .any(|warning| warning.contains("active experts"))
    );
}

#[test]
fn measured_moe_dispatch_overhead_uses_submission_cost() {
    let mut low_overhead = m1_ultra();
    low_overhead.accelerators[0].decode_fixed_overhead_ms = Some(0.002);
    let mut high_overhead = m1_ultra();
    high_overhead.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut moe = dense_model("measured-moe", 4 * GIB, 16, 2048, 4096);
    moe.architecture_class = ModelArchitectureClass::SparseMoeTransformer;
    moe.expert_count = Some(64);
    moe.expert_used_count = Some(8);
    moe.tensor_group_bytes.expert_feed_forward_bytes = 3 * GIB;
    moe.tensor_matmul.expert_bytes = 3 * GIB;
    moe.tensor_matmul.expert_feed_forward.bytes = 3 * GIB;
    moe.tensor_matmul.expert_feed_forward.type_bytes = TensorTypeBytes {
        q4_k_bytes: 3 * GIB,
        ..TensorTypeBytes::default()
    };

    let low_rec = score_model(&low_overhead, &moe, &config);
    let high_rec = score_model(&high_overhead, &moe, &config);

    assert!(
        low_rec.estimated_decode_tokens_per_sec.unwrap()
            > high_rec.estimated_decode_tokens_per_sec.unwrap()
    );
}

#[test]
fn measured_decode_runtime_overhead_reduces_decode_throughput() {
    let without_runtime_overhead = m1_ultra();
    let mut with_runtime_overhead = without_runtime_overhead.clone();
    with_runtime_overhead.accelerators[0].decode_runtime_overhead_ms = Some(0.25);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("runtime-overhead-model", 4 * GIB, 16, 2048, 4096);

    let without_rec = score_model(&without_runtime_overhead, &model, &config);
    let with_rec = score_model(&with_runtime_overhead, &model, &config);

    assert!(
        without_rec.estimated_decode_tokens_per_sec.unwrap()
            > with_rec.estimated_decode_tokens_per_sec.unwrap()
    );
}

#[test]
fn moe_prefill_probe_is_upper_bound_not_free_speedup() {
    let mut without_probe = m1_ultra();
    without_probe.memory.available_system_bytes = None;
    without_probe.accelerators[0].compute_tflops_fp16 = Some(50.0);
    let mut with_probe = without_probe.clone();
    with_probe.accelerators[0].prefill_moe_matmul_tflops_fp16 = Some(1_000.0);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.workload.interaction.expected_prompt_tokens = Some(2048);
    config.weights = config.workload.default_weights();
    let mut moe = dense_model("measured-moe-prefill", 4 * GIB, 16, 2048, 4096);
    moe.architecture_class = ModelArchitectureClass::SparseMoeTransformer;
    moe.expert_count = Some(64);
    moe.expert_used_count = Some(8);
    moe.tensor_group_bytes.expert_feed_forward_bytes = 3 * GIB;
    moe.tensor_matmul.expert_bytes = 3 * GIB;
    moe.tensor_matmul.expert_flops_per_token = 12_000_000_000;

    let fallback = score_model(&without_probe, &moe, &config)
        .estimated_prefill_tokens_per_sec
        .expect("fallback prefill estimate should exist");
    let measured = score_model(&with_probe, &moe, &config)
        .estimated_prefill_tokens_per_sec
        .expect("measured MoE prefill estimate should exist");

    assert!(measured <= fallback * 1.001);
}

#[test]
fn filename_like_identifier_does_not_create_coding_suitability() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::coding_agent(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model(
        "qwen-coder-tool-instruct-name-only.gguf",
        6 * GIB,
        32,
        4096,
        32_768,
    );
    let rec = score_model(&hardware, &model, &config);

    assert!(rec.workload_score < 0.75);
    assert!(
        !rec.reasons
            .iter()
            .any(|reason| reason.contains("fill-in-middle") || reason.contains("tool-call"))
    );
}

#[test]
fn oversized_dense_model_is_rejected_for_local_fit() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::summarization(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("dense-180b", 120 * GIB, 96, 12_288, 32_768);

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.fit_status, FitStatus::Rejected);
}

#[test]
fn partial_transformer_gguf_is_not_ranked_as_standalone_model() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut model = dense_model("stage-artifact", 4 * GIB, 36, 4096, 32_768);
    model.weight_coverage = WeightCoverage::PartialTransformer {
        present_layers: 18,
        expected_layers: 36,
    };

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.fit_status, FitStatus::Rejected);
    assert!(rec.reasons.iter().any(|reason| reason.contains("partial")));
}

#[test]
fn decode_estimate_reports_uncertainty_range() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("dense", 4 * GIB, 32, 4096, 32_768);

    let rec = score_model(&hardware, &model, &config);
    let point = rec
        .estimated_decode_tokens_per_sec
        .expect("decode estimate should exist");
    let range = rec
        .estimated_decode_tokens_per_sec_range
        .expect("decode range should exist");

    assert!(range.lower < point);
    assert!(range.upper > point);
}

#[test]
fn prefill_estimate_reports_first_token_latency_range() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("dense", 4 * GIB, 32, 4096, 32_768);

    let rec = score_model(&hardware, &model, &config);
    let point = rec
        .estimated_first_token_ms
        .expect("first-token estimate should exist");
    let range = rec
        .estimated_first_token_ms_range
        .expect("first-token range should exist");

    assert!(point > 0.0);
    assert!(range.lower_ms < point);
    assert!(range.upper_ms > point);
    assert!(rec.estimated_prefill_tokens_per_sec.unwrap() > 0.0);
}

#[test]
fn prefill_roofline_uses_measured_compute_for_wide_models() {
    let mut slow_compute = m1_ultra();
    slow_compute.memory.available_system_bytes = None;
    slow_compute.accelerators[0].compute_tflops_fp16 = Some(5.0);
    let mut fast_compute = m1_ultra();
    fast_compute.memory.available_system_bytes = None;
    fast_compute.accelerators[0].compute_tflops_fp16 = Some(25.0);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.workload.interaction.expected_prompt_tokens = Some(2048);
    config.weights = config.workload.default_weights();
    let mut model = dense_model("wide-prefill", 4 * GIB, 32, 4096, 32_768);
    model.tensor_matmul.base_flops_per_token = 12_000_000_000;
    model.tensor_matmul.attention.flops_per_token = 2_000_000_000;
    model.tensor_matmul.feed_forward.flops_per_token = 9_000_000_000;
    model.tensor_matmul.output.flops_per_token = 1_000_000_000;

    let slow = score_model(&slow_compute, &model, &config)
        .estimated_prefill_tokens_per_sec
        .expect("slow compute should produce prefill estimate");
    let fast = score_model(&fast_compute, &model, &config)
        .estimated_prefill_tokens_per_sec
        .expect("fast compute should produce prefill estimate");

    assert!(fast > slow);
}

#[test]
fn prefill_roofline_prefers_measured_ubatch_matmul_shape() {
    let mut square_only = m1_ultra();
    square_only.memory.available_system_bytes = None;
    square_only.cpu.memory_bandwidth_bytes_per_sec = None;
    square_only.accelerators[0].prefill_matmul_tflops_fp16 = Some(12.0);
    square_only.accelerators[0].prefill_ubatch_matmul_tflops_fp16 = None;
    let mut ubatch_measured = square_only.clone();
    ubatch_measured.accelerators[0].prefill_ubatch_matmul_tflops_fp16 = Some(1.0);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.workload.interaction.expected_prompt_tokens = Some(4096);
    config.weights = config.workload.default_weights();
    let mut model = dense_model("ubatch-prefill", 4 * GIB, 32, 4096, 32_768);
    model.tensor_matmul.base_flops_per_token = 12_000_000_000;
    model.tensor_matmul.attention.flops_per_token = 2_000_000_000;
    model.tensor_matmul.feed_forward.flops_per_token = 9_000_000_000;
    model.tensor_matmul.output.flops_per_token = 1_000_000_000;

    let square = score_model(&square_only, &model, &config)
        .estimated_prefill_tokens_per_sec
        .expect("square prefill estimate should exist");
    let ubatch = score_model(&ubatch_measured, &model, &config)
        .estimated_prefill_tokens_per_sec
        .expect("ubatch prefill estimate should exist");

    assert!(ubatch < square);
}

#[test]
fn decode_estimate_uses_measured_graph_overhead_for_deeper_shapes() {
    let mut hardware = m1_ultra();
    hardware.memory.available_system_bytes = None;
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let shallow = dense_model("shallow", 3 * GIB, 16, 4096, 32_768);
    let deep = dense_model("deep", 3 * GIB, 40, 4096, 32_768);

    let shallow_rec = score_model(&hardware, &shallow, &config);
    let deep_rec = score_model(&hardware, &deep, &config);

    assert!(deep_rec.estimated_decode_tokens_per_sec < shallow_rec.estimated_decode_tokens_per_sec);
}

#[test]
fn decode_estimate_charges_expanded_ffn_graph_stages_from_shape() {
    let mut hardware = m1_ultra();
    hardware.memory.available_system_bytes = None;
    hardware.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut compact_ffn = dense_model("compact-ffn", 4 * GIB, 28, 2048, 32_768);
    compact_ffn.ffn_size = Some(2048 * 2);
    compact_ffn.tensor_matmul.feed_forward.shape.max_input_width = 4096;
    compact_ffn
        .tensor_matmul
        .feed_forward
        .shape
        .max_output_width = 4096;
    compact_ffn
        .tensor_matmul
        .feed_forward
        .shape
        .weighted_avg_input_width = 2048;
    compact_ffn
        .tensor_matmul
        .feed_forward
        .shape
        .weighted_avg_output_width = 4096;
    let mut expanded_ffn = compact_ffn.clone();
    expanded_ffn.source.id = "expanded-ffn".into();
    expanded_ffn.ffn_size = Some(2048 * 4);
    let feed_forward_delta = compact_ffn.tensor_matmul.feed_forward.bytes;
    expanded_ffn.tensor_matmul.feed_forward.bytes += feed_forward_delta;
    expanded_ffn
        .tensor_matmul
        .feed_forward
        .type_bytes
        .q4_k_bytes += feed_forward_delta;
    expanded_ffn.tensor_matmul.base_bytes += feed_forward_delta;
    expanded_ffn.tensor_matmul.base_type_bytes.q4_k_bytes += feed_forward_delta;
    expanded_ffn.tensor_group_bytes.feed_forward_bytes += feed_forward_delta;
    expanded_ffn
        .tensor_matmul
        .feed_forward
        .shape
        .max_input_width = 8192;
    expanded_ffn
        .tensor_matmul
        .feed_forward
        .shape
        .max_output_width = 8192;

    let compact_rec = score_model(&hardware, &compact_ffn, &config);
    let expanded_rec = score_model(&hardware, &expanded_ffn, &config);

    assert!(
        expanded_rec.estimated_decode_tokens_per_sec.unwrap()
            < compact_rec.estimated_decode_tokens_per_sec.unwrap()
    );
}

#[test]
fn measured_gpu_bandwidth_uses_backend_neutral_efficiency() {
    let metal = m1_ultra();
    let mut cuda = metal.clone();
    cuda.accelerators[0].backend = BackendKind::Cuda;
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("portable", 4 * GIB, 32, 4096, 32_768);

    let metal_tps = score_model(&metal, &model, &config)
        .estimated_decode_tokens_per_sec
        .expect("measured metal estimate should exist");
    let cuda_tps = score_model(&cuda, &model, &config)
        .estimated_decode_tokens_per_sec
        .expect("measured cuda estimate should exist");

    assert!((metal_tps - cuda_tps).abs() < 0.001);
}

#[test]
fn budget_selection_prefers_faster_measured_gpu_over_cpu_headroom() {
    let hardware = discrete_cuda_16g();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("fits-gpu-and-cpu", 7 * GIB, 28, 3584, 32_768);

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.selected_backend, BackendKind::Cuda);
    assert!(
        !rec.warnings
            .iter()
            .any(|warning| warning.contains("memory bandwidth is missing"))
    );
    assert!(rec.estimated_decode_tokens_per_sec.unwrap() > 100.0);
}

#[test]
fn generation_workload_does_not_use_cpu_ram_as_discrete_gpu_fit() {
    let hardware = discrete_cuda_16g();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("too-large-for-vram", 18 * GIB, 48, 4096, 32_768);

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.selected_backend, BackendKind::Cuda);
    assert_eq!(rec.fit_status, FitStatus::Rejected);
    assert!(rec.estimated_runtime_memory_bytes > 15 * GIB);
}

#[test]
fn white_qwen3_moe_fixture_is_rejected_not_cpu_fit() {
    let hardware = discrete_cuda_16g();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = qwen3_30b_a3b_q4_moe();

    let rec = score_model(&hardware, &model, &config);

    assert_eq!(rec.selected_backend, BackendKind::Cuda);
    assert_eq!(rec.fit_status, FitStatus::Rejected);
    assert!(rec.estimated_runtime_memory_bytes > 19 * GIB);
    assert!(
        rec.warnings
            .iter()
            .any(|warning| warning.contains("MoE decode estimate uses active experts"))
    );
}

#[test]
fn q8_decode_uses_ggml_type_kernel_traffic() {
    let hardware = m1_ultra();
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let q4 = dense_model("q4", 8 * GIB, 32, 4096, 32_768);
    let mut q8 = q4.clone();
    q8.source.id = "q8".into();
    q8.quantization = Some("Q8_0".into());
    q8.file_size_bytes = 16 * GIB;
    q8.tensor_bytes = Some(16 * GIB);
    q8.base_resident_bytes = Some(16 * GIB);
    q8.tensor_group_bytes.attention_bytes *= 2;
    q8.tensor_group_bytes.feed_forward_bytes *= 2;
    q8.tensor_group_bytes.output_bytes *= 2;
    q8.tensor_group_bytes.embedding_bytes *= 2;
    q8.tensor_group_bytes.normalization_bytes *= 2;
    q8.tensor_group_bytes.other_bytes *= 2;
    q8.tensor_matmul.base_bytes *= 2;
    q8.tensor_matmul.base_type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.base_type_bytes.q8_0_bytes = q8.tensor_matmul.base_bytes;
    q8.tensor_matmul.attention.bytes *= 2;
    q8.tensor_matmul.feed_forward.bytes *= 2;
    q8.tensor_matmul.output.bytes *= 2;
    q8.tensor_matmul.attention.type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.feed_forward.type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.output.type_bytes.q4_k_bytes = 0;
    q8.tensor_matmul.attention.type_bytes.q8_0_bytes = q8.tensor_matmul.attention.bytes;
    q8.tensor_matmul.feed_forward.type_bytes.q8_0_bytes = q8.tensor_matmul.feed_forward.bytes;
    q8.tensor_matmul.output.type_bytes.q8_0_bytes = q8.tensor_matmul.output.bytes;

    let q4_rec = score_model(&hardware, &q4, &config);
    let q8_rec = score_model(&hardware, &q8, &config);
    let q4_active = q4_rec.estimated_active_decode_bytes_per_token.unwrap();
    let q8_active = q8_rec.estimated_active_decode_bytes_per_token.unwrap();

    assert!(q8_active > q4_active);
    assert!(q8_active > q4_active * 16 / 10);
    assert!(
        q8_rec.estimated_decode_tokens_per_sec.unwrap()
            < q4_rec.estimated_decode_tokens_per_sec.unwrap()
    );
}

#[test]
fn tied_output_projection_charges_embedding_bytes() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "ggml_decode_q4_k_llama_graph_l8_4096_16384".into(),
        tensor_type: "q4_k".into(),
        rows: 16_384,
        cols: 4096,
        batch_tokens: 1,
        graph_features: 0,
        effective_gbps: 180.0,
        tflops: Some(0.8),
        elapsed_ms: None,
        runs: 3,
    }];
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let explicit = dense_model("explicit-output", 8 * GIB, 32, 4096, 32_768);
    let mut tied = explicit.clone();
    tied.source.id = "tied-output".into();
    tied.tensor_matmul.base_bytes = tied
        .tensor_matmul
        .base_bytes
        .saturating_sub(tied.tensor_matmul.output.bytes);
    tied.tensor_matmul.base_type_bytes.q4_k_bytes = tied
        .tensor_matmul
        .base_type_bytes
        .q4_k_bytes
        .saturating_sub(tied.tensor_matmul.output.bytes);
    tied.tensor_matmul.output = TensorMatmulGroupProfile::default();
    tied.tensor_group_bytes.output_bytes = 0;

    let explicit_rec = score_model(&hardware, &explicit, &config);
    let tied_rec = score_model(&hardware, &tied, &config);
    let explicit_bytes = explicit_rec
        .estimated_active_decode_bytes_per_token
        .unwrap();
    let tied_bytes = tied_rec.estimated_active_decode_bytes_per_token.unwrap();

    assert_eq!(explicit_bytes, tied_bytes);
    assert!(
        tied_rec
            .decode_cost_breakdown
            .as_ref()
            .expect("decode cost breakdown")
            .groups
            .iter()
            .any(|group| {
                group.group == "output_matmul"
                    && group.tensor_type == "q4_k"
                    && group.resident_bytes == tied.tensor_group_bytes.embedding_bytes
            })
    );
}

#[test]
fn ggml_decode_kernel_probe_is_required_for_medium_confidence() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "decode_f16_matvec".into(),
        tensor_type: "f16".into(),
        rows: 4096,
        cols: 4096,
        batch_tokens: 1,
        graph_features: 0,
        effective_gbps: 240.0,
        tflops: Some(4.0),
        elapsed_ms: None,
        runs: 20,
    }];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();

    let q4 = dense_model("q4", 8 * GIB, 32, 4096, 32_768);
    let mut f16 = q4.clone();
    f16.source.id = "f16".into();
    f16.quantization = Some("F16".into());
    f16.tensor_matmul.base_type_bytes.q4_k_bytes = 0;
    f16.tensor_matmul.base_type_bytes.f16_bytes = f16.tensor_matmul.base_bytes;
    for group in [
        &mut f16.tensor_matmul.attention,
        &mut f16.tensor_matmul.feed_forward,
        &mut f16.tensor_matmul.output,
    ] {
        group.type_bytes.q4_k_bytes = 0;
        group.type_bytes.f16_bytes = group.bytes;
    }

    let f16_rec = score_model(&hardware, &f16, &config);
    let q4_rec = score_model(&hardware, &q4, &config);

    assert_ne!(f16_rec.estimate_confidence, EstimateConfidence::High);
    assert_ne!(q4_rec.estimate_confidence, EstimateConfidence::High);

    hardware.accelerators[0].decode_kernel_probes[0].name = "ggml_decode_f16_matvec".into();
    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_f16_llama_graph_ffn".into(),
            tensor_type: "f16".into(),
            rows: 16_384,
            cols: 4096,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 280.0,
            tflops: Some(4.5),
            elapsed_ms: None,
            runs: 20,
        });
    let f16_rec = score_model(&hardware, &f16, &config);
    let q4_rec = score_model(&hardware, &q4, &config);

    assert_eq!(f16_rec.estimate_confidence, EstimateConfidence::Medium);
    assert_ne!(q4_rec.estimate_confidence, EstimateConfidence::High);
    assert!(
        f16_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("metadata-only estimates are not yet validated"))
    );
    assert!(
        q4_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("dominant tensor type q4_k"))
    );
}

#[test]
fn decode_kernel_probe_must_match_dominant_matmul_shape() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_effective_bandwidth_bytes_per_sec = Some(400_000_000_000);
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "ggml_decode_q4_k_matvec_square".into(),
        tensor_type: "q4_k".into(),
        rows: 4096,
        cols: 4096,
        batch_tokens: 1,
        graph_features: 0,
        effective_gbps: 25.0,
        tflops: Some(0.1),
        elapsed_ms: None,
        runs: 20,
    }];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("q4", 8 * GIB, 32, 4096, 32_768);

    let off_shape_rec = score_model(&hardware, &model, &config);
    assert_ne!(off_shape_rec.estimate_confidence, EstimateConfidence::High);
    assert!(
        off_shape_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("shape-representative"))
    );
    assert!(
        off_shape_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("shape-representative"))
    );

    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_q4_k_matvec_ffn".into(),
            tensor_type: "q4_k".into(),
            rows: 16_384,
            cols: 4096,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 120.0,
            tflops: Some(0.5),
            elapsed_ms: None,
            runs: 20,
        });
    let matvec_rec = score_model(&hardware, &model, &config);
    assert_ne!(matvec_rec.estimate_confidence, EstimateConfidence::High);
    assert!(
        matvec_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("composite llama decode"))
    );
    assert!(
        (matvec_rec.estimated_decode_tokens_per_sec.unwrap()
            - off_shape_rec.estimated_decode_tokens_per_sec.unwrap())
        .abs()
            < f32::EPSILON
    );

    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_ffn".into(),
            tensor_type: "q4_k".into(),
            rows: 16_384,
            cols: 4096,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 120.0,
            tflops: Some(0.5),
            elapsed_ms: None,
            runs: 20,
        });
    let representative_rec = score_model(&hardware, &model, &config);
    assert_eq!(
        representative_rec.estimate_confidence,
        EstimateConfidence::Medium
    );
    assert!(
        representative_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("metadata-only estimates are not yet validated"))
    );
    assert!(
        representative_rec
            .reasons
            .iter()
            .any(|reason| reason.contains("source-shaped GGML groups"))
    );

    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_stack_4096_kv1024_16384_layers32".into(),
            tensor_type: "q4_k".into(),
            rows: 16_384,
            cols: 4096,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 160.0,
            tflops: Some(0.7),
            elapsed_ms: Some(0.9),
            runs: 20,
        });
    let stack_rec = score_model(&hardware, &model, &config);
    assert_eq!(stack_rec.estimate_confidence, EstimateConfidence::Medium);
    assert!(
        (stack_rec.estimated_decode_tokens_per_sec.unwrap()
            - representative_rec.estimated_decode_tokens_per_sec.unwrap())
        .abs()
            < f32::EPSILON
    );
}

#[test]
fn recurrent_attention_requires_linear_attention_graph_probe() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_effective_bandwidth_bytes_per_sec = Some(400_000_000_000);
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "ggml_decode_q4_k_llama_graph_l24_qknorm_postnorm_gqa_2048_kv512_6144".into(),
        tensor_type: "q4_k".into(),
        rows: 6144,
        cols: 2048,
        batch_tokens: 1,
        graph_features: mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_Q_NORM
            | mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_K_NORM
            | mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_POST_NORM,
        effective_gbps: 220.0,
        tflops: Some(0.4),
        elapsed_ms: Some(6.0),
        runs: 3,
    }];
    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut model = dense_model("recurrent", 3 * GIB, 24, 2048, 32_768);
    model.ffn_size = Some(8192);
    model.dense_graph_features = DenseGraphFeatures {
        attention_q_norm: true,
        attention_k_norm: true,
        attention_post_norm: true,
        feed_forward_post_norm: false,
    };
    model.attention_heads = Some(16);
    model.kv_heads = Some(4);
    model.key_length = Some(128);
    model.value_length = Some(128);
    let mut beta_projection = synthetic_matmul_group(18 * 2048 * 16, 18, 2048, 16);
    beta_projection.shape.min_output_width = 16;
    beta_projection.shape.max_output_width = 16;
    beta_projection.shape.weighted_avg_output_width = 16;
    let mut alpha_projection = synthetic_matmul_group(18 * 2048 * 16, 18, 2048, 16);
    alpha_projection.shape.min_output_width = 16;
    alpha_projection.shape.max_output_width = 16;
    alpha_projection.shape.weighted_avg_output_width = 16;
    model.recurrent_attention = RecurrentAttentionProfile {
        recurrent_layer_count: 18,
        qkv_projection: synthetic_matmul_group(18 * 2048 * 6144, 18, 2048, 6144),
        gate_projection: synthetic_matmul_group(18 * 2048 * 2048, 18, 2048, 2048),
        beta_projection,
        alpha_projection,
        output_projection: synthetic_matmul_group(18 * 2048 * 2048, 18, 2048, 2048),
    };

    let dense_probe_rec = score_model(&hardware, &model, &config);
    assert_ne!(
        dense_probe_rec.estimate_confidence,
        EstimateConfidence::High
    );
    assert!(
        dense_probe_rec
            .decode_cost_breakdown
            .as_ref()
            .expect("decode cost breakdown")
            .groups
            .iter()
            .all(|group| group.probe_name.as_deref()
                != Some("ggml_decode_q4_k_llama_graph_l24_qknorm_postnorm_gqa_2048_kv512_6144"))
    );
    assert!(
        dense_probe_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("linear-attention decode graph probe"))
    );

    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_q4_k_linear_attn_graph_r18_f6_qknorm_postnorm_h2048_qkv6144_gate2048_state16_out2048_kv512_ffn8192".into(),
            tensor_type: "q4_k".into(),
            rows: 8192,
            cols: 2048,
            batch_tokens: 1,
            graph_features: mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_Q_NORM
                | mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_K_NORM
                | mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_POST_NORM,
            effective_gbps: 180.0,
            tflops: Some(0.35),
            elapsed_ms: Some(9.0),
            runs: 3,
        });

    let linear_probe_rec = score_model(&hardware, &model, &config);
    assert_eq!(
        linear_probe_rec.estimate_confidence,
        EstimateConfidence::Medium
    );
    let linear_groups = &linear_probe_rec
        .decode_cost_breakdown
        .as_ref()
        .expect("decode cost breakdown")
        .groups;
    assert!(
        linear_groups.iter().any(|group| {
            group.group == "linear_attention_block"
                && group.source == "probe_linear_attention_block_elapsed"
                && group.probe_name.as_deref().is_some_and(|name| {
                    name.contains("linear_attn_graph_r18_f6")
                        && name.contains("_h2048_qkv6144_gate2048_state16_out2048_kv512_")
                })
        }),
        "groups={linear_groups:#?}; warnings={:#?}",
        linear_probe_rec.warnings
    );
}

#[test]
fn dense_decode_uses_measured_block_graph_depth_curve() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "ggml_decode_q4_k_llama_graph_l8_4096_16384".into(),
        tensor_type: "q4_k".into(),
        rows: 16_384,
        cols: 4096,
        batch_tokens: 1,
        graph_features: 0,
        effective_gbps: 140.0,
        tflops: Some(0.7),
        elapsed_ms: None,
        runs: 3,
    }];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("deep-q4", 4 * GIB, 32, 4096, 32_768);

    let l8_only = score_model(&hardware, &model, &config);
    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l4_4096_16384".into(),
            tensor_type: "q4_k".into(),
            rows: 16_384,
            cols: 4096,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 100.0,
            tflops: Some(0.5),
            elapsed_ms: None,
            runs: 3,
        });
    let depth_curve = score_model(&hardware, &model, &config);

    assert!(
        depth_curve.estimated_decode_tokens_per_sec.unwrap()
            > l8_only.estimated_decode_tokens_per_sec.unwrap()
    );
    assert!(
        depth_curve
            .decode_cost_breakdown
            .as_ref()
            .expect("decode cost breakdown")
            .groups
            .iter()
            .any(|group| group.source == "probe_block" && group.group == "transformer_block")
    );
}

#[test]
fn exact_dense_decode_elapsed_uses_measured_depth_slope() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    hardware.accelerators[0].decode_kernel_probes = vec![
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l4_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 100.0,
            tflops: Some(0.5),
            elapsed_ms: Some(2.25),
            runs: 3,
        },
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l8_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 154.0,
            tflops: Some(0.7),
            elapsed_ms: Some(3.0),
            runs: 3,
        },
    ];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("sublinear-depth-q4", 4 * GIB, 28, 3072, 32_768);

    let rec = score_model(&hardware, &model, &config);
    let block = rec
        .decode_cost_breakdown
        .as_ref()
        .expect("decode breakdown")
        .groups
        .iter()
        .find(|group| group.group == "transformer_block")
        .expect("transformer block group");

    assert_eq!(block.source, "probe_block_depth_elapsed");
    assert!(
        (block.bandwidth_ms - 6.5).abs() < 0.01,
        "expected measured l4->l8 depth slope to extrapolate to 6.5 ms, got {}",
        block.bandwidth_ms
    );
}

#[test]
fn deep_dense_elapsed_extrapolation_uses_measured_envelope() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    hardware.accelerators[0].decode_kernel_probes = vec![
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l4_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 100.0,
            tflops: Some(0.5),
            elapsed_ms: Some(3.25),
            runs: 3,
        },
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l8_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 154.0,
            tflops: Some(0.7),
            elapsed_ms: Some(6.25),
            runs: 3,
        },
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l16_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 220.0,
            tflops: Some(1.0),
            elapsed_ms: Some(8.25),
            runs: 3,
        },
    ];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = dense_model("deep-envelope-q4", 4 * GIB, 28, 3072, 32_768);

    let rec = score_model(&hardware, &model, &config);
    let block = rec
        .decode_cost_breakdown
        .as_ref()
        .expect("decode breakdown")
        .groups
        .iter()
        .find(|group| group.group == "transformer_block")
        .expect("transformer block group");

    assert_eq!(block.source, "probe_block_depth_elapsed");
    assert!(
        (block.bandwidth_ms - 13.0).abs() < 0.01,
        "expected l4->l16 envelope to extrapolate to 13.0 ms, got {}",
        block.bandwidth_ms
    );
}

#[test]
fn exact_dense_block_graph_covers_mixed_residual_tensor_types() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    hardware.accelerators[0].decode_kernel_probes = vec![
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l4_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 100.0,
            tflops: Some(0.5),
            elapsed_ms: Some(2.25),
            runs: 3,
        },
        DecodeKernelProbe {
            name: "ggml_decode_q4_k_llama_graph_l8_3072_12288".into(),
            tensor_type: "q4_k".into(),
            rows: 12_288,
            cols: 3072,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 154.0,
            tflops: Some(0.7),
            elapsed_ms: Some(3.0),
            runs: 3,
        },
    ];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let mut model = dense_model("mixed-q4-q6-block", 4 * GIB, 28, 3072, 32_768);
    let residual_q6 = model.tensor_matmul.attention.bytes / 4;
    model.tensor_matmul.attention.type_bytes.q4_k_bytes =
        model.tensor_matmul.attention.bytes - residual_q6;
    model.tensor_matmul.attention.type_bytes.q6_k_bytes = residual_q6;
    model.tensor_matmul.base_type_bytes.q4_k_bytes -= residual_q6;
    model.tensor_matmul.base_type_bytes.q6_k_bytes = residual_q6;

    let rec = score_model(&hardware, &model, &config);
    let groups = &rec
        .decode_cost_breakdown
        .as_ref()
        .expect("decode breakdown")
        .groups;
    let block = groups
        .iter()
        .find(|group| group.group == "transformer_block")
        .expect("transformer block group");
    let block_matmul_bytes = model
        .tensor_matmul
        .attention
        .bytes
        .saturating_add(model.tensor_matmul.feed_forward.bytes);

    assert_eq!(block.source, "probe_block_depth_elapsed");
    assert_eq!(block.traffic_bytes, block_matmul_bytes);
    assert!(
        groups.iter().all(|group| {
            group.group != "attention_matmul" && group.group != "feed_forward_matmul"
        }),
        "exact block graph should not replay residual transformer groups: {groups:?}"
    );
}

#[test]
fn sparse_moe_requires_composite_routed_expert_decode_probe_for_medium_confidence() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_effective_bandwidth_bytes_per_sec = Some(400_000_000_000);
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "ggml_decode_q4_k_matvec_square_2048".into(),
        tensor_type: "q4_k".into(),
        rows: 2048,
        cols: 2048,
        batch_tokens: 1,
        graph_features: 0,
        effective_gbps: 80.0,
        tflops: Some(0.4),
        elapsed_ms: None,
        runs: 20,
    }];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = qwen3_30b_a3b_q4_moe();

    let dense_probe_rec = score_model(&hardware, &model, &config);
    assert_ne!(
        dense_probe_rec.estimate_confidence,
        EstimateConfidence::High
    );
    assert!(
        dense_probe_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("dominant tensor type q4_k"))
    );

    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_moe_mul_mat_id_q4_k".into(),
            tensor_type: "q4_k".into(),
            rows: 2048,
            cols: 2048,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 120.0,
            tflops: Some(0.8),
            elapsed_ms: None,
            runs: 20,
        });

    let routed_op_rec = score_model(&hardware, &model, &config);
    assert_ne!(routed_op_rec.estimate_confidence, EstimateConfidence::High);

    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_moe_graph_q4_k".into(),
            tensor_type: "q4_k".into(),
            rows: 2048,
            cols: 2048,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 120.0,
            tflops: Some(0.8),
            elapsed_ms: None,
            runs: 20,
        });

    let routed_probe_rec = score_model(&hardware, &model, &config);
    assert_eq!(
        routed_probe_rec.estimate_confidence,
        EstimateConfidence::Medium
    );
    assert!(
        routed_probe_rec
            .warnings
            .iter()
            .any(|warning| warning.contains("metadata-only estimates are not yet validated"))
    );
    assert!(
        routed_probe_rec
            .reasons
            .iter()
            .any(|reason| reason.contains("source-shaped GGML groups"))
    );
}

#[test]
fn sparse_moe_decode_uses_measured_moe_graph_depth_curve() {
    let mut hardware = m1_ultra();
    hardware.accelerators[0].decode_fixed_overhead_ms = Some(0.25);
    hardware.accelerators[0].decode_kernel_probes = vec![DecodeKernelProbe {
        name: "ggml_decode_moe_graph_l1_q4_k_16x4_4096x2048".into(),
        tensor_type: "q4_k".into(),
        rows: 4096,
        cols: 2048,
        batch_tokens: 1,
        graph_features: 0,
        effective_gbps: 80.0,
        tflops: Some(0.4),
        elapsed_ms: None,
        runs: 3,
    }];

    let mut config = SelectionConfig {
        workload: WorkloadProfile::chat(),
        ..SelectionConfig::default()
    };
    config.weights = config.workload.default_weights();
    let model = qwen3_30b_a3b_q4_moe();

    let l1_only = score_model(&hardware, &model, &config);
    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_moe_mul_mat_id_q4_k_16x4_4096x2048".into(),
            tensor_type: "q4_k".into(),
            rows: 4096,
            cols: 2048,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 200.0,
            tflops: Some(1.2),
            elapsed_ms: None,
            runs: 3,
        });
    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_moe_graph_l4_q4_k_16x4_4096x2048".into(),
            tensor_type: "q4_k".into(),
            rows: 4096,
            cols: 2048,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 120.0,
            tflops: Some(0.7),
            elapsed_ms: None,
            runs: 3,
        });
    hardware.accelerators[0]
        .decode_kernel_probes
        .push(DecodeKernelProbe {
            name: "ggml_decode_moe_graph_l8_q4_k_16x4_4096x2048".into(),
            tensor_type: "q4_k".into(),
            rows: 4096,
            cols: 2048,
            batch_tokens: 1,
            graph_features: 0,
            effective_gbps: 150.0,
            tflops: Some(0.9),
            elapsed_ms: None,
            runs: 3,
        });
    let depth_curve = score_model(&hardware, &model, &config);

    assert!(
        depth_curve.estimated_decode_tokens_per_sec.unwrap()
            > l1_only.estimated_decode_tokens_per_sec.unwrap()
    );
    assert!(
        depth_curve
            .decode_cost_breakdown
            .as_ref()
            .expect("decode cost breakdown")
            .groups
            .iter()
            .any(|group| {
                group.group == "routed_expert"
                    && group.source == "probe"
                    && group
                        .probe_name
                        .as_deref()
                        .is_some_and(|name| name.contains("_l8_"))
            })
    );
}

#[test]
fn hardware_profile_uses_mesh_gpu_benchmark_output_as_measured_bandwidth() {
    let hardware = hardware_profile_from_gpu_benchmark(GpuBenchmarkHardwareInput {
        memory: MemoryProfile {
            total_system_bytes: Some(128 * GIB),
            available_system_bytes: Some(110 * GIB),
            total_unified_bytes: Some(128 * GIB),
            available_unified_bytes: Some(110 * GIB),
        },
        cpu: CpuProfile::default(),
        default_backend: BackendKind::Metal,
        accelerators: vec![GpuBenchmarkAcceleratorFacts {
            name: Some("Apple M1 Ultra".into()),
            kind: AcceleratorKind::IntegratedGpu,
            backend: Some(BackendKind::Metal),
            total_memory_bytes: Some(128 * GIB),
            available_memory_bytes: Some(110 * GIB),
            unified_memory: true,
        }],
        benchmark_outputs: vec![GpuBenchmarkOutput {
            device: "Apple M1 Ultra".into(),
            buffer_mb: 1024,
            runs: 7,
            p50_gbps: 710.0,
            p90_gbps: 737.0,
            decode_effective_gbps: Some(295.0),
            decode_fixed_overhead_ms: Some(1.25),
            decode_runtime_overhead_ms: Some(0.125),
            post_prefill_decode_overhead_ms: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            prefill_matmul_tflops_fp16: None,
            prefill_ubatch_matmul_tflops_fp16: None,
            prefill_moe_matmul_tflops_fp16: None,
            sampler_history_us_per_token: None,
            sampler_vocab_us_per_token: None,
            decode_kernel_probes: Vec::new(),
            noise_pct: 1.0,
            runtime_s: 0.25,
            rated_gbps: None,
            rated_estimated: None,
            efficiency_pct: None,
            bus_width_bits: None,
            mem_clock_mhz: None,
            gcn_arch: None,
            hbm: None,
        }],
    })
    .expect("benchmark output should build hardware profile");

    let accelerator = &hardware.accelerators[0];
    assert_eq!(accelerator.bandwidth_source, MeasurementSource::Measured);
    assert_eq!(
        accelerator.memory_bandwidth_bytes_per_sec,
        Some(737_000_000_000)
    );
    assert_eq!(accelerator.benchmark_noise_pct, Some(1.0));
    assert_eq!(accelerator.available_memory_bytes, Some(110 * GIB));
    assert_eq!(accelerator.decode_runtime_overhead_ms, Some(0.125));
}
