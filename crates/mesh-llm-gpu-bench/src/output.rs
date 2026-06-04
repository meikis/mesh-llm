use serde::{Deserialize, Serialize};

pub const GRAPH_FEATURE_ATTENTION_Q_NORM: u32 = 1 << 0;
pub const GRAPH_FEATURE_ATTENTION_K_NORM: u32 = 1 << 1;
pub const GRAPH_FEATURE_ATTENTION_POST_NORM: u32 = 1 << 2;
pub const GRAPH_FEATURE_FFN_POST_NORM: u32 = 1 << 3;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct BenchmarkOutput {
    pub device: String,
    pub buffer_mb: u32,
    pub runs: u32,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    #[serde(default)]
    pub decode_effective_gbps: Option<f64>,
    #[serde(default)]
    pub decode_fixed_overhead_ms: Option<f64>,
    #[serde(default)]
    pub decode_runtime_overhead_ms: Option<f64>,
    #[serde(default)]
    pub post_prefill_decode_overhead_ms: Option<f64>,
    pub compute_tflops_fp32: Option<f64>,
    pub compute_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub prefill_matmul_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub prefill_ubatch_matmul_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub prefill_moe_matmul_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub sampler_history_us_per_token: Option<f64>,
    #[serde(default)]
    pub sampler_vocab_us_per_token: Option<f64>,
    #[serde(default)]
    pub decode_kernel_probes: Vec<DecodeKernelProbe>,
    pub noise_pct: f64,
    pub runtime_s: f64,
    pub rated_gbps: Option<f64>,
    pub rated_estimated: Option<bool>,
    pub efficiency_pct: Option<f64>,
    pub bus_width_bits: Option<u32>,
    pub mem_clock_mhz: Option<u64>,
    pub gcn_arch: Option<String>,
    pub hbm: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DecodeKernelProbe {
    pub name: String,
    pub tensor_type: String,
    pub rows: u32,
    pub cols: u32,
    pub batch_tokens: u32,
    #[serde(default)]
    pub graph_features: u32,
    pub effective_gbps: f64,
    pub tflops: Option<f64>,
    #[serde(default)]
    pub elapsed_ms: Option<f64>,
    pub runs: u32,
}
