mod output;
mod runner;

#[cfg(any(feature = "cuda", feature = "hip", feature = "intel"))]
mod capture;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "intel")]
mod intel;

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
mod ggml_probe;

#[cfg(target_os = "macos")]
mod metal;

pub use output::{
    BenchmarkOutput, DecodeKernelProbe, GRAPH_FEATURE_ATTENTION_K_NORM,
    GRAPH_FEATURE_ATTENTION_POST_NORM, GRAPH_FEATURE_ATTENTION_Q_NORM, GRAPH_FEATURE_FFN_POST_NORM,
};
pub use runner::{
    BenchmarkBackend, BenchmarkOptions, BenchmarkRunner, DenseGraphProbeShape,
    LinearAttentionGraphProbeShape, MoeBlockGraphProbeShape, OutputProjectionProbeShape,
    ProbeDepth, parse_benchmark_output, run_benchmark, run_benchmark_with_options,
    run_model_dense_graph_probe, run_model_linear_attention_graph_probe,
    run_model_moe_block_graph_probe, run_model_moe_graph_probe, run_model_output_projection_probe,
    runner_for,
};
