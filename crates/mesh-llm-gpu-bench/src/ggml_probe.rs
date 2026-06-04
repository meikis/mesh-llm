use crate::{
    BenchmarkBackend, DecodeKernelProbe, DenseGraphProbeShape, LinearAttentionGraphProbeShape,
    MoeBlockGraphProbeShape, OutputProjectionProbeShape, ProbeDepth,
};
use anyhow::{Context, Result, anyhow};
use libc::{c_char, c_int, c_void};
use std::ffi::CStr;

unsafe extern "C" {
    fn mesh_llm_gpu_bench_ggml_decode_probe_json(
        backend_kind: c_int,
        probe_depth: c_int,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_moe_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        expert_count: i64,
        experts_used: i64,
        expert_width: i64,
        hidden: i64,
        repeat_layers: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_moe_block_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        expert_count: i64,
        experts_used: i64,
        expert_width: i64,
        hidden: i64,
        kv_width: i64,
        repeat_layers: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_linear_attention_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        qkv_width: i64,
        gate_width: i64,
        state_width: i64,
        output_input_width: i64,
        ffn: i64,
        recurrent_layers: i64,
        full_attention_layers: i64,
        kv_width: i64,
        graph_features: c_int,
        norm_head_width: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_output_projection_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        vocab: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_decode_probe_free(ptr: *mut c_void);
}

pub fn run(backend: BenchmarkBackend, probe_depth: ProbeDepth) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let probe_depth = match probe_depth {
        ProbeDepth::HardwareOnly => return Ok(Vec::new()),
        ProbeDepth::Standard => 0,
        ProbeDepth::Deep => 1,
    };

    let mut error: *mut c_char = std::ptr::null_mut();
    let json =
        unsafe { mesh_llm_gpu_bench_ggml_decode_probe_json(backend_kind, probe_depth, &mut error) };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML decode probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML decode probe returned invalid output; prefix={preview}")
    })
}

pub fn run_moe_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    expert_count: u32,
    experts_used: u32,
    expert_width: u32,
    hidden: u32,
    repeat_layers: u32,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => 0,
        "q6_k" => 1,
        other => return Err(anyhow!("unsupported MoE graph probe tensor type {other}")),
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_moe_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(expert_count),
            i64::from(experts_used),
            i64::from(expert_width),
            i64::from(hidden),
            i64::from(repeat_layers.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML MoE graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML MoE graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_moe_block_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => 0,
        "q6_k" => 1,
        other => {
            return Err(anyhow!(
                "unsupported MoE block graph probe tensor type {other}"
            ));
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_moe_block_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.expert_count),
            i64::from(shape.experts_used),
            i64::from(shape.expert_width),
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML MoE block graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML MoE block graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseGraphProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_linear_attention_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: LinearAttentionGraphProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_linear_attention_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.qkv_width),
            i64::from(shape.gate_width),
            i64::from(shape.state_width),
            i64::from(shape.output_input_width),
            i64::from(shape.ffn),
            i64::from(shape.recurrent_layers.max(1)),
            i64::from(shape.full_attention_layers),
            i64::from(shape.kv_width.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML linear attention graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML linear attention graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_output_projection_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: OutputProjectionProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_output_projection_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.vocab),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML output projection probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML output projection probe returned invalid output; prefix={preview}")
    })
}

fn dense_graph_tensor_type_kind(tensor_type: &str) -> Result<c_int> {
    match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => Ok(0),
        "q6_k" => Ok(1),
        "q8_0" => Ok(2),
        "f16" => Ok(3),
        other => Err(anyhow!("unsupported dense graph probe tensor type {other}")),
    }
}
