//! `mlx-serve` — run an OpenAI-compatible MLX inference server for one model.
//!
//! This is the runnable entry point mesh orchestrates on an MLX-eligible
//! (Apple Silicon) node: it loads a safetensors model from Hugging Face and
//! serves `/v1/chat/completions` + `/v1/models` on a local address that mesh
//! then routes OpenAI traffic to. No Python, no Swift.
//!
//! Requires the `link-mlx` feature (native MLX Metal engine).
//!
//! Usage:
//!   mlx-serve --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 --addr 127.0.0.1:9999
//!
//! Distributed (pipeline/tensor) launches are coordinated by mesh via the
//! hostfile + MLX backend env; this binary serves the local stage.

#[cfg(feature = "link-mlx")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use mesh_mlx::{Engine, ModelRef, ServerState, mlx_supported, serve};
    use std::net::SocketAddr;

    let mut model = String::from("mlx-community/Qwen2.5-0.5B-Instruct-bf16");
    let mut addr = String::from("127.0.0.1:9999");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().unwrap_or(model),
            "--addr" => addr = args.next().unwrap_or(addr),
            "-h" | "--help" => {
                println!("mlx-serve --model <hf-repo> --addr <ip:port>");
                return Ok(());
            }
            other => eprintln!("ignoring unknown arg: {other}"),
        }
    }

    if !mlx_supported() {
        anyhow::bail!("MLX backend requires Apple Silicon (macOS aarch64)");
    }

    eprintln!("mlx-serve: loading {model} …");
    let model_ref = ModelRef::new(&model);
    let engine = Engine::load_single(&model_ref)
        .await
        .map_err(|e| anyhow::anyhow!("load model: {e}"))?;
    eprintln!(
        "mlx-serve: loaded ({} layers); serving OpenAI API on http://{addr}/v1",
        engine.config.num_hidden_layers
    );

    let state = ServerState::new(engine, model);
    let socket: SocketAddr = addr.parse()?;
    serve(state, socket).await?;
    Ok(())
}

#[cfg(not(feature = "link-mlx"))]
fn main() {
    eprintln!(
        "mlx-serve requires the `link-mlx` feature (native MLX Metal engine). \
         Rebuild with: cargo build -p mesh-mlx --features link-mlx --bin mlx-serve"
    );
    std::process::exit(1);
}
