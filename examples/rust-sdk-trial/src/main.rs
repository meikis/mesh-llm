//! Trial Rust consumer of the mesh-llm SDK.
//!
//! Demonstrates the symmetric-with-Swift consumer experience: depend on
//! `mesh-llm-api-server` + `mesh-llm-host-runtime` as normal Rust source
//! crates, let skippy-ffi's build.rs fetch the prebuilt patched-llama.cpp
//! static archives at consumer build time, and call the public Rust API
//! directly. No FFI, no UniFFI wrappers, no daemon, no `mesh-llm` binary
//! on disk.
//!
//! The flow mirrors `mesh-llm client --auto`:
//!   1. Generate (or load) an owner keypair.
//!   2. Discover public meshes via Nostr relays.
//!   3. Pick the best one and create + start a node against it.
//!   4. List the models the mesh exposes; print them.
//!   5. Stop the node cleanly.

use std::time::Duration;

use mesh_llm_api_server::{
    create_auto_node, AutoNodeResult, MeshApiError, OwnerKeypair, PublicMeshQuery,
};
use tokio::time::timeout;

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> anyhow::Result<()> {
    println!("rust-sdk-trial: starting");

    // 1. Identity. In a real app this is persisted to disk; for the
    //    trial we generate a fresh one per run.
    let owner = OwnerKeypair::generate();
    println!(
        "rust-sdk-trial: owner keypair generated (first 16 hex = {})",
        &owner.to_hex()[..16],
    );

    // 2 + 3. Auto-discover + connect, same as `mesh-llm client --auto`.
    //        Use a generous timeout so a slow Nostr relay or NAT
    //        traversal doesn't kill the trial.
    let auto_query = PublicMeshQuery::default();
    println!("rust-sdk-trial: discovering and joining a public mesh...");
    let AutoNodeResult { node, selected_mesh } =
        match timeout(Duration::from_secs(90), create_auto_node(owner, auto_query)).await {
            Ok(Ok(result)) => result,
            Ok(Err(MeshApiError::Discovery { message })) => {
                println!("rust-sdk-trial: discovery error: {message}");
                println!("rust-sdk-trial: (this is expected if there are no public meshes online)");
                return Ok(());
            }
            Ok(Err(other)) => {
                println!("rust-sdk-trial: create_auto_node failed: {other}");
                return Ok(());
            }
            Err(_) => {
                println!("rust-sdk-trial: timed out waiting for a public mesh");
                return Ok(());
            }
        };

    println!(
        "rust-sdk-trial: selected mesh = {} (nodes={}, vram={:.1} GB, region={:?})",
        selected_mesh.name.as_deref().unwrap_or("(unnamed)"),
        selected_mesh.node_count,
        selected_mesh.total_vram_bytes as f64 / 1e9,
        selected_mesh.region,
    );
    println!(
        "rust-sdk-trial: mesh serving models = {:?}",
        selected_mesh.serving,
    );

    println!("rust-sdk-trial: starting in-process node...");
    if let Err(err) = timeout(Duration::from_secs(30), node.start()).await {
        println!("rust-sdk-trial: start() timed out: {err}");
        return Ok(());
    }
    println!("rust-sdk-trial: node started");

    // 4. List models the mesh exposes through the public API.
    match timeout(Duration::from_secs(15), node.inference().list_models()).await {
        Ok(Ok(models)) => {
            println!("rust-sdk-trial: {} model(s) advertised by the mesh:", models.len());
            for model in models.iter().take(8) {
                println!("  - {}", model.id);
            }
            if models.len() > 8 {
                println!("  ... and {} more", models.len() - 8);
            }
        }
        Ok(Err(err)) => println!("rust-sdk-trial: list_models error: {err}"),
        Err(_) => println!("rust-sdk-trial: list_models timed out"),
    }

    // 5. Clean shutdown.
    let _ = node.stop().await;
    println!("rust-sdk-trial: node stopped");
    Ok(())
}
