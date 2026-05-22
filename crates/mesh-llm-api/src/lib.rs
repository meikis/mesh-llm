#![forbid(unsafe_code)]

mod client;
mod discover;
pub mod events;
mod node;

pub use client::{
    ChatMessage, ChatRequest, ClientBuilder, ClientConfig, MeshApiError, MeshClient, Model,
    RequestId, ResponsesRequest, Status, MAX_RECONNECT_ATTEMPTS,
};
pub use discover::{
    create_auto_client, create_auto_node, discover_public_meshes, AutoConnectResult,
    AutoNodeResult, PublicMesh, PublicMeshQuery,
};
pub use identity::OwnerKeypair;
pub use mesh_llm_node::serving::ServingController;
pub use node::{
    CapabilityLevel, CleanupPolicy, CleanupResult, DeleteModelOptions, DeleteModelResult,
    DevicePolicy, DownloadId, DownloadOptions, DownloadedModel, InstalledModel, LoadModelOptions,
    MeshEvents, MeshInference, MeshModels, MeshNode, MeshNodeBuilder, MeshNodeConfig, MeshServing,
    MeshStatusApi, ModelCacheStatus, ModelCapabilities, ModelDetails, ModelKind, ModelSearchQuery,
    ModelSource, ModelSummary, PrunePolicy, PruneResult, ServedModel, ServingModelState,
    ServingStatus, UnloadModelOptions, UnloadTarget,
};
pub use token::InviteToken;

mod identity;
mod token;
