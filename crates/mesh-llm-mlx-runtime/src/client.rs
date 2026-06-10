//! Thin OpenAI-compatible client to the MLX sidecar.
//!
//! mesh-llm already owns the public `/v1` surface via `openai-frontend`; this
//! client is only what the backend needs internally to (a) health-check the
//! sidecar and (b) confirm the served model id. Request *proxying* for live
//! traffic should reuse mesh's existing OpenAI routing against
//! [`crate::config::MlxRuntimeConfig::endpoint`]; we deliberately do not
//! re-implement chat/stream plumbing here.

use crate::{MlxError, Result};
use std::time::Duration;

/// Minimal client for sidecar lifecycle checks.
#[derive(Debug, Clone)]
pub struct SidecarClient {
    base_url: String,
    http: reqwest::Client,
}

/// Health status of the sidecar's OpenAI endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Health {
    /// `/v1/models` responded and lists at least one model.
    Ready { model_ids: Vec<String> },
    /// The endpoint is reachable but not yet serving a model.
    Starting,
    /// The endpoint could not be reached.
    Unreachable,
}

impl SidecarClient {
    /// `base_url` should be the `…/v1` endpoint.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .expect("reqwest client builds with default config"),
        }
    }

    /// Probe `/v1/models`. Used for readiness polling.
    pub async fn health(&self) -> Health {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        match self.http.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => match resp.json::<ModelsResponse>().await {
                Ok(body) if !body.data.is_empty() => Health::Ready {
                    model_ids: body.data.into_iter().map(|m| m.id).collect(),
                },
                _ => Health::Starting,
            },
            Ok(_) => Health::Starting,
            Err(_) => Health::Unreachable,
        }
    }

    /// Wait until the sidecar reports [`Health::Ready`] or the deadline elapses.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<Vec<String>> {
        let start = std::time::Instant::now();
        let mut backoff = Duration::from_millis(250);
        loop {
            if let Health::Ready { model_ids } = self.health().await {
                return Ok(model_ids);
            }
            if start.elapsed() >= timeout {
                return Err(MlxError::ReadinessTimeout(timeout));
            }
            tokio::time::sleep(backoff).await;
            backoff = (backoff * 2).min(Duration::from_secs(3));
        }
    }
}

#[derive(serde::Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(serde::Deserialize)]
struct ModelEntry {
    id: String,
}
