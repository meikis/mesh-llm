use crate::inference::skippy;
use crate::mesh;

pub(super) async fn standby_message(
    node: &mesh::Node,
    model_ref: &str,
    coordinator: iroh::EndpointId,
) -> String {
    let statuses = node.stage_runtime_statuses().await;
    let active_stage = active_local_stage(node.id(), model_ref, &statuses);
    format_standby_message(
        coordinator,
        active_stage.map(|stage| (stage.stage_id.as_str(), stage.state)),
    )
}

fn active_local_stage<'a>(
    local_node_id: iroh::EndpointId,
    model_ref: &str,
    statuses: &'a [mesh::StageRuntimeStatus],
) -> Option<&'a mesh::StageRuntimeStatus> {
    statuses.iter().find(|status| {
        status.model_id == model_ref
            && status.node_id == Some(local_node_id)
            && matches!(
                status.state,
                skippy::StageRuntimeState::Starting | skippy::StageRuntimeState::Ready
            )
    })
}

fn format_standby_message(
    coordinator: iroh::EndpointId,
    active_stage: Option<(&str, skippy::StageRuntimeState)>,
) -> String {
    let Some((stage_id, stage_state)) = active_stage else {
        return format!(
            "Split runtime coordinator is {}; waiting for a local stage assignment",
            coordinator.fmt_short()
        );
    };
    let state = match stage_state {
        skippy::StageRuntimeState::Starting => "starting",
        skippy::StageRuntimeState::Ready => "ready",
        skippy::StageRuntimeState::Stopping
        | skippy::StageRuntimeState::Stopped
        | skippy::StageRuntimeState::Failed => "inactive",
    };
    format!(
        "Split stage {stage_id} is {state} under coordinator {}; model requests remain routed through the coordinator",
        coordinator.fmt_short()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinator() -> iroh::EndpointId {
        iroh::SecretKey::from_bytes(&[7; 32]).public()
    }

    fn stage_status(
        node_id: iroh::EndpointId,
        model_id: &str,
        state: skippy::StageRuntimeState,
    ) -> mesh::StageRuntimeStatus {
        mesh::StageRuntimeStatus {
            topology_id: "topology-a".to_string(),
            run_id: "run-a".to_string(),
            model_id: model_id.to_string(),
            backend: "skippy".to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            projector_path: None,
            stage_id: "stage-1".to_string(),
            stage_index: 1,
            node_id: Some(node_id),
            layer_start: 16,
            layer_end: 32,
            state,
            bind_addr: "127.0.0.1:1234".to_string(),
            activation_width: 2048,
            wire_dtype: skippy::StageWireDType::F16,
            selected_device: None,
            ctx_size: 4096,
            lane_count: 4,
            n_batch: None,
            n_ubatch: None,
            flash_attn_type: skippy_protocol::FlashAttentionType::Auto,
            error: None,
            shutdown_generation: 1,
        }
    }

    #[test]
    fn reports_pending_assignment() {
        let message = format_standby_message(coordinator(), None);

        assert!(message.contains("waiting for a local stage assignment"));
        assert!(!message.contains("stage is ready"));
    }

    #[test]
    fn reports_active_worker_stage() {
        let message = format_standby_message(
            coordinator(),
            Some(("stage-1", skippy::StageRuntimeState::Ready)),
        );

        assert!(message.contains("Split stage stage-1 is ready"));
        assert!(message.contains("model requests remain routed through the coordinator"));
        assert!(!message.contains("waiting for a local stage assignment"));
    }

    #[test]
    fn selects_only_active_local_stage_for_model() {
        let local_node = iroh::SecretKey::from_bytes(&[8; 32]).public();
        let other_node = iroh::SecretKey::from_bytes(&[9; 32]).public();
        let statuses = vec![
            stage_status(local_node, "model-a", skippy::StageRuntimeState::Failed),
            stage_status(other_node, "model-a", skippy::StageRuntimeState::Ready),
            stage_status(local_node, "model-b", skippy::StageRuntimeState::Ready),
            stage_status(local_node, "model-a", skippy::StageRuntimeState::Ready),
        ];

        let selected = active_local_stage(local_node, "model-a", &statuses).unwrap();

        assert_eq!(selected.model_id, "model-a");
        assert_eq!(selected.node_id, Some(local_node));
        assert_eq!(selected.state, skippy::StageRuntimeState::Ready);
    }
}
