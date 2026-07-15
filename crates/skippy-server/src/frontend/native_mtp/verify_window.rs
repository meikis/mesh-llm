use std::net::TcpStream;

use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_protocol::binary::WireReplyKind;

use super::super::{
    EmbeddedSessionControl, EmbeddedStageZeroGeneration, MtpAnchoredNgramExtender,
    NativeMtpDecodeCounters, NativeMtpDecodeOptions, NativeMtpDraft, NativeMtpDraftOrigin,
    NativeMtpHybridProposal, NativeMtpTrimAction, NativeMtpVerifier, PendingNativeMtpDraft,
    PhaseTimer, ProposalExtender, StageOpenAiBackend, TokenControl, VerifyWindowMessageArgs,
    VerifyWindowScheduler, WireSamplingConfig, classify_native_mtp_verify_window,
    embedded_verify_window_message, ms_to_us, native_mtp_trim_action, token_is_eog_with_runtime,
};

/// Control signal returned after processing a batched native MTP verify step.
pub(in crate::frontend) enum NativeMtpVerifyWindowControl {
    /// The on_token callback returned Stop — outer loop should break.
    ReachedStop,
    /// Continue the outer decode loop normally.
    Continue,
}

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::frontend) fn execute_native_mtp_verify_window(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        downstream: &mut TcpStream,
        session_key: &str,
        request_id: u64,
        session_id: u64,
        prefill_token_count: usize,
        wire_sampling: &Option<WireSamplingConfig>,
        native_mtp_options: &NativeMtpDecodeOptions,
        verify_window_scheduler: &mut VerifyWindowScheduler,
        pending_native_mtp_draft: PendingNativeMtpDraft,
        current: &mut i32,
        decode_step: u32,
        // Mutable decode loop state
        decoded_tokens: &mut usize,
        context_tokens: &mut Vec<i32>,
        exact_replay_tokens: &mut Vec<i32>,
        native_mtp: &mut NativeMtpVerifier,
        native_mtp_counters: &mut NativeMtpDecodeCounters,
        native_mtp_reject_cooldown_remaining: &mut usize,
        native_mtp_suppress_cooldown_drafts_remaining: &mut usize,
        // Mutable decode accumulators
        decode_stage0_compute_ms: &mut f64,
        decode_runtime_lock_wait_ms: &mut f64,
        decode_runtime_lock_wait_max_ms: &mut f64,
        decode_runtime_lock_hold_ms: &mut f64,
        decode_runtime_lock_hold_max_ms: &mut f64,
        decode_runtime_lock_acquires: &mut usize,
        decode_forward_activation_encode_ms: &mut f64,
        decode_output_activation_bytes: &mut usize,
        decode_forward_activation_bytes: &mut usize,
        decode_forward_write_ms: &mut f64,
        decode_downstream_wait_ms: &mut f64,
        // Token emission callback
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<NativeMtpVerifyWindowControl> {
        let verify_window_timer = self.telemetry.is_debug_enabled().then(PhaseTimer::start);
        let native_mtp_remaining = (request.max_tokens as usize).saturating_sub(*decoded_tokens);
        let native_mtp_draft_tokens = pending_native_mtp_draft
            .tokens
            .into_iter()
            .take(native_mtp_options.max_draft_tokens)
            .take(native_mtp_remaining.saturating_sub(1))
            .collect::<Vec<_>>();
        if native_mtp_draft_tokens.is_empty()
            || native_mtp_draft_tokens.len() < native_mtp_options.min_draft_tokens
        {
            native_mtp.clear_pending_draft();
            return Ok(NativeMtpVerifyWindowControl::Continue);
        }
        let native_mtp_draft_origin = pending_native_mtp_draft.origin;
        let native_mtp_proposal =
            if native_mtp_options.ngram_hybrid && native_mtp_draft_tokens.len() == 1 {
                MtpAnchoredNgramExtender::from_options(*native_mtp_options).extend(
                    native_mtp_draft_tokens[0],
                    context_tokens,
                    native_mtp_remaining.saturating_sub(1),
                )
            } else {
                NativeMtpHybridProposal::from_native_mtp_tokens(native_mtp_draft_tokens.clone())
            };
        let verify_inputs = native_mtp_verify_window_inputs(*current, native_mtp_proposal.tokens());
        let window =
            verify_window_scheduler.open(prefill_token_count + *decoded_tokens, *decoded_tokens)?;
        let message = embedded_verify_window_message(
            request.wire_dtype,
            VerifyWindowMessageArgs {
                window_id: window.id,
                request_id,
                session_id,
                prompt_token_count: request.prompt_token_ids.len(),
                pos_start: prefill_token_count + *decoded_tokens,
                decode_step: *decoded_tokens,
                tokens: &verify_inputs,
                sampling: wire_sampling.clone(),
                checkpoint: false,
            },
        )?;
        let verify = self.execute_embedded_stage_message(
            request,
            downstream,
            session_key,
            &message,
            &verify_inputs,
            WireReplyKind::PredictedTokens,
        )?;
        let completed = verify_window_scheduler.complete_next(verify.reply.window.window_id)?;
        if completed != window {
            return Err(OpenAiError::backend(
                "verify window scheduler lost FIFO state",
            ));
        }
        let native_mtp_verify_decision = classify_native_mtp_verify_window(
            native_mtp_proposal.tokens(),
            &verify.reply.predicted_tokens,
            *decoded_tokens,
            request.max_tokens as usize,
            |token| token_is_eog_with_runtime(&self.runtime, token),
        )?;
        let target_token = verify.reply.predicted_tokens[0];
        let verify_next_mtp_draft = NativeMtpDraft::from_verify_prediction_tokens(
            &verify.reply.predicted_tokens,
            verify_inputs.len(),
        );
        let span = native_mtp.observe_taken_draft_span(
            &native_mtp_draft_tokens,
            &verify.reply.predicted_tokens,
            ms_to_us(verify.elapsed_ms),
        );
        let native_mtp_decision = span.first_decision;
        let verified_draft_count = span.accepted_count + usize::from(span.rejected);
        for index in 0..verified_draft_count {
            native_mtp_counters.observe_verify_window_verification(
                native_mtp_draft_origin,
                index < span.accepted_count,
            );
        }
        native_mtp_counters.observe_hybrid_proposal(
            native_mtp_proposal.ngram_span_available(),
            native_mtp_proposal.ngram_anchor_agreed(),
            native_mtp_proposal.ngram_anchor_disagreed(),
            native_mtp_proposal.tokens().len(),
            native_mtp_verify_decision.accepted_proposal_tokens,
        );
        let commit_token_count = native_mtp_verify_decision.commit_count;
        let consumed_positions = verify_inputs.len();
        let mut committed_positions = 0usize;
        let mut reached_stop = false;
        for token in verify
            .reply
            .predicted_tokens
            .iter()
            .copied()
            .take(commit_token_count)
        {
            *current = token;
            *decoded_tokens += 1;
            committed_positions += 1;
            exact_replay_tokens.push(*current);
            context_tokens.push(*current);
            if on_token(*current)? == TokenControl::Stop {
                reached_stop = true;
                break;
            }
            if *decoded_tokens >= request.max_tokens as usize {
                break;
            }
        }
        if native_mtp_verify_decision.rejected && native_mtp_options.reject_cooldown_tokens > 0 {
            *native_mtp_reject_cooldown_remaining = native_mtp_options.reject_cooldown_tokens;
            *native_mtp_suppress_cooldown_drafts_remaining =
                native_mtp_options.suppress_cooldown_draft_limit;
            native_mtp.clear_pending_draft();
        }
        let verify_next_mtp_draft_available = verify_next_mtp_draft.is_some();
        let verify_next_mtp_draft_adopted = !native_mtp_verify_decision.rejected
            && committed_positions == consumed_positions
            && !reached_stop
            && *decoded_tokens < request.max_tokens as usize
            && verify_next_mtp_draft.is_some();
        native_mtp_counters.observe_verify_next_draft(
            verify_next_mtp_draft_available,
            verify_next_mtp_draft_adopted,
        );
        if verify_next_mtp_draft_adopted {
            native_mtp.observe_next_draft(
                verify_next_mtp_draft.clone(),
                NativeMtpDraftOrigin::VerifyNext,
            );
        }
        let mut trim_control: Option<EmbeddedSessionControl> = None;
        match native_mtp_trim_action(committed_positions, consumed_positions) {
            NativeMtpTrimAction::None => {}
            NativeMtpTrimAction::FullSession => {
                let target_token_count = prefill_token_count + *decoded_tokens;
                let trim = self.trim_embedded_stage_session(
                    request,
                    downstream,
                    session_key,
                    request_id,
                    session_id,
                    target_token_count,
                )?;
                trim_control = Some(trim);
            }
        }
        *decode_stage0_compute_ms += verify.stats.stage0_compute_ms;
        *decode_runtime_lock_wait_ms += verify.stats.runtime_lock_wait_ms;
        *decode_runtime_lock_wait_max_ms =
            decode_runtime_lock_wait_max_ms.max(verify.stats.runtime_lock_wait_ms);
        *decode_runtime_lock_hold_ms += verify.stats.runtime_lock_hold_ms;
        *decode_runtime_lock_hold_max_ms =
            decode_runtime_lock_hold_max_ms.max(verify.stats.runtime_lock_hold_ms);
        *decode_runtime_lock_acquires += 1;
        *decode_forward_activation_encode_ms += verify.stats.activation_encode_ms;
        *decode_output_activation_bytes =
            decode_output_activation_bytes.saturating_add(verify.stats.output_activation_bytes);
        *decode_forward_activation_bytes =
            decode_forward_activation_bytes.saturating_add(verify.stats.forward_activation_bytes);
        *decode_forward_write_ms += verify.stats.forward_write_ms;
        *decode_downstream_wait_ms += verify.stats.downstream_wait_ms;

        if let Some(verify_window_timer) = verify_window_timer {
            let mut token_attrs = self.openai_attrs(request.ids);
            token_attrs.insert(
                "llama_stage.decode_step".to_string(),
                serde_json::json!(decode_step),
            );
            token_attrs.insert(
                "llama_stage.message_kind".to_string(),
                serde_json::json!("VerifyWindow"),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_window_batch".to_string(),
                serde_json::json!(true),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verification".to_string(),
                serde_json::json!(native_mtp_decision.label()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_elapsed_ms".to_string(),
                serde_json::json!(verify.elapsed_ms),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.draft_tokens".to_string(),
                serde_json::json!(native_mtp_draft_tokens),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.pending_origin".to_string(),
                serde_json::json!(native_mtp_draft_origin.label()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.target_token".to_string(),
                serde_json::json!(target_token),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.accepted_count".to_string(),
                serde_json::json!(native_mtp_verify_decision.accepted_proposal_tokens),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.hybrid_proposal_len".to_string(),
                serde_json::json!(native_mtp_proposal.tokens().len()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.hybrid_anchor_agreement".to_string(),
                serde_json::json!(native_mtp_proposal.ngram_anchor_agreed()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.hybrid_anchor_disagreement".to_string(),
                serde_json::json!(native_mtp_proposal.ngram_anchor_disagreed()),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_next_draft_available".to_string(),
                serde_json::json!(verify_next_mtp_draft_available),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.verify_next_draft_adopted".to_string(),
                serde_json::json!(verify_next_mtp_draft_adopted),
            );
            if let Some(next_draft) = verify_next_mtp_draft.as_ref() {
                token_attrs.insert(
                    "llama_stage.native_mtp.verify_next_draft_tokens".to_string(),
                    serde_json::json!(next_draft.tokens),
                );
                token_attrs.insert(
                    "llama_stage.native_mtp.verify_next_draft_compute_us".to_string(),
                    serde_json::json!(next_draft.proposal_compute_us),
                );
            }
            token_attrs.insert(
                "llama_stage.native_mtp.consumed_positions".to_string(),
                serde_json::json!(consumed_positions),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.committed_positions".to_string(),
                serde_json::json!(committed_positions),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.reject_cooldown_tokens".to_string(),
                serde_json::json!(native_mtp_options.reject_cooldown_tokens),
            );
            token_attrs.insert(
                "llama_stage.native_mtp.reject_cooldown_remaining".to_string(),
                serde_json::json!(*native_mtp_reject_cooldown_remaining),
            );
            if let Some(trim) = trim_control.as_ref() {
                token_attrs.insert(
                    "llama_stage.native_mtp.trim_ms".to_string(),
                    serde_json::json!(trim.elapsed_ms),
                );
                token_attrs.insert(
                    "llama_stage.native_mtp.trim_local_ms".to_string(),
                    serde_json::json!(trim.local_ms),
                );
                token_attrs.insert(
                    "llama_stage.native_mtp.trim_downstream_write_ms".to_string(),
                    serde_json::json!(trim.downstream_write_ms),
                );
                token_attrs.insert(
                    "llama_stage.native_mtp.trim_downstream_wait_ms".to_string(),
                    serde_json::json!(trim.downstream_wait_ms),
                );
            }
            token_attrs.insert(
                "llama_stage.stage0_compute_ms".to_string(),
                serde_json::json!(verify.stats.stage0_compute_ms),
            );
            token_attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                serde_json::json!(verify.stats.runtime_lock_wait_ms),
            );
            token_attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                serde_json::json!(verify.stats.runtime_lock_hold_ms),
            );
            token_attrs.insert(
                "llama_stage.activation_encode_ms".to_string(),
                serde_json::json!(verify.stats.activation_encode_ms),
            );
            token_attrs.insert(
                "llama_stage.forward_write_ms".to_string(),
                serde_json::json!(verify.stats.forward_write_ms),
            );
            token_attrs.insert(
                "llama_stage.downstream_wait_ms".to_string(),
                serde_json::json!(verify.stats.downstream_wait_ms),
            );
            token_attrs.insert(
                "llama_stage.output_activation_bytes".to_string(),
                serde_json::json!(verify.stats.output_activation_bytes),
            );
            token_attrs.insert(
                "llama_stage.forward_activation_bytes".to_string(),
                serde_json::json!(verify.stats.forward_activation_bytes),
            );
            self.emit_openai_phase(
                "stage.openai_native_mtp_verify",
                verify_window_timer,
                token_attrs,
            );
        }

        if reached_stop {
            return Ok(NativeMtpVerifyWindowControl::ReachedStop);
        }
        Ok(NativeMtpVerifyWindowControl::Continue)
    }
}

fn native_mtp_verify_window_inputs(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len().saturating_add(1));
    tokens.push(current);
    tokens.extend_from_slice(proposals);
    tokens
}

#[cfg(test)]
mod tests {
    use super::native_mtp_verify_window_inputs;

    #[test]
    fn verify_window_inputs_include_every_native_mtp_proposal() {
        assert_eq!(native_mtp_verify_window_inputs(10, &[11, 12]), [10, 11, 12]);
    }
}
