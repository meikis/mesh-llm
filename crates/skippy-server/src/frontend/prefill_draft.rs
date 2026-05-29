use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PrefillDraftDecision {
    pub(super) commit_count: usize,
    pub(super) raw_matches: usize,
    pub(super) tolerated_mismatches: usize,
    pub(super) full_window: bool,
}

pub(super) fn classify_prefill_draft_span(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    max_consecutive_mismatches: usize,
) -> OpenAiResult<PrefillDraftDecision> {
    if predicted_tokens.len() < draft_tokens.len() {
        return Err(OpenAiError::backend(format!(
            "prefill draft verify span returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        )));
    }

    let mut commit_count = 0usize;
    let mut raw_matches = 0usize;
    let mut consecutive_mismatches = 0usize;
    let mut last_match_commit_count = 0usize;
    let mut raw_matches_at_last_match = 0usize;

    for (index, (draft_token, predicted_token)) in
        draft_tokens.iter().zip(predicted_tokens.iter()).enumerate()
    {
        if draft_token == predicted_token {
            commit_count = index + 1;
            raw_matches += 1;
            consecutive_mismatches = 0;
            last_match_commit_count = commit_count;
            raw_matches_at_last_match = raw_matches;
            continue;
        }

        consecutive_mismatches += 1;
        if consecutive_mismatches > max_consecutive_mismatches {
            commit_count = last_match_commit_count;
            raw_matches = raw_matches_at_last_match;
            break;
        }
        commit_count = index + 1;
    }

    Ok(PrefillDraftDecision {
        commit_count,
        raw_matches,
        tolerated_mismatches: commit_count.saturating_sub(raw_matches),
        full_window: commit_count == draft_tokens.len(),
    })
}

pub(super) struct PrefillDraftBurst<'a, 'b> {
    pub(super) request: &'a EmbeddedStageZeroGeneration<'a>,
    pub(super) downstream: &'b mut TcpStream,
    pub(super) session_key: &'b str,
    pub(super) request_id: u64,
    pub(super) session_id: u64,
    pub(super) prefill_token_count: usize,
    pub(super) current: i32,
    pub(super) decoded_tokens: usize,
}

pub(super) struct PrefillDraftBurstOutcome {
    pub(super) current: i32,
    pub(super) decoded_tokens: usize,
    pub(super) reached_stop: bool,
}

struct PrefillDraftVerification {
    draft_tokens: Vec<i32>,
    decision: PrefillDraftDecision,
    verify: EmbeddedStageExecution,
    propose_ms: f64,
}

struct PrefillDraftCommit {
    current: i32,
    decoded_tokens: usize,
    reached_stop: bool,
}

impl StageOpenAiBackend {
    pub(super) fn try_prefill_draft_burst(
        &self,
        mut burst: PrefillDraftBurst<'_, '_>,
        draft: &mut DraftRunner,
        context_tokens: &mut Vec<i32>,
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<PrefillDraftBurstOutcome> {
        let proposal_limit = prefill_draft_proposal_limit(&burst, draft.window);
        if proposal_limit == 0 {
            return Ok(prefill_draft_outcome(&burst));
        }

        let burst_timer = PhaseTimer::start();
        let Some(verification) =
            self.propose_and_verify_prefill_draft(&mut burst, draft, proposal_limit)?
        else {
            return Ok(prefill_draft_outcome(&burst));
        };
        self.repair_prefill_draft_state(
            &mut burst,
            &verification.draft_tokens,
            verification.decision,
        )?;
        let commit = commit_prefill_draft_tokens(
            burst.current,
            burst.decoded_tokens,
            &verification.draft_tokens,
            verification.decision.commit_count,
            context_tokens,
            on_token,
        )?;

        if verification.decision.commit_count < verification.draft_tokens.len()
            || commit.reached_stop
        {
            draft
                .reset_to_context(context_tokens)
                .map_err(openai_backend_error)?;
        }

        self.emit_prefill_draft_metrics(&burst, &verification, &commit, burst_timer);

        Ok(PrefillDraftBurstOutcome {
            current: commit.current,
            decoded_tokens: commit.decoded_tokens,
            reached_stop: commit.reached_stop,
        })
    }

    fn propose_and_verify_prefill_draft(
        &self,
        burst: &mut PrefillDraftBurst<'_, '_>,
        draft: &mut DraftRunner,
        proposal_limit: usize,
    ) -> OpenAiResult<Option<PrefillDraftVerification>> {
        let propose_timer = PhaseTimer::start();
        let draft_tokens = draft
            .propose(burst.current, proposal_limit)
            .map_err(openai_backend_error)?;
        let propose_ms = propose_timer.elapsed_ms();
        if draft_tokens.is_empty() {
            return Ok(None);
        }

        let verify_inputs = verify_inputs_for_proposals(burst.current, &draft_tokens);
        let verify = self.execute_prefill_draft_verify_span(burst, &verify_inputs, true)?;
        let decision = classify_prefill_draft_span(
            &draft_tokens,
            &verify.reply.predicted_tokens,
            burst.request.prefill_draft_max_consecutive_mismatches,
        )?;

        Ok(Some(PrefillDraftVerification {
            draft_tokens,
            decision,
            verify,
            propose_ms,
        }))
    }

    fn execute_prefill_draft_verify_span(
        &self,
        burst: &mut PrefillDraftBurst<'_, '_>,
        verify_inputs: &[i32],
        checkpoint: bool,
    ) -> OpenAiResult<EmbeddedStageExecution> {
        let verify_message = embedded_verify_message(
            burst.request.wire_dtype,
            VerifySpanMessageArgs {
                request_id: burst.request_id,
                session_id: burst.session_id,
                prompt_token_count: burst.request.prompt_token_ids.len(),
                pos_start: burst.prefill_token_count + burst.decoded_tokens,
                decode_step: burst.decoded_tokens,
                tokens: verify_inputs,
                checkpoint,
            },
        )?;

        self.execute_embedded_stage_message(
            burst.request,
            &mut *burst.downstream,
            burst.session_key,
            &verify_message,
            verify_inputs,
            WireReplyKind::PredictedTokens,
        )
    }

    fn repair_prefill_draft_state(
        &self,
        burst: &mut PrefillDraftBurst<'_, '_>,
        draft_tokens: &[i32],
        decision: PrefillDraftDecision,
    ) -> OpenAiResult<()> {
        if !decision.full_window {
            self.restore_embedded_stage_session(
                burst.request,
                &mut *burst.downstream,
                burst.session_key,
                burst.request_id,
                burst.session_id,
            )?;
            if decision.commit_count > 0 {
                let repair_inputs = verify_inputs_for_proposals(
                    burst.current,
                    &draft_tokens[..decision.commit_count],
                );
                self.execute_prefill_draft_verify_span(burst, &repair_inputs, false)?;
            }
        }
        Ok(())
    }

    fn emit_prefill_draft_metrics(
        &self,
        burst: &PrefillDraftBurst<'_, '_>,
        verification: &PrefillDraftVerification,
        commit: &PrefillDraftCommit,
        burst_timer: PhaseTimer,
    ) {
        let mut attrs = self.openai_attrs(burst.request.ids);
        attrs.insert(
            "llama_stage.prefill_draft.proposed".to_string(),
            json!(verification.draft_tokens.len()),
        );
        attrs.insert(
            "llama_stage.prefill_draft.committed".to_string(),
            json!(commit.decoded_tokens.saturating_sub(burst.decoded_tokens)),
        );
        attrs.insert(
            "llama_stage.prefill_draft.raw_matches".to_string(),
            json!(verification.decision.raw_matches),
        );
        attrs.insert(
            "llama_stage.prefill_draft.tolerated_mismatches".to_string(),
            json!(verification.decision.tolerated_mismatches),
        );
        attrs.insert(
            "llama_stage.prefill_draft.max_consecutive_mismatches".to_string(),
            json!(burst.request.prefill_draft_max_consecutive_mismatches),
        );
        attrs.insert(
            "llama_stage.prefill_draft.propose_ms".to_string(),
            json!(verification.propose_ms),
        );
        attrs.insert(
            "llama_stage.stage0_compute_ms".to_string(),
            json!(verification.verify.stats.stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.downstream_wait_ms".to_string(),
            json!(verification.verify.stats.downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.message_kind".to_string(),
            json!("PrefillDraftVerifySpan"),
        );
        self.emit_openai_phase("stage.openai_prefill_draft_burst", burst_timer, attrs);
    }
}

fn prefill_draft_proposal_limit(burst: &PrefillDraftBurst<'_, '_>, draft_window: usize) -> usize {
    let remaining = burst
        .request
        .max_tokens
        .saturating_sub(u32::try_from(burst.decoded_tokens).unwrap_or(u32::MAX))
        as usize;
    remaining
        .min(burst.request.prefill_draft_burst_tokens)
        .min(draft_window)
}

fn prefill_draft_outcome(burst: &PrefillDraftBurst<'_, '_>) -> PrefillDraftBurstOutcome {
    PrefillDraftBurstOutcome {
        current: burst.current,
        decoded_tokens: burst.decoded_tokens,
        reached_stop: false,
    }
}

fn commit_prefill_draft_tokens(
    mut current: i32,
    mut decoded_tokens: usize,
    draft_tokens: &[i32],
    commit_count: usize,
    context_tokens: &mut Vec<i32>,
    on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
) -> OpenAiResult<PrefillDraftCommit> {
    let mut reached_stop = false;
    for token in draft_tokens.iter().take(commit_count).copied() {
        current = token;
        decoded_tokens += 1;
        context_tokens.push(current);
        if on_token(current)? == TokenControl::Stop {
            reached_stop = true;
            break;
        }
    }

    Ok(PrefillDraftCommit {
        current,
        decoded_tokens,
        reached_stop,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_draft_strict_breaks_on_first_mismatch() {
        let decision = classify_prefill_draft_span(&[1, 2, 3], &[1, 9, 3], 0).unwrap();

        assert_eq!(decision.commit_count, 1);
        assert_eq!(decision.raw_matches, 1);
        assert_eq!(decision.tolerated_mismatches, 0);
        assert!(!decision.full_window);
    }

    #[test]
    fn prefill_draft_tolerates_isolated_mismatch() {
        let decision = classify_prefill_draft_span(&[1, 2, 3, 4], &[1, 9, 3, 4], 1).unwrap();

        assert_eq!(decision.commit_count, 4);
        assert_eq!(decision.raw_matches, 3);
        assert_eq!(decision.tolerated_mismatches, 1);
        assert!(decision.full_window);
    }

    #[test]
    fn prefill_draft_rewinds_after_mismatch_run() {
        let decision = classify_prefill_draft_span(&[1, 2, 3, 4, 5], &[1, 9, 8, 4, 5], 1).unwrap();

        assert_eq!(decision.commit_count, 1);
        assert_eq!(decision.raw_matches, 1);
        assert_eq!(decision.tolerated_mismatches, 0);
        assert!(!decision.full_window);
    }
}
