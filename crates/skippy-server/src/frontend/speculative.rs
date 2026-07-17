use super::*;

#[derive(Clone, Default)]
pub(super) struct OpenAiSpeculativeStats {
    pub(super) windows: usize,
    pub(super) draft_tokens: usize,
    pub(super) accepted_tokens: usize,
    pub(super) rejected_tokens: usize,
    pub(super) full_accept_windows: usize,
    pub(super) accepted_stop_windows: usize,
    pub(super) rejected_windows: usize,
    pub(super) early_reject_windows: usize,
    pub(super) tail_reject_windows: usize,
    pub(super) early_reject_stop_windows: usize,
    pub(super) repair_required_windows: usize,
    pub(super) first_reject_position_sum: usize,
    pub(super) primary_verify_requests: usize,
    pub(super) primary_verify_tokens: usize,
    pub(super) primary_verify_elapsed_ms: f64,
    pub(super) primary_verify_stage0_compute_ms: f64,
    pub(super) primary_verify_runtime_lock_wait_ms: f64,
    pub(super) primary_verify_runtime_lock_hold_ms: f64,
    pub(super) primary_verify_activation_encode_ms: f64,
    pub(super) primary_verify_forward_write_ms: f64,
    pub(super) primary_verify_downstream_wait_ms: f64,
    pub(super) primary_verify_output_activation_bytes: usize,
    pub(super) primary_verify_forward_activation_bytes: usize,
    pub(super) checkpoint_ms: f64,
    pub(super) draft_reset_ms: f64,
    pub(super) draft_propose_ms: f64,
    pub(super) recovery_restores: usize,
    pub(super) recovery_decode_repairs: usize,
    pub(super) recovery_decode_elapsed_ms: f64,
    pub(super) recovery_reverify_tokens: usize,
    pub(super) recovery_ms: f64,
    pub(super) recovery_restore_ms: f64,
    pub(super) recovery_restore_local_ms: f64,
    pub(super) recovery_restore_downstream_write_ms: f64,
    pub(super) recovery_restore_downstream_wait_ms: f64,
    pub(super) recovery_reverify_elapsed_ms: f64,
    pub(super) adaptive_window_start: usize,
    pub(super) adaptive_window_final: usize,
    pub(super) adaptive_window_max: usize,
    pub(super) adaptive_window_min: usize,
    pub(super) adaptive_window_max_seen: usize,
    pub(super) adaptive_window_sum: usize,
    pub(super) adaptive_window_grows: usize,
    pub(super) adaptive_window_shrinks: usize,
    pub(super) adaptive_window_enabled: bool,
}

/// Uses llama.cpp's ngram-simple self-speculative proposer. The accepted
/// history includes the current token; the upstream API keeps it separate
/// from the preceding history internally.
pub(super) fn propose_ngram_tokens(
    history: &[i32],
    min_match_tokens: usize,
    max_proposed_tokens: usize,
) -> OpenAiResult<Vec<i32>> {
    skippy_runtime::ngram_simple_draft(history, min_match_tokens, max_proposed_tokens)
        .map_err(openai_backend_error)
}

/// Request-local, cache-based N-gram proposer. It mirrors only committed
/// history into native state; speculative candidates remain read-only inputs.
pub(super) struct CachedNgramProposer {
    cache: skippy_runtime::NgramCache,
    committed_history: Vec<i32>,
}

impl CachedNgramProposer {
    pub(super) fn from_config(config: &SpeculativeDecodeConfig) -> OpenAiResult<Option<Self>> {
        let Some(ngram) = config.ngram.as_ref() else {
            return Ok(None);
        };
        if ngram.kind != NgramProposerKind::Cache {
            return Ok(None);
        }
        Self::new(ngram.min_ngram, ngram.max_ngram).map(Some)
    }

    pub(super) fn new(ngram_min: usize, ngram_max: usize) -> OpenAiResult<Self> {
        let cache =
            skippy_runtime::NgramCache::new(ngram_min, ngram_max).map_err(openai_backend_error)?;
        Ok(Self {
            cache,
            committed_history: Vec::new(),
        })
    }

    pub(super) fn propose(
        &mut self,
        committed_history: &[i32],
        continuation_prefix: &[i32],
        max_proposed_tokens: usize,
    ) -> OpenAiResult<Vec<i32>> {
        self.sync(committed_history)?;
        self.cache
            .draft_after(continuation_prefix, max_proposed_tokens)
            .map_err(openai_backend_error)
    }

    fn sync(&mut self, committed_history: &[i32]) -> OpenAiResult<()> {
        if committed_history.starts_with(&self.committed_history) {
            let appended = &committed_history[self.committed_history.len()..];
            self.cache.append(appended).map_err(openai_backend_error)?;
        } else {
            self.cache
                .reset(committed_history)
                .map_err(openai_backend_error)?;
        }
        self.committed_history.clear();
        self.committed_history.extend_from_slice(committed_history);
        Ok(())
    }
}

impl OpenAiSpeculativeStats {
    pub(super) fn insert_response_timings(&self, timings: &mut BTreeMap<String, Value>) {
        timings.insert(
            "verify_window_verify_elapsed_ms".to_string(),
            json!(self.primary_verify_elapsed_ms),
        );
        timings.insert(
            "verify_window_stage0_compute_ms".to_string(),
            json!(self.primary_verify_stage0_compute_ms),
        );
        timings.insert(
            "verify_window_forward_write_ms".to_string(),
            json!(self.primary_verify_forward_write_ms),
        );
        timings.insert(
            "verify_window_downstream_wait_ms".to_string(),
            json!(self.primary_verify_downstream_wait_ms),
        );
    }

    pub(super) fn observe_verify_decision(
        &mut self,
        decision: VerifyWindowDecision,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        self.accepted_tokens += decision.accepted_before_reject;
        if decision.rejected() {
            self.rejected_tokens += 1;
        }
        self.adaptive_window_sum += *adaptive_window;
        self.adaptive_window_min = nonzero_min(self.adaptive_window_min, *adaptive_window);
        self.adaptive_window_max_seen = self.adaptive_window_max_seen.max(*adaptive_window);
        match decision.kind {
            VerifyWindowDecisionKind::FullAccept => {
                self.full_accept_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifyWindowDecisionKind::AcceptedStop => {
                self.accepted_stop_windows += 1;
            }
            VerifyWindowDecisionKind::TailReject => {
                self.observe_reject(decision);
                self.tail_reject_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifyWindowDecisionKind::EarlyReject => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.repair_required_windows += 1;
                self.shrink_adaptive_window(adaptive_window, adaptive_enabled, decision);
            }
            VerifyWindowDecisionKind::EarlyRejectStop => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.early_reject_stop_windows += 1;
            }
        }
    }

    pub(super) fn observe_reject(&mut self, decision: VerifyWindowDecision) {
        if let Some(repair_input_count) = decision.repair_input_count {
            self.rejected_windows += 1;
            self.first_reject_position_sum += repair_input_count;
        }
    }

    pub(super) fn grow_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        if adaptive_enabled && *adaptive_window < max_speculative_window {
            *adaptive_window += 1;
            self.adaptive_window_grows += 1;
        }
    }

    pub(super) fn shrink_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        decision: VerifyWindowDecision,
    ) {
        if !adaptive_enabled {
            return;
        }
        let Some(repair_input_count) = decision.repair_input_count else {
            return;
        };
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(repair_input_count)
            .max(1);
        if next_window < *adaptive_window {
            *adaptive_window = next_window;
            self.adaptive_window_shrinks += 1;
        }
    }

    pub(super) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        if self.windows == 0 {
            attrs.insert("llama_stage.spec.enabled".to_string(), json!(false));
            return;
        }
        attrs.insert("llama_stage.spec.enabled".to_string(), json!(true));
        attrs.insert("llama_stage.spec.windows".to_string(), json!(self.windows));
        attrs.insert(
            "llama_stage.spec.proposed".to_string(),
            json!(self.draft_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accepted".to_string(),
            json!(self.accepted_tokens),
        );
        attrs.insert(
            "llama_stage.spec.rejected".to_string(),
            json!(self.rejected_tokens),
        );
        attrs.insert(
            "llama_stage.spec.accept_rate".to_string(),
            json!(if self.draft_tokens == 0 {
                0.0
            } else {
                self.accepted_tokens as f64 / self.draft_tokens as f64
            }),
        );
        attrs.insert(
            "llama_stage.spec.full_accept_windows".to_string(),
            json!(self.full_accept_windows),
        );
        attrs.insert(
            "llama_stage.spec.accepted_stop_windows".to_string(),
            json!(self.accepted_stop_windows),
        );
        attrs.insert(
            "llama_stage.spec.rejected_windows".to_string(),
            json!(self.rejected_windows),
        );
        attrs.insert(
            "llama_stage.spec.early_reject_windows".to_string(),
            json!(self.early_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.tail_reject_windows".to_string(),
            json!(self.tail_reject_windows),
        );
        attrs.insert(
            "llama_stage.spec.repair_required_windows".to_string(),
            json!(self.repair_required_windows),
        );
        attrs.insert(
            "llama_stage.spec.draft_reset_ms".to_string(),
            json!(self.draft_reset_ms),
        );
        attrs.insert(
            "llama_stage.spec.draft_propose_ms".to_string(),
            json!(self.draft_propose_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_elapsed_ms".to_string(),
            json!(self.primary_verify_elapsed_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_stage0_compute_ms".to_string(),
            json!(self.primary_verify_stage0_compute_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_wait_ms".to_string(),
            json!(self.primary_verify_runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_runtime_lock_hold_ms".to_string(),
            json!(self.primary_verify_runtime_lock_hold_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_activation_encode_ms".to_string(),
            json!(self.primary_verify_activation_encode_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_write_ms".to_string(),
            json!(self.primary_verify_forward_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_downstream_wait_ms".to_string(),
            json!(self.primary_verify_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_output_activation_bytes".to_string(),
            json!(self.primary_verify_output_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.primary_verify_forward_activation_bytes".to_string(),
            json!(self.primary_verify_forward_activation_bytes),
        );
        attrs.insert(
            "llama_stage.spec.checkpoint_ms".to_string(),
            json!(self.checkpoint_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restores".to_string(),
            json!(self.recovery_restores),
        );
        attrs.insert(
            "llama_stage.spec.recovery_ms".to_string(),
            json!(self.recovery_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_local_ms".to_string(),
            json!(self.recovery_restore_local_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_write_ms".to_string(),
            json!(self.recovery_restore_downstream_write_ms),
        );
        attrs.insert(
            "llama_stage.spec.recovery_restore_downstream_wait_ms".to_string(),
            json!(self.recovery_restore_downstream_wait_ms),
        );
        attrs.insert(
            "llama_stage.spec.adaptive_enabled".to_string(),
            json!(self.adaptive_window_enabled),
        );
        attrs.insert(
            "llama_stage.spec.window_start".to_string(),
            json!(self.adaptive_window_start),
        );
        attrs.insert(
            "llama_stage.spec.window_final".to_string(),
            json!(self.adaptive_window_final),
        );
        attrs.insert(
            "llama_stage.spec.window_max".to_string(),
            json!(self.adaptive_window_max),
        );
        attrs.insert(
            "llama_stage.spec.window_min".to_string(),
            json!(self.adaptive_window_min),
        );
        attrs.insert(
            "llama_stage.spec.window_max_seen".to_string(),
            json!(self.adaptive_window_max_seen),
        );
        attrs.insert(
            "llama_stage.spec.window_grows".to_string(),
            json!(self.adaptive_window_grows),
        );
        attrs.insert(
            "llama_stage.spec.window_shrinks".to_string(),
            json!(self.adaptive_window_shrinks),
        );
    }
}

#[cfg(test)]
mod ngram_tests {
    use super::*;

    #[test]
    fn proposes_tokens_after_latest_matching_suffix() {
        let history = [1, 2, 3, 4, 9, 2, 3, 4];

        assert_eq!(propose_ngram_tokens(&history, 2, 2).unwrap(), vec![9, 2]);
    }

    #[test]
    fn returns_empty_without_enough_history() {
        assert!(propose_ngram_tokens(&[1, 2, 3], 2, 4).unwrap().is_empty());
        assert!(
            propose_ngram_tokens(&[1, 2, 1, 2], 0, 4)
                .unwrap()
                .is_empty()
        );
        assert!(
            propose_ngram_tokens(&[1, 2, 1, 2], 1, 0)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn cache_proposer_syncs_only_the_committed_prefix() {
        let mut proposer = CachedNgramProposer::new(2, 2).unwrap();
        let history = [1, 2, 3, 1, 2, 3, 1, 2];

        assert_eq!(proposer.propose(&history, &[], 2).unwrap(), vec![3, 1]);
        assert_eq!(
            proposer.propose(&history, &[9], 2).unwrap(),
            Vec::<i32>::new()
        );
        assert_eq!(proposer.propose(&history, &[], 2).unwrap(), vec![3, 1]);
    }
}

pub(super) fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len());
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerifyWindowDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowDecision {
    pub(super) kind: VerifyWindowDecisionKind,
    pub(super) accepted_before_reject: usize,
    pub(super) repair_input_count: Option<usize>,
    pub(super) commit_count: usize,
}

impl VerifyWindowDecision {
    pub(super) fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifyWindowDecisionKind::TailReject
                | VerifyWindowDecisionKind::EarlyReject
                | VerifyWindowDecisionKind::EarlyRejectStop
        )
    }

    pub(super) fn requires_repair(self) -> bool {
        self.kind == VerifyWindowDecisionKind::EarlyReject
    }
}

pub(super) fn classify_verify_window<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<VerifyWindowDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        return Err(OpenAiError::backend(format!(
            "verify window returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        )));
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if (reached_eog || reached_limit) && commit_count < draft_tokens.len() {
                return Ok(VerifyWindowDecision {
                    kind: VerifyWindowDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    repair_input_count: None,
                    commit_count,
                });
            }
            continue;
        }

        let repair_input_count = accepted_before_reject + 1;
        let kind = if repair_input_count == draft_tokens.len() {
            VerifyWindowDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifyWindowDecisionKind::EarlyRejectStop
        } else {
            VerifyWindowDecisionKind::EarlyReject
        };
        return Ok(VerifyWindowDecision {
            kind,
            accepted_before_reject,
            repair_input_count: Some(repair_input_count),
            commit_count,
        });
    }

    Ok(VerifyWindowDecision {
        kind: VerifyWindowDecisionKind::FullAccept,
        accepted_before_reject,
        repair_input_count: None,
        commit_count,
    })
}

pub(super) fn repaired_commit_tokens(
    draft_tokens: &[i32],
    accepted_before_reject: usize,
    repair_input_count: usize,
    repaired_predictions: &[i32],
) -> OpenAiResult<Vec<i32>> {
    if repaired_predictions.len() < repair_input_count {
        return Err(OpenAiError::backend(format!(
            "recovery verify returned too few tokens: expected {} got {:?}",
            repair_input_count, repaired_predictions
        )));
    }
    if accepted_before_reject > 0
        && repaired_predictions[..accepted_before_reject] != draft_tokens[..accepted_before_reject]
    {
        eprintln!(
            "recovery verify changed accepted prefix; committing restored target tokens: accepted {:?}, repaired {:?}",
            &draft_tokens[..accepted_before_reject],
            &repaired_predictions[..accepted_before_reject]
        );
    }
    Ok(repaired_predictions[..repair_input_count].to_vec())
}

pub(super) fn nonzero_min(current: usize, candidate: usize) -> usize {
    if current == 0 {
        candidate
    } else {
        current.min(candidate)
    }
}
