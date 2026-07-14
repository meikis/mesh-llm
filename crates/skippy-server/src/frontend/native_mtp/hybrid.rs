use openai_frontend::{OpenAiError, OpenAiResult};

use super::NativeMtpDecodeOptions;
use crate::frontend::speculative::propose_ngram_tokens;

/// Widens a native-MTP anchor without ever replacing it.
pub(in crate::frontend) trait ProposalExtender {
    fn extend(
        &self,
        anchor: i32,
        context_tokens: &[i32],
        max_proposal_tokens: usize,
    ) -> NativeMtpHybridProposal;
}

#[derive(Debug, Clone, Copy)]
pub(in crate::frontend) struct MtpAnchoredNgramExtender {
    enabled: bool,
    ngram_size: usize,
    max_proposal_tokens: usize,
}

impl MtpAnchoredNgramExtender {
    pub(in crate::frontend) fn from_options(options: NativeMtpDecodeOptions) -> Self {
        Self {
            enabled: options.ngram_hybrid,
            ngram_size: options.ngram_size,
            max_proposal_tokens: options.ngram_max_proposal_tokens,
        }
    }
}

impl ProposalExtender for MtpAnchoredNgramExtender {
    fn extend(
        &self,
        anchor: i32,
        context_tokens: &[i32],
        max_proposal_tokens: usize,
    ) -> NativeMtpHybridProposal {
        let max_proposal_tokens = max_proposal_tokens.max(1);
        if !self.enabled || self.max_proposal_tokens == 0 {
            return NativeMtpHybridProposal::anchor_only(anchor);
        }

        let proposal_limit = max_proposal_tokens.min(self.max_proposal_tokens);
        let ngram_tokens = propose_ngram_tokens(context_tokens, self.ngram_size, proposal_limit);
        let ngram_span_available = !ngram_tokens.is_empty();
        let ngram_anchor_agreed = ngram_tokens.first().is_some_and(|token| *token == anchor);
        let ngram_anchor_disagreed = ngram_span_available && !ngram_anchor_agreed;
        if ngram_anchor_agreed {
            return NativeMtpHybridProposal {
                tokens: ngram_tokens,
                ngram_span_available,
                ngram_anchor_agreed,
                ngram_anchor_disagreed,
            };
        }

        NativeMtpHybridProposal {
            tokens: vec![anchor],
            ngram_span_available,
            ngram_anchor_agreed,
            ngram_anchor_disagreed,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpHybridProposal {
    tokens: Vec<i32>,
    ngram_span_available: bool,
    ngram_anchor_agreed: bool,
    ngram_anchor_disagreed: bool,
}

impl NativeMtpHybridProposal {
    pub(in crate::frontend) fn from_native_mtp_tokens(tokens: Vec<i32>) -> Self {
        Self {
            tokens,
            ngram_span_available: false,
            ngram_anchor_agreed: false,
            ngram_anchor_disagreed: false,
        }
    }

    pub(in crate::frontend) fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    pub(in crate::frontend) fn ngram_span_available(&self) -> bool {
        self.ngram_span_available
    }

    pub(in crate::frontend) fn ngram_anchor_agreed(&self) -> bool {
        self.ngram_anchor_agreed
    }

    pub(in crate::frontend) fn ngram_anchor_disagreed(&self) -> bool {
        self.ngram_anchor_disagreed
    }

    fn anchor_only(anchor: i32) -> Self {
        Self {
            tokens: vec![anchor],
            ngram_span_available: false,
            ngram_anchor_agreed: false,
            ngram_anchor_disagreed: false,
        }
    }
}

pub(in crate::frontend) fn native_mtp_verify_inputs_for_proposals(
    current: i32,
    proposals: &[i32],
) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len().saturating_add(1));
    tokens.push(current);
    tokens.extend_from_slice(proposals);
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) struct NativeMtpBatchedDecision {
    pub(in crate::frontend) accepted_proposal_tokens: usize,
    pub(in crate::frontend) commit_count: usize,
    pub(in crate::frontend) rejected: bool,
}

pub(in crate::frontend) fn classify_native_mtp_batched_verify<F>(
    proposal_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> OpenAiResult<NativeMtpBatchedDecision>
where
    F: FnMut(i32) -> OpenAiResult<bool>,
{
    let required_predictions = proposal_tokens.len().saturating_add(1);
    if predicted_tokens.len() < required_predictions {
        return Err(OpenAiError::backend(format!(
            "native MTP verify window returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            required_predictions
        )));
    }

    let mut accepted_proposal_tokens = 0usize;
    for (index, proposal_token) in proposal_tokens.iter().enumerate() {
        let predicted = predicted_tokens[index];
        let commit_count = index + 1;
        if predicted != *proposal_token {
            return Ok(NativeMtpBatchedDecision {
                accepted_proposal_tokens,
                commit_count,
                rejected: true,
            });
        }

        accepted_proposal_tokens += 1;
        if token_is_eog(predicted)? || generated_len + commit_count >= max_new_tokens {
            return Ok(NativeMtpBatchedDecision {
                accepted_proposal_tokens,
                commit_count,
                rejected: false,
            });
        }
    }

    let extra_commit_count = proposal_tokens.len().saturating_add(1);
    Ok(NativeMtpBatchedDecision {
        accepted_proposal_tokens,
        commit_count: extra_commit_count.min(max_new_tokens.saturating_sub(generated_len)),
        rejected: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> NativeMtpDecodeOptions {
        NativeMtpDecodeOptions {
            batched_verify: true,
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: 0,
            defer_reject_trim: false,
            suppress_cooldown_drafts: false,
            suppress_cooldown_draft_limit: 0,
            ngram_hybrid: true,
            ngram_size: 2,
            ngram_max_proposal_tokens: 4,
        }
    }

    #[test]
    fn extends_only_when_ngram_agrees_with_mtp_anchor() {
        let extender = MtpAnchoredNgramExtender::from_options(options());
        let proposal = extender.extend(3, &[1, 2, 3, 4, 5, 1, 2], 4);

        assert_eq!(proposal.tokens(), &[3, 4, 5, 1]);
        assert!(proposal.ngram_span_available());
        assert!(proposal.ngram_anchor_agreed());
    }

    #[test]
    fn disagreement_keeps_the_mtp_anchor() {
        let extender = MtpAnchoredNgramExtender::from_options(options());
        let proposal = extender.extend(9, &[1, 2, 3, 4, 5, 1, 2], 4);

        assert_eq!(proposal.tokens(), &[9]);
        assert!(proposal.ngram_span_available());
        assert!(proposal.ngram_anchor_disagreed());
    }

    #[test]
    fn verification_commits_the_correction_after_rejection() {
        let decision =
            classify_native_mtp_batched_verify(&[11, 12], &[11, 42, 99], 0, 8, |_| Ok(false))
                .unwrap();

        assert_eq!(decision.accepted_proposal_tokens, 1);
        assert_eq!(decision.commit_count, 2);
        assert!(decision.rejected);
    }
}
