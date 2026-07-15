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

#[cfg(test)]
mod tests {
    use super::*;

    fn options() -> NativeMtpDecodeOptions {
        NativeMtpDecodeOptions {
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: 0,
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
}
