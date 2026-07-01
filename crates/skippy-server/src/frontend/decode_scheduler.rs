use std::collections::VecDeque;

use super::{OpenAiError, OpenAiResult, speculative::VerifyWindowDecision};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DecodeWindow {
    pub(super) window_id: i32,
    pub(super) base_position: usize,
    pub(super) decode_step: usize,
    pub(super) token_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DecodeWindowOutcome {
    pub(super) accepted_len: usize,
    pub(super) rejected: bool,
}

impl DecodeWindowOutcome {
    pub(super) fn accepted(accepted_len: usize) -> Self {
        Self {
            accepted_len,
            rejected: false,
        }
    }

    pub(super) fn rejected(accepted_len: usize) -> Self {
        Self {
            accepted_len,
            rejected: true,
        }
    }

    pub(super) fn from_verify_decision(decision: VerifyWindowDecision) -> Self {
        Self {
            accepted_len: decision.accepted_before_reject,
            rejected: decision.rejected(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DecodeWindowCompletion {
    pub(super) window_id: i32,
    pub(super) accepted_len: usize,
    pub(super) in_flight_after: usize,
    pub(super) stale_discarded: usize,
    pub(super) provider_reset_required: bool,
}

#[derive(Debug, Default)]
pub(super) struct DecodeWindowScheduler {
    max_inflight: usize,
    next_window_id: i32,
    in_flight: VecDeque<DecodeWindow>,
    stale_discard_count: usize,
}

impl DecodeWindowScheduler {
    pub(super) fn new(max_inflight: usize) -> Self {
        Self {
            max_inflight: max_inflight.max(1),
            next_window_id: 1,
            in_flight: VecDeque::new(),
            stale_discard_count: 0,
        }
    }

    #[cfg(test)]
    pub(super) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn stale_discard_count(&self) -> usize {
        self.stale_discard_count
    }

    pub(super) fn open_window(
        &mut self,
        base_position: usize,
        decode_step: usize,
        token_count: usize,
    ) -> OpenAiResult<DecodeWindow> {
        if token_count == 0 {
            return Err(OpenAiError::backend(
                "decode verify window requires at least one token",
            ));
        }
        if self.in_flight.len() >= self.max_inflight {
            return Err(OpenAiError::backend("decode verify window depth exceeded"));
        }
        let window_id = self.next_window_id;
        self.next_window_id = self
            .next_window_id
            .checked_add(1)
            .ok_or_else(|| OpenAiError::backend("decode verify window id overflow"))?;
        let window = DecodeWindow {
            window_id,
            base_position,
            decode_step,
            token_count,
        };
        self.in_flight.push_back(window.clone());
        Ok(window)
    }

    pub(super) fn complete_window(
        &mut self,
        window_id: i32,
        outcome: DecodeWindowOutcome,
    ) -> OpenAiResult<DecodeWindowCompletion> {
        let Some(index) = self
            .in_flight
            .iter()
            .position(|window| window.window_id == window_id)
        else {
            self.stale_discard_count = self.stale_discard_count.saturating_add(1);
            return Ok(DecodeWindowCompletion {
                window_id,
                accepted_len: 0,
                in_flight_after: self.in_flight.len(),
                stale_discarded: 1,
                provider_reset_required: true,
            });
        };

        let stale_before = index;
        self.in_flight.drain(..index);
        self.in_flight.pop_front();
        let stale_after = if outcome.rejected {
            let discarded = self.in_flight.len();
            self.in_flight.clear();
            discarded
        } else {
            0
        };
        let stale_discarded = stale_before.saturating_add(stale_after);
        self.stale_discard_count = self.stale_discard_count.saturating_add(stale_discarded);
        Ok(DecodeWindowCompletion {
            window_id,
            accepted_len: outcome.accepted_len,
            in_flight_after: self.in_flight.len(),
            stale_discarded,
            provider_reset_required: outcome.rejected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigns_monotonic_window_ids_and_tracks_depth() {
        let mut scheduler = DecodeWindowScheduler::new(2);

        let first = scheduler.open_window(10, 0, 2).unwrap();
        let second = scheduler.open_window(12, 2, 2).unwrap();

        assert_eq!(first.window_id, 1);
        assert_eq!(second.window_id, 2);
        assert_eq!(scheduler.in_flight_len(), 2);
        assert!(scheduler.open_window(14, 4, 1).is_err());
    }

    #[test]
    fn completion_by_window_id_discards_older_stale_work() {
        let mut scheduler = DecodeWindowScheduler::new(3);
        let first = scheduler.open_window(10, 0, 1).unwrap();
        let second = scheduler.open_window(11, 1, 1).unwrap();

        let completion = scheduler
            .complete_window(second.window_id, DecodeWindowOutcome::accepted(1))
            .unwrap();

        assert_eq!(first.window_id, 1);
        assert_eq!(completion.window_id, second.window_id);
        assert_eq!(completion.stale_discarded, 1);
        assert_eq!(completion.in_flight_after, 0);
        assert_eq!(scheduler.stale_discard_count(), 1);
    }

    #[test]
    fn rejection_discards_newer_windows_and_requires_provider_reset() {
        let mut scheduler = DecodeWindowScheduler::new(3);
        let first = scheduler.open_window(10, 0, 2).unwrap();
        scheduler.open_window(12, 2, 2).unwrap();

        let completion = scheduler
            .complete_window(first.window_id, DecodeWindowOutcome::rejected(1))
            .unwrap();

        assert_eq!(completion.accepted_len, 1);
        assert_eq!(completion.stale_discarded, 1);
        assert!(completion.provider_reset_required);
        assert_eq!(scheduler.in_flight_len(), 0);
    }

    #[test]
    fn unknown_window_reply_counts_as_stale() {
        let mut scheduler = DecodeWindowScheduler::new(1);

        let completion = scheduler
            .complete_window(99, DecodeWindowOutcome::accepted(1))
            .unwrap();

        assert_eq!(completion.window_id, 99);
        assert_eq!(completion.stale_discarded, 1);
        assert!(completion.provider_reset_required);
        assert_eq!(scheduler.stale_discard_count(), 1);
    }
}
