use std::collections::{BTreeMap, VecDeque};

use super::{OpenAiError, OpenAiResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowPipelineConfig {
    depth: usize,
}

impl VerifyWindowPipelineConfig {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
        }
    }

    pub(super) fn depth(self) -> usize {
        self.depth
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct VerifyWindowPipelineStats {
    depth: usize,
    direct_prediction_return: bool,
    opened_windows: usize,
    max_in_flight: usize,
    stale_discarded: usize,
    stale_drain_ms: f64,
}

impl VerifyWindowPipelineStats {
    pub(super) fn insert_response_timings(self, timings: &mut BTreeMap<String, serde_json::Value>) {
        timings.insert(
            "verify_window_depth".to_string(),
            serde_json::json!(self.depth),
        );
        timings.insert(
            "verify_window_direct_prediction_return".to_string(),
            serde_json::json!(self.direct_prediction_return),
        );
        timings.insert(
            "verify_window_opened".to_string(),
            serde_json::json!(self.opened_windows),
        );
        timings.insert(
            "verify_window_max_in_flight".to_string(),
            serde_json::json!(self.max_in_flight),
        );
        timings.insert(
            "verify_window_stale_discarded".to_string(),
            serde_json::json!(self.stale_discarded),
        );
        timings.insert(
            "verify_window_stale_drain_ms".to_string(),
            serde_json::json!(self.stale_drain_ms),
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerifyWindow {
    pub(super) id: i32,
    pub(super) base_position: usize,
    pub(super) decode_step: usize,
}

#[derive(Debug)]
pub(super) struct VerifyWindowScheduler {
    config: VerifyWindowPipelineConfig,
    next_id: i32,
    in_flight: VecDeque<VerifyWindow>,
    stats: VerifyWindowPipelineStats,
}

impl VerifyWindowScheduler {
    pub(super) fn new(config: VerifyWindowPipelineConfig) -> Self {
        Self {
            config,
            next_id: 1,
            in_flight: VecDeque::new(),
            stats: VerifyWindowPipelineStats {
                depth: config.depth(),
                ..VerifyWindowPipelineStats::default()
            },
        }
    }

    pub(super) fn has_capacity(&self) -> bool {
        self.in_flight.len() < self.config.depth
    }

    pub(super) fn depth(&self) -> usize {
        self.config.depth()
    }

    pub(super) fn mark_direct_prediction_return(&mut self) {
        self.stats.direct_prediction_return = true;
    }

    pub(super) fn open(
        &mut self,
        base_position: usize,
        decode_step: usize,
    ) -> OpenAiResult<VerifyWindow> {
        if !self.has_capacity() {
            return Err(OpenAiError::backend(
                "verify window pipeline depth exceeded",
            ));
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| OpenAiError::backend("verify window id overflow"))?;
        let window = VerifyWindow {
            id,
            base_position,
            decode_step,
        };
        self.in_flight.push_back(window.clone());
        self.stats.opened_windows = self.stats.opened_windows.saturating_add(1);
        self.stats.max_in_flight = self.stats.max_in_flight.max(self.in_flight.len());
        Ok(window)
    }

    pub(super) fn complete_next(&mut self, reply_window_id: i32) -> OpenAiResult<VerifyWindow> {
        let Some(window) = self.in_flight.front() else {
            return Err(OpenAiError::backend(
                "verify window reply arrived with no in-flight window",
            ));
        };
        if window.id != reply_window_id {
            return Err(OpenAiError::backend(format!(
                "verify window reply out of order: got {reply_window_id}, expected {}",
                window.id
            )));
        }
        Ok(self.in_flight.pop_front().expect("checked non-empty queue"))
    }

    #[cfg(test)]
    pub(super) fn discard_stale(&mut self) -> usize {
        let discarded = self.in_flight.len();
        self.in_flight.clear();
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(discarded);
        discarded
    }

    pub(super) fn record_stale_discarded(&mut self, count: usize, drain_ms: f64) {
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(count);
        self.stats.stale_drain_ms += drain_ms;
    }

    pub(super) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn stale_discard_count(&self) -> usize {
        self.stats.stale_discarded
    }

    pub(super) fn stats(&self) -> VerifyWindowPipelineStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_depth_and_requires_fifo_reply_ids() {
        let config = VerifyWindowPipelineConfig { depth: 2 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        let first = scheduler.open(10, 0).unwrap();
        let second = scheduler.open(11, 1).unwrap();

        assert!(scheduler.open(12, 2).is_err());
        assert!(scheduler.complete_next(second.id).is_err());
        assert_eq!(scheduler.in_flight_len(), 2);
        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
        assert_eq!(first.id, 1);
        assert_eq!(scheduler.stats().depth, 2);
        assert_eq!(scheduler.stats().opened_windows, 2);
        assert_eq!(scheduler.stats().max_in_flight, 2);
        assert!(!scheduler.stats().direct_prediction_return);
    }

    #[test]
    fn discards_stale_windows_after_divergence() {
        let config = VerifyWindowPipelineConfig { depth: 3 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        scheduler.open(10, 0).unwrap();
        scheduler.open(11, 1).unwrap();
        scheduler.open(12, 2).unwrap();

        assert_eq!(scheduler.discard_stale(), 3);
        assert_eq!(scheduler.stale_discard_count(), 3);
        assert_eq!(scheduler.in_flight_len(), 0);
        assert_eq!(scheduler.stats().stale_discarded, 3);
    }
}
