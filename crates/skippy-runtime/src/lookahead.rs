use std::collections::{BTreeMap, VecDeque};

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LookaheadConfig {
    pub ngram_size: usize,
    pub window_size: usize,
    pub max_candidates: usize,
    pub candidates_per_token: usize,
    pub jacobi_on_miss: bool,
}

impl LookaheadConfig {
    pub fn validate(self) -> Result<Self> {
        if self.ngram_size < 2 {
            bail!("lookahead ngram_size must be at least two");
        }
        if self.window_size == 0 {
            bail!("lookahead window_size must be greater than zero");
        }
        if self.max_candidates == 0 || self.candidates_per_token == 0 {
            bail!("lookahead candidate limits must be greater than zero");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookaheadBranchPlan {
    pub token_ids: Vec<i32>,
    pub position_offsets: Vec<u32>,
    pub sequence_offsets: Vec<u32>,
    pub sequence_ids: Vec<u32>,
    pub sequence_count: u32,
    branches: Vec<LookaheadBranch>,
    lookahead_sequence_ids: Vec<u32>,
}

impl LookaheadBranchPlan {
    pub fn candidate_count(&self) -> usize {
        self.branches
            .iter()
            .filter(|branch| branch.expected_tokens.is_some())
            .count()
    }

    pub fn lookahead_count(&self) -> usize {
        self.lookahead_sequence_ids.len()
    }

    pub fn branch_row_indices(&self, sequence_id: u32) -> Option<&[usize]> {
        self.branches
            .get(sequence_id as usize)
            .map(|branch| branch.row_indices.as_slice())
    }

    pub fn branch_input_tokens(&self, sequence_id: u32) -> Option<&[i32]> {
        self.branches
            .get(sequence_id as usize)
            .map(|branch| branch.input_tokens.as_slice())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LookaheadBranch {
    input_tokens: Vec<i32>,
    expected_tokens: Option<Vec<i32>>,
    row_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookaheadDecision {
    pub sequence_id: u32,
    pub accepted_candidate_tokens: usize,
    pub commit_input_tokens: Vec<i32>,
    pub emitted_target_tokens: Vec<i32>,
    pub candidate_count: usize,
    pub branch_rows: usize,
}

#[derive(Debug, Clone)]
pub struct LookaheadState {
    config: LookaheadConfig,
    history: Vec<i32>,
    levels: Vec<Vec<i32>>,
    pool: BTreeMap<i32, VecDeque<Vec<i32>>>,
}

impl LookaheadState {
    pub fn new(config: LookaheadConfig, context_tokens: &[i32]) -> Result<Self> {
        let config = config.validate()?;
        if context_tokens.is_empty() {
            bail!("lookahead context must contain at least one token");
        }
        let mut state = Self {
            config,
            history: context_tokens.to_vec(),
            levels: seed_levels(context_tokens, config.ngram_size - 1, config.window_size),
            pool: BTreeMap::new(),
        };
        state.record_all_history_ngrams();
        Ok(state)
    }

    pub fn record_candidate(&mut self, key: i32, continuation: &[i32]) -> Result<()> {
        if continuation.is_empty() {
            bail!("lookahead candidate continuation must not be empty");
        }
        let continuation = continuation
            .iter()
            .copied()
            .take(self.config.ngram_size - 1)
            .collect::<Vec<_>>();
        let entries = self.pool.entry(key).or_default();
        if let Some(index) = entries
            .iter()
            .position(|candidate| candidate == &continuation)
        {
            entries.remove(index);
        }
        entries.push_back(continuation);
        while entries.len() > self.config.candidates_per_token {
            entries.pop_front();
        }
        Ok(())
    }

    pub fn plan(&self, current: i32, max_tokens: usize) -> Result<LookaheadBranchPlan> {
        if max_tokens == 0 {
            bail!("lookahead plan requires a positive token limit");
        }
        if self.history.last().copied() != Some(current) {
            bail!("lookahead current token does not match committed history");
        }

        let mut candidates = Vec::with_capacity(self.config.max_candidates);
        let history_end = self.history.len().saturating_sub(1);
        let history_candidate = crate::ngram_simple_draft(
            &self.history[..history_end],
            current,
            self.config.ngram_size,
            self.config.window_size.min(max_tokens),
        )?;
        if !history_candidate.is_empty() {
            candidates.push(history_candidate);
        }
        for candidate in history_ngram_candidates(
            &self.history,
            self.config.ngram_size,
            self.config.window_size.min(max_tokens),
            self.config.max_candidates,
        ) {
            if candidates.len() == self.config.max_candidates {
                break;
            }
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
        let pooled_candidates = self
            .pool
            .get(&current)
            .into_iter()
            .flat_map(|entries| entries.iter().rev())
            .map(|candidate| {
                candidate
                    .iter()
                    .copied()
                    .take(max_tokens)
                    .collect::<Vec<_>>()
            })
            .filter(|candidate| !candidate.is_empty())
            .collect::<Vec<_>>();
        for candidate in pooled_candidates {
            if candidates.len() == self.config.max_candidates {
                break;
            }
            if !candidates.contains(&candidate) {
                candidates.push(candidate);
            }
        }
        let lookahead_width = if candidates.is_empty() && self.config.jacobi_on_miss {
            self.config.window_size.min(max_tokens).max(1)
        } else {
            0
        };
        let candidate_mode = !candidates.is_empty();
        let base_sequence_count = if candidate_mode { candidates.len() } else { 1 };
        let lookahead_sequence_start =
            u32::try_from(base_sequence_count).context("lookahead sequence id exceeds u32")?;
        let sequence_count = lookahead_sequence_start
            .checked_add(u32::try_from(lookahead_width).context("lookahead width exceeds u32")?)
            .context("lookahead sequence count overflow")?;

        let mut token_ids = vec![current];
        let mut position_offsets = vec![0];
        let mut memberships = vec![Vec::new()];
        let mut branches = Vec::with_capacity(sequence_count as usize);
        if !candidate_mode {
            memberships[0].push(0);
            branches.push(LookaheadBranch {
                input_tokens: vec![current],
                expected_tokens: None,
                row_indices: vec![0],
            });
        }

        for (candidate_index, candidate) in candidates.into_iter().enumerate() {
            let sequence_id =
                u32::try_from(candidate_index).context("candidate sequence id exceeds u32")?;
            memberships[0].push(sequence_id);
            let mut input_tokens = vec![current];
            input_tokens.extend(
                candidate
                    .iter()
                    .copied()
                    .take(candidate.len().saturating_sub(1)),
            );
            let mut row_indices = vec![0];
            append_branch_rows(
                sequence_id,
                &input_tokens[1..],
                &mut token_ids,
                &mut position_offsets,
                &mut memberships,
                &mut row_indices,
                1,
            )?;
            branches.push(LookaheadBranch {
                input_tokens,
                expected_tokens: Some(candidate),
                row_indices,
            });
        }

        let mut lookahead_sequence_ids = Vec::with_capacity(lookahead_width);
        for column in 0..lookahead_width {
            let sequence_id = lookahead_sequence_start
                .checked_add(u32::try_from(column).context("lookahead column exceeds u32")?)
                .context("lookahead sequence id overflow")?;
            lookahead_sequence_ids.push(sequence_id);

            let mut input_tokens = Vec::with_capacity(self.config.ngram_size - 1);
            let mut row_indices = Vec::with_capacity(self.config.ngram_size - 1);
            if column == 0 {
                memberships[0].push(sequence_id);
                input_tokens.push(current);
                row_indices.push(0);
            } else {
                input_tokens.push(
                    *self.levels[0]
                        .get(column - 1)
                        .context("lookahead level zero is shorter than its window")?,
                );
            }
            for level in self.levels.iter().skip(1) {
                input_tokens.push(
                    *level
                        .get(column)
                        .context("lookahead level is shorter than its window")?,
                );
            }

            let first_unshared = usize::from(column == 0);
            append_branch_rows(
                sequence_id,
                &input_tokens[first_unshared..],
                &mut token_ids,
                &mut position_offsets,
                &mut memberships,
                &mut row_indices,
                u32::try_from(first_unshared).context("lookahead position exceeds u32")?,
            )?;
            branches.push(LookaheadBranch {
                input_tokens,
                expected_tokens: None,
                row_indices,
            });
        }

        let mut sequence_offsets = Vec::with_capacity(token_ids.len() + 1);
        let mut sequence_ids = Vec::new();
        sequence_offsets.push(0);
        for row_memberships in memberships {
            sequence_ids.extend(row_memberships);
            sequence_offsets.push(
                u32::try_from(sequence_ids.len()).context("lookahead memberships exceed u32")?,
            );
        }

        Ok(LookaheadBranchPlan {
            token_ids,
            position_offsets,
            sequence_offsets,
            sequence_ids,
            sequence_count,
            branches,
            lookahead_sequence_ids,
        })
    }

    pub fn observe(
        &mut self,
        plan: &LookaheadBranchPlan,
        predictions: &[i32],
    ) -> Result<LookaheadDecision> {
        if predictions.len() != plan.token_ids.len() {
            bail!("lookahead prediction count must match branch row count");
        }
        let mut lookahead_predictions = Vec::with_capacity(plan.lookahead_sequence_ids.len());
        for sequence_id in &plan.lookahead_sequence_ids {
            let branch = plan
                .branches
                .get(*sequence_id as usize)
                .context("lookahead branch is missing from plan")?;
            let branch_outputs = branch_predictions(branch, predictions)?;
            let new_result = *branch_outputs
                .last()
                .context("lookahead branch produced no prediction")?;
            let mut continuation = branch.input_tokens[1..].to_vec();
            continuation.push(new_result);
            self.record_candidate(branch.input_tokens[0], &continuation)?;
            lookahead_predictions.push(new_result);
        }

        let mut best_sequence_id = 0_u32;
        let mut best_accepted = 0_usize;
        for (sequence_id, branch) in plan.branches.iter().enumerate() {
            let Some(expected) = branch.expected_tokens.as_deref() else {
                continue;
            };
            let branch_predictions = branch_predictions(branch, predictions)?;
            let accepted = expected
                .iter()
                .zip(branch_predictions.iter())
                .take_while(|(expected, predicted)| expected == predicted)
                .count();
            if accepted > best_accepted {
                best_accepted = accepted;
                best_sequence_id = u32::try_from(sequence_id)
                    .context("accepted lookahead sequence id exceeds u32")?;
            }
        }

        let selected = plan
            .branches
            .get(best_sequence_id as usize)
            .context("selected lookahead branch is missing")?;
        let selected_predictions = branch_predictions(selected, predictions)?;
        let commit_count = selected
            .expected_tokens
            .as_ref()
            .map_or(1, |_| (best_accepted + 1).min(selected.input_tokens.len()));
        let commit_input_tokens = selected.input_tokens[..commit_count].to_vec();
        let emitted_target_tokens = selected_predictions[..commit_count].to_vec();

        if !lookahead_predictions.is_empty() {
            self.advance_levels(&lookahead_predictions);
        }
        let prior_history_len = self.history.len();
        self.history.extend_from_slice(&emitted_target_tokens);
        self.record_new_history_ngrams(prior_history_len);

        Ok(LookaheadDecision {
            sequence_id: best_sequence_id,
            accepted_candidate_tokens: best_accepted,
            commit_input_tokens,
            emitted_target_tokens,
            candidate_count: plan.candidate_count(),
            branch_rows: plan.token_ids.len(),
        })
    }

    pub fn observe_serial(&mut self, current: i32, predicted: i32) -> Result<()> {
        if self.history.last().copied() != Some(current) {
            bail!("lookahead serial token does not match committed history");
        }
        let prior_history_len = self.history.len();
        self.history.push(predicted);
        self.record_new_history_ngrams(prior_history_len);
        Ok(())
    }

    fn record_all_history_ngrams(&mut self) {
        let ngram_size = self.config.ngram_size;
        let ngrams = self
            .history
            .windows(ngram_size)
            .map(<[i32]>::to_vec)
            .collect::<Vec<_>>();
        for ngram in ngrams {
            let _ = self.record_candidate(ngram[0], &ngram[1..]);
        }
    }

    fn record_new_history_ngrams(&mut self, prior_history_len: usize) {
        let ngram_size = self.config.ngram_size;
        let first_end = prior_history_len.saturating_add(1).max(ngram_size);
        let ngrams = (first_end..=self.history.len())
            .map(|end| self.history[end - ngram_size..end].to_vec())
            .collect::<Vec<_>>();
        for ngram in ngrams {
            let _ = self.record_candidate(ngram[0], &ngram[1..]);
        }
    }

    fn advance_levels(&mut self, predictions: &[i32]) {
        if self.levels.len() == 1 {
            self.levels[0] = predictions.to_vec();
            return;
        }
        let previous = self.levels.clone();
        self.levels[0] = previous[1].iter().copied().skip(1).collect();
        let last = self.levels.len() - 1;
        self.levels[1..last].clone_from_slice(&previous[2..]);
        self.levels[last] = predictions.to_vec();
    }
}

fn append_branch_rows(
    sequence_id: u32,
    tokens: &[i32],
    token_ids: &mut Vec<i32>,
    position_offsets: &mut Vec<u32>,
    memberships: &mut Vec<Vec<u32>>,
    row_indices: &mut Vec<usize>,
    first_position: u32,
) -> Result<()> {
    for (position, token) in tokens.iter().copied().enumerate() {
        row_indices.push(token_ids.len());
        token_ids.push(token);
        position_offsets.push(
            first_position
                .checked_add(u32::try_from(position).context("lookahead position exceeds u32")?)
                .context("lookahead position overflow")?,
        );
        memberships.push(vec![sequence_id]);
    }
    Ok(())
}

fn branch_predictions(branch: &LookaheadBranch, predictions: &[i32]) -> Result<Vec<i32>> {
    branch
        .row_indices
        .iter()
        .map(|row| {
            predictions
                .get(*row)
                .copied()
                .context("lookahead branch row is outside predictions")
        })
        .collect()
}

fn seed_levels(context_tokens: &[i32], level_count: usize, window_size: usize) -> Vec<Vec<i32>> {
    (0..level_count)
        .map(|level| {
            let level_size = if level_count > 1 && level == 0 {
                window_size.saturating_sub(1)
            } else {
                window_size
            };
            context_tokens
                .iter()
                .rev()
                .copied()
                .cycle()
                .skip(level)
                .take(level_size)
                .collect()
        })
        .collect()
}

fn history_ngram_candidates(
    history: &[i32],
    ngram_size: usize,
    candidate_window: usize,
    max_candidates: usize,
) -> Vec<Vec<i32>> {
    let Some((current, prefix)) = history.split_last() else {
        return Vec::new();
    };
    if ngram_size == 0 || candidate_window == 0 || prefix.len() <= ngram_size + 1 {
        return Vec::new();
    }

    let pattern_start = prefix.len() + 1 - ngram_size;
    let mut pattern = prefix[pattern_start..].to_vec();
    pattern.push(*current);
    let mut candidates = Vec::with_capacity(max_candidates);
    let latest_start = prefix.len().saturating_sub(ngram_size + 1);
    for match_start in (1..=latest_start).rev() {
        if prefix.get(match_start..match_start + ngram_size) != Some(pattern.as_slice()) {
            continue;
        }
        let continuation_start = match_start + ngram_size;
        let continuation_len = candidate_window.min(prefix.len() - continuation_start);
        if continuation_len < ngram_size {
            continue;
        }
        let candidate = prefix[continuation_start..continuation_start + continuation_len].to_vec();
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
            if candidates.len() == max_candidates {
                break;
            }
        }
    }
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> LookaheadConfig {
        LookaheadConfig {
            ngram_size: 4,
            window_size: 4,
            max_candidates: 4,
            candidates_per_token: 8,
            jacobi_on_miss: true,
        }
    }

    #[test]
    fn plan_encodes_shared_root_candidate_branch() -> Result<()> {
        let mut state = LookaheadState::new(config(), &[7, 8, 9, 10])?;
        state.record_candidate(10, &[20, 21, 22])?;

        let plan = state.plan(10, 4)?;

        assert_eq!(plan.sequence_count, 1);
        assert_eq!(plan.token_ids[0], 10);
        assert_eq!(&plan.sequence_ids[..1], &[0]);
        assert_eq!(plan.branch_input_tokens(0), Some(&[10, 20, 21][..]));
        assert_eq!(plan.branch_row_indices(0), Some(&[0, 1, 2][..]));
        Ok(())
    }

    #[test]
    fn plan_encodes_independent_jacobi_diagonals_on_candidate_miss() -> Result<()> {
        let state = LookaheadState::new(config(), &[7, 8, 9, 10])?;

        let plan = state.plan(10, 4)?;

        assert_eq!(plan.sequence_count, 5);
        assert_eq!(&plan.sequence_ids[..2], &[0, 1]);
        assert_eq!(plan.branch_row_indices(1), Some(&[0, 1, 2][..]));
        assert_eq!(plan.branch_row_indices(2), Some(&[3, 4, 5][..]));
        Ok(())
    }

    #[test]
    fn full_candidate_accept_commits_one_target_forward_without_repair() -> Result<()> {
        let mut state = LookaheadState::new(config(), &[7, 8, 9, 10])?;
        state.record_candidate(10, &[20, 21, 22])?;
        let plan = state.plan(10, 4)?;
        let mut predictions = vec![0; plan.token_ids.len()];
        for (row, token) in plan
            .branch_row_indices(0)
            .context("candidate branch missing")?
            .iter()
            .zip([20, 21, 22])
        {
            predictions[*row] = token;
        }

        let decision = state.observe(&plan, &predictions)?;

        assert_eq!(decision.sequence_id, 0);
        assert_eq!(decision.accepted_candidate_tokens, 3);
        assert_eq!(decision.commit_input_tokens, vec![10, 20, 21]);
        assert_eq!(decision.emitted_target_tokens, vec![20, 21, 22]);
        Ok(())
    }

    #[test]
    fn rejection_commits_target_mismatch_without_restore_or_repair() -> Result<()> {
        let mut state = LookaheadState::new(config(), &[7, 8, 9, 10])?;
        state.record_candidate(10, &[20, 21, 22])?;
        let plan = state.plan(10, 4)?;
        let mut predictions = vec![0; plan.token_ids.len()];
        for (row, token) in plan
            .branch_row_indices(0)
            .context("candidate branch missing")?
            .iter()
            .zip([20, 99, 100])
        {
            predictions[*row] = token;
        }

        let decision = state.observe(&plan, &predictions)?;

        assert_eq!(decision.sequence_id, 0);
        assert_eq!(decision.accepted_candidate_tokens, 1);
        assert_eq!(decision.commit_input_tokens, vec![10, 20]);
        assert_eq!(decision.emitted_target_tokens, vec![20, 99]);
        Ok(())
    }

    #[test]
    fn history_lookup_returns_distinct_recent_continuations() {
        let history = [1, 7, 8, 20, 21, 22, 7, 8, 30, 31, 32, 7, 8];

        let candidates = history_ngram_candidates(&history, 2, 3, 4);

        assert_eq!(candidates, vec![vec![30, 31, 32], vec![20, 21, 22]]);
    }
}
