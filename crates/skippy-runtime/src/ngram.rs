use std::ptr;

use anyhow::{Context, Result, bail};

/// Uses llama.cpp's ngram-simple proposer against accepted history, including
/// the current sampled token as the final item in `history`.
pub fn simple_draft(
    history: &[i32],
    ngram_size: usize,
    max_draft_tokens: usize,
) -> Result<Vec<i32>> {
    if ngram_size == 0 || max_draft_tokens == 0 || history.len() < 2 {
        return Ok(Vec::new());
    }
    let ngram_size = u16::try_from(ngram_size).context("ngram size exceeds llama.cpp limit")?;
    let max_draft_tokens =
        u16::try_from(max_draft_tokens).context("N-gram draft limit exceeds llama.cpp limit")?;
    let (sampled_token, token_ids) = history
        .split_last()
        .expect("history length is checked above");
    let mut output_tokens = vec![0_i32; usize::from(max_draft_tokens)];
    let mut output_token_count = 0_usize;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_ngram_simple_draft(
            token_ids.as_ptr(),
            token_ids.len(),
            *sampled_token,
            ngram_size,
            max_draft_tokens,
            output_tokens.as_mut_ptr(),
            output_tokens.len(),
            &mut output_token_count,
            &mut error,
        )
    };
    super::ensure_ok(status, error)?;
    if output_token_count > output_tokens.len() {
        bail!("llama.cpp N-gram proposer exceeded its requested draft limit");
    }
    output_tokens.truncate(output_token_count);
    Ok(output_tokens)
}

#[cfg(test)]
mod tests {
    use super::simple_draft;

    #[test]
    fn drafts_the_continuation_from_the_latest_matching_ngram() {
        let history = [1, 2, 3, 4, 9, 2, 3, 4];

        assert_eq!(simple_draft(&history, 2, 2).unwrap(), vec![9, 2]);
    }

    #[test]
    fn respects_zero_limits_without_entering_the_native_abi() {
        assert!(simple_draft(&[1, 2, 1, 2], 0, 4).unwrap().is_empty());
        assert!(simple_draft(&[1, 2, 1, 2], 1, 0).unwrap().is_empty());
    }
}
