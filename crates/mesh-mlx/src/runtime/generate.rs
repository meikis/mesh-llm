//! Token generation loop with single-node and pipeline-parallel execution.
//!
//! The forward pass per step, on rank `r` of a pipeline of size `N`:
//!   1. The first-forward stage (highest rank) embeds the input ids.
//!      Other stages `recv_like` the hidden state from rank `r+1`.
//!   2. Run this rank's owned layers.
//!   3. If not the output stage, `send` the hidden state to rank `r-1`.
//!   4. The output stage (rank 0) runs norm + lm_head → logits, samples the
//!      next token, then `all_gather` distributes it so every stage advances.
//!
//! For `N == 1` this collapses to a plain local forward.

use crate::Result;
use crate::array::{Array, Stream};
use crate::distributed::{Group, Pipeline};
use crate::models::{LayerCache, LlamaModel};
use crate::ops;

/// Greedy argmax sampling over the last position's logits.
fn sample_greedy(logits: &Array, s: &Stream) -> Result<i32> {
    // logits: [B, L, vocab] — flatten to [B*L, vocab] and argmax each row;
    // the last row corresponds to the most recent position.
    let shape = logits.shape();
    let (b, l, vocab) = (shape[0], shape[1], shape[2]);
    let flat = ops::reshape(logits, &[b * l, vocab], s)?;
    let arg = ops::argmax(&flat, s)?; // [b*l]
    let ids = arg.to_vec_i32()?;
    Ok(*ids.last().unwrap_or(&0))
}

/// Generate up to `max_tokens` greedily from a prompt token sequence,
/// single-node (no group). Returns the generated token ids.
pub fn generate_local(
    model: &LlamaModel<'_>,
    _pipeline: &Pipeline,
    prompt_ids: &[i32],
    max_tokens: usize,
    eos: impl Fn(i32) -> bool,
    s: &Stream,
) -> Result<Vec<i32>> {
    let mut cache = model.new_cache();

    // Prefill the prompt in one forward.
    let ids = Array::from_i32(prompt_ids, &[1, prompt_ids.len() as i32])?;
    let h = model.embed(&ids, s)?;
    let h = model.forward_layers(h, &mut cache, s)?;
    let logits = model.head(&h, s)?;
    let mut next = sample_greedy(&logits, s)?;

    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        if eos(next) {
            break;
        }
        out.push(next);
        // Decode one token at a time: embed → layers → head → sample.
        let step_ids = Array::from_i32(&[next], &[1, 1])?;
        let h = model.embed(&step_ids, s)?;
        let h = model.forward_layers(h, &mut cache, s)?;
        let logits = model.head(&h, s)?;
        next = sample_greedy(&logits, s)?;
    }
    Ok(out)
}

/// Generate greedily across a pipeline-parallel [`Group`].
///
/// Every rank runs this in lock-step. The token-id sequence is replicated on
/// all ranks (kept in sync by the per-step `all_gather` of the chosen token),
/// so each rank can embed/locate positions identically. Only the output stage
/// (rank 0) computes logits; the chosen token is broadcast so all ranks advance
/// their KV-cache offset together.
pub fn generate_distributed(
    model: &LlamaModel<'_>,
    pipeline: &Pipeline,
    group: &Group,
    prompt_ids: &[i32],
    max_tokens: usize,
    eos: impl Fn(i32) -> bool,
    s: &Stream,
) -> Result<Vec<i32>> {
    let mut cache = model.new_cache();

    // Prefill: run the whole prompt through the pipeline once.
    let prompt = Array::from_i32(prompt_ids, &[1, prompt_ids.len() as i32])?;
    let mut next = pipeline_forward(model, pipeline, group, &prompt, &mut cache, s)?;

    let mut out = Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        if eos(next) {
            break;
        }
        out.push(next);
        let step = Array::from_i32(&[next], &[1, 1])?;
        next = pipeline_forward(model, pipeline, group, &step, &mut cache, s)?;
    }
    Ok(out)
}

/// One pipeline forward over `token_ids`, returning the next token (valid on
/// every rank after the broadcast).
fn pipeline_forward(
    model: &LlamaModel<'_>,
    pipeline: &Pipeline,
    group: &Group,
    token_ids: &Array,
    cache: &mut [LayerCache],
    s: &Stream,
) -> Result<i32> {
    // First-forward stage embeds; later stages receive the hidden state.
    let mut h = if pipeline.is_first_forward_stage() {
        model.embed(token_ids, s)?
    } else {
        let src = pipeline.recv_from().expect("non-first stage has a source");
        // Template shaped like the hidden state for this input length.
        let template = model.embed(token_ids, s)?;
        group.recv_like(&template, src, s)?
    };

    h = model.forward_layers(h, cache, s)?;

    if let Some(dst) = pipeline.send_to() {
        let dep = group.send(&h, dst, s)?;
        dep.eval()?;
    }

    // Output stage (rank 0) computes logits and samples.
    let chosen = if pipeline.is_output_stage() {
        let logits = model.head(&h, s)?;
        sample_greedy(&logits, s)?
    } else {
        0
    };

    // Broadcast the chosen token: rank 0 holds the real value, others 0; sum
    // reduces to the real token on every rank.
    let tok = Array::from_i32(&[chosen], &[1])?;
    let summed = group.all_sum(&tok, s)?;
    let vals = summed.to_vec_i32()?;
    Ok(*vals.first().unwrap_or(&chosen))
}

#[cfg(test)]
mod tests {
    // Sampling/forward require the engine; these are exercised by the
    // `link-mlx` integration test. Pure-logic coverage lives in the pipeline,
    // loader, planner, and config modules.
}
