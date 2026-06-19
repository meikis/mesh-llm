#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Train an SPD head from native product rows without loading base weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from train_product_activation_head import (
    PRODUCT_SCHEMA,
    TEACHER_SCHEMA,
    batch_indices,
    batch_product_cur_in,
    patch_reference_checkout,
    read_safetensors_metadata,
    resolve_device,
    resolve_model_dtype,
    validate_input_mode,
    validate_metadata,
    validate_product_convention,
    validate_sample_alignment,
    validate_tensor_shapes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-train an SPD speculation head from raw product tap rows and "
            "native verifier logits. This path loads AutoConfig only, not the "
            "base model weights."
        )
    )
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--product-corpus", required=True)
    parser.add_argument("--teacher-logits", required=True)
    parser.add_argument("--out-checkpoint", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--stage-layer-boundaries", required=True)
    parser.add_argument("--num-spec-layers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--hard-label-weight", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--model-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--rope-theta", type=int, default=0)
    parser.add_argument("--rotary-dim", type=int, default=0)
    parser.add_argument("--summary-json")
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    patch_reference_checkout(Path(args.reference_dir))
    sys.path.insert(0, str(Path(args.reference_dir)))

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from pipeline_model import (  # type: ignore[import-not-found]
        SpeculationHeadTransformer,
        _decoder_relevant_config,
        _get_apply_rotary_pos_emb,
    )

    product_tensors = load_file(args.product_corpus)
    teacher_tensors = load_file(args.teacher_logits)
    product_metadata = read_safetensors_metadata(Path(args.product_corpus))
    teacher_metadata = read_safetensors_metadata(Path(args.teacher_logits))
    validate_metadata(product_metadata, PRODUCT_SCHEMA, args.product_corpus)
    validate_metadata(teacher_metadata, TEACHER_SCHEMA, args.teacher_logits)
    validate_product_convention(product_metadata, args.product_corpus)
    validate_tensor_shapes(product_tensors, teacher_tensors)
    validate_sample_alignment(product_tensors, teacher_tensors)
    validate_input_mode("fresh", "raw", product_metadata, args.product_corpus)

    set_torch_seed(int(args.seed), torch)
    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.model_torch_dtype, device)
    hf_config = AutoConfig.from_pretrained(
        args.base_model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    dec_cfg = _decoder_relevant_config(hf_config)
    normalize_dense_sidecar_config(dec_cfg, args.attn_implementation)
    boundaries = parse_stage_layer_boundaries(args.stage_layer_boundaries)
    shallow_rows = product_row_hf_indices(product_tensors)
    draft_token_ids = teacher_tensors["teacher_logit_token_ids"].cpu().tolist()
    hidden_size = int(product_tensors["final_norm_weight"].shape[0])
    if int(getattr(dec_cfg, "hidden_size")) != hidden_size:
        raise ValueError(
            f"AutoConfig hidden_size={getattr(dec_cfg, 'hidden_size')} does not match "
            f"product final_norm_weight length {hidden_size}"
        )
    if len(draft_token_ids) != int(teacher_tensors["teacher_logits"].shape[1]):
        raise ValueError("teacher_logit_token_ids width does not match teacher_logits")
    rotary = build_rotary_embedding(dec_cfg, device)
    head = SpeculationHeadTransformer(
        dec_cfg,
        model_dtype,
        device,
        base_rotary_emb=rotary,
        apply_rotary_fn=_get_apply_rotary_pos_emb(hf_config),
        stage_feature_hf_indices=shallow_rows,
        num_spec_layers=int(args.num_spec_layers),
        init_weights_from_base_layer_indices=None,
        base_decoder_layers=None,
        draft_vocab_size=len(draft_token_ids),
    ).to(device)
    head.train()
    final_norm_weight = product_tensors["final_norm_weight"].to(device=device, dtype=model_dtype)
    projected_cur_in = product_tensors["cur_in"].to(device=device, dtype=model_dtype)
    raw_tap_concat = product_tensors["raw_tap_concat"].to(device=device, dtype=model_dtype)
    raw_tap_offsets = product_tensors["raw_tap_offsets"].cpu().tolist()
    raw_row_stage_ids = product_tensors["row_i_stages"].cpu().tolist()
    position_ids = product_tensors["position_ids"].to(device=device)
    teacher_logits = teacher_tensors["teacher_logits"].to(device=device, dtype=torch.float32)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    label_draft_indices = product_tensors["label_draft_indices"].to(device=device, dtype=torch.long)
    pipeline_view = HeadOnlyPipeline(head)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    sample_count = int(projected_cur_in.shape[0])
    losses: list[float] = []
    accs: list[float] = []
    hard_label_accs: list[float] = []
    steps = 0
    for _epoch in range(int(args.epochs)):
        for batch_idx in batch_indices(sample_count, int(args.batch_size)):
            batch_cur_in = batch_product_cur_in(
                pipeline_view,
                "raw",
                projected_cur_in,
                raw_tap_concat,
                raw_tap_offsets,
                raw_row_stage_ids,
                batch_idx,
            )
            student_logits = forward_head_logits(
                head,
                batch_cur_in,
                position_ids[batch_idx],
                final_norm_weight,
            )
            batch_teacher = teacher_logits[batch_idx]
            loss_parts = []
            if args.kl_weight > 0.0:
                teacher_probs = F.softmax(batch_teacher / float(args.temperature), dim=-1)
                student_log_probs = F.log_softmax(
                    student_logits / float(args.temperature),
                    dim=-1,
                )
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                loss_parts.append(float(args.kl_weight) * kl_loss * (float(args.temperature) ** 2))
            if args.hard_label_weight > 0.0:
                batch_labels = label_draft_indices[batch_idx]
                in_scope = batch_labels >= 0
                if bool(in_scope.any()):
                    hard_label_loss = F.cross_entropy(
                        student_logits[in_scope],
                        batch_labels[in_scope],
                    )
                    loss_parts.append(float(args.hard_label_weight) * hard_label_loss)
            if not loss_parts:
                continue
            loss = sum(loss_parts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = student_logits.argmax(dim=-1)
                accs.append(float((pred == teacher_argmax[batch_idx]).float().mean().cpu()))
                batch_labels = label_draft_indices[batch_idx]
                in_scope = batch_labels >= 0
                if bool(in_scope.any()):
                    hard_label_accs.append(
                        float((pred[in_scope] == batch_labels[in_scope]).float().mean().cpu())
                    )
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if args.max_steps > 0 and steps >= int(args.max_steps):
                break
        if args.max_steps > 0 and steps >= int(args.max_steps):
            break

    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config(
        args=args,
        dec_cfg=dec_cfg,
        boundaries=boundaries,
        shallow_rows=shallow_rows,
        draft_token_ids=draft_token_ids,
        hidden_size=hidden_size,
    )
    torch.save({"state_dict": head.state_dict(), "config": config}, out_checkpoint)
    write_manifest(args, out_checkpoint, Path(args.manifest_out), config)
    summary = {
        "schema": "skippy-spd-head-only-train-summary/v1",
        "base_model_load": "skipped",
        "base_model_path": args.base_model_path,
        "out_checkpoint": str(out_checkpoint),
        "manifest": str(args.manifest_out),
        "product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits),
        "input_mode": "raw",
        "sample_count": sample_count,
        "steps_completed": steps,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "initial_argmax_acc": accs[0] if accs else None,
        "final_argmax_acc": accs[-1] if accs else None,
        "initial_hard_label_acc": hard_label_accs[0] if hard_label_accs else None,
        "final_hard_label_acc": hard_label_accs[-1] if hard_label_accs else None,
        "draft_vocab_size": len(draft_token_ids),
        "stage_layer_boundaries": boundaries,
        "shallow_hidden_layer_indices": shallow_rows,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


class HeadOnlyPipeline:
    def __init__(self, head: Any) -> None:
        self.speculation_module = head


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.kl_weight < 0.0 or args.hard_label_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.kl_weight == 0.0 and args.hard_label_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")
    if args.num_spec_layers <= 0:
        raise ValueError("--num-spec-layers must be positive")
    if (args.rope_theta == 0) ^ (args.rotary_dim == 0):
        raise ValueError("--rope-theta and --rotary-dim must be provided together")


def set_torch_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def normalize_dense_sidecar_config(dec_cfg: Any, attn_implementation: str) -> None:
    setattr(dec_cfg, "_attn_implementation", attn_implementation or "sdpa")
    if not hasattr(dec_cfg, "intermediate_size"):
        fallback = getattr(dec_cfg, "moe_intermediate_size", None)
        if fallback is None:
            fallback = int(getattr(dec_cfg, "hidden_size")) * 4
        setattr(dec_cfg, "intermediate_size", int(fallback))
    if not hasattr(dec_cfg, "attention_dropout"):
        setattr(dec_cfg, "attention_dropout", 0.0)
    if not hasattr(dec_cfg, "sliding_window"):
        setattr(dec_cfg, "sliding_window", None)
    if not hasattr(dec_cfg, "rms_norm_eps"):
        setattr(dec_cfg, "rms_norm_eps", 1.0e-6)
    if not hasattr(dec_cfg, "head_dim"):
        setattr(dec_cfg, "head_dim", int(dec_cfg.hidden_size) // int(dec_cfg.num_attention_heads))


def build_rotary_embedding(dec_cfg: Any, device: Any) -> Any:
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    except ImportError:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as Qwen3RotaryEmbedding

    try:
        rotary = Qwen3RotaryEmbedding(config=dec_cfg)
    except TypeError:
        try:
            rotary = Qwen3RotaryEmbedding(dec_cfg)
        except TypeError:
            rotary = Qwen3RotaryEmbedding(
                dim=int(dec_cfg.head_dim),
                max_position_embeddings=int(getattr(dec_cfg, "max_position_embeddings", 32768)),
                base=float(getattr(dec_cfg, "rope_theta", 10000.0)),
            )
    return rotary.to(device=device)


def product_row_hf_indices(product_tensors: dict[str, Any]) -> list[list[int]]:
    flat = product_tensors["row_hf_indices_flat"].cpu().tolist()
    offsets = product_tensors["row_hf_indices_offsets"].cpu().tolist()
    rows = []
    for index in range(len(offsets) - 1):
        rows.append([int(value) for value in flat[int(offsets[index]) : int(offsets[index + 1])]])
    return rows


def forward_head_logits(
    head: Any,
    cur_in: Any,
    position_ids: Any,
    final_norm_weight: Any,
) -> Any:
    proc = head.forward_inference_g1_only_with_rotary(
        cur_in,
        position_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
    )
    final_hidden = qwen_rms_norm(proc, final_norm_weight)
    return head.lm_head(final_hidden[:, -1:, :]).squeeze(1).float()


def qwen_rms_norm(values: Any, weight: Any) -> Any:
    import torch

    scale = torch.rsqrt(values.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    return values * scale * weight


def parse_stage_layer_boundaries(value: str) -> list[int]:
    boundaries = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not boundaries:
        raise ValueError("--stage-layer-boundaries must not be empty")
    if any(boundary <= 0 for boundary in boundaries):
        raise ValueError(f"stage boundaries must be positive: {boundaries}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError(f"stage boundaries must be strictly increasing: {boundaries}")
    return boundaries


def checkpoint_config(
    *,
    args: argparse.Namespace,
    dec_cfg: Any,
    boundaries: list[int],
    shallow_rows: list[list[int]],
    draft_token_ids: list[int],
    hidden_size: int,
) -> dict[str, Any]:
    rotary = rotary_metadata(args, dec_cfg)
    return {
        "hidden_size": hidden_size,
        "vocab_size": int(getattr(dec_cfg, "vocab_size")),
        "draft_vocab_size": len(draft_token_ids),
        "num_stages": len(boundaries),
        "stage_layer_boundaries": boundaries,
        "num_spec_layers": int(args.num_spec_layers),
        "version": 10,
        "trained_with_use_deepest": False,
        "shallow_hidden_layer_indices": shallow_rows,
        "base_model_path": args.base_model_path,
        "draft_token_ids": [int(token) for token in draft_token_ids],
        **rotary,
    }


def rotary_metadata(args: argparse.Namespace, dec_cfg: Any) -> dict[str, int]:
    if args.rope_theta and args.rotary_dim:
        return {"rope_theta": int(args.rope_theta), "rotary_dim": int(args.rotary_dim)}
    head_dim = int(getattr(dec_cfg, "head_dim"))
    rope_parameters = getattr(dec_cfg, "rope_parameters", None) or {}
    rope_theta = getattr(dec_cfg, "rope_theta", None) or rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise RuntimeError("could not resolve rope_theta from AutoConfig")
    partial = getattr(dec_cfg, "partial_rotary_factor", None)
    if partial is None:
        partial = rope_parameters.get("partial_rotary_factor")
    rotary_dim = head_dim if partial is None else int(round(head_dim * float(partial)))
    return {"rope_theta": int(rope_theta), "rotary_dim": int(rotary_dim)}


def write_manifest(
    args: argparse.Namespace,
    checkpoint: Path,
    manifest_path: Path,
    config: dict[str, Any],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "skippy-spd-head/v1",
        "checkpoint": {
            "path": checkpoint.name,
            "sha256": file_sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "source": {
            "format": "torch-speculation-head-v10",
            "reference_repo": "https://github.com/yuyijiong/speculative_pipeline_decoding.git",
            "base_model_path": args.base_model_path,
            "model_type": None,
            "checkpoint_version": 10,
        },
        "topology": {
            "hidden_size": int(config["hidden_size"]),
            "vocab_size": int(config["vocab_size"]),
            "draft_vocab_size": int(config["draft_vocab_size"]),
            "num_stages": int(config["num_stages"]),
            "stage_layer_boundaries": config["stage_layer_boundaries"],
            "num_spec_layers": int(config["num_spec_layers"]),
            "trained_with_use_deepest": False,
            "shallow_hidden_layer_indices": config["shallow_hidden_layer_indices"],
            "spec_init_from_base_layers": None,
            "draft_token_ids": config["draft_token_ids"],
            "rope_theta": config.get("rope_theta"),
            "rotary_dim": config.get("rotary_dim"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    main()
