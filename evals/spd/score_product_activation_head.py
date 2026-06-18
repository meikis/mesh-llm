#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Score an SPD head on captured product activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from train_product_activation_head import (
    PRODUCT_SCHEMA,
    TEACHER_SCHEMA,
    batch_indices,
    patch_reference_checkout,
    read_safetensors_metadata,
    resolve_device,
    resolve_model_dtype,
    validate_metadata,
    validate_product_convention,
    validate_sample_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an SPD speculation head on product-captured cur_in rows and "
            "report native-target and optional teacher agreement."
        )
    )
    parser.add_argument("--reference-dir", required=True, help="Reference SPD repo checkout")
    parser.add_argument("--checkpoint", required=True, help="speculation_head_final.pt")
    parser.add_argument("--product-corpus", required=True, help="Product corpus safetensors")
    parser.add_argument("--teacher-logits", help="Optional teacher logits safetensors")
    parser.add_argument("--base-model-path", help="Override checkpoint config base_model_path")
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--model-torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    patch_reference_checkout(Path(args.reference_dir))
    sys.path.insert(0, str(Path(args.reference_dir)))

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from pipeline_inference import (  # type: ignore[import-not-found]
        _infer_pipeline_kind,
        _read_spec_config,
        build_pipeline_from_spec_ckpt,
    )

    product_tensors = load_file(args.product_corpus)
    product_metadata = read_safetensors_metadata(Path(args.product_corpus))
    validate_metadata(product_metadata, PRODUCT_SCHEMA, args.product_corpus)
    validate_product_convention(product_metadata, args.product_corpus)
    validate_product_tensors(product_tensors, args.product_corpus)

    teacher_tensors: dict[str, Any] | None = None
    teacher_metadata: dict[str, str] | None = None
    if args.teacher_logits:
        teacher_tensors = load_file(args.teacher_logits)
        teacher_metadata = read_safetensors_metadata(Path(args.teacher_logits))
        validate_metadata(teacher_metadata, TEACHER_SCHEMA, args.teacher_logits)
        validate_teacher_tensors(product_tensors, teacher_tensors, args.teacher_logits)
        validate_sample_alignment(product_tensors, teacher_tensors)

    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.model_torch_dtype, device)
    checkpoint_path = Path(args.checkpoint)
    spec_cfg = _read_spec_config(str(checkpoint_path))
    _infer_pipeline_kind(spec_cfg)
    base_model_path = args.base_model_path or str(spec_cfg["base_model_path"])
    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs).to(device)
    base_model.eval()
    pipeline = build_pipeline_from_spec_ckpt(
        base_model,
        str(checkpoint_path),
        spec_cfg,
        map_location="cpu",
    ).to(device)
    pipeline.eval()
    pipeline.base_model.requires_grad_(False)
    pipeline.speculation_module.eval()

    metrics = score_product_rows(
        args,
        pipeline,
        product_tensors,
        teacher_tensors,
        device,
        model_dtype,
        torch,
        F,
    )
    summary = {
        "schema": "skippy-spd-product-activation-score-summary/v1",
        "checkpoint": str(checkpoint_path),
        "product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits) if args.teacher_logits else None,
        "cur_in_convention": product_metadata.get("cur_in_convention"),
        "teacher_source": teacher_metadata.get("teacher_source") if teacher_metadata else None,
        "native_product_teacher_logits": (
            teacher_metadata.get("native_product_teacher_logits") if teacher_metadata else None
        ),
        "base_model_path": base_model_path,
        "device": str(device),
        "model_torch_dtype": str(model_dtype).replace("torch.", ""),
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        **metrics,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def validate_product_tensors(product: dict[str, Any], path: str) -> None:
    required = {"cur_in", "position_ids", "label_draft_indices"}
    missing = sorted(required.difference(product))
    if missing:
        raise ValueError(f"{path} missing tensors: {missing}")
    sample_count = int(product["cur_in"].shape[0])
    if sample_count == 0:
        raise ValueError(f"{path} has no rows")
    if int(product["position_ids"].shape[0]) != sample_count:
        raise ValueError("position_ids sample count does not match cur_in")
    if int(product["label_draft_indices"].shape[0]) != sample_count:
        raise ValueError("label_draft_indices sample count does not match cur_in")


def validate_teacher_tensors(
    product: dict[str, Any],
    teacher: dict[str, Any],
    path: str,
) -> None:
    required = {"teacher_logits", "teacher_argmax_indices"}
    missing = sorted(required.difference(teacher))
    if missing:
        raise ValueError(f"{path} missing tensors: {missing}")
    if int(product["cur_in"].shape[0]) != int(teacher["teacher_logits"].shape[0]):
        raise ValueError("teacher logits sample count does not match cur_in")


def score_product_rows(
    args: argparse.Namespace,
    pipeline: Any,
    product_tensors: dict[str, Any],
    teacher_tensors: dict[str, Any] | None,
    device: Any,
    model_dtype: Any,
    torch: Any,
    F: Any,
) -> dict[str, Any]:
    cur_in = product_tensors["cur_in"].to(device=device, dtype=model_dtype)
    position_ids = product_tensors["position_ids"].to(device=device)
    labels = product_tensors["label_draft_indices"].to(device=device, dtype=torch.long)
    teacher_logits = None
    teacher_argmax = None
    if teacher_tensors is not None:
        teacher_logits = teacher_tensors["teacher_logits"].to(device=device, dtype=torch.float32)
        teacher_argmax = teacher_tensors["teacher_argmax_indices"].to(device=device)

    sample_count = int(cur_in.shape[0])
    top_k = min(int(args.top_k), int(pipeline.speculation_module.lm_head.out_features))
    totals = ScoreTotals()
    with torch.inference_mode():
        for batch_idx in batch_indices(sample_count, int(args.batch_size)):
            logits = score_batch(pipeline, cur_in[batch_idx], position_ids[batch_idx])
            batch_labels = labels[batch_idx]
            totals.add_native(logits, batch_labels, top_k, torch, F)
            if teacher_logits is not None and teacher_argmax is not None:
                totals.add_teacher(
                    logits,
                    teacher_logits[batch_idx],
                    teacher_argmax[batch_idx],
                    batch_labels,
                    top_k,
                    torch,
                    F,
                )
    return totals.to_summary(sample_count)


def score_batch(pipeline: Any, cur_in: Any, position_ids: Any) -> Any:
    proc = pipeline.speculation_module.forward_inference_g1_only_with_rotary(
        cur_in,
        position_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
    )
    final_hidden = pipeline.final_norm(proc)
    return pipeline.speculation_module.lm_head(final_hidden[:, -1:, :]).squeeze(1).float()


class ScoreTotals:
    def __init__(self) -> None:
        self.labels_in_scope = 0
        self.native_top1 = 0
        self.native_topk = 0
        self.teacher_top1 = 0
        self.teacher_topk = 0
        self.teacher_vs_native = 0
        self.teacher_topk_native = 0
        self.hard_ce_sum = 0.0
        self.hard_ce_count = 0
        self.kl_sum = 0.0
        self.kl_count = 0

    def add_native(self, logits: Any, labels: Any, top_k: int, torch: Any, F: Any) -> None:
        pred = logits.argmax(dim=-1)
        topk_indices = logits.topk(k=top_k, dim=-1).indices
        in_scope = labels >= 0
        self.labels_in_scope += int(in_scope.sum().detach().cpu())
        if bool(in_scope.any()):
            self.native_top1 += int((pred[in_scope] == labels[in_scope]).sum().detach().cpu())
            self.native_topk += int(
                (topk_indices[in_scope] == labels[in_scope].unsqueeze(-1))
                .any(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
            hard_ce = F.cross_entropy(logits[in_scope], labels[in_scope], reduction="sum")
            self.hard_ce_sum += float(hard_ce.detach().cpu())
            self.hard_ce_count += int(in_scope.sum().detach().cpu())

    def add_teacher(
        self,
        logits: Any,
        teacher_logits: Any,
        teacher_argmax: Any,
        labels: Any,
        top_k: int,
        torch: Any,
        F: Any,
    ) -> None:
        pred = logits.argmax(dim=-1)
        topk_indices = logits.topk(k=top_k, dim=-1).indices
        teacher_topk = teacher_logits.topk(k=top_k, dim=-1).indices
        self.teacher_top1 += int((pred == teacher_argmax).sum().detach().cpu())
        self.teacher_topk += int(
            (topk_indices == teacher_argmax.unsqueeze(-1)).any(dim=-1).sum().detach().cpu()
        )
        in_scope = labels >= 0
        if bool(in_scope.any()):
            self.teacher_vs_native += int(
                (teacher_argmax[in_scope] == labels[in_scope]).sum().detach().cpu()
            )
            self.teacher_topk_native += int(
                (teacher_topk[in_scope] == labels[in_scope].unsqueeze(-1))
                .any(dim=-1)
                .sum()
                .detach()
                .cpu()
            )
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(logits, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        self.kl_sum += float(kl.detach().cpu()) * int(logits.shape[0])
        self.kl_count += int(logits.shape[0])

    def to_summary(self, sample_count: int) -> dict[str, Any]:
        out = {
            "sample_count": sample_count,
            "labels_in_draft_scope": self.labels_in_scope,
            "head_top1_vs_native_target": count_rate(self.native_top1, sample_count),
            "head_top1_vs_native_in_scope": count_rate(
                self.native_top1,
                self.labels_in_scope,
            ),
            "head_topk_contains_native_target": count_rate(self.native_topk, sample_count),
            "head_topk_contains_native_in_scope": count_rate(
                self.native_topk,
                self.labels_in_scope,
            ),
            "hard_label_ce_mean": mean_or_none(self.hard_ce_sum, self.hard_ce_count),
        }
        if self.kl_count > 0:
            out.update(
                {
                    "head_top1_vs_teacher_top1": count_rate(
                        self.teacher_top1,
                        sample_count,
                    ),
                    "head_topk_contains_teacher_top1": count_rate(
                        self.teacher_topk,
                        sample_count,
                    ),
                    "teacher_top1_vs_native_in_scope": count_rate(
                        self.teacher_vs_native,
                        self.labels_in_scope,
                    ),
                    "teacher_topk_contains_native_in_scope": count_rate(
                        self.teacher_topk_native,
                        self.labels_in_scope,
                    ),
                    "kl_mean": mean_or_none(self.kl_sum, self.kl_count),
                }
            )
        return out


def count_rate(matched: int, total: int) -> dict[str, Any]:
    return {
        "matched": int(matched),
        "total": int(total),
        "rate": (float(matched) / float(total)) if total else None,
    }


def mean_or_none(total: float, count: int) -> float | None:
    if count == 0:
        return None
    return float(total) / float(count)


if __name__ == "__main__":
    main()
