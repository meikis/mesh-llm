#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Fine-tune an SPD head on captured product activations and teacher logits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from hf_train_eval_qwen06 import patch_reference_for_transformers


PRODUCT_SCHEMA = "skippy-spd-product-activation-safetensors/v1"
TEACHER_SCHEMA = "skippy-spd-product-activation-teacher-logits/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an SPD speculation head on product-captured cur_in rows with "
            "precomputed teacher logits."
        )
    )
    parser.add_argument("--reference-dir", required=True, help="Reference SPD repo checkout")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Input speculation_head_final.pt. In checkpoint mode the head weights are "
            "loaded and fine-tuned; in fresh mode only the config/topology is reused."
        ),
    )
    parser.add_argument("--product-corpus", required=True, help="Product corpus safetensors")
    parser.add_argument("--teacher-logits", required=True, help="Teacher logits safetensors")
    parser.add_argument("--out-checkpoint", required=True, help="Output speculation head checkpoint")
    parser.add_argument("--base-model-path", help="Override checkpoint config base_model_path")
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    parser.add_argument(
        "--init-mode",
        choices=("checkpoint", "fresh"),
        default="checkpoint",
        help=(
            "checkpoint: load and fine-tune the input head weights. fresh: construct "
            "the same topology from checkpoint config without loading head weights."
        ),
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "projected", "raw"),
        default="auto",
        help=(
            "projected: train on pre-projected cur_in rows. raw: train through "
            "stage_projs from raw terminal-normalized tap-concat rows. auto keeps "
            "projected rows for checkpoint mode and uses raw rows for fresh mode."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch/random seed")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=1.0,
        help="Weight for KL loss against teacher logits.",
    )
    parser.add_argument(
        "--hard-label-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for cross-entropy against product native target labels. "
            "Samples whose target token is outside the draft vocab are ignored."
        ),
    )
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
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be positive")
    if args.kl_weight < 0.0:
        raise ValueError("--kl-weight must be non-negative")
    if args.hard_label_weight < 0.0:
        raise ValueError("--hard-label-weight must be non-negative")
    if args.kl_weight == 0.0 and args.hard_label_weight == 0.0:
        raise ValueError("at least one of --kl-weight or --hard-label-weight must be positive")
    patch_reference_checkout(Path(args.reference_dir))
    sys.path.insert(0, str(Path(args.reference_dir)))

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from pipeline_inference import (  # type: ignore[import-not-found]
        _infer_pipeline_kind,
        _pipeline_init_kwargs,
        _read_spec_config,
        build_pipeline_from_spec_ckpt,
    )
    from pipeline_model import Qwen3SpeculativePipelineModel  # type: ignore[import-not-found]

    product_tensors = load_file(args.product_corpus)
    teacher_tensors = load_file(args.teacher_logits)
    product_metadata = read_safetensors_metadata(Path(args.product_corpus))
    teacher_metadata = read_safetensors_metadata(Path(args.teacher_logits))
    validate_metadata(product_metadata, PRODUCT_SCHEMA, args.product_corpus)
    validate_metadata(teacher_metadata, TEACHER_SCHEMA, args.teacher_logits)
    validate_product_convention(product_metadata, args.product_corpus)
    validate_tensor_shapes(product_tensors, teacher_tensors)
    validate_sample_alignment(product_tensors, teacher_tensors)
    input_mode = resolve_input_mode(args.input_mode, args.init_mode, product_tensors)
    validate_input_mode(args.init_mode, input_mode, product_metadata, args.product_corpus)

    set_torch_seed(int(args.seed), torch)
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
    set_torch_seed(int(args.seed), torch)
    if args.init_mode == "checkpoint":
        pipeline = build_pipeline_from_spec_ckpt(
            base_model,
            str(checkpoint_path),
            spec_cfg,
            map_location="cpu",
        ).to(device)
    else:
        pipeline = Qwen3SpeculativePipelineModel(
            base_model=base_model,
            **_pipeline_init_kwargs(spec_cfg),
        ).to(device)
    pipeline.base_model.requires_grad_(False)
    pipeline.speculation_module.train()

    projected_cur_in = product_tensors["cur_in"].to(device=device, dtype=model_dtype)
    raw_tap_concat = product_tensors.get("raw_tap_concat")
    if raw_tap_concat is not None:
        raw_tap_concat = raw_tap_concat.to(device=device, dtype=model_dtype)
    raw_tap_offsets = product_tensors.get("raw_tap_offsets")
    raw_row_stage_ids = product_tensors.get("row_i_stages")
    if raw_tap_offsets is not None:
        raw_tap_offsets = raw_tap_offsets.cpu().tolist()
    if raw_row_stage_ids is not None:
        raw_row_stage_ids = raw_row_stage_ids.cpu().tolist()
    position_ids = product_tensors["position_ids"].to(device=device)
    teacher_logits = teacher_tensors["teacher_logits"].to(device=device, dtype=torch.float32)
    teacher_argmax = teacher_logits.argmax(dim=-1)
    label_draft_indices = product_tensors.get("label_draft_indices")
    if args.hard_label_weight > 0.0 and label_draft_indices is None:
        raise ValueError(
            "--hard-label-weight requires label_draft_indices in the product corpus"
        )
    if label_draft_indices is not None:
        label_draft_indices = label_draft_indices.to(device=device, dtype=torch.long)
    optimizer = torch.optim.AdamW(
        pipeline.speculation_module.parameters(),
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
                pipeline,
                input_mode,
                projected_cur_in,
                raw_tap_concat,
                raw_tap_offsets,
                raw_row_stage_ids,
                batch_idx,
            )
            batch_position_ids = position_ids[batch_idx]
            batch_teacher = teacher_logits[batch_idx]
            proc = pipeline.speculation_module.forward_inference_g1_only_with_rotary(
                batch_cur_in,
                batch_position_ids,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )
            final_hidden = pipeline.final_norm(proc)
            student_logits = pipeline.speculation_module.lm_head(final_hidden[:, -1:, :])
            student_logits = student_logits.squeeze(1).float()
            if student_logits.shape != batch_teacher.shape:
                raise ValueError(
                    f"student logits shape {tuple(student_logits.shape)} does not match "
                    f"teacher logits shape {tuple(batch_teacher.shape)}"
                )
            loss_parts = []
            if args.kl_weight > 0.0:
                teacher_probs = F.softmax(batch_teacher / float(args.temperature), dim=-1)
                student_log_probs = F.log_softmax(
                    student_logits / float(args.temperature),
                    dim=-1,
                )
                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                kl_loss = kl_loss * (float(args.temperature) ** 2)
                loss_parts.append(float(args.kl_weight) * kl_loss)
            if args.hard_label_weight > 0.0:
                assert label_draft_indices is not None
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
                acc = (pred == teacher_argmax[batch_idx]).float().mean()
                if label_draft_indices is not None:
                    batch_labels = label_draft_indices[batch_idx]
                    in_scope = batch_labels >= 0
                    if bool(in_scope.any()):
                        hard_acc = (pred[in_scope] == batch_labels[in_scope]).float().mean()
                        hard_label_accs.append(float(hard_acc.detach().cpu()))
            losses.append(float(loss.detach().cpu()))
            accs.append(float(acc.detach().cpu()))
            steps += 1
            if args.max_steps > 0 and steps >= int(args.max_steps):
                break
        if args.max_steps > 0 and steps >= int(args.max_steps):
            break

    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pipeline.save_speculation_head(str(out_checkpoint))
    summary = {
        "schema": "skippy-spd-product-activation-train-summary/v1",
        "input_checkpoint": str(checkpoint_path),
        "init_mode": str(args.init_mode),
        "input_mode": input_mode,
        "seed": int(args.seed),
        "out_checkpoint": str(out_checkpoint),
        "product_corpus": str(args.product_corpus),
        "teacher_logits": str(args.teacher_logits),
        "cur_in_convention": product_metadata.get("cur_in_convention"),
        "teacher_source": teacher_metadata.get("teacher_source"),
        "native_product_teacher_logits": teacher_metadata.get("native_product_teacher_logits"),
        "base_model_path": base_model_path,
        "device": str(device),
        "model_torch_dtype": str(model_dtype).replace("torch.", ""),
        "sample_count": sample_count,
        "batch_size": int(args.batch_size),
        "epochs_requested": int(args.epochs),
        "steps_completed": steps,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "temperature": float(args.temperature),
        "kl_weight": float(args.kl_weight),
        "hard_label_weight": float(args.hard_label_weight),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "initial_argmax_acc": accs[0] if accs else None,
        "final_argmax_acc": accs[-1] if accs else None,
        "initial_hard_label_acc": hard_label_accs[0] if hard_label_accs else None,
        "final_hard_label_acc": hard_label_accs[-1] if hard_label_accs else None,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def patch_reference_checkout(reference_dir: Path) -> None:
    patch_reference_for_transformers(reference_dir)


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    import json as json_module

    with path.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json_module.loads(handle.read(header_len))
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def validate_metadata(metadata: dict[str, str], schema: str, path: str) -> None:
    if metadata.get("schema") != schema:
        raise ValueError(f"{path} has schema {metadata.get('schema')!r}, expected {schema!r}")


def validate_product_convention(metadata: dict[str, str], path: str) -> None:
    convention = metadata.get("cur_in_convention")
    if convention in {"terminal_final_normed_cur_in", "not_available_zero_placeholder"}:
        return
    if convention != "terminal_final_normed_cur_in":
        raise ValueError(
            f"{path} has cur_in_convention {convention!r}, expected "
            "'terminal_final_normed_cur_in' or 'not_available_zero_placeholder'"
        )


def resolve_input_mode(
    requested: str,
    init_mode: str,
    product_tensors: dict[str, Any],
) -> str:
    if requested != "auto":
        return requested
    if init_mode == "fresh" and "raw_tap_concat" in product_tensors:
        return "raw"
    return "projected"


def validate_input_mode(
    init_mode: str,
    input_mode: str,
    metadata: dict[str, str],
    path: str,
) -> None:
    if input_mode == "raw":
        convention = metadata.get("raw_tap_convention")
        if convention != "terminal_final_normed_tap_concat":
            raise ValueError(
                f"{path} has raw_tap_convention {convention!r}, expected "
                "'terminal_final_normed_tap_concat'"
            )
        return
    if metadata.get("cur_in_convention") == "not_available_zero_placeholder":
        raise ValueError(
            f"{path} stores zero placeholder cur_in rows; use --input-mode raw"
        )
    if init_mode == "fresh":
        source_dir = metadata.get("source_corpus_dir", path)
        raise ValueError(
            "--init-mode fresh is not valid for projected product corpora. "
            f"{path} was converted from {source_dir} and stores rows after the "
            "manifest's SPD input projection. Use --init-mode checkpoint with the "
            "same projection basis, or add a raw terminal-normalized tap-concat "
            "corpus/trainer path so fresh stage_projs are trained and used "
            "consistently."
        )


def validate_tensor_shapes(product: dict[str, Any], teacher: dict[str, Any]) -> None:
    required_product = {"cur_in", "position_ids"}
    required_teacher = {"teacher_logits"}
    missing_product = sorted(required_product.difference(product))
    missing_teacher = sorted(required_teacher.difference(teacher))
    if missing_product:
        raise ValueError(f"product corpus missing tensors: {missing_product}")
    if missing_teacher:
        raise ValueError(f"teacher logits missing tensors: {missing_teacher}")
    if int(product["cur_in"].shape[0]) != int(teacher["teacher_logits"].shape[0]):
        raise ValueError(
            "product corpus sample count does not match teacher logits sample count"
        )
    if int(product["cur_in"].shape[0]) == 0:
        raise ValueError("product corpus is empty")
    if "raw_tap_concat" in product:
        raw_required = {"raw_tap_offsets", "row_i_stages"}
        missing_raw = sorted(raw_required.difference(product))
        if missing_raw:
            raise ValueError(f"raw product corpus missing tensors: {missing_raw}")
        if int(product["raw_tap_concat"].shape[0]) != int(product["cur_in"].shape[0]):
            raise ValueError("raw_tap_concat sample count does not match cur_in")


def validate_sample_alignment(product: dict[str, Any], teacher: dict[str, Any]) -> None:
    for product_name, teacher_name in [
        ("query_positions", "query_positions"),
        ("target_positions", "target_positions"),
        ("query_row_indices", "query_row_indices"),
    ]:
        if product_name not in product or teacher_name not in teacher:
            continue
        product_tensor = product[product_name].cpu()
        teacher_tensor = teacher[teacher_name].cpu()
        if product_tensor.shape != teacher_tensor.shape:
            raise ValueError(
                f"{product_name} shape {tuple(product_tensor.shape)} does not match "
                f"{teacher_name} shape {tuple(teacher_tensor.shape)}"
            )
        if bool((product_tensor != teacher_tensor).any()):
            raise ValueError(f"{product_name} does not match teacher {teacher_name}")


def resolve_device(value: str) -> Any:
    import torch

    if value == "cuda":
        return torch.device("cuda")
    if value == "mps":
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_dtype(value: str, device: Any) -> Any:
    import torch

    if value != "auto":
        return torch_dtype(value)
    if device.type in {"cuda", "mps"}:
        return torch.bfloat16
    return torch.float32


def torch_dtype(value: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def set_torch_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    mps = getattr(torch_module, "mps", None)
    manual_seed = getattr(mps, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)


def batch_product_cur_in(
    pipeline: Any,
    input_mode: str,
    projected_cur_in: Any,
    raw_tap_concat: Any | None,
    raw_tap_offsets: list[int] | None,
    raw_row_stage_ids: list[int] | None,
    batch_idx: slice,
) -> Any:
    if input_mode == "projected":
        return projected_cur_in[batch_idx]
    if raw_tap_concat is None or raw_tap_offsets is None or raw_row_stage_ids is None:
        raise ValueError("--input-mode raw requires raw_tap_concat/raw_tap_offsets")
    return project_raw_tap_concat(
        pipeline,
        raw_tap_concat[batch_idx],
        raw_tap_offsets,
        raw_row_stage_ids,
    )


def project_raw_tap_concat(
    pipeline: Any,
    raw_batch: Any,
    raw_tap_offsets: list[int],
    row_stage_ids: list[int],
) -> Any:
    if len(raw_tap_offsets) != len(row_stage_ids) + 1:
        raise ValueError("raw_tap_offsets must have row_stage_ids + 1 entries")
    rows = []
    num_stages = len(pipeline.speculation_module.stage_projs)
    for row_index, stage_id in enumerate(row_stage_ids):
        start = int(raw_tap_offsets[row_index])
        end = int(raw_tap_offsets[row_index + 1])
        row = raw_batch[:, start:end]
        if int(stage_id) == 0:
            projected = pipeline.speculation_module.g0_proj(row)
        else:
            block = num_stages - int(stage_id)
            if block < 0 or block >= num_stages:
                raise ValueError(
                    f"row {row_index} stage_id {stage_id} is outside num_stages={num_stages}"
                )
            projected = pipeline.speculation_module.stage_projs[block](row)
        rows.append(projected.unsqueeze(1))
    import torch

    return torch.cat(rows, dim=1)


def batch_indices(sample_count: int, batch_size: int) -> list[slice]:
    return [
        slice(start, min(start + batch_size, sample_count))
        for start in range(0, sample_count, batch_size)
    ]


if __name__ == "__main__":
    main()
