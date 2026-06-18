#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
#   "transformers>=5.6.0",
# ]
# ///
"""Attach HF teacher logits to a Skippy SPD product activation corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_product_activation_corpus import (
    CORPUS_SCHEMA,
    load_draft_token_ids,
    read_json,
    read_rows,
)


OUT_SCHEMA = "skippy-spd-product-activation-teacher-logits/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen HF teacher model on product-corpus context tokens and "
            "save next-token logits aligned to each captured SPD query row."
        )
    )
    parser.add_argument("--corpus-dir", required=True, help="Input product corpus directory")
    parser.add_argument("--out", required=True, help="Output teacher safetensors path")
    parser.add_argument(
        "--base-model-path",
        help=(
            "HF id or local path for the frozen teacher. Defaults to source.base_model_path "
            "from --spd-manifest or from the corpus manifest_path when readable."
        ),
    )
    parser.add_argument(
        "--spd-manifest",
        help="Optional SPD manifest JSON used for base-model and draft-token metadata.",
    )
    parser.add_argument(
        "--logit-scope",
        choices=("draft", "full"),
        default="draft",
        help="Save logits over draft_token_ids or the full base vocabulary.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--start-sample", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
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
    parser.add_argument(
        "--save-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--trust-remote-code",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--schema-smoke",
        action="store_true",
        help="Validate corpus/teacher shape planning without importing torch or loading a model.",
    )
    parser.add_argument("--summary-json", help="Optional JSON summary output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.corpus_dir)
    corpus_manifest = read_json(corpus_dir / "manifest.json")
    if corpus_manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported corpus schema: {corpus_manifest.get('schema')!r}")
    rows = read_rows(corpus_dir / "rows.jsonl")
    selected_rows = select_rows(rows, args.start_sample, args.max_samples)
    spd_manifest = load_spd_manifest(args.spd_manifest, corpus_manifest)
    base_model_path = resolve_base_model_path(args.base_model_path, spd_manifest)
    draft_token_ids = load_teacher_draft_token_ids(args, corpus_manifest, spd_manifest)
    plan = build_teacher_plan(
        rows=selected_rows,
        corpus_manifest=corpus_manifest,
        base_model_path=base_model_path,
        draft_token_ids=draft_token_ids,
        logit_scope=args.logit_scope,
        top_k=args.top_k,
        save_dtype=args.save_dtype,
        start_sample=args.start_sample,
    )
    if args.schema_smoke:
        emit_summary(plan, args.summary_json)
        return
    tensors, runtime_summary = run_teacher(args, selected_rows, plan)
    summary = {**plan, **runtime_summary}
    save_teacher_tensors(args, tensors, summary)
    emit_summary(summary, args.summary_json)


def select_rows(
    rows: list[dict[str, Any]],
    start_sample: int,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if start_sample < 0:
        raise ValueError("--start-sample must be non-negative")
    selected = rows[start_sample:]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided")
        selected = selected[:max_samples]
    if not selected:
        raise ValueError("selected product corpus row range is empty")
    return selected


def load_spd_manifest(
    explicit_path: str | None,
    corpus_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    path: str | None = explicit_path
    if path is None:
        candidate = corpus_manifest.get("manifest_path")
        if isinstance(candidate, str) and Path(candidate).is_file():
            path = candidate
    if path is None:
        return None
    return read_json(Path(path))


def resolve_base_model_path(
    explicit_path: str | None,
    spd_manifest: dict[str, Any] | None,
) -> str:
    if explicit_path:
        return explicit_path
    if spd_manifest is not None:
        source = spd_manifest.get("source", {})
        base_model_path = source.get("base_model_path")
        if base_model_path:
            return str(base_model_path)
    raise ValueError(
        "--base-model-path is required when no readable SPD manifest source.base_model_path exists"
    )


def load_teacher_draft_token_ids(
    args: argparse.Namespace,
    corpus_manifest: dict[str, Any],
    spd_manifest: dict[str, Any] | None,
) -> list[int]:
    if spd_manifest is not None:
        ids = [
            int(token)
            for token in spd_manifest.get("topology", {}).get("draft_token_ids", [])
        ]
    else:
        ids = load_draft_token_ids(None, corpus_manifest)
    if args.logit_scope == "draft" and not ids:
        raise ValueError("--logit-scope=draft requires draft_token_ids from a manifest")
    return ids


def build_teacher_plan(
    *,
    rows: list[dict[str, Any]],
    corpus_manifest: dict[str, Any],
    base_model_path: str,
    draft_token_ids: list[int],
    logit_scope: str,
    top_k: int,
    save_dtype: str,
    start_sample: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive")
    logit_width = (
        len(draft_token_ids)
        if logit_scope == "draft"
        else int(corpus_manifest.get("topology", {}).get("vocab_size", 0))
    )
    if logit_width <= 0:
        raise ValueError("could not infer positive teacher logit width")
    validate_teacher_rows(rows)
    sample_count = len(rows)
    bytes_per_value = {"float32": 4, "float16": 2, "bfloat16": 2}[save_dtype]
    return {
        "schema": OUT_SCHEMA,
        "source_schema": CORPUS_SCHEMA,
        "base_model_path": base_model_path,
        "sample_count": sample_count,
        "start_sample": start_sample,
        "logit_scope": logit_scope,
        "logit_width": logit_width,
        "top_k": top_k,
        "save_dtype": save_dtype,
        "estimated_teacher_logits_bytes": sample_count * logit_width * bytes_per_value,
        "target_logits_available": True,
        "teacher_source": "hf_base_model_aligned_to_product_context_tokens",
        "native_product_teacher_logits": False,
        "paper_kl_training_ready": True,
    }


def validate_teacher_rows(rows: list[dict[str, Any]]) -> None:
    for idx, row in enumerate(rows):
        context_tokens = row.get("context_tokens")
        if not isinstance(context_tokens, list) or not context_tokens:
            raise ValueError(f"sample {idx} has no context_tokens")
        target_position = int(row.get("target_position", row["context_token_count_before"]))
        if target_position <= 0:
            raise ValueError(f"sample {idx} has invalid target_position={target_position}")
        if target_position > len(context_tokens):
            raise ValueError(
                f"sample {idx} target_position={target_position} exceeds "
                f"context length {len(context_tokens)}"
            )
        query_row_index = int(row.get("query_row_index", len(row["row_positions"]) - 1))
        row_positions = [int(value) for value in row["row_positions"]]
        if query_row_index < 0 or query_row_index >= len(row_positions):
            raise ValueError(f"sample {idx} has invalid query_row_index={query_row_index}")


def run_teacher(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(args.device)
    model_dtype = resolve_model_dtype(args.model_torch_dtype, device)
    save_dtype = torch_dtype(args.save_dtype)
    model_kwargs: dict[str, Any] = {
        "dtype": model_dtype,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(
        plan["base_model_path"],
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    model = AutoModelForCausalLM.from_pretrained(plan["base_model_path"], **model_kwargs)
    model = model.to(device)
    model.eval()
    draft_token_ids = plan_draft_token_tensor(args, plan, device)

    logits_out = []
    argmax_token_ids = []
    argmax_indices = []
    target_logit_indices = []
    label_in_logit_scope = []
    topk_token_ids = []
    topk_indices = []
    topk_logits = []
    sample_indices = []
    target_positions = []
    logit_positions = []
    query_row_indices = []
    query_positions = []
    with torch.no_grad():
        for batch in batched(rows, args.batch_size):
            batch_tensors = teacher_batch_tensors(batch, pad_token_id, device)
            outputs = model(
                input_ids=batch_tensors["input_ids"],
                attention_mask=batch_tensors["attention_mask"],
                return_dict=True,
            )
            batch_logits = outputs.logits[
                torch.arange(len(batch), device=device),
                batch_tensors["logit_positions"],
            ].float()
            full_argmax = batch_logits.argmax(dim=-1)
            if args.logit_scope == "draft":
                scoped_logits = batch_logits.index_select(dim=-1, index=draft_token_ids)
                scoped_argmax = scoped_logits.argmax(dim=-1)
                scoped_argmax_tokens = draft_token_ids.index_select(0, scoped_argmax)
                target_draft_indices = token_ids_to_draft_indices(
                    batch_tensors["target_token_ids"],
                    draft_token_ids,
                )
                in_scope = target_draft_indices >= 0
            else:
                scoped_logits = batch_logits
                scoped_argmax = full_argmax
                scoped_argmax_tokens = full_argmax
                target_draft_indices = batch_tensors["target_token_ids"]
                in_scope = torch.ones_like(batch_tensors["target_token_ids"], dtype=torch.bool)
            values, indices = torch.topk(scoped_logits, k=min(args.top_k, scoped_logits.shape[-1]))
            if args.logit_scope == "draft":
                tokens = draft_token_ids.index_select(0, indices.reshape(-1)).reshape_as(indices)
            else:
                tokens = indices
            logits_out.append(scoped_logits.to(dtype=save_dtype).cpu())
            argmax_token_ids.append(scoped_argmax_tokens.to(dtype=torch.long).cpu())
            argmax_indices.append(scoped_argmax.to(dtype=torch.long).cpu())
            target_logit_indices.append(target_draft_indices.to(dtype=torch.long).cpu())
            label_in_logit_scope.append(in_scope.to(dtype=torch.long).cpu())
            topk_token_ids.append(tokens.to(dtype=torch.long).cpu())
            topk_indices.append(indices.to(dtype=torch.long).cpu())
            topk_logits.append(values.to(dtype=save_dtype).cpu())
            sample_indices.append(batch_tensors["sample_indices"].cpu())
            target_positions.append(batch_tensors["target_positions"].cpu())
            logit_positions.append(batch_tensors["logit_positions"].cpu())
            query_row_indices.append(batch_tensors["query_row_indices"].cpu())
            query_positions.append(batch_tensors["query_positions"].cpu())

    tensors = {
        "teacher_logits": torch.cat(logits_out, dim=0).contiguous(),
        "teacher_argmax_token_ids": torch.cat(argmax_token_ids, dim=0).contiguous(),
        "teacher_argmax_indices": torch.cat(argmax_indices, dim=0).contiguous(),
        "target_logit_indices": torch.cat(target_logit_indices, dim=0).contiguous(),
        "teacher_label_in_logit_scope": torch.cat(label_in_logit_scope, dim=0).contiguous(),
        "teacher_topk_token_ids": torch.cat(topk_token_ids, dim=0).contiguous(),
        "teacher_topk_indices": torch.cat(topk_indices, dim=0).contiguous(),
        "teacher_topk_logits": torch.cat(topk_logits, dim=0).contiguous(),
        "sample_indices": torch.cat(sample_indices, dim=0).contiguous(),
        "target_positions": torch.cat(target_positions, dim=0).contiguous(),
        "logit_positions": torch.cat(logit_positions, dim=0).contiguous(),
        "query_row_indices": torch.cat(query_row_indices, dim=0).contiguous(),
        "query_positions": torch.cat(query_positions, dim=0).contiguous(),
    }
    if args.logit_scope == "draft":
        tensors["teacher_logit_token_ids"] = draft_token_ids.to(dtype=torch.long).cpu()
    labels_in_scope = int(tensors["teacher_label_in_logit_scope"].sum().item())
    runtime_summary = {
        "device": str(device),
        "model_torch_dtype": str(model_dtype).replace("torch.", ""),
        "labels_in_logit_scope": labels_in_scope,
        "labels_missing_from_logit_scope": len(rows) - labels_in_scope,
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
    }
    return tensors, runtime_summary


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


def plan_draft_token_tensor(args: argparse.Namespace, plan: dict[str, Any], device: Any) -> Any:
    import torch

    if args.logit_scope != "draft":
        return torch.empty((0,), dtype=torch.long, device=device)
    corpus_dir = Path(args.corpus_dir)
    manifest = read_json(corpus_dir / "manifest.json")
    spd_manifest = load_spd_manifest(args.spd_manifest, manifest)
    draft_token_ids = load_teacher_draft_token_ids(args, manifest, spd_manifest)
    return torch.tensor(draft_token_ids, dtype=torch.long, device=device)


def teacher_batch_tensors(
    rows: list[dict[str, Any]],
    pad_token_id: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    max_len = max(len(row["context_tokens"]) for row in rows)
    input_ids = torch.full((len(rows), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    logit_positions = []
    target_positions = []
    target_token_ids = []
    sample_indices = []
    query_row_indices = []
    query_positions = []
    for idx, row in enumerate(rows):
        tokens = torch.tensor(row["context_tokens"], dtype=torch.long, device=device)
        input_ids[idx, : tokens.numel()] = tokens
        attention_mask[idx, : tokens.numel()] = 1
        target_position = int(row.get("target_position", row["context_token_count_before"]))
        query_row_index = int(row.get("query_row_index", len(row["row_positions"]) - 1))
        logit_positions.append(target_position - 1)
        target_positions.append(target_position)
        target_token_ids.append(int(row["target_token"]))
        sample_indices.append(int(row["sample_index"]))
        query_row_indices.append(query_row_index)
        query_positions.append(int(row.get("query_position", row["row_positions"][query_row_index])))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logit_positions": torch.tensor(logit_positions, dtype=torch.long, device=device),
        "target_positions": torch.tensor(target_positions, dtype=torch.long),
        "target_token_ids": torch.tensor(target_token_ids, dtype=torch.long, device=device),
        "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        "query_row_indices": torch.tensor(query_row_indices, dtype=torch.long),
        "query_positions": torch.tensor(query_positions, dtype=torch.long),
    }


def token_ids_to_draft_indices(token_ids: Any, draft_token_ids: Any) -> Any:
    import torch

    if draft_token_ids.numel() == 0:
        return torch.full_like(token_ids, -1)
    out = torch.full_like(token_ids, -1)
    for draft_index, token_id in enumerate(draft_token_ids.tolist()):
        out[token_ids == int(token_id)] = int(draft_index)
    return out


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def save_teacher_tensors(
    args: argparse.Namespace,
    tensors: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    from safetensors.torch import save_file

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": OUT_SCHEMA,
        "base_model_path": str(summary["base_model_path"]),
        "sample_count": str(summary["sample_count"]),
        "logit_scope": str(summary["logit_scope"]),
        "logit_width": str(summary["logit_width"]),
        "save_dtype": str(summary["save_dtype"]),
        "teacher_source": str(summary["teacher_source"]),
        "native_product_teacher_logits": "false",
        "paper_kl_training_ready": "true",
    }
    save_file(tensors, out, metadata=metadata)
    summary["out"] = str(out)
    summary["bytes"] = out.stat().st_size


def emit_summary(summary: dict[str, Any], summary_json: str | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if summary_json:
        Path(summary_json).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
