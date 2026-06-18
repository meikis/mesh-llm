#!/usr/bin/env python3
"""Diagnose SPD product-row teacher, verifier, and proposal alignment.

This script answers a narrow but important question for product-tap SPD
training: are we failing because the sidecar has not learned the teacher, or
because the HF teacher disagrees with the native Q4 verifier target?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare product-row target tokens, HF teacher top-k, and optional live head proposals."
    )
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--teacher-logits", required=True, type=Path)
    parser.add_argument(
        "--live-tap-report",
        type=Path,
        help="Optional spd-live-tap-parity JSON report for the head being diagnosed.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.corpus_dir / "rows.jsonl")
    teacher = read_teacher(args.teacher_logits)
    if len(rows) != len(teacher["argmax_token_ids"]):
        raise SystemExit(
            f"row/teacher length mismatch: {len(rows)} rows vs "
            f"{len(teacher['argmax_token_ids'])} teacher samples"
        )

    target_tokens = [int(row["target_token"]) for row in rows]
    old_proposals = [int(row["proposal_top_k"]["token_ids"][0]) for row in rows]
    report: dict[str, Any] = {
        "schema": "skippy-spd-product-teacher-alignment/v1",
        "corpus_dir": str(args.corpus_dir),
        "teacher_logits": str(args.teacher_logits),
        "sample_count": len(rows),
        "labels_in_draft_scope": int(sum(teacher["label_in_scope"])),
        "teacher_top1_vs_q4_target": count_matches(
            teacher["argmax_token_ids"], target_tokens
        ),
        "teacher_top4_contains_q4_target": count_topk_contains(
            teacher["topk_token_ids"], target_tokens
        ),
        "corpus_head_top1_vs_q4_target": count_matches(old_proposals, target_tokens),
        "corpus_head_top1_vs_teacher_top1": count_matches(
            old_proposals, teacher["argmax_token_ids"]
        ),
        "per_prompt": per_prompt_counts(
            rows,
            teacher_top1=teacher["argmax_token_ids"],
            target_tokens=target_tokens,
            corpus_head=old_proposals,
        ),
    }

    if args.live_tap_report:
        live_proposals = read_live_proposals(args.live_tap_report)
        if len(live_proposals) != len(rows):
            raise SystemExit(
                f"row/live length mismatch: {len(rows)} rows vs "
                f"{len(live_proposals)} live proposals"
            )
        report["live_tap_report"] = str(args.live_tap_report)
        report["live_head_top1_vs_q4_target"] = count_matches(
            live_proposals, target_tokens
        )
        report["live_head_top1_vs_teacher_top1"] = count_matches(
            live_proposals, teacher["argmax_token_ids"]
        )
        report["per_prompt"] = per_prompt_counts(
            rows,
            teacher_top1=teacher["argmax_token_ids"],
            target_tokens=target_tokens,
            corpus_head=old_proposals,
            live_head=live_proposals,
        )

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_teacher(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        return {
            "argmax_token_ids": tensor_list(tensors.get_tensor("teacher_argmax_token_ids")),
            "label_in_scope": tensor_list(tensors.get_tensor("teacher_label_in_logit_scope")),
            "topk_token_ids": tensors.get_tensor("teacher_topk_token_ids").tolist(),
        }


def read_live_proposals(path: Path) -> list[int | None]:
    report = json.loads(path.read_text(encoding="utf-8"))
    generations = sorted(
        report.get("verified_generations", []), key=lambda item: int(item["prompt_index"])
    )
    proposals: list[int | None] = []
    for generation in generations:
        steps = sorted(generation.get("steps", []), key=lambda item: int(item["step_index"]))
        for step in steps:
            tokens = step.get("proposal_tokens") or []
            proposals.append(int(tokens[0]) if tokens else None)
    return proposals


def tensor_list(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.tolist()]


def count_matches(predicted: list[int | None], target: list[int]) -> dict[str, Any]:
    compared = [(pred, gold) for pred, gold in zip(predicted, target) if pred is not None]
    matched = sum(1 for pred, gold in compared if int(pred) == int(gold))
    total = len(compared)
    return {
        "matched": matched,
        "total": total,
        "rate": matched / total if total else None,
    }


def count_topk_contains(topk_rows: list[list[int]], target: list[int]) -> dict[str, Any]:
    matched = 0
    for topk, gold in zip(topk_rows, target):
        if int(gold) in {int(token) for token in topk}:
            matched += 1
    total = len(target)
    return {
        "matched": matched,
        "total": total,
        "rate": matched / total if total else None,
    }


def per_prompt_counts(
    rows: list[dict[str, Any]],
    *,
    teacher_top1: list[int],
    target_tokens: list[int],
    corpus_head: list[int],
    live_head: list[int | None] | None = None,
) -> list[dict[str, Any]]:
    by_prompt: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        prompt_index = int(row["prompt_index"])
        entry = by_prompt.setdefault(
            prompt_index,
            {
                "prompt_index": prompt_index,
                "sample_count": 0,
                "teacher_top1_matches_q4_target": 0,
                "corpus_head_matches_q4_target": 0,
                "live_head_matches_q4_target": 0,
            },
        )
        entry["sample_count"] += 1
        entry["teacher_top1_matches_q4_target"] += int(
            int(teacher_top1[index]) == int(target_tokens[index])
        )
        entry["corpus_head_matches_q4_target"] += int(
            int(corpus_head[index]) == int(target_tokens[index])
        )
        if live_head is not None:
            proposal = live_head[index]
            entry["live_head_matches_q4_target"] += int(
                proposal is not None and int(proposal) == int(target_tokens[index])
            )
    if live_head is None:
        for entry in by_prompt.values():
            entry.pop("live_head_matches_q4_target", None)
    return [by_prompt[index] for index in sorted(by_prompt)]


if __name__ == "__main__":
    main()
