#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers>=5.6.0",
# ]
# ///
"""Build tokenized product prompt files for SPD product-corpus capture.

The output is JSONL accepted by:

    skippy-bench spd-live-tap-parity --prompt-token-file ...

Rows are rendered with the target tokenizer chat template and
``enable_thinking=False`` so product tap capture can reuse the same prompt
surface as the OpenAI no-thinking SPD smokes.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import median
from typing import Any


SOURCE_FILES = {
    "mt_bench": Path("mt_bench/question.jsonl"),
    "gsm8k": Path("gsm8k/question.jsonl"),
    "humaneval": Path("humaneval/question.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize SPD train/held-out prompts from reference eval JSONL files."
    )
    parser.add_argument("--eval-data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--train-per-set", type=int, default=16)
    parser.add_argument("--heldout-per-set", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=480)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-second-turn",
        action="store_true",
        help="Use every turn from each reference row instead of only the first user turn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_per_set < 0 or args.heldout_per_set < 0:
        raise SystemExit("--train-per-set and --heldout-per-set must be non-negative")
    if args.train_per_set == 0 and args.heldout_per_set == 0:
        raise SystemExit("nothing to write: both split counts are zero")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    rng = random.Random(args.seed)
    for dataset, rel_path in SOURCE_FILES.items():
        source_path = args.eval_data_dir / rel_path
        rows = read_prompt_rows(source_path, dataset, include_second_turn=args.include_second_turn)
        if args.shuffle:
            rng.shuffle(rows)
        tokenized = []
        for row in rows:
            tokens = render_prompt_tokens(tokenizer, row["messages"])
            row = {**row, "prompt_token_ids": tokens, "prompt_token_count": len(tokens)}
            if args.max_prompt_tokens > 0 and len(tokens) > args.max_prompt_tokens:
                skipped.append(row)
                continue
            tokenized.append(row)
        train_rows.extend(tokenized[: args.train_per_set])
        heldout_rows.extend(
            tokenized[args.train_per_set : args.train_per_set + args.heldout_per_set]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_tokens = args.out_dir / "train-prompt-token-ids.jsonl"
    heldout_tokens = args.out_dir / "heldout-prompt-token-ids.jsonl"
    train_prompts = args.out_dir / "train-prompts.jsonl"
    heldout_prompts = args.out_dir / "heldout-prompts.jsonl"
    summary = args.out_dir / "summary.json"

    write_jsonl(train_tokens, [token_row(row) for row in train_rows])
    write_jsonl(heldout_tokens, [token_row(row) for row in heldout_rows])
    write_jsonl(train_prompts, [prompt_row(row) for row in train_rows])
    write_jsonl(heldout_prompts, [prompt_row(row) for row in heldout_rows])
    summary_obj = {
        "model_name": args.model_name,
        "eval_data_dir": str(args.eval_data_dir),
        "train_prompt_token_file": str(train_tokens),
        "heldout_prompt_token_file": str(heldout_tokens),
        "train_prompt_count": len(train_rows),
        "heldout_prompt_count": len(heldout_rows),
        "skipped_prompt_count": len(skipped),
        "max_prompt_tokens": args.max_prompt_tokens,
        "shuffle": bool(args.shuffle),
        "seed": args.seed,
        "include_second_turn": bool(args.include_second_turn),
        "train_token_stats": token_stats(train_rows),
        "heldout_token_stats": token_stats(heldout_rows),
        "skipped_token_stats": token_stats(skipped),
        "sources": source_counts(train_rows, heldout_rows, skipped),
    }
    summary.write_text(json.dumps(summary_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_obj, indent=2, ensure_ascii=False))


def read_prompt_rows(path: Path, dataset: str, *, include_second_turn: bool) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            turns = obj.get("turns")
            if not isinstance(turns, list) or not turns:
                continue
            usable_turns = turns if include_second_turn else turns[:1]
            messages = [
                {"role": "user", "content": str(turn)}
                for turn in usable_turns
                if str(turn).strip()
            ]
            if not messages:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "source_index": index,
                    "question_id": obj.get("question_id"),
                    "category": obj.get("category"),
                    "prompt": str(usable_turns[0]),
                    "messages": messages,
                }
            )
    return rows


def render_prompt_tokens(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(ids, dict):
        ids = ids.get("input_ids")
    elif hasattr(ids, "get"):
        ids = ids.get("input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list):
        raise TypeError(f"unexpected chat-template token output: {type(ids).__name__}")
    return [int(token) for token in ids]


def token_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_token_ids": row["prompt_token_ids"],
        "dataset": row["dataset"],
        "source_index": row["source_index"],
        "question_id": row["question_id"],
        "category": row["category"],
        "prompt_token_count": row["prompt_token_count"],
    }


def prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": row["prompt"],
        "messages": row["messages"],
        "dataset": row["dataset"],
        "source_index": row["source_index"],
        "question_id": row["question_id"],
        "category": row["category"],
        "prompt_token_count": row["prompt_token_count"],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def token_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = sorted(int(row["prompt_token_count"]) for row in rows)
    if not counts:
        return {"count": 0}
    return {
        "count": len(counts),
        "min": counts[0],
        "p50": median(counts),
        "max": counts[-1],
        "mean": sum(counts) / len(counts),
    }


def source_counts(*groups: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rows in groups:
        for row in rows:
            dataset = str(row["dataset"])
            counts[dataset] = counts.get(dataset, 0) + 1
    return counts


if __name__ == "__main__":
    main()
