#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets>=3.0.0",
# ]
# ///
"""Build mixed conversation JSONL for SPD sidecar training.

The reference SPD trainer expects one JSON object per line with a `messages`
array in OpenAI role/content shape. This helper mixes general instruction data
with public agent/task trace windows. The labels are not used; the frozen target
model supplies teacher logits during SPD KD training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any


DEFAULT_ULTRACHAT = "HuggingFaceH4/ultrachat_200k"
DEFAULT_ULTRACHAT_SPLIT = "train_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed SPD conversation JSONL")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--ultrachat-dataset", default=DEFAULT_ULTRACHAT)
    parser.add_argument("--ultrachat-split", default=DEFAULT_ULTRACHAT_SPLIT)
    parser.add_argument("--ultrachat-rows", type=int, default=32768)
    parser.add_argument("--trace-csv", action="append", default=[], type=Path)
    parser.add_argument("--trace-rows", type=int, default=32768)
    parser.add_argument("--max-trace-chars", type=int, default=12000)
    parser.add_argument(
        "--trace-as-system",
        action="store_true",
        help="Write trace task/context as a system+user pair instead of a single user turn.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ultrachat_rows < 0 or args.trace_rows < 0:
        raise ValueError("row counts must be non-negative")
    if args.ultrachat_rows == 0 and args.trace_rows == 0:
        raise ValueError("nothing to write: both row counts are zero")

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}

    if args.ultrachat_rows:
        ultra = load_ultrachat_rows(
            args.ultrachat_dataset,
            args.ultrachat_split,
            args.ultrachat_rows,
        )
        rows.extend(ultra)
        source_counts["ultrachat"] = len(ultra)

    if args.trace_rows:
        trace_rows = load_trace_rows(
            args.trace_csv,
            args.trace_rows,
            rng=rng,
            max_chars=args.max_trace_chars,
            trace_as_system=bool(args.trace_as_system),
        )
        rows.extend(trace_rows)
        source_counts["trace_windows"] = len(trace_rows)

    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "schema": "skippy-spd-training-conversations/v1",
        "out": str(args.out),
        "seed": args.seed,
        "row_count": len(rows),
        "source_counts": source_counts,
        "ultrachat_dataset": args.ultrachat_dataset,
        "ultrachat_split": args.ultrachat_split,
        "trace_csv": [str(path) for path in args.trace_csv],
        "max_trace_chars": args.max_trace_chars,
        "trace_as_system": bool(args.trace_as_system),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def load_ultrachat_rows(dataset_name: str, split: str, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    wanted = int(limit)
    ds = load_dataset(dataset_name, split=f"{split}[:{wanted}]")
    rows: list[dict[str, Any]] = []
    for raw in ds:
        messages = normalize_messages(raw.get("messages") or raw.get("conversations"))
        if messages:
            rows.append({"messages": messages, "source": "ultrachat"})
        if len(rows) >= wanted:
            break
    if len(rows) < wanted:
        raise RuntimeError(f"only found {len(rows)} usable UltraChat rows, wanted {wanted}")
    return rows


def normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        if role is None or content is None:
            return []
        role = str(role)
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            continue
        text = str(content).strip()
        if text:
            out.append({"role": role, "content": text})
    return out


def load_trace_rows(
    paths: list[Path],
    limit: int,
    *,
    rng: random.Random,
    max_chars: int,
    trace_as_system: bool,
) -> list[dict[str, Any]]:
    if not paths:
        raise ValueError("--trace-rows requires at least one --trace-csv")
    candidates = reservoir_trace_windows(paths, limit, rng=rng, max_chars=max_chars)
    rows: list[dict[str, Any]] = []
    for item in candidates:
        text = item["window_text"]
        if trace_as_system:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI coding and terminal assistant continuing a task trace.",
                },
                {"role": "user", "content": text},
            ]
        else:
            messages = [{"role": "user", "content": text}]
        rows.append(
            {
                "messages": messages,
                "source": f"trace:{item.get('source', '')}",
                "trace_id": item.get("trace_id"),
                "task_id": item.get("task_id"),
            }
        )
    if len(rows) < int(limit):
        raise RuntimeError(f"only found {len(rows)} trace rows, wanted {limit}")
    return rows


def reservoir_trace_windows(
    paths: list[Path],
    limit: int,
    *,
    rng: random.Random,
    max_chars: int,
) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    sample: list[dict[str, str]] = []
    seen = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "window_text" not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no window_text column")
            for row in reader:
                text = str(row.get("window_text") or "").strip()
                if not text:
                    continue
                if max_chars > 0 and len(text) > max_chars:
                    text = text[:max_chars]
                item = {
                    "window_text": text,
                    "source": str(row.get("source") or ""),
                    "trace_id": str(row.get("trace_id") or ""),
                    "task_id": str(row.get("task_id") or ""),
                }
                seen += 1
                if len(sample) < limit:
                    sample.append(item)
                    continue
                slot = rng.randrange(seen)
                if slot < limit:
                    sample[slot] = item
    rng.shuffle(sample)
    return sample


if __name__ == "__main__":
    main()
