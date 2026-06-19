#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "safetensors>=0.5.0",
#   "torch>=2.8.0",
# ]
# ///
"""Export the minimal fixture tensors needed by Rust SPD serving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_product_activation_head import (
    PRODUCT_SCHEMA,
    read_safetensors_metadata,
    validate_metadata,
    validate_product_convention,
)


OUT_SCHEMA = "skippy-spd-serving-fixture/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a serving-only SPD fixture from a product activation corpus. "
            "This is not a Python parity fixture; it carries row metadata and "
            "final norm weights for Rust request-path serving."
        )
    )
    parser.add_argument("--product-corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from safetensors.torch import load_file, save_file

    product = load_file(args.product_corpus)
    metadata = read_safetensors_metadata(Path(args.product_corpus))
    validate_metadata(metadata, PRODUCT_SCHEMA, args.product_corpus)
    validate_product_convention(metadata, args.product_corpus)
    tensors = build_fixture_tensors(product)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        out_path,
        metadata={
            "schema": OUT_SCHEMA,
            "source_schema": PRODUCT_SCHEMA,
            "source_product_corpus": str(args.product_corpus),
            "parity_fixture": "false",
            "serving_fixture": "true",
        },
    )
    summary = {
        "schema": OUT_SCHEMA,
        "out": str(out_path),
        "bytes": out_path.stat().st_size,
        "source_product_corpus": str(args.product_corpus),
        "tensor_shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "parity_fixture": False,
        "serving_fixture": True,
    }
    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_fixture_tensors(product: dict[str, Any]) -> dict[str, Any]:
    import torch

    required = {
        "cur_in",
        "final_norm_weight",
        "position_ids",
        "row_positions",
        "row_i_stages",
        "row_hf_indices_flat",
        "row_hf_indices_offsets",
    }
    missing = sorted(required.difference(product))
    if missing:
        raise ValueError(f"product corpus missing tensors required for serving fixture: {missing}")
    cur_in = product["cur_in"]
    if len(cur_in.shape) != 3 or int(cur_in.shape[0]) == 0:
        raise ValueError("cur_in must be [sample_count, row_count, hidden] with samples")
    row_count = int(cur_in.shape[1])
    tensors: dict[str, Any] = {
        "cur_in": cur_in[:1].to(dtype=torch.float32).contiguous(),
        "final_norm_weight": product["final_norm_weight"].to(dtype=torch.float32).contiguous(),
        "row_i_stages": product["row_i_stages"].to(dtype=torch.long).contiguous(),
        "row_positions": product["row_positions"][0].to(dtype=torch.long).contiguous(),
        "position_ids": product["position_ids"][0].to(dtype=torch.long).contiguous(),
        "prompt_input_ids": torch.zeros((1, 1), dtype=torch.long),
    }
    rows = row_hf_indices(product, row_count)
    for row_index, indices in enumerate(rows):
        tensors[f"tap_row_{row_index}_hf_indices"] = torch.tensor(indices, dtype=torch.long)
    return tensors


def row_hf_indices(product: dict[str, Any], row_count: int) -> list[list[int]]:
    flat = [int(value) for value in product["row_hf_indices_flat"].cpu().tolist()]
    offsets = [int(value) for value in product["row_hf_indices_offsets"].cpu().tolist()]
    if len(offsets) != row_count + 1:
        raise ValueError(
            f"row_hf_indices_offsets length {len(offsets)} does not match row_count {row_count}"
        )
    rows = []
    for row_index in range(row_count):
        rows.append(flat[offsets[row_index] : offsets[row_index + 1]])
    return rows


if __name__ == "__main__":
    main()
