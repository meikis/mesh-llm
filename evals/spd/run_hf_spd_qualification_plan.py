#!/usr/bin/env python3
"""Run command groups emitted by plan_hf_spd_qualification.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_GROUP_ORDER = [
    "setup",
    "write_physical_stage_ms",
    "build_prompts",
    "capture",
    "convert",
    "train",
    "score",
    "export_serving_bundle",
    "export_and_parity",
    "rust_fixture_parity",
    "package_smoke",
    "latency_simulation",
    "upload",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Planner JSON file")
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUP_ORDER),
        help="Comma-separated command group order to run.",
    )
    parser.add_argument(
        "--script-out",
        default="",
        help="Optional path to write the generated bash script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    commands = selected_commands(plan, groups)
    script = render_bash(commands)
    script_path = Path(args.script_out) if args.script_out else Path(args.plan).with_suffix(".sh")
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    subprocess.run(["bash", str(script_path)], check=True, env=os.environ.copy())


def selected_commands(plan: dict[str, Any], groups: list[str]) -> list[tuple[str, str]]:
    command_groups = plan.get("commands")
    if not isinstance(command_groups, dict):
        raise ValueError("plan is missing commands object")
    selected: list[tuple[str, str]] = []
    for group in groups:
        entries = command_groups.get(group)
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"commands.{group} must be a list")
        for index, command in enumerate(entries):
            if not isinstance(command, str):
                raise ValueError(f"commands.{group}[{index}] must be a string")
            selected.append((f"{group}[{index}]", command))
    if not selected:
        raise ValueError(f"no commands selected from groups {groups}")
    return selected


def render_bash(commands: list[tuple[str, str]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export PYTHONUNBUFFERED=1",
    ]
    for label, command in commands:
        lines.extend(
            [
                "",
                f"echo '===== SPD HF command {shell_single_quote(label)} ====='",
                command,
            ]
        )
    lines.append("")
    return "\n".join(lines)


def shell_single_quote(value: str) -> str:
    return value.replace("'", "'\"'\"'")


if __name__ == "__main__":
    main()
