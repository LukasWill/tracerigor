"""Command-line entry point for TraceRigor's public workflows."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Sequence

from tracerigor.analysis import analyze_records, load_records


def _analyze(args: argparse.Namespace) -> int:
    result = analyze_records(load_records(args.input), group_by=args.group_by)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


def _list_environments(args: argparse.Namespace) -> int:
    from tracerigor.env import environment_status

    status = environment_status()
    if args.json:
        sys.stdout.write(json.dumps(status, indent=2, sort_keys=True) + "\n")
    else:
        for name, detail in status.items():
            suffix = f" ({detail['detail']})" if args.verbose and detail["detail"] else ""
            print(f"{name:20} {detail['status']}{suffix}")
    return 0


def _generate_data(args: argparse.Namespace) -> int:
    from tracerigor.env.create_dataset import create_dataset_from_yaml

    create_dataset_from_yaml(
        args.config,
        force_gen=args.force,
        seed=args.seed,
        train_path=args.train_path,
        test_path=args.test_path,
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    sys.argv = ["tracerigor.trainer.main_ppo", *args.overrides]
    runpy.run_module("tracerigor.trainer.main_ppo", run_name="__main__")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracerigor",
        description="Train, verify, and analyze multi-turn agent trajectories.",
    )
    parser.add_argument("--version", action="version", version="TraceRigor 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="summarize judged trajectory records")
    analyze.add_argument("input", help="JSON or JSONL trajectory records")
    analyze.add_argument("--output", "-o", help="write summary JSON to this path")
    analyze.add_argument("--group-by", help="optional record or metadata field")
    analyze.set_defaults(func=_analyze)

    envs = commands.add_parser("envs", help="show environment availability")
    envs.add_argument("--json", action="store_true", help="emit JSON")
    envs.add_argument("--verbose", action="store_true", help="show missing dependencies")
    envs.set_defaults(func=_list_environments)

    data = commands.add_parser("data", help="generate train/evaluation seed datasets")
    data.add_argument("config", help="environment dataset YAML")
    data.add_argument("--train-path", default="data/train.parquet")
    data.add_argument("--test-path", default="data/test.parquet")
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--force", action="store_true")
    data.set_defaults(func=_generate_data)

    train = commands.add_parser("train", help="run the Hydra/VERL training entry point")
    train.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides")
    train.set_defaults(func=_train)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
