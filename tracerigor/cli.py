"""Command-line entry point for TraceRigor's public workflows."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Sequence

from tracerigor import __version__
from tracerigor.analysis import analyze_records, load_records


def _analyze(args: argparse.Namespace) -> int:
    result = analyze_records(load_records(args.input), group_by=args.group_by)
    # Keep insertion order so numeric groups stay in numeric order.
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


def _judge(args: argparse.Namespace) -> int:
    from tracerigor.judge.batch import judge_turns
    from tracerigor.judge.config import JudgeConfig

    cfg = JudgeConfig.from_yaml(args.config)
    cfg.audit.enabled = False
    if args.base_url:
        cfg.provider.base_url = args.base_url
    if args.model:
        cfg.provider.model = args.model
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"error: environment variable {args.api_key_env} is not set")
        cfg.provider.api_key = key
    if args.no_structured_output:
        cfg.provider.use_structured_output = False

    try:
        records = judge_turns(
            load_records(args.input),
            cfg,
            dry_run=args.dry_run,
            include_prompt=args.show_prompt,
        )
    except ImportError as exc:
        raise SystemExit(f"error: {exc}\nLLM judging needs: pip install -e '.[judge]'") from exc
    rendered = "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)

    routed = sum(
        bool(r.get("response", {}).get("short_circuited")) or r.get("route") == "heuristic_short_circuit"
        for r in records
    )
    summary = f"{len(records)} turns: {routed} short-circuited by heuristics, {len(records) - routed} "
    if args.dry_run:
        summary += "would be sent to the LLM judge (dry run, no requests made)"
    else:
        failed = sum(
            not r["response"]["short_circuited"] and not r["response"]["query_success"]
            for r in records
        )
        summary += f"sent to the LLM judge ({failed} failed requests)"
    print(summary, file=sys.stderr)
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
        description="Train multi-turn agents with RLVR and audit the reliability of their reasoning traces.",
    )
    parser.add_argument("--version", action="version", version=f"TraceRigor {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="summarize judged trajectory records")
    analyze.add_argument("input", help="JSON or JSONL trajectory records")
    analyze.add_argument("--output", "-o", help="write summary JSON to this path")
    analyze.add_argument("--group-by", help="optional record or metadata field")
    analyze.set_defaults(func=_analyze)

    judge = commands.add_parser(
        "judge",
        help="judge recorded agent turns with heuristics and an LLM judge",
        description=(
            "Score recorded turns for observation grounding, action coherence, and "
            "temporal consistency. Output JSONL can be summarized with 'tracerigor analyze'."
        ),
    )
    judge.add_argument("input", help="JSON or JSONL turn records (see tracerigor/judge/batch.py)")
    judge.add_argument("--output", "-o", help="write judged JSONL to this path")
    judge.add_argument("--config", help="judge YAML (default: tracerigor/judge/default_config.yaml)")
    judge.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8001/v1")
    judge.add_argument("--model", help="model name as served by the endpoint")
    judge.add_argument("--api-key-env", metavar="VAR", help="read the API key from this environment variable")
    judge.add_argument(
        "--no-structured-output",
        action="store_true",
        help="do not request JSON-schema guided decoding (for providers without support)",
    )
    judge.add_argument(
        "--dry-run",
        action="store_true",
        help="run Layer-0 heuristics only and make no network requests",
    )
    judge.add_argument("--show-prompt", action="store_true", help="include rendered judge prompts")
    judge.set_defaults(func=_judge)

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
