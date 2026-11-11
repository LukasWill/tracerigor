"""Offline batch judging: heuristic routing, CLI, and a stub-endpoint round trip."""

import json
from pathlib import Path

import pytest

from tracerigor.analysis import analyze_records, load_records
from tracerigor.cli import main
from tracerigor.judge.batch import judge_turns
from tracerigor.judge.config import JudgeConfig

SAMPLE_TURNS = Path(__file__).resolve().parents[1] / "examples" / "judge" / "sample_turns.jsonl"


def _offline_config(base_url: str = "http://127.0.0.1:9/v1") -> JudgeConfig:
    cfg = JudgeConfig.from_dict(
        {"provider": {"base_url": base_url, "model": "stub", "max_retries": 0, "max_concurrency": 2}}
    )
    cfg.audit.enabled = False
    return cfg


def test_dry_run_routes_mechanical_failures_to_heuristics() -> None:
    records = judge_turns(load_records(SAMPLE_TURNS), _offline_config(), dry_run=True)
    routes = {record["id"]: record["route"] for record in records}
    assert routes == {
        "grounded": "llm_judge",
        "hallucinated_state": "llm_judge",
        "incoherent_action": "llm_judge",
        "invalid_action": "heuristic_short_circuit",
        "format_violation": "heuristic_short_circuit",
        "stale_belief": "llm_judge",
    }
    # Pass-through fields survive; packet fields are not echoed back.
    assert all("scenario" in record and "observation" not in record for record in records)


def test_dry_run_prompt_includes_history_and_ground_truth() -> None:
    records = judge_turns(
        load_records(SAMPLE_TURNS), _offline_config(), dry_run=True, include_prompt=True
    )
    stale = next(record for record in records if record["id"] == "stale_belief")
    user_prompt = stale["prompt"][1]["content"]
    assert "[Initial Observation]" in user_prompt  # history entry from the previous turn
    assert "target0 is 1 cell(s) below and at the same column as box0" in user_prompt
    assert "{" not in user_prompt.split("Reasoning to verify")[0]  # no unfilled placeholders


def test_cli_dry_run_writes_jsonl(tmp_path, capsys) -> None:
    out = tmp_path / "judged.jsonl"
    assert main(["judge", str(SAMPLE_TURNS), "--dry-run", "-o", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 6
    assert "2 short-circuited by heuristics" in capsys.readouterr().err


def test_round_trip_against_stub_endpoint(requires_judge_client, stub_judge_server, tmp_path) -> None:
    base_url, requests = stub_judge_server

    records = judge_turns(load_records(SAMPLE_TURNS), _offline_config(base_url))
    by_id = {record["id"]: record["response"] for record in records}

    # Heuristics answered two turns without an LLM call.
    assert len(requests) == 4
    assert by_id["invalid_action"]["short_circuited"]
    assert by_id["format_violation"]["short_circuited"]
    for turn_id in ("grounded", "hallucinated_state", "incoherent_action", "stale_belief"):
        assert by_id[turn_id]["query_success"] and by_id[turn_id]["parse_success"]
        assert by_id[turn_id]["rubrics"]["observation_grounding"]["score"] == 1.0
    # Structured output is requested with the universal rubric schema.
    assert requests[0]["response_format"]["type"] == "json_schema"

    path = tmp_path / "judged.jsonl"
    path.write_text("".join(json.dumps(record, default=str) + "\n" for record in records))
    summary = analyze_records(load_records(path))
    assert summary["scored_record_count"] == 4
    assert summary["short_circuit_rate"] == pytest.approx(2 / 6)


def test_client_reuse_across_event_loops(requires_judge_client, stub_judge_server) -> None:
    """Regression: batches larger than max_concurrency must not fail after the first.

    The concurrency semaphore used to stay bound to the first ``asyncio.run``
    loop, so later batches silently returned ``success=False`` for most items.
    """
    from tracerigor.judge.client import JudgeClient

    base_url, _ = stub_judge_server
    client = JudgeClient(_offline_config(base_url).provider)
    batch = [[{"role": "user", "content": "judge this"}]] * 6
    for _ in range(3):
        results = client.judge_batch_sync(batch)
        assert [r["error"] for r in results if not r["success"]] == []
