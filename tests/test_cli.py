import json

import pytest

import tracerigor
from tracerigor.analysis import analyze_records
from tracerigor.cli import main
from tracerigor.judge.config import JudgeConfig


def test_version_flag_matches_package(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert tracerigor.__version__ in capsys.readouterr().out


def test_numeric_groups_are_ordered_numerically(tmp_path, capsys) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "".join(
            json.dumps({"training_step": step, "rubrics": {"action_coherence": 1}}) + "\n"
            for step in (10, 2, 1)
        )
    )
    assert main(["analyze", str(path), "--group-by", "training_step"]) == 0
    assert list(json.loads(capsys.readouterr().out)["groups"]) == ["1", "2", "10"]


def test_short_circuit_rate_is_reported() -> None:
    summary = analyze_records(
        [{"response": {"short_circuited": True}}, {"response": {"short_circuited": False}}]
    )
    assert summary["short_circuit_rate"] == pytest.approx(0.5)


def test_judge_yaml_defaults_and_env_expansion(tmp_path, monkeypatch) -> None:
    default = JudgeConfig.from_yaml()
    assert default.rubrics == ["observation_grounding", "action_coherence", "temporal_consistency"]

    monkeypatch.setenv("TRACERIGOR_TEST_KEY", "from-env")
    path = tmp_path / "judge.yaml"
    path.write_text("provider:\n  api_key: ${TRACERIGOR_TEST_KEY}\n  model: m\n")
    assert JudgeConfig.from_yaml(path).provider.api_key == "from-env"
