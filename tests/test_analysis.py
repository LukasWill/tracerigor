import json

import pytest

from tracerigor.analysis import analyze_records, load_records


def test_load_and_analyze_jsonl(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "training_step": 0,
                    "rubrics": {
                        "grounding": "pass",
                        "action_coherence": {"score": 1},
                        "temporal_consistency": "uncertain",
                    },
                    "parse_success": True,
                },
                {
                    "training_step": 10,
                    "response": {
                        "rubrics": {
                            "grounding": {"label": "fail"},
                            "action_coherence": {"score": 0},
                            "temporal_consistency": {"score": 1},
                        },
                        "parse_success": False,
                    },
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = analyze_records(load_records(path), group_by="training_step")
    assert summary["record_count"] == 2
    assert summary["scored_record_count"] == 2
    assert summary["parse_success_rate"] == pytest.approx(0.5)
    assert sorted(summary["groups"]) == ["0", "10"]


def test_invalid_jsonl_reports_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_records(path)
