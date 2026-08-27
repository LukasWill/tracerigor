"""Schema-tolerant summaries for judged RLVR trajectory records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


DEFAULT_RUBRICS = (
    "observation_grounding",
    "action_coherence",
    "temporal_consistency",
)


def load_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load records from a JSON array/object or newline-delimited JSON file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"trajectory input does not exist: {source}")

    if source.suffix.lower() == ".jsonl":
        records = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {source}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} of {source} is not a JSON object")
                records.append(value)
        return records

    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{source} must contain only JSON objects")
        return value
    if isinstance(value, dict):
        for key in ("records", "samples", "data"):
            nested = value.get(key)
            if isinstance(nested, list) and all(isinstance(item, dict) for item in nested):
                return nested
        return [value]
    raise ValueError(f"{source} must contain a JSON object or array")


def _nested(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _as_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "yes", "true"}:
            return 1.0
        if normalized in {"fail", "no", "false"}:
            return 0.0
        if normalized in {"uncertain", "unknown"}:
            return 0.5
        try:
            return float(normalized)
        except ValueError:
            return None
    if isinstance(value, Mapping):
        for key in ("score", "value", "label", "yes_no", "verdict"):
            score = _as_score(value.get(key))
            if score is not None:
                return score
    return None


def _rubric_score(record: Mapping[str, Any], rubric: str) -> float | None:
    aliases = {
        "observation_grounding": ("observation_grounding", "grounding"),
        "grounding": ("grounding", "observation_grounding"),
        "action_coherence": ("action_coherence", "action"),
        "temporal_consistency": ("temporal_consistency", "temporal"),
    }.get(rubric, (rubric,))
    for alias in aliases:
        candidates = (
            ("response", "rubrics", alias),
            ("rubrics", alias),
            ("eval_results", alias),
            (alias,),
            (f"{alias}_score",),
        )
        for path in candidates:
            score = _as_score(_nested(record, path))
            if score is not None:
                return score
    return None


def _boolean_rate(records: Iterable[Mapping[str, Any]], paths: Sequence[Sequence[str]]) -> float | None:
    values: List[float] = []
    for record in records:
        for path in paths:
            value = _nested(record, path)
            if isinstance(value, bool):
                values.append(float(value))
                break
    return fmean(values) if values else None


def analyze_records(
    records: Sequence[Mapping[str, Any]],
    *,
    rubrics: Sequence[str] = DEFAULT_RUBRICS,
    group_by: str | None = None,
) -> Dict[str, Any]:
    """Compute coverage and reliability summaries without assuming one run schema."""
    rubric_summary: Dict[str, Dict[str, Any]] = {}
    per_record_means: List[float] = []

    for rubric in rubrics:
        scores = [score for record in records if (score := _rubric_score(record, rubric)) is not None]
        if scores:
            rubric_summary[rubric] = {
                "count": len(scores),
                "coverage": len(scores) / len(records) if records else 0.0,
                "mean": fmean(scores),
                "pass_rate": sum(score >= 0.75 for score in scores) / len(scores),
                "uncertain_rate": sum(0.25 < score < 0.75 for score in scores) / len(scores),
            }

    for record in records:
        scores = [score for rubric in rubrics if (score := _rubric_score(record, rubric)) is not None]
        if scores:
            per_record_means.append(fmean(scores))

    result: Dict[str, Any] = {
        "record_count": len(records),
        "scored_record_count": len(per_record_means),
        "mean_trace_reliability": fmean(per_record_means) if per_record_means else None,
        "parse_success_rate": _boolean_rate(
            records,
            (("response", "parse_success"), ("parse_success",), ("parse_ok",)),
        ),
        "query_success_rate": _boolean_rate(
            records,
            (("response", "query_success"), ("query_success",)),
        ),
        "rubrics": rubric_summary,
    }

    if group_by:
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            value = record.get(group_by)
            if value is None:
                value = _nested(record, ("metadata", group_by))
            if value is not None:
                grouped[str(value)].append(record)
        result["groups"] = {
            key: analyze_records(group, rubrics=rubrics)
            for key, group in sorted(grouped.items())
        }

    return result
