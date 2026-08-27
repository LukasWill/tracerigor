
from __future__ import annotations
import json, re, logging
from typing import Any, Dict, Optional, Tuple, Callable


class BaseParser:  # minimal shim
    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x, **kwargs):
        self.logger = logging.getLogger(f"tracerigor.verifier.parsers.{self.__class__.__name__}")
        self.extract_fn = extract_fn
        for k, v in kwargs.items():
            setattr(self, k, v)

    def parse(self, text: str) -> Any:
        return self.extract_fn(text)

    def parse_answer(self, completion: Any) -> Any:
        if isinstance(completion, str):
            return self.parse(completion)
        return self.parse(completion[-1]["content"])

YESNO_RE = re.compile(r"<think>(?P<think>.*?)</think>\s*<answer>(?P<ans>YES|NO)</answer>", re.IGNORECASE | re.DOTALL)

def _safe_json_loads(s: str) -> Dict[str, Any]:
    # Strip code fences or surrounding whitespace
    s = s.strip()
    if s.startswith("```"):
        # remove triple backticks and optional leading 'json'
        lines = [ln for ln in s.splitlines() if ln.strip("`").strip()]
        if lines and lines[0].lstrip("`").strip().lower() in ("json",):
            lines = lines[1:]
        # remove closing fence if present
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines)
    return json.loads(s)

def _yes_no_to_score(label: Optional[str]) -> Optional[float]:
    if not label:
        return None
    lab = label.strip().upper()
    if lab == "YES":
        return 1.0
    if lab == "NO":
        return 0.0
    # treat "N/A" or unknown as None; aggregator decides what to do
    return None

def _to01(yes_no: str) -> int:
    return 1 if str(yes_no).strip().upper() == "YES" else 0

def _aggregate(scores: Dict[str, int]) -> float:
    vals = [v for v in scores.values() if v is not None]
    return round(sum(vals) / max(1, len(vals)), 6)

class UniversalVerifierParser(BaseParser):
    """
    Parse the Universal Verifier JSON:
    {
      "factual_grounding": {"yes_no":"YES|NO","evidence":"..."},
      "action_reasoning_consistency": {"yes_no":"YES|NO","evidence":"..."},
      "history_consistency": {"yes_no":"YES|NO|N/A","evidence":"..."},
      "score":{"grounding":0|1,"behavioral":0|1,"history":0|1,"aggregate":0.0},
      "meta":{"judged_step": int}
    }
    """
    def parse(self, text: str) -> Dict[str, Any]:  # type: ignore[override]
        try:
            obj = _safe_json_loads(text)
        except Exception as e:
            self.logger.error("JSON parse error: %s", e)
            raise
        # Normalize & patch score if missing / inconsistent
        fg = obj.get("factual_grounding", {})
        ar = obj.get("action_reasoning_consistency", {})
        hc = obj.get("history_consistency", {})
        scores = obj.get("score", {})

        g = _to01(fg.get("yes_no", "NO"))
        b = _to01(ar.get("yes_no", "NO"))
        h_raw = str(hc.get("yes_no", "N/A")).upper()
        h = None if h_raw == "N/A" else _to01(h_raw)

        # Build score dict consistently
        out_scores: Dict[str, Any] = {
            "grounding": g,
            "behavioral": b,
            "history": h if h is not None else None,
        }
        # Compute aggregate over available
        aggr = _aggregate({k: v for k, v in out_scores.items() if v is not None})
        obj["score"] = {
            "grounding": g,
            "behavioral": b,
            "history": -1 if h is None else h,  # keep N/A visible as -1
            "aggregate": aggr,
        }
        return obj

    def make_reward_fn(self, weights: Dict[str, float] | None = None):
        """
        Returns a function: text -> (reward, metrics).
        weights keys: 'grounding', 'behavioral', 'history' (history ignored if N/A).
        Default: equal weights on available signals.
        """
        def reward_fn(text: str):
            obj = self.parse(text)
            g = obj["score"]["grounding"]
            b = obj["score"]["behavioral"]
            h = obj["score"]["history"]  # -1 means N/A
            # Dynamic equal weights if not provided
            if weights is None:
                terms = [("grounding", g), ("behavioral", b)]
                if h != -1:
                    terms.append(("history", h))
                w = 1.0 / len(terms)
                reward = sum(v * w for _, v in terms)
            else:
                _w = dict(weights)
                total_w = sum(v for k, v in _w.items() if k in ("grounding","behavioral","history"))
                if total_w <= 0:
                    raise ValueError("weights must sum to > 0")
                # ignore history if N/A
                eff_total = total_w - (_w.get("history", 0.0) if h == -1 else 0.0)
                if eff_total <= 0:
                    # fallback to equal over present signals
                    terms = [("grounding", g), ("behavioral", b)]
                    if h != -1:
                        terms.append(("history", h))
                    w = 1.0 / len(terms)
                    reward = sum(v * w for _, v in terms)
                else:
                    reward = (g * _w.get("grounding", 0.0) + b * _w.get("behavioral", 0.0) + (0 if h == -1 else h * _w.get("history", 0.0))) / eff_total
            metrics = {
                "verifier/grounding": g,
                "verifier/behavioral": b,
                "verifier/history": None if h == -1 else h,
                "verifier/aggregate": obj["score"]["aggregate"],
            }
            # bubble up judged step if present
            if isinstance(obj.get("meta"), dict) and "judged_step" in obj["meta"]:
                metrics["verifier/judged_step"] = obj["meta"]["judged_step"]
            return float(reward), metrics
        return reward_fn

class YesNoXMLParser(BaseParser):
    """
    Parse binary outputs of the form:
      <think>…</think><answer>YES|NO</answer>
    Returns dict with {'yes_no': 'YES'|'NO', 'score': 1|0, 'think': '...'}
    """
    def parse(self, text: str) -> Dict[str, Any]:  # type: ignore[override]
        m = YESNO_RE.search(text)
        if not m:
            raise ValueError("Could not extract <think>...</think><answer>YES|NO</answer> from text.")
        ans = m.group("ans").upper()
        think = m.group("think").strip()
        return {"yes_no": ans, "score": 1 if ans == "YES" else 0, "think": think}

def default_universal_reward(text: str) -> float:
    """Convenience: equal-weight reward from universal JSON output."""
    return UniversalVerifierParser().make_reward_fn()(text)[0]


# For prediction:
# If block is missing or yes_no == "N/A" → None.
# For observation/reasoning:
# Missing/invalid is pessimistically treated as 0 (but should rarely happen).
class SelfConsistencySubtagParser(BaseParser):
    """
    Parse structured self-consistency JSON for Sokoban:

    {
      "observation_consistency": {"yes_no":"YES|NO","evidence":[...]},
      "reasoning_consistency":   {"yes_no":"YES|NO","evidence":[...]},
      "prediction_consistency":  {"yes_no":"YES|NO|N/A","evidence":[...]}  # optional
    }

    We normalize:
      - yes_no -> 0.0/1.0 (or None for N/A / missing),
      - scalar_scores: flat view for downstream code.
    """
    def parse(self, text: str) -> Dict[str, Any]:  # type: ignore[override]
        try:
            obj = _safe_json_loads(text)
        except Exception as e:
            self.logger.error("SelfConsistency JSON parse error: %s", e)
            raise

        def _to_score(block: Dict[str, Any], *, allow_missing: bool = False) -> Optional[float]:
            """
            - If allow_missing is True and block is empty: return None.
            - Otherwise, use _yes_no_to_score; if that returns None and
              allow_missing is False, fall back to 0.0.
            """
            if not block and allow_missing:
                return None
            yn = block.get("yes_no")
            score = _yes_no_to_score(yn)  # YES->1.0, NO->0.0, N/A/unknown->None
            if score is None and not allow_missing:
                # For observation/reasoning, treat invalid/missing as NO.
                return 0.0
            return score

        oc = obj.get("observation_consistency") or {}
        rc = obj.get("reasoning_consistency") or {}
        pc = obj.get("prediction_consistency") or {}

        obs = _to_score(oc, allow_missing=False)  # 0.0 / 1.0
        rea = _to_score(rc, allow_missing=False)  # 0.0 / 1.0
        # For prediction, allow:
        # - missing block, or
        # - yes_no == "N/A"
        pred = _to_score(pc, allow_missing=True)  # 0.0 / 1.0 / None

        vals = [v for v in (obs, rea, pred) if v is not None]
        aggregate = round(float(sum(vals) / len(vals)), 6) if vals else 0.0

        obj["scalar_scores"] = {
            "observation": obs,      # 0.0 or 1.0
            "reasoning":   rea,      # 0.0 or 1.0
            "prediction":  pred,     # 0.0, 1.0, or None
            "aggregate":   aggregate,
        }
        return obj

# If <prediction> is missing / empty / nonsense, the verifier just answers "NO".
# Then prediction_score is always 0 or 1, never None.
# Aggregation is simpler; no N/A needed.
# class SelfConsistencySubtagParser(BaseParser):
#     """
#     Parse structured self-consistency JSON for Sokoban:

#     {
#       "observation_consistency": {"yes_no":"YES|NO","evidence":[...]},
#       "reasoning_consistency":   {"yes_no":"YES|NO","evidence":[...]},
#       "prediction_consistency":  {"yes_no":"YES|NO|N/A","evidence":[...]}
#     }

#     We normalize:
#       - yes_no -> 0/1 (or None for N/A),
#       - scalar_scores: flat view for downstream code.
#     """
#     def parse(self, text: str) -> Dict[str, Any]:  # type: ignore[override]
#         try:
#             obj = _safe_json_loads(text)
#         except Exception as e:
#             self.logger.error("SelfConsistency JSON parse error: %s", e)
#             raise

#         def _to_score(block: Dict[str, Any], *, allow_na: bool = False):
#             yn = str(block.get("yes_no", "NO")).strip().upper()
#             if allow_na and yn == "N/A":
#                 return None
#             return 1 if yn == "YES" else 0

#         oc = obj.get("observation_consistency", {}) or {}
#         rc = obj.get("reasoning_consistency", {}) or {}
#         pc = obj.get("prediction_consistency", {}) or {}

#         obs = _to_score(oc, allow_na=False)
#         rea = _to_score(rc, allow_na=False)
#         pred = _to_score(pc, allow_na=True)

#         # simple aggregate: mean over available dimensions (ignore None)
#         vals = [v for v in (obs, rea, pred) if v is not None]
#         aggregate = round(float(sum(vals) / len(vals)), 6) if vals else 0.0

#         obj["scalar_scores"] = {
#             "observation": obs,      # 0 or 1
#             "reasoning": rea,        # 0 or 1
#             "prediction": pred,      # 0, 1, or None
#             "aggregate": aggregate,  # float in [0,1]
#         }
#         return obj

def _selfconsistency_structured_score(text: str, mode: str = "mean"):
    """
    Return (score, parse_success, extra_dict) for the 'self_consistency' rubric.

    - score: float in [0,1], aggregated from observation/reasoning/prediction
      via combine_selfconsistency_scores.
    - extra_dict: full parsed JSON including 'scalar_scores'.
    """
    try:
        parser = SelfConsistencySubtagParser()
        obj = parser.parse(text)
    except Exception:
        return 0.0, False, None

    scalars = obj.get("scalar_scores", {})
    score = combine_selfconsistency_scores(scalars, mode=mode)

    score_f = float(score if score is not None else 0.0)
    return score_f, True, obj


def combine_selfconsistency_scores(
    scalars: dict,
    mode: str = "reasoning_primary"
) -> float:
    o = scalars.get("observation")
    r = scalars.get("reasoning")
    p = scalars.get("prediction")  # 0/1 or None

    # Ensure numeric defaults
    r = 0 if r is None else r

    if mode == "and":
        # AND over available dimensions
        vals = [v for v in (o, r, p) if v is not None]
        return float(1.0 if vals and all(v == 1 for v in vals) else 0.0)

    if mode == "mean":
        # mean over available dimensions
        vals = [v for v in (o, r, p) if v is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0

    if mode == "reasoning_only":
        # exactly what you had before
        return float(r)

    # default: reasoning primary, penalize inconsistent obs/pred a bit
    penalty = 0.0
    if o == 0:
        penalty += 0.25
    if p == 0:
        penalty += 0.25
    return max(0.0, min(1.0, float(r) - penalty))

def _yesno_score(text: str):
    m = re.search(r"<answer>(YES|NO)</answer>", text, re.IGNORECASE)
    if not m:
        return 0.0, False
    return (1.0 if m.group(1).upper() == "YES" else 0.0), True

def _universal_score(text: str):
    try:
        s = text.strip()
        if s.startswith("```"):
            lines = [ln for ln in s.splitlines() if ln.strip("`").strip()]
            if lines and lines[0].lstrip("`").strip().lower() in ("json",):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines)
        obj = json.loads(s)
        score = obj.get("score", {}).get("aggregate")
        if score is None:
            parts = []
            for k in ("factual_grounding","action_reasoning_consistency","history_consistency"):
                v = obj.get(k, {}).get("yes_no")
                if v and v.upper() in ("YES","NO"):
                    parts.append(1.0 if v.upper()=="YES" else 0.0)
            score = sum(parts)/len(parts) if parts else 0.0
        return float(score), True
    except Exception:
        return 0.0, False


def _sciworld_universal_score(text: str):
    """
    Parse SciWorld universal verifier JSON output:
    {
      "observation_grounding": {"yes_no": "YES|NO", "evidence": "..."},
      "action_coherence": {"yes_no": "YES|NO", "evidence": "..."},
      "temporal_consistency": {"yes_no": "YES|NO", "evidence": "..."}
    }

    Returns (score, parse_success, extra_dict).
    """
    try:
        s = text.strip()
        if s.startswith("```"):
            lines = [ln for ln in s.splitlines() if ln.strip("`").strip()]
            if lines and lines[0].lstrip("`").strip().lower() in ("json",):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines)
        obj = json.loads(s)

        # Extract scores from SciWorld-specific keys
        parts = []
        scalar_scores = {}
        for k in ("observation_grounding", "action_coherence", "temporal_consistency"):
            v = obj.get(k, {}).get("yes_no")
            if v and v.upper() in ("YES", "NO"):
                score_val = 1.0 if v.upper() == "YES" else 0.0
                parts.append(score_val)
                # Map to shorter names for scalar_scores
                short_key = k.replace("_", "")  # observationgrounding -> observationgrounding
                if k == "observation_grounding":
                    scalar_scores["grounding"] = score_val
                elif k == "action_coherence":
                    scalar_scores["action"] = score_val
                elif k == "temporal_consistency":
                    scalar_scores["temporal"] = score_val

        aggregate = sum(parts) / len(parts) if parts else 0.0
        scalar_scores["aggregate"] = aggregate

        # Add scalar_scores to obj for downstream use
        obj["scalar_scores"] = scalar_scores

        return float(aggregate), True, obj
    except Exception:
        return 0.0, False, None


if __name__ == "__main__":
    # Universal JSON → reward + metrics
    uvp = UniversalVerifierParser()
    reward_fn = uvp.make_reward_fn(weights={"grounding":1.0, "behavioral":1.0, "history":1.0})

    # Binary YES/NO outputs
    yn = YesNoXMLParser()
    yn_out = yn.parse("<think>...</think><answer>YES</answer>")
    # -> {'yes_no': 'YES', 'score': 1, 'think': '...'}
