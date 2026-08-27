
from __future__ import annotations
import base64, io, os
import math, random
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig
import wandb

# persistent per-process store of step-row tables
_WANDB_VERIFIER_STEP_TABLES = {}  # pid -> { rubric -> { bucket -> wandb.Table } }
_WANDB_RESPONSES_TABLES = {}  # (pid, rubric) -> wandb.Table

# Per-sample columns (order matters!)
STEP_ROW_FIELDS = [
    "id",
    "obs_image",          # W&B Image cell (first obs image if present)
    "reasoning_tokens",
    "action_tokens",
    "response",
    "query_success",
    "parse_success",
]

def _get_table_freq(config: DictConfig, default:int=10) -> int:
    try:
        return int(getattr(config, "wandb", {}).get("table_logging_frequency", default))
    except Exception:
        return default

def _get_table_k(config: DictConfig, default:int=8) -> int:
    try:
        return int(getattr(config, "wandb", {}).get("table_samples", default))
    except Exception:
        return default

def _bucketize(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ok = [r for r in results if r.get("query_success")]
    succ = [r for r in ok if r.get("parse_success") and float(r.get("score", 0.0)) >= 0.5]
    fail = [r for r in ok if r.get("parse_success") and float(r.get("score", 0.0)) < 0.5]
    parse_failed = [r for r in ok if not r.get("parse_success")]
    api_err = [r for r in results if not r.get("query_success")]
    return {
        "success": succ,
        "fail": fail,
        "parse_failed": parse_failed,
        "api_err": api_err,
    }

def _maybe(val, default=None):
    return val if val is not None else default

def _pick(rows: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0 or not rows:
        return []
    if len(rows) <= k:
        return rows
    return random.sample(rows, k)

def _ensure_steprow_table(pid: int, rubric: str, bucket: str, k: int) -> tuple[wandb.Table, list[str]]:
    """
    Ensure we have a persistent table for (pid, rubric, bucket) with K-sample columns.
    Returns (table, columns).
    """
    # fields = [
    #     "id","reasoning_tokens","action_tokens","response",
    #     "query_success","parse_success","verdict_prob_full","p_yes","p_no","binary_entropy",
    # ]
    cols = ["step","model"] + [f"sample_{i}_{f}" for i in range(1, k+1) for f in STEP_ROW_FIELDS]

    store = _WANDB_VERIFIER_STEP_TABLES.setdefault(pid, {}).setdefault(rubric, {})
    tbl = store.get(bucket)

    # Recreate if first time or K changed (column count mismatch)
    if tbl is None or len(getattr(tbl, "columns", [])) != len(cols):
        tbl = wandb.Table(columns=cols)
        store[bucket] = tbl

    return tbl, cols

# place this tiny helper above _row_for_step (or inline if you prefer)
def _first_image_cell(sample: dict):
    """
    Build a wandb.Image (or empty string) from sample['current_observation_image'].
    Uses your existing _to_wandb_image helper.
    """
    imgs = sample.get("current_observation_image")
    if not imgs:
        return None

    if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
        return _to_wandb_image(imgs[0], caption=str(sample.get("id","")))
    return _to_wandb_image(imgs, caption=str(sample.get("id","")))


def _row_for_step(step:int, model:str, samples:list, k:int) -> list:
    row = [step, model]
    # flatten each picked sample into the row
    per_sample_cols = len(STEP_ROW_FIELDS)
    for i in range(k):
        if i < len(samples):
            s = samples[i] or {}
            img_cell = _first_image_cell(s)
            row.extend([
                s.get("id", ""),                      # id: always string
                img_cell,                             # obs_image: wandb.Image or None
                s.get("reasoning_tokens", ""),        # string
                s.get("action_tokens", ""),           # string
                s.get("response", ""),                # string
                s.get("query_success", None),         # bool or None
                s.get("parse_success", None),         # bool or None
            ])
        else:
            # pad this slot to match STEP_ROW_FIELDS
            # pad this slot with type-consistent defaults
            row.extend([
                "",        # id (string)
                None,      # obs_image
                "",        # reasoning_tokens
                "",        # action_tokens
                "",        # response
                None,      # query_success
                None,      # parse_success
            ])
    return row

def log_verifier_examples_steprow(
    results: list[dict],
    rubric: str,
    step: int,
    model_name: str,
    *,
    samples_per_bucket: int = 8,
) -> None:
    """
    Log one step-row into 4 tables (success/fail/parse_failed/api_err).
    Keys:
      verifier_examples/{rubric}/success
      verifier_examples/{rubric}/fail
      verifier_examples/{rubric}/parse_failed
      verifier_examples/{rubric}/api_err
    """
    pid = os.getpid()
    buckets = _bucketize(results)
    for bucket_name, items in buckets.items():
        tbl, cols = _ensure_steprow_table(pid, rubric, bucket_name, samples_per_bucket)
        picked = _pick(items, samples_per_bucket)
        row = _row_for_step(step, model_name, picked, samples_per_bucket)
        tbl.add_data(*row)

        # log a snapshot so W&B keeps history (accumulated rows)
        snapshot = wandb.Table(columns=cols, data=tbl.data)
        wandb.log({f"verifier_examples/{rubric}/{bucket_name}": snapshot}, step=step)

def log_metrics(metrics: Dict[str, Any], run=None, step: Optional[int] = None, prefix: str = "verifier", commit: bool = True) -> None:
    """Log metrics to Weights & Biases if available.

    - metrics: dict of simple scalars (nested dicts are flattened under `prefix/`).
    - run: optional wandb.Run; if None, uses global `wandb.log` if wandb is initialized.
    - step: optional global step.
    - prefix: a namespace prefix (default 'verifier').
    - commit: wandb commit flag.
    """

    def _flatten(d: Dict[str, Any], base: str = "") -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            nk = f"{base}/{k}" if base else k
            if isinstance(v, dict):
                out.update(_flatten(v, nk))
            else:
                out[nk] = v
        return out

    payload = _flatten(metrics, prefix)
    if run is not None:
        run.log(payload, step=step, commit=commit)
    else:
        wandb.log(payload, step=step, commit=commit)

def reward_and_log(response_text: str, weights: Dict[str, float] | None = None, run=None, step: Optional[int] = None, prefix: str = "verifier") -> float:
    """Parse universal JSON, compute reward, and log metrics. Returns the scalar reward."""
    from .parsers import UniversalVerifierParser
    parser = UniversalVerifierParser()
    reward_fn = parser.make_reward_fn(weights=weights)
    reward, metrics = reward_fn(response_text)
    # Add reward into metrics
    metrics = dict(metrics)
    metrics["reward"] = reward
    log_metrics(metrics, run=run, step=step, prefix=prefix)
    return reward

def log_verifier_batch_to_wandb(
    results: List[Dict[str, Any]],
    rubric: str,
    step: Optional[int] = None,
    duration: Optional[float] = None,
) -> None:
    """Log a batch of verifier results to Weights & Biases."""
    # ---------- scalars (unchanged, plus a couple optional summaries) ----------
    # log scalars
    num_queried = len(results)
    num_query_success = sum(r["query_success"] for r in results)
    query_succ  = num_query_success / num_queried if num_queried else 0.0
    avg_verifier_score = sum(r["score"] for r in results) / num_queried if num_queried else 0.0
    # new: self-consistency sub-scores (if present)
    extra_scalars = {}
    if rubric == "self_consistency":
        obs_vals = [r.get("sub_observation_score") for r in results
                    if r.get("sub_observation_score") is not None]
        rea_vals = [r.get("sub_reasoning_score") for r in results
                    if r.get("sub_reasoning_score") is not None]
        pred_vals = [r.get("sub_prediction_score") for r in results
                     if r.get("sub_prediction_score") is not None]

        if obs_vals:
            extra_scalars["selfconsistency_avg_observation"] = sum(obs_vals) / len(obs_vals)
        if rea_vals:
            extra_scalars["selfconsistency_avg_reasoning"] = sum(rea_vals) / len(rea_vals)
        if pred_vals:
            extra_scalars["selfconsistency_avg_prediction"] = sum(pred_vals) / len(pred_vals)

    parse_succ = sum(r["parse_success"] for r in results) / num_queried if num_queried else 0.0
    parse_succ_rate_on_query_successes = sum(r["parse_success"] for r in results) / num_query_success if num_query_success else 0.0
    if rubric == "simulatability":
        # only consider entries where ratio is not None
        simula_ratios = [r["ratio"] for r in results if r.get("ratio") is not None]
        avg_simula_ratio = sum(simula_ratios) / len(simula_ratios) if simula_ratios else 0.0

    # token-level summaries (present only when available)
    probs_yes = [r["p_yes"] for r in results if r.get("p_yes") is not None]  # p_yes and p_no comes from a two-class renormalization over {YES, NO} using the top-k subset, ignoring all other tokens
    probs_no = [r["p_no"] for r in results if r.get("p_no") is not None]
    ents  = [r["binary_entropy"] for r in results if r.get("binary_entropy") is not None]
    yes_flags = [1.0 if (r.get("verdict") == "YES") else 0.0 for r in results if r.get("verdict") is not None]

    # verdict_logprob is the true full-vocab logprob for the chosen token (present only when available)
    vlogps = [r["verdict_logprob"]   for r in results if r.get("verdict_logprob")   is not None and math.isfinite(r["verdict_logprob"])]
    vprobs = [r["verdict_prob_full"] for r in results if r.get("verdict_prob_full") is not None]

    avg_p_yes = (sum(probs_yes) / len(probs_yes)) if probs_yes else None
    avg_p_no = (sum(probs_no) / len(probs_no)) if probs_no else None
    avg_bin_entropy = (sum(ents) / len(ents)) if ents else None
    yes_rate = (sum(yes_flags) / len(yes_flags)) if yes_flags else None

    avg_vlogp  = (sum(vlogps) / len(vlogps)) if vlogps else None
    avg_vprob  = (sum(vprobs) / len(vprobs)) if vprobs else None

    wandb.log({
        "step": step,
        "duration": duration,
        "num_queried": num_queried,
        "query_success_rate": query_succ,
        "avg_verifier_score": avg_verifier_score,
        "parse_succ_rate_all": parse_succ,
        "parse_succ_rate_on_successes": parse_succ_rate_on_query_successes,
        **({"avg_simula_ratio": avg_simula_ratio} if rubric == "simulatability" else {}),
        # : token-level scalar summaries (only log when present)
        **({"avg_p_yes": avg_p_yes} if avg_p_yes is not None else {}),
        **({"avg_p_no": avg_p_no} if avg_p_no is not None else {}),
        **({"avg_binary_entropy": avg_bin_entropy} if avg_bin_entropy is not None else {}),
        **({"verdict_yes_rate": yes_rate} if yes_rate is not None else {}),
        **({"avg_verdict_logprob": avg_vlogp}  if avg_vlogp  is not None else {}),
        **({"avg_verdict_prob":    avg_vprob}  if avg_vprob  is not None else {}),
        **extra_scalars,
    }, step=step)

    # ---------- dynamic columns ----------
    base_cols = [
        'step', 'sample_id', 'reasoning_trace', 'chosen_actions', 'raw_response', 'verifier_score'
    ]
    tail_cols = ['parse_ok', 'model_name']

    # Detect if any row has images
    any_images = any(bool(r.get("current_observation_image")) for r in results)

    # Only add these if any row actually has a non-None value
    optional_map = [
        ("verdict",               "verdict"),
        ("p_yes",                 "p_yes"),
        ("p_no",                  "p_no"),
        ("binary_entropy",        "binary_entropy"),
        ("topk_entropy_at_label", "topk_entropy_at_label"),
        ("verdict_logprob",       "verdict_logprob"),     # <-- NEW
        ("verdict_prob_full",     "verdict_prob_full"),   # <-- NEW
        ("logprobs",              "logprobs"),  # can be a dict; W&B Tables accept JSON-like values
        # NEW: per-subtag self-consistency scores
        ("sub_observation_score", "sub_observation_score"),
        ("sub_reasoning_score",   "sub_reasoning_score"),
        ("sub_prediction_score",  "sub_prediction_score"),
    ]
    optional_present = [
        (k, header)
        for (k, header) in optional_map
        if any((k in r) and (r.get(k) is not None) for r in results)
    ]

    cols = list(base_cols)
    if any_images:
        cols.insert(4, "observation_image")  # put image just before 'raw_response' for visibility
    if rubric == "simulatability":
        # insert 'simula_ratio' right after 'verifier_score'
        cols.append('simula_ratio')
    cols += tail_cols
    cols += [header for _, header in optional_present]

    # table = wandb.Table(columns=cols)
    # ACCUMULATE rather than recreate
    pid = os.getpid()
    key = (pid, rubric)
    table = _WANDB_RESPONSES_TABLES.get(key)

    if table is None or list(getattr(table, "columns", [])) != cols:
        table = wandb.Table(columns=cols)
        _WANDB_RESPONSES_TABLES[key] = table

    for r in results:
        # row = [step, r['id'], r['reasoning_tokens'], r['action_tokens'],
        #     r['response'], r['score']
        # ]
        row = [
            step,
            r['id'],
            r['reasoning_tokens'],
            r['action_tokens'],
        ]
        # If we have images, add the first one (table cell expects one object)
        if any_images:
            img_cell = None
            imgs = r.get("current_observation_image")
            if isinstance(imgs, (list, tuple)) and len(imgs) > 0:
                img_cell = _to_wandb_image(imgs[0], caption=f"{r['id']}")
            elif imgs:
                img_cell = _to_wandb_image(imgs, caption=f"{r['id']}")
            row.append(img_cell)

        # raw_response + score
        row += [r['response'], r['score']]

        if rubric == "simulatability":
            row.append(r.get('ratio', None))
        row += [
            r['parse_success'], r['model'],
        ]
        # append optionals in the same order as columns (may be None for gpt-5 runs with no logprobs)
        row += [r.get(k) for (k, _) in optional_present]

        table.add_data(*row)

    # wandb.log({'responses': table}, step=step)
    # log a snapshot that contains ALL accumulated rows
    snapshot = wandb.Table(columns=cols, data=table.data)
    wandb.log({'responses': snapshot}, step=step)


def _to_wandb_image(img_src: Any, *, caption: Optional[str] = None):
    """
    Convert one image source (data URL, bytes, local path) to wandb.Image.
    Returns None if it cannot be read (we fail soft).
    """
    try:
        # Lazy import to avoid hard dep if images aren't logged
        from PIL import Image  # pip install pillow
        import wandb

        # bytes / bytearray
        if isinstance(img_src, (bytes, bytearray)):
            im = Image.open(io.BytesIO(img_src)).convert("RGB")
            return wandb.Image(im, caption=caption)

        # strings (data URL or path)
        if isinstance(img_src, str):
            s = img_src.strip()

            # data URL
            if s.startswith("data:"):
                try:
                    header, b64 = s.split(",", 1)
                except ValueError:
                    return None
                raw = base64.b64decode(b64)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                return wandb.Image(im, caption=caption)

            # local path (only if exists on THIS machine)
            if os.path.exists(s):
                return wandb.Image(s, caption=caption)

            # if it's an http(s) URL, we skip (no remote fetch in this minimal edit)
            return None

        # dicts like {"type":"image_url","image_url":{"url": ...}} (from some mm libs)
        if isinstance(img_src, dict):
            url = img_src.get("image_url", {}).get("url")
            if url and url.startswith("data:"):
                header, b64 = url.split(",", 1)
                raw = base64.b64decode(b64)
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                return wandb.Image(im, caption=caption)
            # non-data URLs are skipped in this minimal edit
            return None

        # numpy array / PIL.Image
        try:
            import numpy as np  # type: ignore
            if isinstance(img_src, np.ndarray):
                return wandb.Image(img_src, caption=caption)
        except Exception:
            pass

        try:
            from PIL.Image import Image as PILImage  # type: ignore
            if isinstance(img_src, PILImage):
                return wandb.Image(img_src, caption=caption)
        except Exception:
            pass

    except Exception:
        # fail soft — we don't want logging to kill the run
        return None

    return None