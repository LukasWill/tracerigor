# Verifier_api.py
import os
import time
import threading
import uuid
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Type

import hydra
import wandb
import asyncio
from fastapi import FastAPI, HTTPException
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from ollama import AsyncClient, ResponseError

from tracerigor.verifier.prompt.sokoban import *
from tracerigor.verifier.utils.parsers import _universal_score, _yesno_score
# from tracerigor.verifier.prompt.verifier_template_base import VerifierTemplate
from tracerigor.verifier.verifier.verifier_base import BaseVerifier

import logging
log = logging.getLogger("x_reasoner.ollama_verifier_api")
log.setLevel(logging.DEBUG)

# ------------------------------------------------------------------------
# Per-process globals for WandB + Hydra
# ------------------------------------------------------------------------
_WANDB_INITIALIZED: Dict[int, bool] = {}
_GLOBAL_STEPS: Dict[int, int] = {}
_PROCESS_LOCKS: Dict[int, threading.Semaphore] = {}
_HYDRA_LOCKS: Dict[int, threading.Lock] = {}
_HYDRA_INITIALIZED: Dict[int, bool] = {}
_PID_CONFIG: Dict[int, DictConfig] = {}
_WANDB_TABLES: Dict[int, Dict[str, wandb.Table]] = {}

@contextmanager
def wandb_run_context():
    """Ensure WandB run is finished and flag reset per request."""
    try:
        yield
    finally:
        if wandb.run is not None:
            pid = os.getpid()
            _WANDB_TABLES.pop(pid, None)
            wandb.finish()
            # mark that next time we need to init again
            _WANDB_INITIALIZED[pid] = False

def _get_hydra_config(pid: int) -> DictConfig:
    # thread/process-safe Hydra initialization
    if pid not in _HYDRA_LOCKS:
        _HYDRA_LOCKS[pid] = threading.Lock()
    with _HYDRA_LOCKS[pid]:
        if not _HYDRA_INITIALIZED.get(pid, False):
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            two_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_dir = os.path.join(two_level_up, "config")
            # use initialize_config_dir for absolute path
            with hydra.initialize_config_dir(config_dir=config_dir, job_name="ollama_verifier"):
                cfg = hydra.compose(config_name="verifier_rubrics")
            _HYDRA_INITIALIZED[pid] = True
        if pid not in _PID_CONFIG:
            _PID_CONFIG[pid] = cfg
        return _PID_CONFIG[pid]

# -----------------------------------------------------------------------------
# Ollama verifier class and specific rubrics
# -----------------------------------------------------------------------------
class OllamaBaseVerifier(BaseVerifier):
    """
    Base class for verifiers that use an Ollama backend.
    """
    def __init__(self, config: DictConfig, model_name: str, client: AsyncClient, model_params: Dict[str, Any]):
        super().__init__(config, model_name, client, model_params)

    async def evaluate_batch(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prompts = [self.get_template(item) for item in input_data]
        metadata = input_data

        ## combined chunk + semaphore
        # CHUNK_SIZE = 50
        # MAX_CONCURRENCY = 8
        # sem = asyncio.Semaphore(MAX_CONCURRENCY)
        # all_raw = []

        # for i in range(0, len(prompts), CHUNK_SIZE):
        #     chunk = prompts[i: i+CHUNK_SIZE]

        #     async def single_call(p: str):
        #         async with sem:
        #             return await self.client.chat(
        #                 model=self.model_name,
        #                 messages=[
        #                     {"role": "user",   "content": p}
        #                 ],
        #                 options={
        #                     "temperature": self.config.api.temperature,
        #                     **self.model_params
        #                 }
        #             )

        #     raw_chunk = await asyncio.gather(*(single_call(p) for p in chunk), return_exceptions=True)
        #     all_raw.extend(raw_chunk)

        ## process in chunks
        CHUNK_SIZE = 50  # tune this up/down
        all_raw = []
        for i in range(0, len(prompts), CHUNK_SIZE):
            chunk = prompts[i : i + CHUNK_SIZE]
            tasks = []
            for p in chunk:
                # normalize messages depending on type
                if isinstance(p, str):
                    messages = [{"role": "user", "content": p}]
                elif isinstance(p, list) and all(isinstance(m, dict) for m in p):
                        messages = p
                else:
                    raise TypeError(f"Unexpected prompt format: {type(p)}")
                # base payload for the chat request
                payload = {
                    "model":      self.model_name,
                    "messages": messages,
                    # nest all generation args here
                    "options": {
                        "temperature": self.config.api.temperature,
                        "logprobs": 5,   # request top-5 logprobs
                        **self.model_params  # allow overriding model_params per request
                    }
                }
                # --- no streaming
                tasks.append(
                    self.client.chat(**payload)
                )
                # --- turn on streaming
                # async def stream_chat(payload):
                #     tokens, tops = [], []
                #     full_text = ""
                #     async for chunk in await self.client.chat(**payload, stream=True):  # stream=True
                #         # chunk.message.content may be incremental text
                #         if chunk.message and chunk.message.content:
                #             full_text += chunk.message.content

                #         # chunk may carry per-token logprobs
                #         if hasattr(chunk, "logprobs") and chunk.logprobs:
                #             tokens.append(chunk.logprobs.get("token"))
                #             tops.append(chunk.logprobs.get("top_logprobs", []))
                #         elif isinstance(chunk, dict) and "logprobs" in chunk:
                #             tokens.append(chunk["logprobs"].get("token"))
                #             tops.append(chunk["logprobs"].get("top_logprobs", []))

                #     # Return normalized object
                #     return {
                #         "text": full_text,
                #         "tokens": tokens,
                #         "tops": tops,
                #     }
                # tasks.append(stream_chat(payload))

            raw_chunk = await asyncio.gather(*tasks, return_exceptions=True)
            all_raw.extend(raw_chunk)

        results = []
        for md, resp in zip(metadata, all_raw):
            is_exc = isinstance(resp, Exception)
            if is_exc:
                # Model call itself failed
                logging.debug(f"[{self.rubric}][{self.model_name}] model call ERROR for id={md['id']}: {resp}")
            text = "" if is_exc else resp.message.content or ""

            logprobs = None
            if not is_exc and hasattr(resp.message, "logprobs"):
                logprobs = resp.message.logprobs  # backend-dependent structure
            # if no text, treat as failure
            query_success = (not is_exc) and (len(text.strip())>0)
            if not query_success:
                score, parse_success, extra = 0.0, False, None
                error = str(resp) if is_exc else "empty response"
            else:
                logging.debug(f"Invoking parser for rubric={self.rubric}")
                # Parse the response based on the rubric
                parsed = self.parse_response(text, metadata_item=md) \
                         if self.rubric == "simulatability" else self.parse_response(text)
                # unpack 2- or 3-tuple
                score, parse_success, *maybe_ratio = parsed
                extra = maybe_ratio[0] if maybe_ratio else None
                error = None

            entry = {
                **md,
                "prompt": (
                    None if not getattr(self.config.api, "keep_prompts", False)
                    else (
                        p if isinstance(p, str)
                        else next((m["content"] for m in p if m.get("role") == "user"), None)
                    )
                ),
                "response":          text,
                "query_success":     query_success,
                "score":             score,
                "parse_success":     parse_success,
                "error":             error,
                "logprobs":          logprobs,
            }

            if extra is not None:
            # if getattr(self, "rubric", None) == "simulatability":
                entry['ratio'] = extra

            results.append(entry)

        return results

# -----------------------------------------------------------------------------
# class FaithfulnessVerifier(BaseVerifier):
#     system_prompt = "You are an expert evaluator. Think step by step, then <think>…</think><answer>YES/NO</answer>."
#     rubric = "faithfulness"
#     def get_template(self, item):
#         tmpl = self.config.prompt_templates.default_env.faithfulness
#         return tmpl.format(**item)

# class SimulatabilityVerifier(BaseVerifier):
#     system_prompt = "You are an expert evaluator. Provide <predicted_action>…</predicted_action><implicit>YES/NO</implicit>."
#     rubric = "simulatability"
#     def get_template(self, item):
#         tmpl = self.config.prompt_templates.default_env.simulatability
#         return tmpl.format(**item)
#     def parse_response(self, text, metadata_item):
#         # Print out the raw text and ground truth so we can see failures
#         log.debug(f"\n[SimulatabilityVerifier] raw response text:\n{text}\n")
#         log.debug(f"[SimulatabilityVerifier] ground-truth action_tokens: {metadata_item.get('action_tokens')}\n")

#         # Extract predicted actions and implicit flag
#         m_pred = re.search(r'<predicted_action>(.*?)</predicted_action>', text, re.IGNORECASE)
#         if not m_pred:
#             log.debug(f"[SimulatabilityVerifier] missing <predicted_action> in text:\n{text}")

#         m_imp  = re.search(r'<implicit>(YES|NO)</implicit>', text, re.IGNORECASE)
#         if not m_imp:
#             log.debug(f"[SimulatabilityVerifier] missing <implicit> in text:\n{text}")
#         predicted = [a.strip() for a in m_pred.group(1).split(',')] if m_pred else []
#         implic   = m_imp.group(1).upper() if m_imp else "NO"

#         log.debug(f"[SimulatabilityVerifier] parsed predicted={predicted}, implicit={implic}")

#         # Ground truth actions from metadata_item["action_tokens"]
#         gt = re.search(r'<answer>(.*?)</answer>', metadata_item.get("action_tokens",""), re.IGNORECASE)
#         if not gt:
#             log.debug(f"[SimulatabilityVerifier] missing ground-truth <answer> in action_tokens:\n{metadata_item.get('action_tokens')}")
#         actual = [a.strip() for a in gt.group(1).split(',')] if gt else []

#         # Compute ordered match ratio (prefix-wise):
#         match_count = sum(1 for (p, a) in zip(predicted, actual) if p == a)
#         ratio = match_count / len(actual) if actual else 0.0

#         log.debug(f"[SimulatabilityVerifier] match_count={match_count}, ratio={ratio}\n")

#         # Score based on implicit yes/no
#         score = 1.0 if implic == "YES" else 0.0
#         parse_success = bool(m_pred and m_imp and gt)
#         return score, parse_success, ratio

# class LogicalnessVerifier(BaseVerifier):
#     system_prompt = "You are an expert evaluator. Think step by step, then <think>…</think><answer>YES/NO</answer>."
#     rubric = "logicalness"
#     def get_template(self, item):
#         tmpl = self.config.prompt_templates.default_env.logicalness
#         return tmpl.format(**item)

def build_sokoban_verifier(rubric: str) -> Type[OllamaBaseVerifier]:
    class SokobanVerifier(OllamaBaseVerifier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rubric = rubric
            self._templates = {
                "universal": SokobanUniversalTemplate,
                "grounding": SokobanGroundingTemplate,
                "self_consistency": SokobanActionReasoningConsistencyTemplate,
                "history_consistency": SokobanHistoryConsistencyTemplate,
            }

        def get_template(self, item: Dict[str, Any]) -> str:
            if self.rubric not in self._templates:
                raise ValueError(f"Unknown rubric: {self.rubric}")
            return self._templates[self.rubric]().build_messages(item)

        def parse_response(self, text: str):
            if self.rubric == "universal":
                return _universal_score(text)
            return _yesno_score(text)

    return SokobanVerifier

# -----------------------------------------------------------------------------
# Async run & WandB logging
# -----------------------------------------------------------------------------
async def run_ollama_verifier(
    input_data: List[Dict[str, Any]],
    rubric: str,
    model_name: str,
    model_params: Dict[str, Any]
) -> List[Dict[str, Any]]:
    pid = os.getpid()
    with _PROCESS_LOCKS.setdefault(pid, threading.Semaphore(1)):
        # step & config
        step = _GLOBAL_STEPS.setdefault(pid, -1) + 1
        _GLOBAL_STEPS[pid] = step
        config = _get_hydra_config(pid)
        config.api.name = model_name
        # init WandB
        if not _WANDB_INITIALIZED.get(pid, False):
            run_id = uuid.uuid4().hex[:8]
            wandb.init(
                project=config.wandb.project,
                name=f"{config.wandb.run_name}_{rubric}_{run_id}",
                config=OmegaConf.to_container(config, resolve=True)
            )
            _WANDB_INITIALIZED[pid] = True

        # actual LLM calls
        client = AsyncClient(host="http://localhost:11434", timeout=500)
        # pick verifier
        # verifier_cls = {
        #     'faithfulness': FaithfulnessVerifier,
        #     'simulatability': SimulatabilityVerifier,
        #     'logicalness': LogicalnessVerifier
        # }[rubric]
        verifier_cls = build_sokoban_verifier(rubric)
        verifier: BaseVerifier = verifier_cls(config, model_name, client, model_params)
        start = time.time()
        results = await verifier.evaluate_batch(input_data)
        for r in results:
            r["model"] = model_name
        duration = time.time() - start

        # log scalars
        num_queried = len(results)
        num_query_success = sum(r["query_success"] for r in results)
        query_succ  = num_query_success / num_queried if num_queried else 0.0
        avg_verifier_score = sum(r["score"] for r in results) / num_queried if num_queried else 0.0
        parse_succ = sum(r["parse_success"] for r in results) / num_queried if num_queried else 0.0
        parse_succ_rate_on_query_successes = sum(r["parse_success"] for r in results) / num_query_success if num_query_success else 0.0
        if rubric == "simulatability":
            # only consider entries where ratio is not None
            simula_ratios = [r["ratio"] for r in results if r.get("ratio") is not None]
            avg_simula_ratio = sum(simula_ratios) / len(simula_ratios) if simula_ratios else 0.0
        wandb.log({
            "step": step,
            "duration": duration,
            "num_queried": num_queried,
            "query_success_rate": query_succ,
            "avg_verifier_score": avg_verifier_score,
            "parse_succ_rate_all": parse_succ,
            "parse_succ_rate_on_successes": parse_succ_rate_on_query_successes,
            **({"avg_simula_ratio": avg_simula_ratio} if rubric == "simulatability" else {}),
        }, step=step)

        cols = [
            'step', 'sample_id', 'reasoning_trace', 'chosen_actions', 'raw_response', 'verifier_score'
        ]
        if rubric == "simulatability":
            # insert 'simula_ratio' right after 'verifier_score'
            cols.append('simula_ratio')
        cols += [
            'parse_ok', 'model_name'
        ]
        table = wandb.Table(columns=cols)

        for r in results:
            row = [step, r['id'], r['reasoning_tokens'], r['action_tokens'],
                r['response'], r['score']
            ]
            if rubric == "simulatability":
                row.append(r.get('ratio', None))
            row += [
                r['parse_success'], r['model']
            ]
            table.add_data(*row)

        wandb.log({'responses': table}, step=step)
        return results

# -----------------------------------------------------------------------------
# FastAPI endpoint
# -----------------------------------------------------------------------------

# Minimal history unit used by universal/history verifiers
class HistoryItem(BaseModel):
    observation_text: Optional[str] = None
    # If this history step was image-based, set True; upstream can render "<image>"
    observation_image: Optional[bool] = False
    reasoning_tokens: Optional[str] = None
    action_tokens: Optional[str] = None

class BatchItem(BaseModel):
    id: str
    reasoning_tokens: str
    action_tokens: str

    # Keep but normalize to lowercase for consistency with prompts
    admissible_actions: List[str] = Field(
        default_factory=lambda: ["up", "down", "left", "right"]
    )

    # Step index used by universal/history prompts.
    current_step: Optional[int] = Field(default=None, description="Preferred.")

    # Preferred structured current observation
    current_observation_text: Optional[str] = None
    current_observation_image: Optional[bool] = False

    # Back-compat: if provided, overrides the computed value used in the prompt
    _current_observation_text_or_image: Optional[str] = ""

    # Optional short history (last 2–3 usually enough)
    history: List[HistoryItem] = Field(default_factory=list)

class BatchRequest(BaseModel):
    items: List[BatchItem]
    rubric: str  # 'self-consistency'|'simulatability'|'logicalness'
    models: List[str] = ["llama3.3", "deepseek-r1:32b"]
    model_params: Optional[Dict[str, Any]] = {}  # e.g. {"temperature":0.2, "top_p":0.8}


app = FastAPI(title="LLM-Ollama-Verifier (multi-rubrics)",
            #   description="Batch verification of LLM reasoning traces using multiple rubrics.",
            #   version="0.1.0",
            #   contact={
            #       "name": "",
            #       "email": ""
            #   }
              )

@app.post("/batch_verify")
async def batch_verify(req: BatchRequest):
    try:
        data = [i.model_dump() for i in req.items]
        for d in data:
            # normalize admissible actions to lowercase to match templates
            if "admissible_actions" in d and isinstance(d["admissible_actions"], list):
                d["admissible_actions"] = [str(a).lower() for a in d["admissible_actions"]]
        # all_results = []
        # # fan-out per model
        # for model in req.models:
        #     # override config.api.name for this run
        #     # simplest: monkey-patch the hydra config
        #     pid = os.getpid()
        #     cfg = _get_hydra_config(pid)
        #     cfg.api.name = model
        #     with wandb_run_context():
        #         res = await run_llm_verifier(data)
        #     all_results.extend(res)
        # return {"results": all_results}

        # launch all model‐runs in parallel
        async def single_model_run(model: str):
            with wandb_run_context():
                res = await run_ollama_verifier(data, req.rubric, model, req.model_params or {})
            return res

        batches = await asyncio.gather(
            *[single_model_run(m) for m in req.models],
            return_exceptions=False
        )
        # flatten list of lists
        results = [r for batch in batches for r in batch]
        return {"results": results}
    except ResponseError as e:
        raise HTTPException(status_code=e.status_code, detail=e.error)
    except Exception as e:
        # print full traceback to the server console
        import traceback; traceback.print_exc()
        # return the exception type and message in the 500 response
        detail = f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=detail)

# if __name__ == "__main__":
#     import asyncio

#     items = [{
#         "id":"ex-1",
#         "current_step":17,
#         "history":[
#             {"observation_text":"...", "reasoning_tokens":"<think>t-1</think>", "action_tokens":"left"},
#             {"observation_text":"...", "reasoning_tokens":"<think>t-2</think>", "action_tokens":"up"}
#         ],
#         "current_observation_text":"#####\n#_PXO#\n#####",
#         "reasoning_tokens":"<think>now…</think>",
#         "action_tokens":"left",
#         "admissible_actions": ["up","down","left","right"]
#     }]

#     # Ollama backend
#     async def demo_ollama():
#         res = await run_ollama_sokoban(items, rubric="universal",
#                                     model_name="deepseek-r1:32b",
#                                     model_params={"temperature":0.0})
#         print(res[0]["response"], res[0]["score"])

#     # OpenAI backend
#     async def demo_openai():
#         res = await run_openai_sokoban(items, rubric="history",
#                                     model_name="gpt-4.1",
#                                     model_params={"temperature":0.0})
#         print(res[0]["response"], res[0]["score"])

#     asyncio.run(demo_ollama())
#     asyncio.run(demo_openai())
