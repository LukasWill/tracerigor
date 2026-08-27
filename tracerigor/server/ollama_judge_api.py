# judge_api.py
import os
import time
import re
import threading
import uuid
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

import hydra
import wandb
import asyncio
from fastapi import FastAPI, HTTPException
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel
from ollama import AsyncClient, ResponseError

# -----------------------------------------------------------------------------
# ———  Per-process globals for WandB + Hydra ————————————————
# -----------------------------------------------------------------------------
_WANDB_INITIALIZED = {}
_GLOBAL_STEPS = {}
_PROCESS_LOCKS = {}
_HYDRA_LOCKS = {}
_HYDRA_INITIALIZED = {}
_PID_CONFIG = {}
_WANDB_TABLES: Dict[int, Dict[str, wandb.Table]] = {}

@contextmanager
def wandb_run_context():
    """Ensure WandB run is finished at the end of each request."""
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
            hydra.initialize(config_path="config")
            _HYDRA_INITIALIZED[pid] = True
        if pid not in _PID_CONFIG:
            _PID_CONFIG[pid] = hydra.compose(config_name="verifier_rubrics")
        return _PID_CONFIG[pid]

# -----------------------------------------------------------------------------
# ———  Core LLM-as-Judge logic (adapted) ——————————————————————
# -----------------------------------------------------------------------------
async def process_llm_judgments_ollama(
    input_data: List[Dict[str, Any]],
    client: AsyncClient,
    config: DictConfig,
    model_name: str,
    model_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build prompts from input_data (with reasoning_tokens & action_tokens),
    send them to Ollama, and return structured results.
    """
    prompts = []
    metadata = []
    tmpl = config.prompt_templates.default_env.faithfulness

    for item in input_data:
        prompt = tmpl.format(
            reasoning_tokens=item["reasoning_tokens"],
            action_tokens=item["action_tokens"]
        )
        prompts.append(prompt)
        metadata.append(item)

    # # Fire off all requests in parallel
    # async def _batch():
    #     tasks = [
    #         client.chat(
    #             model=model_name,
    #             messages=[
    #                 {"role": "user",   "content": p}
    #             ],
    #         )
    #         for p in prompts
    #     ]
    #     return await asyncio.gather(*tasks, return_exceptions=True)

    ## ─── batch sender ────────────────────────────────
    # async def _batch():
    #     tasks = []
    #     for p in prompts:
    #         # base payload for the chat request
    #         payload = {
    #             "model":      model_name,
    #             "messages": [
    #                 {"role": "user",   "content": p}
    #             ],
    #             # nest all generation args here
    #             "options": {
    #                 "temperature": config.api.temperature,
    #                 **model_params  # allow overriding model_params per request
    #             }
    #         }
    #         tasks.append(
    #             client.chat(**payload)
    #         )
    #     return await asyncio.gather(*tasks, return_exceptions=True)

    # # directly await your batch coroutine
    # raw = await _batch()

    ## ─── chunked batch sender ────────────────────────────────
    async def _batch_chunked():
        CHUNK_SIZE = 50  # tune this up/down
        all_raw = []
        for i in range(0, len(prompts), CHUNK_SIZE):
            chunk = prompts[i : i + CHUNK_SIZE]
            tasks = []
            for p in chunk:
                # base payload for the chat request
                payload = {
                    "model":      model_name,
                    "messages": [
                        {"role": "user",   "content": p}
                    ],
                    # nest all generation args here
                    "options": {
                        "temperature": config.api.temperature,
                        **model_params  # allow overriding model_params per request
                    }
                }
                tasks.append(
                    client.chat(**payload)
                )
            raw_chunk = await asyncio.gather(*tasks, return_exceptions=True)
            all_raw.extend(raw_chunk)
        return all_raw

    # directly await your batch coroutine
    raw = await _batch_chunked()

    ## ─── batch sender with concurrency limit ────────────────────────────────
    # async def _batch_limited():
    #     MAX_CONCURRENCY = 8  # tune this up/down
    #     sem = asyncio.Semaphore(MAX_CONCURRENCY)

    #     async def single_call(p: str):
    #         async with sem:
    #             return await client.chat(
    #                 model=model_name,
    #                 messages=[
    #                     {"role":"user",  "content": p}
    #                 ],
    #                 options={
    #                     "temperature": config.api.temperature,
    #                     **model_params
    #                 }
    #             )

    #     tasks = [single_call(p) for p in prompts]
    #     return await asyncio.gather(*tasks, return_exceptions=True)

    # raw = await _batch_limited()

    ## ─── batch sender with chunk+semaphore concurrency limit ────────────────────────────────
    # async def _batch_combined():
    #     CHUNK_SIZE     = 50   # number of prompts per mini‐batch
    #     MAX_CONCURRENCY = 8   # how many in‐flight RPCs at once
    #     sem = asyncio.Semaphore(MAX_CONCURRENCY)
    #     all_raw = []

    #     for i in range(0, len(prompts), CHUNK_SIZE):
    #         chunk = prompts[i : i + CHUNK_SIZE]

    #         async def single_call(p: str):
    #             async with sem:
    #                 return await client.chat(
    #                     model=model_name,
    #                     messages=[
    #                         {"role": "user",   "content": p}
    #                     ],
    #                     options={
    #                         "temperature": config.api.temperature,
    #                         **model_params
    #                     }
    #                 )

    #         # schedule and run this mini‐batch
    #         raw_chunk = await asyncio.gather(
    #             *(single_call(p) for p in chunk),
    #             return_exceptions=True
    #         )
    #         all_raw.extend(raw_chunk)

    #     return all_raw

    # # execute the combined chunk+semaphore batch
    # raw = await _batch_combined()

    results = []
    for md, resp in zip(metadata, raw):
        is_exc = isinstance(resp, Exception)
        text = "" if is_exc else resp.message.content or ""
        # if no text, treat as failure
        success = (not is_exc) and (len(text.strip())>0)
        if not success:
            # you could re-queue md for a retry here
            score = 0.0
            parse_success = False
            error = str(resp) if is_exc else "empty response"
        else:
            m = re.search(r'<answer>(YES|NO)</answer>', text, re.IGNORECASE)
            score = 1.0 if (m and m.group(1).upper()=="YES") else 0.0
            parse_success = bool(m)
            error = None

        results.append({
            **md,
            "prompt": None,       # omit bulky prompt
            "response": text,
            "success": success,
            "score": score,
            "parse_success": parse_success,
            "error": error
        })

    return results

async def run_llm_judge_ollama(input_data: List[Dict[str, Any]], model_name: str, model_params: Dict[str, Any] ) -> List[Dict[str, Any]]:
    """
    Wrap process_llm_judgments_ollama with WandB logging, step counting, etc.
    """
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
                name=f"{config.wandb.run_name}_{run_id}",
                config=OmegaConf.to_container(config, resolve=True)
            )
            _WANDB_INITIALIZED[pid] = True

        # actual LLM calls
        client = AsyncClient(host="http://localhost:11434", timeout=500)
        start = time.time()
        results = await process_llm_judgments_ollama(input_data, client, config, model_name, model_params)
        for r in results:
            r["model"] = model_name
        duration = time.time() - start

        # log scalars
        total = len(results)
        succ  = sum(r["success"] for r in results)
        parse_succ = sum(r["parse_success"] for r in results)
        wandb.log({
            "step": step,
            "duration": duration,
            "total": total,
            "success_rate": succ / total if total else 0.0,
            "avg_score": sum(r["score"] for r in results) / total if total else 0.0,
            "parse_success_rate": parse_succ / total if total else 0.0
        }, step=step)

        table = wandb.Table(columns=["step","id","reasoning_tokens","action_tokens","response","score","parse_success","model"])
        for r in results:
            table.add_data(step, r["id"], r["reasoning_tokens"], r["action_tokens"], r["response"], r["score"], r["parse_success"], r["model"])
        wandb.log({"responses": table}, step=step)

        return results

# -----------------------------------------------------------------------------
# FastAPI wiring with multi-model support
# -----------------------------------------------------------------------------
class BatchItem(BaseModel):
    id: str
    reasoning_tokens: str
    action_tokens: str

class BatchRequest(BaseModel):
    items: List[BatchItem]
    models: List[str] = ["llama3.3", "deepseek-r1:32b"]
    model_params: Optional[Dict[str, Any]] = {}  # e.g. {"temperature":0.2, "top_p":0.8}

app = FastAPI(title="LLM-as-Judge (Ollama)")

@app.post("/batch_judge")
async def batch_judge(req: BatchRequest):
    try:
        data = [i.dict() for i in req.items]
        # all_results = []
        # # fan-out per model
        # for model in req.models:
        #     # override config.api.name for this run
        #     # simplest: monkey-patch the hydra config
        #     pid = os.getpid()
        #     cfg = _get_hydra_config(pid)
        #     cfg.api.name = model
        #     with wandb_run_context():
        #         res = await run_llm_judge_ollama(data)
        #     all_results.extend(res)
        # return {"results": all_results}

        # launch all model‐runs in parallel
        async def single_model_run(model: str):
            with wandb_run_context():
                # res = await run_llm_judge_ollama(data, model)
                # pass through the model_params dict
                res = await run_llm_judge_ollama(data, model, req.model_params or {})
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