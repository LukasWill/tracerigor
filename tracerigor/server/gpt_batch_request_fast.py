import asyncio
import random
import time
from typing import List, Dict, Any, Union, Optional
import httpx
from openai import AsyncOpenAI

from tracerigor.server.verdict_logprob_helper import (
    extract_verdict_and_offset,
    find_token_index_at_offset_tail,
    find_label_token_index,
    binary_probs_from_toplogprobs,
    binary_entropy,
    topk_entropy_from_toplogprobs,
    to_responses_input,
    safe_exp_logprob
)

_async_client: Optional[AsyncOpenAI] = None

def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        # Generous pool; tune as needed.
        limits  = httpx.Limits(max_connections=128, max_keepalive_connections=64)
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=60.0)
        _async_client = AsyncOpenAI(http_client=httpx.AsyncClient(limits=limits, timeout=timeout))
    return _async_client

class RateLimiter:
    """Rate limiter for OpenAI GPT API"""
    def __init__(self, qps_limit=70, rpm_limit=4000, tps_limit=15000, max_concurrency: int = 64):
        self.qps_limit = qps_limit
        self.rpm_limit = rpm_limit
        self.tps_limit = tps_limit
        self.request_timestamps = []
        self.token_counts = []
        # self.semaphore = asyncio.Semaphore(qps_limit)
        # concurrency should be its own knob (not = qps)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def wait_if_needed(self, estimated_tokens=500):
        now = time.time()
        # Clean up old timestamps (older than 60 seconds)
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]

        # Check RPM limit
        rpm_current = len(self.request_timestamps)
        if rpm_current >= self.rpm_limit:
            oldest = self.request_timestamps[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.wait_if_needed(estimated_tokens)

        # Check QPS limit (last 1 second)
        recent_requests = sum(1 for ts in self.request_timestamps if now - ts < 1)
        if recent_requests >= self.qps_limit:
            await asyncio.sleep(0.1)
            return await self.wait_if_needed(estimated_tokens)

        # Check TPS limit (last 1 second)
        recent_tokens = sum(tokens for ts, tokens in zip(self.request_timestamps, self.token_counts) if now - ts < 1)
        if recent_tokens + estimated_tokens >= self.tps_limit:
            await asyncio.sleep(0.2)
            return await self.wait_if_needed(estimated_tokens)

        # Update tracking
        self.request_timestamps.append(now)
        self.token_counts.append(estimated_tokens)

# ---------------------------
# One batch (async)
# ---------------------------
async def _process_batch_async(prompts: List[Union[str, List[Dict[str, str]]]], config) -> List[Dict[str, Any]]:
    """Process a single batch with rate limiting"""
    async_client = _get_async_client()
    rate_limiter = RateLimiter(
        qps_limit=config.get("qps_limit", 70),
        rpm_limit=config.get("rpm_limit", 4000),
        tps_limit=config.get("tps_limit", 15000),
        max_concurrency=config.get("max_concurrency", 64),
    )

    per_req_timeout = int(config.get("per_request_timeout", 45))
    launch_spread_s = float(config.get("launch_spread_s", 0.0))  # 0.5–0.8 recommended for big waves; 0 by default

    results = [{"response": "", "success": False, "retries": 0, "error": None} for _ in prompts]

    async def process_prompt(prompt: Union[str, List[Dict[str, str]]], index: int) -> None:
        # Estimate tokens (1 token ≈ 4 chars) (string only)
        if isinstance(prompt, str):
            estimated_prompt_tokens = len(prompt) // 4
            messages = [{"role": "user", "content": prompt}]
        else:
            # concatenate all contents for rough token estimation
            estimated_prompt_tokens = sum(len(m.get("content", "")) for m in prompt) // 4
            messages = prompt   # already list of {"role":..., "content":...}
        estimated_completion_tokens = config.get("max_completion_tokens", 500)
        total_estimated_tokens = estimated_prompt_tokens + estimated_completion_tokens

        model_name = config.get("name", "gpt-5-nano-2025-08-07")
        use_responses = bool(config.get("use_responses_api", True)) and model_name.startswith("gpt-5")

        if use_responses:
            # Responses API: max_output_tokens already includes reasoning + visible output
            out_budget = int(config.get("max_output_tokens", 1280))
            total_estimated_tokens = estimated_prompt_tokens + out_budget

        retries = 0
        max_retries = int(config.get("max_retries", 3))
        retry_delay = float(config.get("retry_delay", 1.0))

        while retries <= max_retries:
            try:
                # 1) throttle BEFORE occupying a concurrency slot
                await rate_limiter.wait_if_needed(total_estimated_tokens)
                # 2) take a concurrency slot only for the API call
                async with rate_limiter.semaphore:

                    if use_responses:
                        sys_txt, input_payload = to_responses_input(messages)
                        coro = async_client.responses.create(
                            model=model_name,
                            instructions=sys_txt,                           # <- system here
                            input=input_payload,                            # <- user here
                            reasoning={
                                "effort":  config.get("reasoning_effort",  "minimal"),  # minimal|low|medium|high
                                # "summary": config.get("reasoning_summary", "concise"), # auto|concise|detailed
                            },
                            max_output_tokens=int(config.get("max_output_tokens", 1280)),
                        )
                        response = await asyncio.wait_for(coro, timeout=per_req_timeout)

                        # Prefer the convenience accessor
                        text = getattr(response, "output_text", None) or ""
                        if not text and hasattr(response, "output") and response.output:
                            # very defensive fallback: stitch any text segments
                            try:
                                parts = []
                                for item in response.output:
                                    if hasattr(item, "content"):
                                        for c in item.content:
                                            t = getattr(c, "text", None)
                                            if t:
                                                parts.append(t)
                                text = "".join(parts)
                            except Exception:
                                text = ""

                        results[index] = {
                            "response": text,
                            "success": True,
                            "retries": retries,
                            "error": None,
                        }
                    else:
                        # Non-gpt-5 path: Chat Completions as before (no unsupported args)
                        # response = await async_client.chat.completions.create(
                        coro = async_client.chat.completions.create(
                            model=config.get("name", "gpt-4.1-nano-2025-04-14"),
                            # text only
                            messages=messages,
                            temperature=config.get("temperature", 0.1),
                            max_completion_tokens=estimated_completion_tokens,
                            logprobs=True,  # not supported for gpt-5 series
                            top_logprobs=config.get("top_logprobs", 20),  # the number of most likely tokens to return at each token position
                        )
                        response = await asyncio.wait_for(coro, timeout=per_req_timeout)

                        choice  = response.choices[0]
                        text    = choice.message.content or ""

                        # --- pull token-by-token info (OpenAI schema) ---
                        tokens = []
                        tops   = []
                        chosen_lps = []  # chosen token logprobs (full-vocab)
                        if choice.logprobs and choice.logprobs.content:
                            for item in choice.logprobs.content[-10:]:  # last few tokens
                                # item.token: str, item.logprob: float, item.top_logprobs: list[TopLogprob]
                                tokens.append(item.token)
                                chosen_lps.append(item.logprob)
                                if item.top_logprobs:
                                    tops.append([{"token": t.token, "logprob": t.logprob} for t in item.top_logprobs])
                                else:
                                    tops.append([])
                        # --- locate the verdict and compute binary probs/entropies ---
                        verdict, ch_start = extract_verdict_and_offset(text)
                        p_yes = p_no = H_bin = H_topk_labelpos = None

                        if verdict and tokens:
                            i0 = find_token_index_at_offset_tail(tokens, text, ch_start)
                            if i0 is None:
                                i0 = find_label_token_index(tokens)
                            if i0 is not None and 0 <= i0 < len(tops):
                                # Binary probs from top-k (approx over YES/NO)
                                p_yes, p_no = binary_probs_from_toplogprobs(tops[i0])
                                if p_yes is not None:
                                    H_bin = binary_entropy(p_yes)
                                H_topk_labelpos = topk_entropy_from_toplogprobs(tops[i0])
                                # chosen token logprob (full vocabulary)
                                verdict_logprob = chosen_lps[i0] if i0 < len(chosen_lps) else None
                                verdict_prob_full = safe_exp_logprob(verdict_logprob)

                        results[index] = {
                            "response": text,
                            "success": True,
                            "retries": retries,
                            "error": None,
                            "verdict": verdict,               # 'YES' | 'NO' (parsed from <answer>..</answer>)
                            "p_yes": p_yes,                   # probability mass for YES (approx, first-token)
                            "p_no":  p_no,                    # probability mass for NO
                            "binary_entropy": H_bin,          # H over {YES, NO}
                            "topk_entropy_at_label": H_topk_labelpos,  # approx entropy at first label token
                            "verdict_logprob": verdict_logprob,  # full-vocab ln P - logprob of the chosen token
                            "verdict_prob_full": verdict_prob_full,
                            "logprobs": {                    # keep raw for debugging if you want
                                "tokens": tokens,
                                "topk": tops,
                            }
                        }
                    return
            except asyncio.TimeoutError:
                # Hard timeout around the single API call
                retries += 1
                if retries > max_retries:
                    results[index] = {
                        "response": "",
                        "success": False,
                        "retries": retries,
                        "error": "timeout",
                        "error_type": "Timeout",
                        "status_code": None,
                    }
                    return
                # exponential backoff with jitter (short bucket)
                delay = min(5.0, retry_delay * (2 ** (retries - 1))) + random.random()
                await asyncio.sleep(delay)
            except Exception as e:
                # ---------- Rich introspection (your style) ----------
                etype = type(e).__name__
                msg   = str(e) or repr(e)
                status_code = getattr(e, "status_code", None)
                code        = getattr(e, "code", None)
                resp        = getattr(e, "response", None)

                # Try to pull HTTP status/body if this is an HTTPX/OpenAI-wrapped error
                resp_text = None
                retry_after_hint = None
                try:
                    if resp is not None:
                        # httpx.Response or similar
                        if status_code is None:
                            status_code = getattr(resp, "status_code", None)
                        # body text (best-effort; keep short to avoid huge logs)
                        try:
                            resp_text = getattr(resp, "text", None)
                            # If it's bytes-like, decode safely
                            if resp_text is None and hasattr(resp, "content"):
                                rt = resp.content
                                if isinstance(rt, (bytes, bytearray)):
                                    resp_text = rt[:2048].decode("utf-8", "ignore")
                        except Exception:
                            pass
                        # Retry-After header
                        try:
                            headers = getattr(resp, "headers", None)
                            if headers:
                                ra = headers.get("Retry-After") or headers.get("retry-after")
                                if ra:
                                    try:
                                        retry_after_hint = float(ra)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    pass

                # ---------- Classification (my style) ----------
                low = (msg or "").lower()
                is_timeout   = ("timeout" in low) or ("timed out" in low)
                is_rate      = (status_code == 429) or ("rate" in low and "limit" in low)
                is_server    = (status_code is not None and 500 <= int(status_code) < 600)
                is_client_nr = (status_code in (400, 403, 404, 422))  # non-retryable client errors
                is_auth_maybe_transient = (status_code == 401)  # 401 can be transient under batch load
                is_net_transient = any(s in low for s in [
                    "connection reset", "temporar", "unavailable",
                    "server disconnected", "connection aborted",
                    "remote protocol", "stream closed"
                ])

                # Some OpenAI SDKs expose specific exception classes; try best-effort detection
                # (Keep generic checks above so this still works if imports differ.)
                retryable = False
                backoff_bucket = "short"  # short|medium|long

                if is_rate:
                    retryable = True
                    backoff_bucket = "long"
                elif is_server or is_net_transient:
                    retryable = True
                    backoff_bucket = "medium"
                elif is_timeout:
                    retryable = True
                    backoff_bucket = "short"
                elif is_auth_maybe_transient:
                    retryable = True
                    backoff_bucket = "medium"  # 401 can be transient under concurrent load
                elif is_client_nr:
                    retryable = False  # immediate fail
                else:
                    # Unknown: treat as retryable a few times, then give up
                    retryable = True
                    backoff_bucket = "short"

                # ---------- Decide retry vs fail ----------
                retries += 1
                if (not retryable) or (retries > max_retries):
                    # Final failure for this prompt
                    results[index] = {
                        "response": "",
                        "success": False,
                        "retries": retries,
                        "error": msg,
                        "error_type": etype,
                        "status_code": status_code,
                        **({"response_text": resp_text[:2048]} if resp_text else {}),
                        **({"error_code": code} if code else {}),
                    }
                    return

                # ---------- Respect Retry-After if present ----------
                if retry_after_hint is not None:
                    delay = float(retry_after_hint) + random.random()
                else:
                    # bucketed exponential backoff with jitter
                    if backoff_bucket == "long":     # rate-limit
                        delay = min(30.0, retry_delay * (2 ** (retries - 1))) + random.random()
                    elif backoff_bucket == "medium": # 5xx / transient net
                        delay = min(15.0, retry_delay * (2 ** (retries - 1))) + random.random()
                    else:                             # short: generic/timeout-ish
                        delay = min(5.0,  retry_delay * (2 ** (retries - 1))) + random.random()

                # Optional: print one concise line (avoid spamming logs)
                print(f"[retry {retries}/{max_retries}] {etype} (HTTP {status_code or '-'}): {msg[:160]} — sleeping {delay:.2f}s")
                await asyncio.sleep(delay)

    # tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]
    # tasks = []
    # for i, prompt in enumerate(prompts):
    #     if launch_spread_s > 0:
    #         # spread the wave over launch_spread_s seconds
    #         await asyncio.sleep(launch_spread_s / max(1, len(prompts)))
    #     tasks.append(process_prompt(prompt, i))

    # try:
    #     await asyncio.gather(*tasks, return_exceptions=False)
    # except asyncio.TimeoutError:
    #     pass
    tasks: List[asyncio.Task] = []
    if launch_spread_s > 0 and len(prompts) > 1:
        for i, p in enumerate(prompts):
            await asyncio.sleep(launch_spread_s / len(prompts))
            tasks.append(asyncio.create_task(process_prompt(p, i)))
    else:
        tasks = [asyncio.create_task(process_prompt(p, i)) for i, p in enumerate(prompts)]

    # No outer batch timeout → each task returns success or timeout
    await asyncio.gather(*tasks, return_exceptions=False)
    return results

# ---------------------------
# Async entrypoint (new)
# ---------------------------
async def run_gpt_request_async(prompts: List[Union[str, List[Dict[str, str]]]], config) -> List[Dict[str, Any]]:
    batch_size = int(config.get("batch_size", 20))
    if len(prompts) <= batch_size:
        return await _process_batch_async(prompts, config)

    all_results: List[Dict[str, Any]] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        if len(prompts) > batch_size:
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts)+batch_size-1)//batch_size} ({len(batch)} prompts)")
        all_results.extend(await _process_batch_async(batch, config))
        if i + batch_size < len(prompts):
            await asyncio.sleep(0.5)  # tiny pause between waves
    return all_results

# ---------------------------
# Sync wrapper (kept for legacy callers)
# ---------------------------
def run_gpt_request(prompts: List[Union[str, List[Dict[str, str]]]], config) -> List[Dict[str, Any]]:
    # robust: works even if called from inside a running loop (e.g., FastAPI)
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(asyncio.run, run_gpt_request_async(prompts, config))
                return fut.result()
        else:
            return asyncio.run(run_gpt_request_async(prompts, config))
    except Exception as e:
        return [{"response": f"Global error: {str(e)}", "success": False, "retries": 0, "error": str(e)}
                for _ in prompts]