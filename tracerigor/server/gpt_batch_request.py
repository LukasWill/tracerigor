import asyncio
import math
import random
import time
from typing import List, Dict, Any, Union
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

class RateLimiter:
    """Rate limiter for OpenAI GPT API"""
    def __init__(self, qps_limit=70, rpm_limit=4000, tps_limit=15000):
        self.qps_limit = qps_limit
        self.rpm_limit = rpm_limit
        self.tps_limit = tps_limit
        self.request_timestamps = []
        self.token_counts = []
        self.semaphore = asyncio.Semaphore(qps_limit)

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

def run_gpt_request(prompts: List[Union[str, List[Dict[str, str]]]], config) -> List[Dict[str, Any]]:
    """
    Process prompts with OpenAI GPT API, handling rate limits.

    Args:
        prompts: List of prompt strings to process
        config: Config object that supports config.get() method

    Returns:
        List of dictionaries with results for each prompt
    """
    # Process in batches if needed
    batch_size = config.get("batch_size", 20)
    if len(prompts) <= batch_size:
        return _process_batch(prompts, config)

    # For larger sets, process in batches
    all_results = []
    batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

    for i, batch in enumerate(batches):
        if len(batches) > 1:
            print(f"Processing batch {i+1}/{len(batches)} ({len(batch)} prompts)")
        batch_results = _process_batch(batch, config)
        all_results.extend(batch_results)
        if i < len(batches) - 1:
            time.sleep(0.5)

    return all_results

def _process_batch(prompts: List[Union[str, List[Dict[str, str]]]], config) -> List[Dict[str, Any]]:
    """Process a single batch with rate limiting"""
    async def _async_batch_completions():
        async with AsyncOpenAI() as async_client:
            rate_limiter = RateLimiter(
                qps_limit=config.get("qps_limit", 70),
                rpm_limit=config.get("rpm_limit", 4000),
                tps_limit=config.get("tps_limit", 15000)
            )

            results = [{"response": "", "success": False, "retries": 0, "error": None} for _ in prompts]

            async def process_prompt(prompt: Union[str, List[Dict[str, str]]], index: int) -> None:
                retries = 0
                # Estimate tokens (1 token ≈ 4 chars) (string only)
                if isinstance(prompt, str):
                    estimated_prompt_tokens = len(prompt) // 4
                else:
                    # concatenate all contents for rough token estimation
                    estimated_prompt_tokens = sum(len(m.get("content", "")) for m in prompt) // 4
                estimated_completion_tokens = config.get("max_completion_tokens", 500)
                total_estimated_tokens = estimated_prompt_tokens + estimated_completion_tokens

                model_name = config.get("name", "gpt-5-nano-2025-08-07")
                use_responses = bool(config.get("use_responses_api", True)) and model_name.startswith("gpt-5")

                if use_responses:
                    # Responses API: max_output_tokens already includes reasoning + visible output
                    out_budget = int(config.get("max_output_tokens", 1280))
                    total_estimated_tokens = estimated_prompt_tokens + out_budget


                while retries <= config.get("max_retries", 3):
                    try:
                        async with rate_limiter.semaphore:
                            await rate_limiter.wait_if_needed(total_estimated_tokens)

                            # detect type: str -> wrap, list[dict] -> pass directly
                            if isinstance(prompt, str):
                                messages = [{"role": "user", "content": prompt}]
                            else:
                                messages = prompt   # already list of {"role":..., "content":...}

                            if use_responses:
                                sys_txt, input_payload = to_responses_input(messages)

                                # One-sentence, cost-saving constraint (keep <think>, ensure clean output):
                                # instr = (sys_txt + " Do not output any text before <think> or after </answer>.").strip()

                                response = await async_client.responses.create(
                                    model=model_name,
                                    instructions=sys_txt,                           # <- system here
                                    input=input_payload,                            # <- user here
                                    reasoning={
                                        "effort":  config.get("reasoning_effort",  "minimal"),  # minimal|low|medium|high
                                        # "summary": config.get("reasoning_summary", "concise"), # auto|concise|detailed
                                    },
                                    max_output_tokens=int(config.get("max_output_tokens", 1280)),
                                )

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
                            else:
                                # Non-gpt-5 path: Chat Completions as before (no unsupported args)
                                response = await async_client.chat.completions.create(
                                    model=config.get("name", "gpt-4.1-nano-2025-04-14"),
                                    # text only
                                    messages=messages,
                                    # # text + image
                                    # messages=[
                                    #     {
                                    #         "role": "user",
                                    #         "content": [
                                    #             {"type": "text", "text": prompt},
                                    #             {
                                    #                 "type": "image_url",
                                    #                 "image_url": {
                                    #                     "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                                    #                 }
                                    #             },
                                    #         ],
                                    #     }
                                    # ],
                                    temperature=config.get("temperature", 0.1),
                                    max_completion_tokens=estimated_completion_tokens,
                                    logprobs=True,  # not supported for gpt-5 series
                                    top_logprobs=config.get("top_logprobs", 20),  # the number of most likely tokens to return at each token position
                                )

                                choice  = response.choices[0]
                                message = choice.message
                                text    = message.content or ""

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

                            if use_responses:
                                results[index] = {
                                    "response": text,
                                    "success": True,
                                    "retries": retries,
                                    "error": None,
                                }
                            else:
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
                    except Exception as e:
                        # The OpenAI SDK usually carries a structured error
                        print("TYPE:", type(e).__name__, "| MSG:", str(e))
                        # Some SDK versions expose .status_code / .code / .message
                        for attr in ("status_code","code","message","response"):
                            if hasattr(e, attr):
                                print(attr, "=>", getattr(e, attr))
                        # If there is an HTTPX response attached:
                        resp = getattr(e, "response", None)
                        if resp is not None:
                            print("HTTP", resp.status_code, resp.text)
                        error_str = str(e)
                        retries += 1

                        # Exponential backoff for rate limit errors
                        if "rate_limit" in error_str.lower():
                            backoff_time = config.get("retry_delay", 1) * (2 ** (retries - 1))
                            backoff_time += random.uniform(0, 1)  # Add jitter
                            backoff_time = min(backoff_time, 30)  # Cap at 30s
                            await asyncio.sleep(backoff_time)
                        elif retries <= config.get("max_retries", 3):
                            await asyncio.sleep(config.get("retry_delay", 1))
                        else:
                            results[index] = {
                                "response": f"Error after {retries} attempts",
                                "success": False,
                                "retries": retries,
                                "error": error_str
                            }
                            return

            tasks = [process_prompt(prompt, i) for i, prompt in enumerate(prompts)]

            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=config.get("request_timeout", 120)
                )
            except asyncio.TimeoutError:
                pass

            return results

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(_async_batch_completions())
                loop.close()
            else:
                results = loop.run_until_complete(_async_batch_completions())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_async_batch_completions())
            loop.close()
    except Exception as e:
        return [{"response": f"Global error: {str(e)}", "success": False, "retries": 0, "error": str(e)}
                for _ in prompts]

    return results