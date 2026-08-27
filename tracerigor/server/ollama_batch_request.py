import asyncio
import random
import time
from typing import List, Dict, Any
from ollama import AsyncClient, ResponseError

class RateLimiter:
    def __init__(self, qps_limit=70, rpm_limit=4000, tps_limit=15000):
        self.qps_limit = qps_limit
        self.rpm_limit = rpm_limit
        self.tps_limit = tps_limit
        self.request_timestamps: List[float] = []
        self.token_counts: List[int] = []
        self.semaphore = asyncio.Semaphore(qps_limit)

    async def wait_if_needed(self, estimated_tokens=500):
        now = time.time()
        # Cleanup & enforce RPM
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]
        if len(self.request_timestamps) >= self.rpm_limit:
            await asyncio.sleep(60 - (now - self.request_timestamps[0]))
            return await self.wait_if_needed(estimated_tokens)
        # Check QPS limit (last 1 second)
        recent_reqs = sum(1 for ts in self.request_timestamps if now - ts < 1)
        if recent_reqs >= self.qps_limit:
            await asyncio.sleep(0.1)
            return await self.wait_if_needed(estimated_tokens)
        # Check TPS limit (last 1 second)
        recent_toks = sum(tok for ts, tok in zip(self.request_timestamps, self.token_counts) if now - ts < 1)
        if recent_toks + estimated_tokens >= self.tps_limit:
            await asyncio.sleep(0.2)
            return await self.wait_if_needed(estimated_tokens)
        # Update tracking
        self.request_timestamps.append(now)
        self.token_counts.append(estimated_tokens)

async def judge_single(prompt: str,
                       client: AsyncClient,
                       rate_limiter: RateLimiter,
                       retries: int = 3) -> Dict[str, Any]:
    """Send one judge request, with simple retry/backoff."""
    attempt = 0
    est_tokens = len(prompt) // 4 + 500
    while attempt <= retries:
        attempt += 1
        try:
            async with rate_limiter.semaphore:
                await rate_limiter.wait_if_needed(est_tokens)
            resp = await client.chat(
                model="llama3.3",
                messages=[
                    {"role": "system",
                     "content": "You are an expert evaluator. Score the student's answer from 1-10, and explain your reasoning."},
                    {"role": "user", "content": prompt}
                ]
            )
            return {
                "response": resp.message.content,
                "success": True,
                "retries": attempt - 1,
                "error": None
            }
        # except ResponseError as e:
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err and attempt <= retries:
                # exponential backoff with jitter
                backoff = min(30, (2 ** (attempt-1)) + random.random())
                await asyncio.sleep(backoff)
            elif attempt <= retries:
                await asyncio.sleep(2 ** (attempt - 1))
            else:
                return {
                    "response": "",
                    "success": False,
                    "retries": attempt - 1,
                    "error": str(e)
                }

async def judge_batch(prompts: List[str],
                      qps_limit: float = 70.0,
                      rpm_limit: int = 4000,
                      tps_limit: int = 15000,
                      max_concurrency: int = 10) -> List[Dict[str, Any]]:
    client = AsyncClient("http://localhost:11434")
    rate_limiter = RateLimiter(qps_limit, rpm_limit, tps_limit)
    sem = asyncio.Semaphore(max_concurrency)

    async def worker(p: str, idx: int, out: List):
        async with sem:
            out[idx] = await judge_single(p, client, rate_limiter)

    results: List[Dict[str, Any]] = [{}] * len(prompts)
    tasks = [worker(p, i, results) for i, p in enumerate(prompts)]
    await asyncio.gather(*tasks)
    return results

def run(prompts: List[str]):
    return asyncio.run(judge_batch(prompts))

# Example usage:
if __name__ == "__main__":
    sample_prompts = [
        "Explain the difference between supervised and unsupervised learning.",
        "What are the ethical concerns around LLM deployment?",
        # …more prompts…
    ]
    out = run(sample_prompts)
    for r in out:
        print(r)