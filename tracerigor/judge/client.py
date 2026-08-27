"""
Provider-agnostic async LLM client for the judge stack.

Reuses proven patterns from gpt_batch_request_fast.py:
  - Proper rate limiting (QPS / RPM / TPS)
  - Semaphore-based concurrency control
  - Exponential backoff with jitter and bucketed delays
  - Dual API support (Chat Completions + Responses API for gpt-5)
  - Launch spread for large batches
  - Rich error classification & retry logic

Supports any OpenAI-compatible endpoint:
  - Self-hosted vLLM (local or remote)
  - OpenAI (GPT-4o, GPT-5, etc.)
  - Together AI, OpenRouter, Fireworks, etc.
  - Any provider exposing /v1/chat/completions
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

try:
    import httpx
    from openai import AsyncOpenAI
except ImportError:
    httpx = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from tracerigor.judge.config import ProviderConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter (ported from gpt_batch_request_fast.py)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket style rate limiter tracking QPS, RPM, and TPS."""

    def __init__(
        self,
        qps_limit: int = 70,
        rpm_limit: int = 4000,
        tps_limit: int = 15000,
        max_concurrency: int = 64,
    ):
        self.qps_limit = qps_limit
        self.rpm_limit = rpm_limit
        self.tps_limit = tps_limit
        self.request_timestamps: List[float] = []
        self.token_counts: List[int] = []
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def wait_if_needed(self, estimated_tokens: int = 500) -> None:
        now = time.time()
        # Clean timestamps older than 60 s
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        self.token_counts = self.token_counts[-len(self.request_timestamps):]

        # RPM check
        if len(self.request_timestamps) >= self.rpm_limit:
            oldest = self.request_timestamps[0]
            wait = 60 - (now - oldest)
            if wait > 0:
                await asyncio.sleep(wait)
                return await self.wait_if_needed(estimated_tokens)

        # QPS check (last 1 s)
        recent_reqs = sum(1 for ts in self.request_timestamps if now - ts < 1)
        if recent_reqs >= self.qps_limit:
            await asyncio.sleep(0.1)
            return await self.wait_if_needed(estimated_tokens)

        # TPS check (last 1 s)
        recent_tokens = sum(
            tok for ts, tok in zip(self.request_timestamps, self.token_counts)
            if now - ts < 1
        )
        if recent_tokens + estimated_tokens >= self.tps_limit:
            await asyncio.sleep(0.2)
            return await self.wait_if_needed(estimated_tokens)

        self.request_timestamps.append(now)
        self.token_counts.append(estimated_tokens)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class JudgeClient:
    """Async LLM client for judge calls against one provider endpoint."""

    def __init__(self, provider: ProviderConfig):
        if AsyncOpenAI is None:
            raise ImportError(
                "openai and httpx are required for JudgeClient. "
                "Install with: pip install openai httpx"
            )
        self.provider = provider

        limits = httpx.Limits(
            max_connections=provider.max_concurrency + 16,
            max_keepalive_connections=provider.max_concurrency,
        )
        timeout = httpx.Timeout(
            connect=10.0,
            read=provider.timeout_s,
            write=10.0,
            pool=provider.timeout_s,
        )
        self._client = AsyncOpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            http_client=httpx.AsyncClient(limits=limits, timeout=timeout),
        )
        self._rate_limiter = _RateLimiter(
            qps_limit=getattr(provider, "qps_limit", 70),
            rpm_limit=getattr(provider, "rpm_limit", 4000),
            tps_limit=getattr(provider, "tps_limit", 15000),
            max_concurrency=provider.max_concurrency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def judge_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Send a batch of message lists to the judge endpoint.

        For large batches, applies launch spread to avoid thundering herd.
        """
        n = len(messages_batch)
        launch_spread_s = getattr(self.provider, "launch_spread_s", 0.0)
        batch_size = getattr(self.provider, "batch_size", 0)

        # Chunked processing for very large batches
        if batch_size and batch_size > 0 and n > batch_size:
            all_results: List[Dict[str, Any]] = []
            for i in range(0, n, batch_size):
                chunk = messages_batch[i : i + batch_size]
                chunk_results = await self._launch_batch(chunk, json_schema, i, launch_spread_s)
                all_results.extend(chunk_results)
                if i + batch_size < n:
                    await asyncio.sleep(0.5)
            return all_results

        return await self._launch_batch(messages_batch, json_schema, 0, launch_spread_s)

    def judge_batch_sync(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Blocking wrapper for non-async call sites."""
        return _run_async(self.judge_batch(messages_batch, json_schema))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _launch_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
        json_schema: Optional[Dict[str, Any]],
        offset: int,
        launch_spread_s: float,
    ) -> List[Dict[str, Any]]:
        """Launch a batch of requests with optional spread."""
        tasks: List[asyncio.Task] = []
        if launch_spread_s > 0 and len(messages_batch) > 1:
            for i, msgs in enumerate(messages_batch):
                await asyncio.sleep(launch_spread_s / len(messages_batch))
                tasks.append(asyncio.create_task(
                    self._call_one(msgs, json_schema, offset + i)
                ))
        else:
            tasks = [
                asyncio.create_task(self._call_one(msgs, json_schema, offset + i))
                for i, msgs in enumerate(messages_batch)
            ]
        return list(await asyncio.gather(*tasks))

    async def _call_one(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
        index: int,
    ) -> Dict[str, Any]:
        """Single request with rate limiting, retry, and error classification."""
        p = self.provider
        retries = 0
        last_error: Optional[str] = None
        retry_delay = 1.0

        # Estimate tokens for rate limiter
        est_tokens = sum(
            len(str(m.get("content", ""))) for m in messages
        ) // 4 + p.max_completion_tokens

        use_responses = p.use_responses_api and p.model.startswith("gpt-5")

        while retries <= p.max_retries:
            try:
                # Rate limit before taking concurrency slot
                await self._rate_limiter.wait_if_needed(est_tokens)

                async with self._rate_limiter.semaphore:
                    t0 = time.monotonic()

                    if use_responses:
                        result = await self._call_responses_api(messages, p, index)
                    else:
                        result = await self._call_chat_api(messages, json_schema, p, index)

                    result["latency_ms"] = (time.monotonic() - t0) * 1000
                    return result

            except asyncio.TimeoutError:
                last_error = "timeout"
                retries += 1
                if retries <= p.max_retries:
                    delay = min(5.0, retry_delay * (2 ** (retries - 1))) + random.random()
                    logger.warning(
                        "[JudgeClient] retry %d/%d idx=%d: timeout (wait %.1fs)",
                        retries, p.max_retries, index, delay,
                    )
                    await asyncio.sleep(delay)

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retries += 1

                # Classify error and determine retry strategy
                backoff_bucket, retryable = _classify_error(exc)

                if not retryable or retries > p.max_retries:
                    break

                if backoff_bucket == "long":
                    delay = min(30.0, retry_delay * (2 ** (retries - 1))) + random.random()
                elif backoff_bucket == "medium":
                    delay = min(15.0, retry_delay * (2 ** (retries - 1))) + random.random()
                else:
                    delay = min(5.0, retry_delay * (2 ** (retries - 1))) + random.random()

                logger.warning(
                    "[JudgeClient] retry %d/%d idx=%d: %s (wait %.1fs)",
                    retries, p.max_retries, index, last_error[:160], delay,
                )
                await asyncio.sleep(delay)

        return {
            "index": index,
            "response": "",
            "success": False,
            "error": last_error,
            "latency_ms": 0.0,
            "model": p.model,
        }

    async def _call_chat_api(
        self,
        messages: List[Dict[str, Any]],
        json_schema: Optional[Dict[str, Any]],
        p: ProviderConfig,
        index: int,
    ) -> Dict[str, Any]:
        """Standard Chat Completions API call."""
        kwargs: Dict[str, Any] = {
            "model": p.model,
            "messages": messages,
            "temperature": p.temperature,
            "max_completion_tokens": p.max_completion_tokens,
        }

        if json_schema and p.use_structured_output:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        resp = await asyncio.wait_for(
            self._client.chat.completions.create(**kwargs),
            timeout=p.timeout_s,
        )
        content = resp.choices[0].message.content or ""

        return {
            "index": index,
            "response": content,
            "success": True,
            "error": None,
            "model": p.model,
        }

    async def _call_responses_api(
        self,
        messages: List[Dict[str, Any]],
        p: ProviderConfig,
        index: int,
    ) -> Dict[str, Any]:
        """OpenAI Responses API call (for gpt-5 models)."""
        sys_txt, input_payload = _messages_to_responses_input(messages)
        out_budget = p.max_completion_tokens

        resp = await asyncio.wait_for(
            self._client.responses.create(
                model=p.model,
                instructions=sys_txt,
                input=input_payload,
                reasoning={
                    "effort": getattr(p, "reasoning_effort", "minimal"),
                },
                max_output_tokens=out_budget,
            ),
            timeout=p.timeout_s,
        )

        text = getattr(resp, "output_text", None) or ""
        if not text and hasattr(resp, "output") and resp.output:
            try:
                parts = []
                for item in resp.output:
                    if hasattr(item, "content"):
                        for c in item.content:
                            t = getattr(c, "text", None)
                            if t:
                                parts.append(t)
                text = "".join(parts)
            except Exception:
                text = ""

        return {
            "index": index,
            "response": text,
            "success": True,
            "error": None,
            "model": p.model,
        }


# ---------------------------------------------------------------------------
# Error classification (ported from gpt_batch_request_fast.py)
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> tuple:
    """
    Classify an exception into (backoff_bucket, retryable).

    backoff_bucket: "short" | "medium" | "long"
    retryable: bool
    """
    msg = (str(exc) or repr(exc)).lower()
    status_code = getattr(exc, "status_code", None)

    is_rate = (status_code == 429) or ("rate" in msg and "limit" in msg)
    is_server = status_code is not None and 500 <= int(status_code) < 600
    is_client_nr = status_code in (400, 401, 403, 404, 422)
    is_net_transient = any(s in msg for s in [
        "connection reset", "temporar", "unavailable",
        "server disconnected", "connection aborted",
        "remote protocol", "stream closed",
    ])

    if is_rate:
        return "long", True
    if is_server or is_net_transient:
        return "medium", True
    if is_client_nr:
        return "short", False
    return "short", True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _messages_to_responses_input(messages: List[Dict[str, Any]]):
    """Convert Chat-style messages to Responses API format."""
    sys_txt = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    )
    user = next(
        (m for m in messages if m.get("role") == "user"), {"content": ""}
    )
    ucont = user.get("content", "")

    parts = []
    if isinstance(ucont, list):
        for p in ucont:
            ptype = p.get("type")
            if ptype == "text":
                txt = p.get("text") or ""
                if txt:
                    parts.append({"type": "input_text", "text": txt})
            elif ptype == "image_url":
                iu = p.get("image_url")
                if isinstance(iu, dict):
                    iu = iu.get("url")
                if iu:
                    parts.append({"type": "input_image", "image_url": iu})
    else:
        parts.append({"type": "input_text", "text": str(ucont)})

    return sys_txt, [{"role": "user", "content": parts}]


def _run_async(coro):
    """Run an async coroutine from sync code, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
        # Already in an async context — run in a worker thread
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)
