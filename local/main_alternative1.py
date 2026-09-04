"""
Phase 6 — FastAPI async service with in-memory rate limiting (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

Same API surface as main.py but uses a Python dict for rate limiting and caching.
No Redis dependency — useful for understanding the algorithm before adding Redis.

When you'd prefer this over main.py:
- Local testing without Docker (no Redis required)
- Understanding the sliding window algorithm in pure Python before Redis
- Single-process deployments where memory is shared

Trade-offs vs Redis (main.py):
- NOT suitable for multi-process or multi-machine deployments:
  each worker process has its own in-memory counter, so a client can
  make N requests to each of K workers for N*K total (bypasses the limit)
- Data is lost on restart (rate limit counters, cache entries)
- In-memory cache grows unboundedly without LRU eviction (simple version here)

Run:
    uvicorn local.main_alternative1:app --reload --port 8001
"""

import hashlib
import json
import time
import asyncio
from collections import defaultdict, deque
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from local.agent import run_query
from local.resilience import get_breaker_state

load_dotenv()

RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
CACHE_TTL_SECONDS = 300

# In-memory stores (not thread-safe at extreme concurrency, fine for demo)
_rate_counters: dict[str, deque] = defaultdict(deque)
_cache: dict[str, tuple[dict, float]] = {}   # key -> (value, expiry_timestamp)

app = FastAPI(title="ChipMate API (in-memory)", version="1.0.0-alt")


class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    intent: str
    cached: bool
    breaker_state: str


def check_rate_limit_memory(ip: str) -> None:
    """
    Sliding window using a deque of timestamps.
    deque stores request times; we pop from the left anything older than the window.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    counter = _rate_counters[ip]

    # Remove expired entries from the left side of the deque
    while counter and counter[0] < window_start:
        counter.popleft()

    if len(counter) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: {RATE_LIMIT_REQUESTS} req/{RATE_LIMIT_WINDOW_SECONDS}s",
        )

    counter.append(now)


def cache_key(query: str) -> str:
    return hashlib.md5(query.lower().encode()).hexdigest()


def get_cached(query: str) -> Optional[dict]:
    key = cache_key(query)
    if key in _cache:
        value, expiry = _cache[key]
        if time.time() < expiry:
            return value
        del _cache[key]
    return None


def set_cached(query: str, result: dict) -> None:
    _cache[cache_key(query)] = (result, time.time() + CACHE_TTL_SECONDS)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    client_ip = request.client.host if request.client else "unknown"
    check_rate_limit_memory(client_ip)

    cached = get_cached(body.query)
    if cached:
        return QueryResponse(**cached, cached=True, breaker_state=get_breaker_state())

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_query, body.query)

    response_data = {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "intent": result.get("intent", ""),
    }
    set_cached(body.query, response_data)

    return QueryResponse(**response_data, cached=False, breaker_state=get_breaker_state())


@app.get("/health")
async def health_endpoint():
    return {
        "status": "ok",
        "redis": "not used (in-memory mode)",
        "circuit_breaker": get_breaker_state(),
        "cache_entries": len(_cache),
    }
