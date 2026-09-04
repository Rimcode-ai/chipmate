"""
Phase 6 — FastAPI async service with Redis rate limiting and semantic caching (RECOMMENDED).
Used by the full ChipMate pipeline.

Endpoints:
    POST /query   — submit a query, get an answer
    GET  /health  — service health + circuit breaker state

Rate limiting: sliding window counter in Redis (10 req/min per IP).
Semantic caching: MD5(query.lower()) -> cached answer (TTL 5 min).

Why Redis for both rate limiting and caching:
- Redis sorted sets (ZADD/ZRANGEBYSCORE) implement sliding window without a background job
- Redis is already in docker-compose.yml for the local stack
- A single Redis client handles both concerns; no extra dependency

Why sliding window over fixed window:
- Fixed window: 10 requests in the last 60 seconds resets at the :00 mark.
  A client can make 10 requests at :59 and 10 more at :01 — 20 in 2 seconds.
- Sliding window: the 60-second window moves with each request, so the burst is capped.

Why MD5 cache key over embedding similarity:
- Embedding similarity requires an embedding call to look up the cache, adding 50-200ms
- MD5 on normalized query text is free and works for exact/near-exact repeats
- For a demo/learning project, exact-match caching is sufficient and more explainable

Run:
    uvicorn local.main:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
         -d '{"query": "What is the voltage of TPS62902?"}'
"""

import hashlib
import json
import time
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
import os

from local.agent import run_query
from local.resilience import get_breaker_state

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

redis_client: aioredis.Redis = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    yield
    await redis_client.aclose()


app = FastAPI(title="ChipMate API", version="1.0.0", lifespan=lifespan)


# --- Request/response models ---

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


# --- Rate limiting ---

async def check_rate_limit(ip: str) -> None:
    """
    Sliding window rate limiter using Redis sorted set.
    Key: ratelimit:<ip>
    Members: request timestamps (stored as both score and value for uniqueness)
    Algorithm:
      1. Remove entries older than window_start
      2. Add current timestamp
      3. Count entries in window
      4. If count > limit, raise 429
    """
    key = f"ratelimit:{ip}"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, 0, window_start)          # remove old entries
        pipe.zadd(key, {f"{now}": now})                       # add current request
        pipe.zcard(key)                                        # count in window
        pipe.expire(key, RATE_LIMIT_WINDOW_SECONDS)           # auto-expire key
        results = await pipe.execute()

    request_count = results[2]
    if request_count > RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SECONDS}s",
        )


# --- Semantic caching ---

def cache_key(query: str) -> str:
    return f"cache:{hashlib.md5(query.lower().encode()).hexdigest()}"


async def get_cached_answer(query: str) -> dict | None:
    val = await redis_client.get(cache_key(query))
    return json.loads(val) if val else None


async def set_cached_answer(query: str, result: dict) -> None:
    await redis_client.setex(cache_key(query), CACHE_TTL_SECONDS, json.dumps(result))


# --- Endpoints ---

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: Request, body: QueryRequest):
    # 1. Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    await check_rate_limit(client_ip)

    # 2. Cache check
    cached = await get_cached_answer(body.query)
    if cached:
        return QueryResponse(**cached, cached=True, breaker_state=get_breaker_state())

    # 3. Run agent (blocking call — wrap in thread pool since run_query is synchronous)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_query, body.query)

    # 4. Build response and cache it
    response_data = {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "intent": result.get("intent", ""),
    }
    await set_cached_answer(body.query, response_data)

    return QueryResponse(
        **response_data,
        cached=False,
        breaker_state=get_breaker_state(),
    )


@app.get("/health")
async def health_endpoint():
    redis_ok = False
    try:
        await redis_client.ping()
        redis_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "circuit_breaker": get_breaker_state(),
    }
