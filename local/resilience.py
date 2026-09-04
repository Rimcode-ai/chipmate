"""
Phase 7 — Circuit breaker around the LLM call using pybreaker (RECOMMENDED).
Used by the full ChipMate pipeline (imported by agent.py).

Circuit breaker pattern:
  CLOSED   — normal operation, calls pass through
  OPEN     — calls are rejected immediately (fail fast), no wait for timeout
  HALF_OPEN — one test call allowed; if it succeeds -> CLOSED, if not -> OPEN again

States transition:
  CLOSED -> OPEN:      after fail_max consecutive failures
  OPEN -> HALF_OPEN:   after reset_timeout seconds
  HALF_OPEN -> CLOSED: if the test call succeeds
  HALF_OPEN -> OPEN:   if the test call fails

Why pybreaker over manual state machine (alternative1):
- pybreaker is battle-tested and handles threading edge cases
- The @breaker decorator syntax is minimal overhead on existing functions
- pybreaker fires listeners on state changes so you can log/alert when breaker opens

Why the fallback matters:
- Without a fallback, a breaker open circuit returns an exception to the user
- Our fallback returns the retrieved context directly — not as good as an LLM answer,
  but informative and always available
- In a production system the fallback might call a secondary model or a cached answer

Demo: kill Ollama (pkill ollama), run 4 calls, watch the breaker open on call 3.
    from local.resilience import call_llm_with_fallback
    for i in range(4):
        answer, method = call_llm_with_fallback("What is voltage?", "context text")
        print(i, method, answer[:60])
"""

import httpx
from pybreaker import CircuitBreaker, CircuitBreakerError
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

# fail_max=3: open after 3 consecutive failures
# reset_timeout=30: try again after 30 seconds
ollama_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)


@ollama_breaker
def _call_ollama(prompt: str, context: str) -> str:
    """Primary LLM call. Decorated with the breaker — any exception increments the failure count."""
    system_prompt = (
        "You are an electrical engineering assistant. "
        "Answer only from the provided context. "
        "If the context does not contain the answer, say so explicitly."
    )
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "system": system_prompt,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def _fallback_response(prompt: str, context: str) -> str:
    """
    Fallback: no LLM call. Return the raw context with a disclaimer.
    This is always available even when Ollama is down.
    In a real system you might call a secondary model or return a cached answer.
    """
    context_preview = context[:400].strip() if context else "(no context available)"
    return (
        f"[Primary LLM unavailable — circuit breaker open]\n"
        f"Retrieved context for your query:\n{context_preview}"
    )


def call_llm_with_fallback(prompt: str, context: str) -> tuple[str, str]:
    """
    Returns (answer_text, method) where method is 'primary' or 'fallback'.
    Callers can use the method field to tag responses in traces/logs.
    """
    try:
        answer = _call_ollama(prompt, context)
        return answer, "primary"
    except CircuitBreakerError:
        # Breaker is OPEN — skip the call entirely, use fallback immediately
        return _fallback_response(prompt, context), "fallback"
    except Exception as exc:
        # Other failures (network error, timeout) — pybreaker already counted this
        # as a failure; if we've hit fail_max the next call will hit CircuitBreakerError
        return _fallback_response(prompt, context), f"fallback (error: {type(exc).__name__})"


def get_breaker_state() -> str:
    """Utility to inspect current breaker state. Useful in /health endpoint."""
    return ollama_breaker.current_state
