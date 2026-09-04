"""
Phase 7 — Circuit breaker as a manual state machine (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

This re-implements the circuit breaker from scratch so you can see exactly
what pybreaker is doing internally. No external dependency.

Reading this alongside resilience.py makes the pybreaker decorator magic visible:
  CLOSED: _call_ollama() in resilience.py executes normally
  OPEN:   CircuitBreakerError is raised BEFORE _call_ollama() is even called
  HALF_OPEN: one test call is allowed through

States:
    CLOSED    -- all calls pass through; failures increment counter
    OPEN      -- all calls rejected immediately (fail fast); check reset timer
    HALF_OPEN -- one call passes through as a test; success -> CLOSED, fail -> OPEN

When you'd prefer this over pybreaker (resilience.py):
- Educational: you want to understand the pattern deeply
- You need custom behavior on state transitions (e.g., emit a metric, alert)
- The framework adds no real value once you understand the pattern

Run:
    python -m local.resilience_alternative1
    (runs a self-contained demo that triggers all three states)
"""

import time
import httpx
from enum import Enum
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")


class CircuitState(Enum):
    CLOSED    = "CLOSED"
    OPEN      = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ManualCircuitBreaker:
    def __init__(self, fail_max: int = 3, reset_timeout: float = 30.0):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> CircuitState:
        # OPEN -> HALF_OPEN transition: check if reset_timeout has elapsed
        if (
            self._state == CircuitState.OPEN
            and time.time() - self._last_failure_time >= self.reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            print(f"[CB] OPEN -> HALF_OPEN (testing recovery after {self.reset_timeout}s)")
        return self._state

    def call(self, fn, *args, **kwargs):
        current = self.state

        if current == CircuitState.OPEN:
            raise RuntimeError(
                f"[CB] Circuit is OPEN — rejecting call immediately "
                f"(will retry in {self.reset_timeout - (time.time() - self._last_failure_time):.1f}s)"
            )

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self):
        if self._state in (CircuitState.HALF_OPEN, CircuitState.CLOSED):
            prev = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            if prev == CircuitState.HALF_OPEN:
                print("[CB] HALF_OPEN -> CLOSED (recovered)")

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.fail_max:
            prev = self._state
            self._state = CircuitState.OPEN
            print(f"[CB] {prev.value} -> OPEN ({self._failure_count}/{self.fail_max} failures)")


manual_breaker = ManualCircuitBreaker(fail_max=3, reset_timeout=30.0)


def _call_ollama(prompt: str, context: str) -> str:
    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": f"{context}\n\n{prompt}", "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def call_llm_with_fallback(prompt: str, context: str) -> tuple[str, str]:
    try:
        answer = manual_breaker.call(_call_ollama, prompt, context)
        return answer, "primary"
    except RuntimeError as e:
        if "Circuit is OPEN" in str(e):
            return f"[Circuit OPEN] {context[:200]}", "fallback"
        raise
    except Exception:
        return f"[Error] {context[:200]}", "fallback"


if __name__ == "__main__":
    print("Demo: manually triggering all three circuit breaker states\n")
    print("Stop Ollama before running: pkill ollama\n")

    for i in range(1, 6):
        print(f"--- Call {i} (state={manual_breaker.state.value}) ---")
        answer, method = call_llm_with_fallback("What is voltage?", "Test context")
        print(f"  method={method}  answer={answer[:60]}\n")
        time.sleep(0.5)
