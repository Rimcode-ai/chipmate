"""
Phase 8 — Observability with Langfuse callback handler (RECOMMENDED).
Used by the full ChipMate pipeline.

Langfuse captures:
- One Trace per query (router + all downstream spans)
- One Span per LangGraph node (router, retrieval, analysis, grounding)
- Custom metadata on each span: intent, chunk_count, confidence, method (primary/fallback)

Why Langfuse over OpenTelemetry (alternative1):
- Langfuse is purpose-built for LLM observability: it understands prompts, completions,
  token counts, and model names natively
- The Langfuse UI shows cost per trace, which is directly useful for tuning
- The LangChain/LangGraph callback protocol is one import and one kwarg
- OpenTelemetry is the right choice if you need traces to flow into existing infra
  (Datadog, Jaeger, Honeycomb) — it's the portable standard

How this works with LangGraph:
  LangGraph nodes call LangChain runnables, which fire callbacks.
  Langfuse's CallbackHandler listens to those callbacks and creates spans.
  You pass the handler at invoke time: app.invoke(state, config={"callbacks": [handler]})

Env vars needed in .env:
    LANGFUSE_PUBLIC_KEY=pk-...
    LANGFUSE_SECRET_KEY=sk-...
    LANGFUSE_HOST=http://localhost:3000

Usage:
    from local.tracing import make_langfuse_handler
    handler = make_langfuse_handler(trace_name="query-trace", user_id="test")
    result = app.invoke(state, config={"callbacks": [handler]})
    handler.flush()  # ensure spans are sent before process exits
"""

import os
from dotenv import load_dotenv

load_dotenv()

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")


def make_langfuse_handler(trace_name: str = "chipmate-query", user_id: str = "local"):
    """
    Create a Langfuse CallbackHandler for a single trace.
    Call once per query, pass to app.invoke(config={"callbacks": [handler]}).
    """
    from langfuse.callback import CallbackHandler

    handler = CallbackHandler(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
        host=LANGFUSE_HOST,
        trace_name=trace_name,
        user_id=user_id,
    )
    return handler


def trace_query(query: str, app, initial_state: dict) -> dict:
    """
    Run the LangGraph app with Langfuse tracing attached.
    Returns the result dict; trace appears in Langfuse UI at LANGFUSE_HOST.
    """
    if not LANGFUSE_PUBLIC_KEY:
        print("[tracing] LANGFUSE_PUBLIC_KEY not set — running without tracing")
        return app.invoke(initial_state)

    handler = make_langfuse_handler(trace_name=f"query: {query[:50]}")
    result = app.invoke(initial_state, config={"callbacks": [handler]})
    handler.flush()

    trace_url = f"{LANGFUSE_HOST}/traces/{handler.get_trace_id()}"
    print(f"[tracing] Trace: {trace_url}")
    return result
