"""
Phase 8 — Observability with OpenTelemetry (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

OpenTelemetry (OTel) is the vendor-neutral tracing standard.
The same instrumentation code sends traces to Jaeger, Zipkin, Datadog, Honeycomb,
or AWS X-Ray just by swapping the exporter.

When you'd prefer this over Langfuse (tracing.py):
- You need traces in an existing observability stack (Datadog, Jaeger, etc.)
- You want standardized spans that non-LLM services can emit too
- You're building something that others will deploy in their own infra

Trade-offs vs Langfuse (tracing.py):
- OTel has no built-in LLM concepts (prompt, completion, token count, model)
- You add those as span attributes manually — more code, but more portable
- The Langfuse UI shows cost per trace; OTel-based UIs generally don't unless you add it

This file exports to console (easy to verify locally) and optionally to OTLP endpoint.
For AWS, you'd swap ConsoleSpanExporter for OTLPSpanExporter pointing at X-Ray OTLP.

Install: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc

Usage:
    from local.tracing_alternative1 import tracer
    with tracer.start_as_current_span("router_node") as span:
        span.set_attribute("intent", intent)
        ...
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
import os

# Resource identifies the service in the trace backend
resource = Resource.create({"service.name": "chipmate-local"})

# Provider holds all registered exporters and processors
provider = TracerProvider(resource=resource)

# BatchSpanProcessor sends spans asynchronously — better for production than SimpleSpanProcessor
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# To also send to an OTLP endpoint (Jaeger, X-Ray OTLP, etc.):
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
# provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317")))

trace.set_tracer_provider(provider)
tracer = trace.get_tracer("chipmate")


def wrap_node_with_span(node_name: str, fn, state: dict) -> dict:
    """
    Wrap a LangGraph node call in an OTel span.
    Usage in agent.py:
        result = wrap_node_with_span("router", router_node, state)
    """
    with tracer.start_as_current_span(node_name) as span:
        span.set_attribute("query", state.get("query", ""))
        span.set_attribute("intent.before", state.get("intent", ""))

        result = fn(state)

        # Update span with post-node state
        merged = {**state, **result}
        span.set_attribute("intent.after", merged.get("intent", ""))
        span.set_attribute("confidence", merged.get("confidence", 0.0))
        span.set_attribute("chunks.count", len(merged.get("retrieved_chunks", [])))

        return result


# Example showing how spans look in console output:
if __name__ == "__main__":
    print("OTel span demo — check console output for span JSON\n")

    with tracer.start_as_current_span("demo-query") as root:
        root.set_attribute("query", "What is the voltage of TPS62902?")

        with tracer.start_as_current_span("router_node") as router_span:
            router_span.set_attribute("intent", "spec_lookup")

        with tracer.start_as_current_span("retrieval_node") as ret_span:
            ret_span.set_attribute("chunks.retrieved", 5)
            ret_span.set_attribute("model", "nomic-embed-text")

        with tracer.start_as_current_span("analysis_node") as analysis_span:
            analysis_span.set_attribute("model", "mistral:7b")
            analysis_span.set_attribute("method", "primary")

        with tracer.start_as_current_span("grounding_node") as ground_span:
            ground_span.set_attribute("confidence", 0.85)
            ground_span.set_attribute("passed", True)
