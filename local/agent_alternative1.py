"""
Phase 4 — Multi-agent orchestration as a plain sequential pipeline (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

This implements the exact same logic as agent.py but using plain Python
functions — no LangGraph, no framework.

When you'd prefer this over agent.py:
- You want to understand what LangGraph is actually doing under the hood
- Simpler debugging: just add print() anywhere, no framework wrapping
- Fewer dependencies

Trade-offs vs LangGraph (agent.py):
- Conditional branching becomes if/else — readable but not visualizable
- No built-in interrupt/resume for human-in-the-loop (you'd need to persist state yourself)
- No automatic state checkpointing — if a node crashes mid-flow, you restart from scratch
- Harder to extend: adding a new node means editing the run_pipeline() function
  vs. adding g.add_node() and wiring edges in LangGraph

Reading both side-by-side shows exactly what LangGraph automates.

Run:
    python -m local.agent_alternative1 --query "What is the voltage of TPS62902?"
"""

import re
import argparse
from dataclasses import dataclass, field

import psycopg
from dotenv import load_dotenv
import os

from local.retrieve import hybrid_search
from local.graph import get_driver, find_replacements, find_pin_compatible
from local.resilience import call_llm_with_fallback

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://chipmate:chipmate@localhost:5432/chipmate")
CONFIDENCE_THRESHOLD = 0.6

RELATIONSHIP_PATTERNS = re.compile(
    r"\b(replac|substitut|swap|equivalent|pin.compat|drop.in|alternative)\b",
    re.IGNORECASE,
)
SPEC_PATTERNS = re.compile(
    r"\b(voltage|current|power|watt|ampere|frequency|efficiency|temperature|"
    r"resolution|accuracy|gain|pin|package|datasheet|spec|rating|range)\b",
    re.IGNORECASE,
)


@dataclass
class PipelineState:
    """Plain dataclass instead of TypedDict. Mutable, easier to debug with print()."""
    query: str
    intent: str = ""
    retrieved_chunks: list = field(default_factory=list)
    graph_results: list = field(default_factory=list)
    context: str = ""
    answer: str = ""
    confidence: float = 0.0
    grounding_passed: bool = False
    requires_human_review: bool = False
    human_decision: str = ""


def router_step(state: PipelineState) -> None:
    if RELATIONSHIP_PATTERNS.search(state.query):
        state.intent = "relationship_lookup"
    elif SPEC_PATTERNS.search(state.query):
        state.intent = "spec_lookup"
    else:
        state.intent = "general"
    print(f"[router] intent={state.intent}")


def retrieval_step(state: PipelineState) -> None:
    conn = psycopg.connect(POSTGRES_URL)
    chunks = hybrid_search(state.query, top_k=5, conn=conn)
    conn.close()
    state.retrieved_chunks = chunks
    state.context = "\n\n".join(f"[{c.component} / {c.section}]\n{c.text}" for c in chunks)
    print(f"[retrieval] {len(chunks)} chunks")


def graph_step(state: PipelineState) -> None:
    driver = get_driver()
    match = re.search(r'\b([A-Z]{2,}[\d]{3,}[A-Z0-9]*)\b', state.query)
    if match:
        part = match.group(1)
        replacements = find_replacements(driver, part)
        pin_compat = find_pin_compatible(driver, part)
        state.graph_results = replacements + pin_compat
        lines = []
        if replacements:
            lines.append(f"{part} is replaced by: {', '.join(replacements)}")
        if pin_compat:
            lines.append(f"{part} is pin-compatible with: {', '.join(pin_compat)}")
        state.context = "\n".join(lines) if lines else "No relationships found."
    else:
        state.context = "No component part number found."
    driver.close()
    print(f"[graph] results={state.graph_results}")


def analysis_step(state: PipelineState) -> None:
    answer, method = call_llm_with_fallback(state.query, state.context)
    state.answer = answer
    print(f"[analysis] method={method}")


def grounding_step(state: PipelineState) -> None:
    answer = state.answer.lower()
    context = state.context.lower()
    claims = re.findall(r'(\d+\.?\d*)\s*(v|mv|a|ma|ua|mhz|khz|hz|w|mw|%|bit|sps|°c|c)\b', answer)

    if not claims:
        state.confidence = 0.7 if state.graph_results else 0.5
        state.grounding_passed = True
    else:
        verified = sum(
            1 for val, unit in claims
            if re.search(rf'{re.escape(val)}\s*{re.escape(unit)}', context, re.IGNORECASE)
        )
        state.confidence = verified / len(claims)
        state.grounding_passed = state.confidence >= CONFIDENCE_THRESHOLD

    state.requires_human_review = not state.grounding_passed
    print(f"[grounding] confidence={state.confidence:.2f}")


def human_approval_step(state: PipelineState) -> None:
    print("\n--- HUMAN REVIEW REQUIRED ---")
    print(f"Query:      {state.query}")
    print(f"Answer:     {state.answer}")
    print(f"Confidence: {state.confidence:.2f}")
    decision = input("Decision [approve/escalate]: ").strip().lower()
    state.human_decision = decision if decision in ("approve", "escalate") else "escalate"
    print(f"[human_approval] decision={state.human_decision}")


def run_pipeline(query: str) -> PipelineState:
    state = PipelineState(query=query)

    # Step 1: route
    router_step(state)

    # Step 2: retrieve (branch based on intent)
    if state.intent == "relationship_lookup":
        graph_step(state)
    else:
        retrieval_step(state)

    # Step 3: analyse
    analysis_step(state)

    # Step 4: ground
    grounding_step(state)

    # Step 5: human review if needed
    if state.requires_human_review:
        human_approval_step(state)

    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChipMate agent (plain pipeline)")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    result = run_pipeline(args.query)
    print("\n=== RESULT ===")
    print(f"Answer:     {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Intent:     {result.intent}")
