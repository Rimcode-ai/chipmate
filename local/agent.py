"""
Phase 4 — Multi-agent orchestration with LangGraph StateGraph (RECOMMENDED).
Used by the full ChipMate pipeline.

Agent flow:
    router -> retrieval OR graph_search -> analysis -> grounding -> END or human_approval

Why LangGraph over a plain sequential pipeline (alternative1):
- Conditional routing (different paths based on intent) needs branching — a for-loop can't do that cleanly
- State is explicit and typed — every node sees the same dict, no hidden globals
- The compiled graph is pausable (for human-in-the-loop) via interrupt_before= parameter
- LangGraph produces a visual graph you can export (good for interviews)
- The provider-agnostic point: only analysis_node touches the LLM call function.
  Swapping Ollama for Bedrock only changes _call_llm() — nothing else.

Why the router is deterministic (not an LLM call):
- An LLM router adds ~500ms latency per query and can misclassify
- Regex patterns on electrical engineering terminology are reliable and auditable
- "I can explain exactly why this query was routed to the graph" is a strong interview answer

Run:
    python -m local.agent --query "What is the voltage of TPS62902?"
    python -m local.agent --query "What replaces TPS62902?"
    python -m local.agent --query "Does TPS62902 support power save mode?"
"""

import re
import argparse
from typing import TypedDict

import httpx
import psycopg
from dotenv import load_dotenv
import os

from local.retrieve import hybrid_search
from local.graph import get_driver, find_replacements, find_pin_compatible, find_all_related
from local.resilience import call_llm_with_fallback

load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://chipmate:chipmate@localhost:5432/chipmate")
CONFIDENCE_THRESHOLD = 0.6   # below this score, route to human_approval


# --- State definition ---
# TypedDict makes the state shape explicit; LangGraph passes it between nodes as a plain dict.

class AgentState(TypedDict):
    query: str
    intent: str              # spec_lookup | relationship_lookup | general
    retrieved_chunks: list   # list of Chunk namedtuples from retrieve.py
    graph_results: list      # list of strings (component names)
    context: str             # combined text passed to the LLM
    answer: str
    confidence: float        # 0.0–1.0
    grounding_passed: bool
    requires_human_review: bool
    human_decision: str      # approve | escalate | ""


# --- Routing patterns (deterministic, no LLM) ---
RELATIONSHIP_PATTERNS = re.compile(
    r"\b(replac|substitut|swap|equivalent|pin.compat|drop.in|alternative)\b",
    re.IGNORECASE,
)
SPEC_PATTERNS = re.compile(
    r"\b(voltage|current|power|watt|ampere|frequency|efficiency|temperature|"
    r"resolution|accuracy|gain|pin|package|datasheet|spec|rating|range)\b",
    re.IGNORECASE,
)


# --- Node functions ---
# Each node receives the full state dict and returns a PARTIAL dict (only updated keys).
# LangGraph merges the return value into the existing state.

def router_node(state: AgentState) -> dict:
    query = state["query"]
    if RELATIONSHIP_PATTERNS.search(query):
        intent = "relationship_lookup"
    elif SPEC_PATTERNS.search(query):
        intent = "spec_lookup"
    else:
        intent = "general"
    print(f"[router] intent={intent}")
    return {"intent": intent}


def retrieval_node(state: AgentState) -> dict:
    conn = psycopg.connect(POSTGRES_URL)
    chunks = hybrid_search(state["query"], top_k=5, conn=conn)
    conn.close()
    context = "\n\n".join(f"[{c.component} / {c.section}]\n{c.text}" for c in chunks)
    print(f"[retrieval] {len(chunks)} chunks retrieved")
    return {"retrieved_chunks": chunks, "context": context}


def graph_node(state: AgentState) -> dict:
    driver = get_driver()
    # Extract component name from query: look for known patterns like TPS62902
    component_match = re.search(r'\b([A-Z]{2,}[\d]{3,}[A-Z0-9]*)\b', state["query"])
    graph_results = []
    context = ""

    if component_match:
        part = component_match.group(1)
        replacements = find_replacements(driver, part)
        pin_compat = find_pin_compatible(driver, part)
        related = find_all_related(driver, part)
        graph_results = replacements + pin_compat
        lines = []
        if replacements:
            lines.append(f"{part} is replaced by: {', '.join(replacements)}")
        if pin_compat:
            lines.append(f"{part} is pin-compatible with: {', '.join(pin_compat)}")
        if related:
            lines.append(f"Related components: {related}")
        context = "\n".join(lines)
    else:
        context = "No specific component part number found in the query."

    driver.close()
    print(f"[graph] results={graph_results}")
    return {"graph_results": graph_results, "context": context}


def analysis_node(state: AgentState) -> dict:
    answer, method = call_llm_with_fallback(
        prompt=state["query"],
        context=state["context"],
    )
    print(f"[analysis] method={method} answer_len={len(answer)}")
    return {"answer": answer}


def grounding_node(state: AgentState) -> dict:
    """
    Deterministic grounding check:
    1. Extract any numeric value+unit from the answer.
    2. Check if that value appears in the source context.
    3. Set confidence based on whether the check passes.

    This is the key insight: we don't trust the LLM's claim — we verify it against
    the retrieved text. If the LLM says 3.3V but the context says 5V, grounding fails.
    """
    answer = state["answer"].lower()
    context = state["context"].lower()

    # Extract numeric claims: "3.3 v", "2a", "2.2mhz", "60ua"
    numeric_claims = re.findall(r'(\d+\.?\d*)\s*(v|mv|a|ma|ua|mhz|khz|hz|w|mw|%|bit|sps|°c|c)\b', answer)

    if not numeric_claims:
        # No numeric claims to verify — moderate confidence for relationship/general queries
        confidence = 0.7 if state["graph_results"] else 0.5
        grounding_passed = True
    else:
        verified = 0
        for value, unit in numeric_claims:
            # Check if "value unit" appears in the source context
            pattern = re.compile(rf'{re.escape(value)}\s*{re.escape(unit)}', re.IGNORECASE)
            if pattern.search(context):
                verified += 1
        confidence = verified / len(numeric_claims)
        grounding_passed = confidence >= CONFIDENCE_THRESHOLD

    requires_human_review = not grounding_passed
    print(f"[grounding] confidence={confidence:.2f} grounding_passed={grounding_passed}")
    return {
        "confidence": confidence,
        "grounding_passed": grounding_passed,
        "requires_human_review": requires_human_review,
    }


def human_approval_node(state: AgentState) -> dict:
    """
    Human-in-the-loop node. Prints the low-confidence answer and waits for approval.
    In production (async API), this would write to a queue and wait for a webhook.
    Here it reads stdin directly for demo purposes.
    """
    print("\n--- HUMAN REVIEW REQUIRED ---")
    print(f"Query: {state['query']}")
    print(f"Answer: {state['answer']}")
    print(f"Confidence: {state['confidence']:.2f}")
    print("\nOptions: [approve] Accept this answer  |  [escalate] Mark as unresolved")
    decision = input("Decision: ").strip().lower()
    if decision not in ("approve", "escalate"):
        decision = "escalate"
    print(f"[human_approval] decision={decision}")
    return {"human_decision": decision}


# --- Conditional routing functions ---

def route_after_router(state: AgentState) -> str:
    return "graph_search" if state["intent"] == "relationship_lookup" else "retrieval"


def route_after_grounding(state: AgentState) -> str:
    return "human_approval" if state["requires_human_review"] else "__end__"


# --- Build the graph ---

from langgraph.graph import StateGraph, END

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("router",         router_node)
    g.add_node("retrieval",      retrieval_node)
    g.add_node("graph_search",   graph_node)
    g.add_node("analysis",       analysis_node)
    g.add_node("grounding",      grounding_node)
    g.add_node("human_approval", human_approval_node)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {"retrieval": "retrieval", "graph_search": "graph_search"},
    )
    g.add_edge("retrieval",    "analysis")
    g.add_edge("graph_search", "analysis")
    g.add_edge("analysis",     "grounding")
    g.add_conditional_edges(
        "grounding",
        route_after_grounding,
        {"human_approval": "human_approval", "__end__": END},
    )
    g.add_edge("human_approval", END)

    return g.compile()


app = build_graph()


def run_query(query: str) -> dict:
    initial_state: AgentState = {
        "query": query,
        "intent": "",
        "retrieved_chunks": [],
        "graph_results": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "grounding_passed": False,
        "requires_human_review": False,
        "human_decision": "",
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChipMate agent (LangGraph)")
    parser.add_argument("--query", required=True)
    args = parser.parse_args()

    result = run_query(args.query)
    print("\n=== RESULT ===")
    print(f"Answer:     {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Intent:     {result['intent']}")
    if result["human_decision"]:
        print(f"Human:      {result['human_decision']}")
