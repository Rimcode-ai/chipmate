"""
AWS Phase 4 — LangGraph agent with Amazon Bedrock (Claude 3 Haiku).
Same graph structure as local/agent.py — only _call_llm() changes.

Interview talking point:
  "My orchestration layer is provider-agnostic. The LangGraph state machine,
   the router logic, the grounding check — none of that changes. Only the
   function that calls the LLM swaps from Ollama to Bedrock. That's the design
   I'd argue for in any production system."

Cost: Haiku is ~$0.25/1M input tokens. 10 test queries ≈ cents.
"""

import re
import json
import boto3
import psycopg
from typing import TypedDict
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

from aws.retrieve_aws import hybrid_search_aws
from aws.graph_neptune import get_client as get_neptune_client, find_replacements, find_pin_compatible
from aws.resilience_aws import call_bedrock_with_fallback

load_dotenv()

AURORA_URL = os.getenv("AURORA_URL", "")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
CONFIDENCE_THRESHOLD = 0.6

RELATIONSHIP_PATTERNS = re.compile(
    r"\b(replac|substitut|swap|equivalent|pin.compat|drop.in|alternative)\b",
    re.IGNORECASE,
)
SPEC_PATTERNS = re.compile(
    r"\b(voltage|current|power|watt|ampere|frequency|efficiency|temperature|"
    r"resolution|accuracy|gain|pin|package|spec|rating|range)\b",
    re.IGNORECASE,
)


class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: list
    graph_results: list
    context: str
    answer: str
    confidence: float
    grounding_passed: bool
    requires_human_review: bool
    human_decision: str


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
    conn = psycopg.connect(AURORA_URL)
    chunks = hybrid_search_aws(state["query"], top_k=5, conn=conn)
    conn.close()
    context = "\n\n".join(f"[{c.component} / {c.section}]\n{c.text}" for c in chunks)
    return {"retrieved_chunks": chunks, "context": context}


def graph_node(state: AgentState) -> dict:
    c = get_neptune_client()
    match = re.search(r'\b([A-Z]{2,}[\d]{3,}[A-Z0-9]*)\b', state["query"])
    graph_results = []
    context = ""
    if match:
        part = match.group(1)
        replacements = find_replacements(c, part)
        pin_compat = find_pin_compatible(c, part)
        graph_results = replacements + pin_compat
        lines = []
        if replacements:
            lines.append(f"{part} is replaced by: {', '.join(replacements)}")
        if pin_compat:
            lines.append(f"{part} is pin-compatible with: {', '.join(pin_compat)}")
        context = "\n".join(lines) or "No relationships found."
    c.close()
    return {"graph_results": graph_results, "context": context}


def analysis_node(state: AgentState) -> dict:
    answer, method = call_bedrock_with_fallback(state["query"], state["context"])
    print(f"[analysis] method={method}")
    return {"answer": answer}


def grounding_node(state: AgentState) -> dict:
    answer = state["answer"].lower()
    context = state["context"].lower()
    claims = re.findall(
        r'(\d+\.?\d*)\s*(v|mv|a|ma|ua|mhz|khz|hz|w|mw|%|bit|sps|°c|c)\b',
        answer,
    )
    if not claims:
        confidence = 0.7 if state["graph_results"] else 0.5
        grounding_passed = True
    else:
        verified = sum(
            1 for val, unit in claims
            if re.search(rf'{re.escape(val)}\s*{re.escape(unit)}', context, re.IGNORECASE)
        )
        confidence = verified / len(claims)
        grounding_passed = confidence >= CONFIDENCE_THRESHOLD

    return {
        "confidence": confidence,
        "grounding_passed": grounding_passed,
        "requires_human_review": not grounding_passed,
    }


def human_approval_node(state: AgentState) -> dict:
    print("\n--- HUMAN REVIEW REQUIRED ---")
    print(f"Query: {state['query']}")
    print(f"Confidence: {state['confidence']:.2f}")
    decision = input("Decision [approve/escalate]: ").strip().lower()
    return {"human_decision": decision if decision in ("approve", "escalate") else "escalate"}


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
        lambda s: "graph_search" if s["intent"] == "relationship_lookup" else "retrieval",
        {"retrieval": "retrieval", "graph_search": "graph_search"},
    )
    g.add_edge("retrieval",    "analysis")
    g.add_edge("graph_search", "analysis")
    g.add_edge("analysis",     "grounding")
    g.add_conditional_edges(
        "grounding",
        lambda s: "human_approval" if s["requires_human_review"] else "__end__",
        {"human_approval": "human_approval", "__end__": END},
    )
    g.add_edge("human_approval", END)
    return g.compile()


app = build_graph()


def run_query(query: str) -> dict:
    return app.invoke({
        "query": query, "intent": "", "retrieved_chunks": [],
        "graph_results": [], "context": "", "answer": "",
        "confidence": 0.0, "grounding_passed": False,
        "requires_human_review": False, "human_decision": "",
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    result = run_query(args.query)
    print(f"\nAnswer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
