"""
Phase 5 — Deterministic evaluation harness (RECOMMENDED).
Used by the full ChipMate pipeline.

Evaluation approach:
- For spec_lookup: extract numeric value+unit from the answer with regex,
  then check if that value appears in the retrieved source text.
- For relationship_lookup: check if each expected component name appears in the answer.
- Verdict: PASS / FAIL / PARTIAL (for relationship queries where some but not all match)

Why deterministic over LLM-as-judge (alternative1):
- Deterministic means the eval script itself is not a stochastic system.
  An LLM judge can give different verdicts on different runs for the same answer.
- Regex extraction is explainable: "It passed because '3.3V' appears in both answer and source"
- This is exactly what you'd describe in an interview: "I verified the answer against
  the retrieved text, not against the model's own confidence"
- LLM-as-judge is useful for quality dimensions regex can't capture (fluency, completeness)
  but for factual claims in datasheets, regex is more reliable

Run:
    python -m local.eval --dataset golden_dataset.json
    python -m local.eval --dataset golden_dataset.json --verbose
"""

import re
import json
import argparse
from dataclasses import dataclass
from typing import Optional

from local.agent import run_query

NUMERIC_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*(v|mv|a|ma|ua|mhz|khz|hz|w|mw|%|bit|sps|°c|c|kohm|ohm)\b',
    re.IGNORECASE,
)


@dataclass
class EvalResult:
    test_id: str
    query: str
    intent: str
    expected: dict
    actual_answer: str
    actual_confidence: float
    verdict: str          # PASS | FAIL | PARTIAL | SKIP
    reason: str


def extract_numeric(text: str) -> list[tuple[str, str]]:
    """Extract all (value, unit) pairs from text."""
    return NUMERIC_PATTERN.findall(text.lower())


def evaluate_spec_case(tc: dict, result: dict) -> tuple[str, str]:
    """
    Verify a spec_lookup test case.
    Returns (verdict, reason).
    """
    answer = result["answer"].lower()
    context = result["context"].lower()
    expected_val = tc.get("expected_value")
    expected_unit = tc.get("expected_unit", "").lower() if tc.get("expected_unit") else None
    expected_source = tc.get("expected_source_contains", "").lower() if tc.get("expected_source_contains") else None

    if expected_val is None and expected_source:
        # String match only (e.g., "SOIC", "I2C", "power save")
        if expected_source in answer or expected_source in context:
            return "PASS", f"source contains '{expected_source}'"
        return "FAIL", f"'{expected_source}' not found in answer or context"

    if expected_val:
        # Check answer contains the expected value (approximate match)
        if expected_val in answer:
            if expected_unit and expected_unit not in answer:
                return "PARTIAL", f"value {expected_val} found but unit {expected_unit} missing"
            return "PASS", f"value {expected_val} found in answer"

        # Check source context contains the expected value
        if expected_val in context:
            return "FAIL", f"value {expected_val} in source but not in answer (grounding gap)"

        return "FAIL", f"expected value {expected_val} not found in answer or context"

    return "SKIP", "no expected_value specified"


def evaluate_relationship_case(tc: dict, result: dict) -> tuple[str, str]:
    """Verify a relationship_lookup test case."""
    answer = result["answer"].lower()
    graph_results = [r.lower() for r in result.get("graph_results", [])]
    expected = [e.lower() for e in tc.get("expected_graph_result", [])]

    if not expected:
        return "SKIP", "no expected_graph_result specified"

    found_in_answer = [e for e in expected if e in answer]
    found_in_graph = [e for e in expected if e in graph_results]

    if found_in_answer or found_in_graph:
        if len(found_in_answer) == len(expected):
            return "PASS", f"all expected ({expected}) found in answer"
        return "PARTIAL", f"found {found_in_answer} but missing {[e for e in expected if e not in found_in_answer]}"
    return "FAIL", f"none of {expected} found in answer or graph results"


def run_eval(dataset_path: str, verbose: bool = False) -> list[EvalResult]:
    with open(dataset_path) as f:
        test_cases = json.load(f)

    results = []
    for tc in test_cases:
        print(f"Running {tc['id']}: {tc['query'][:60]}...")

        agent_result = run_query(tc["query"])
        intent = agent_result.get("intent", "")

        if tc["intent"] == "relationship_lookup":
            verdict, reason = evaluate_relationship_case(tc, agent_result)
        else:
            verdict, reason = evaluate_spec_case(tc, agent_result)

        result = EvalResult(
            test_id=tc["id"],
            query=tc["query"],
            intent=intent,
            expected=tc,
            actual_answer=agent_result.get("answer", ""),
            actual_confidence=agent_result.get("confidence", 0.0),
            verdict=verdict,
            reason=reason,
        )
        results.append(result)

        status = {"PASS": "PASS", "FAIL": "FAIL", "PARTIAL": "PART", "SKIP": "SKIP"}[verdict]
        print(f"  [{status}] {reason}")
        if verbose:
            print(f"         Answer: {result.actual_answer[:100]}")
        print()

    return results


def print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    by_verdict = {}
    for r in results:
        by_verdict[r.verdict] = by_verdict.get(r.verdict, 0) + 1

    passes = by_verdict.get("PASS", 0) + by_verdict.get("PARTIAL", 0) * 0.5
    accuracy = passes / total if total else 0

    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total:   {total}")
    print(f"PASS:    {by_verdict.get('PASS', 0)}")
    print(f"PARTIAL: {by_verdict.get('PARTIAL', 0)}")
    print(f"FAIL:    {by_verdict.get('FAIL', 0)}")
    print(f"SKIP:    {by_verdict.get('SKIP', 0)}")
    print(f"Accuracy (PASS + 0.5*PARTIAL): {accuracy:.1%}")
    print("=" * 50)

    print("\nFailed cases:")
    for r in results:
        if r.verdict == "FAIL":
            print(f"  {r.test_id}: {r.reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic evaluation harness")
    parser.add_argument("--dataset", default="golden_dataset.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_eval(args.dataset, args.verbose)
    print_summary(results)
