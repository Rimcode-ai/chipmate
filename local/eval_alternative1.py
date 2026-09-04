"""
Phase 5 — LLM-as-judge evaluation (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

Strategy: send the question, expected answer, and actual answer to the LLM
and ask it to score correctness on a 1-5 scale with a justification.

When you'd prefer this over regex (eval.py):
- Evaluating fluency, completeness, or explanation quality (not just factual correctness)
- The answers involve complex reasoning rather than extractable numeric values
- You're evaluating a generative task where there is no single correct string

Trade-offs vs deterministic (eval.py):
- Non-deterministic: same input can produce different scores on different runs
- Adds LLM cost to the eval pipeline (eval should be cheap)
- The judge can be wrong: LLMs exhibit "sycophancy" toward longer/confident-sounding answers
- Slower: every test case requires an LLM call
- Use as a complement to deterministic eval, not a replacement

When Amazon asks about eval in interviews: mention BOTH approaches and why you use
deterministic checks as the primary gate and LLM-as-judge only for qualitative dimensions.

Run:
    python -m local.eval_alternative1 --dataset golden_dataset.json
"""

import json
import argparse
import httpx
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

JUDGE_PROMPT = """You are an expert electrical engineer evaluating an AI assistant's answer.

Question: {question}

Expected answer guidance: {expected_notes}

Actual answer: {actual_answer}

Score the answer on a scale of 1-5:
5 = Completely correct, all values and facts accurate
4 = Mostly correct, minor imprecision
3 = Partially correct, some key facts missing or wrong
2 = Mostly incorrect but shows some relevant knowledge
1 = Completely wrong or refused to answer

Respond with ONLY:
SCORE: <1-5>
REASON: <one sentence>
"""


@dataclass
class LLMJudgeResult:
    test_id: str
    query: str
    actual_answer: str
    score: int
    reason: str
    verdict: str   # PASS (>=4), PARTIAL (3), FAIL (<=2)


def call_judge(question: str, expected_notes: str, actual_answer: str) -> tuple[int, str]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_notes=expected_notes,
        actual_answer=actual_answer,
    )
    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    text = response.json()["response"].strip()

    # Parse SCORE and REASON from response
    score = 3  # default if parsing fails
    reason = text
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                pass
        if line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return score, reason


def run_eval(dataset_path: str) -> list[LLMJudgeResult]:
    from local.agent import run_query

    with open(dataset_path) as f:
        test_cases = json.load(f)

    results = []
    for tc in test_cases:
        print(f"Judging {tc['id']}: {tc['query'][:60]}...")
        agent_result = run_query(tc["query"])
        actual_answer = agent_result.get("answer", "")

        expected_notes = tc.get("notes", "")
        if tc.get("expected_value"):
            expected_notes += f" Expected value: {tc['expected_value']} {tc.get('expected_unit','')}"

        score, reason = call_judge(tc["query"], expected_notes, actual_answer)
        verdict = "PASS" if score >= 4 else ("PARTIAL" if score == 3 else "FAIL")

        result = LLMJudgeResult(
            test_id=tc["id"],
            query=tc["query"],
            actual_answer=actual_answer,
            score=score,
            reason=reason,
            verdict=verdict,
        )
        results.append(result)
        print(f"  [{verdict}] score={score}/5  {reason}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge eval")
    parser.add_argument("--dataset", default="golden_dataset.json")
    args = parser.parse_args()

    results = run_eval(args.dataset)
    passed = sum(1 for r in results if r.verdict == "PASS")
    partial = sum(1 for r in results if r.verdict == "PARTIAL")
    total = len(results)
    print(f"\nLLM Judge: {passed}/{total} PASS, {partial}/{total} PARTIAL")
    print("(Compare with deterministic eval.py — differences reveal LLM judge bias)")
