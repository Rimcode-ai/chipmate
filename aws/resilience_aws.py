"""
AWS Phase 7 — Circuit breaker around Bedrock calls.
Same pybreaker pattern as local/resilience.py — now wrapping boto3 instead of httpx.

Bedrock has real rate limits (requests per minute per model per account).
You can actually trigger ThrottlingException under load — the breaker catches it.

Interview talking point:
  "The circuit breaker pattern is identical to my local version — only the
   wrapped function changes. On AWS I catch botocore ClientError with
   ThrottlingException code, which is the real Bedrock rate limit response."
"""

import json
import boto3
from botocore.exceptions import ClientError
from pybreaker import CircuitBreaker, CircuitBreakerError
from dotenv import load_dotenv
import os

load_dotenv()

BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
# Use Haiku for all dev iterations — cheapest Claude model
PRIMARY_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
FALLBACK_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # same model, different region in real scenario

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

bedrock_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)


def _invoke_claude(model_id: str, prompt: str, context: str) -> str:
    body = {
        "anthropic_version": "bedrock-2023-06-01",
        "max_tokens": 1024,
        "system": (
            "You are an electrical engineering assistant. "
            "Answer only from the provided context."
        ),
        "messages": [{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {prompt}",
        }],
    }
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


@bedrock_breaker
def _call_bedrock_primary(prompt: str, context: str) -> str:
    return _invoke_claude(PRIMARY_MODEL, prompt, context)


def call_bedrock_with_fallback(prompt: str, context: str) -> tuple[str, str]:
    try:
        return _call_bedrock_primary(prompt, context), "primary"
    except CircuitBreakerError:
        return f"[Circuit breaker open] {context[:200]}", "fallback"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ThrottlingException":
            # Bedrock throttle — breaker already counted this failure
            return f"[Bedrock throttled] {context[:200]}", "fallback"
        raise
