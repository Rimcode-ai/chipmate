"""
AWS Phase 6 — Lambda handler for the API Gateway endpoint.
Mirrors local/main.py but packaged as a Lambda function.

API Gateway -> Lambda -> LangGraph (Bedrock + Aurora) -> response

Interview talking point:
  "Lambda auto-scales to handle concurrent requests. Each invocation is stateless.
   I'm not running the FastAPI server inside Lambda — I use a plain handler function.
   API Gateway handles HTTP routing; Lambda handles the logic."

Rate limiting: done at API Gateway usage plans (not reimplemented in Lambda).
Caching: DynamoDB with TTL instead of Redis.

Deploy: See PLAN.md AWS Phase 6.
"""

import json
import boto3
import hashlib
import time
from dotenv import load_dotenv
import os

load_dotenv()

DYNAMODB_TABLE = os.getenv("DYNAMODB_CACHE_TABLE", "chipmate-cache")
CACHE_TTL_SECONDS = 300

dynamodb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))


def get_cached(query: str) -> dict | None:
    """Check DynamoDB for a cached answer."""
    table = dynamodb.Table(DYNAMODB_TABLE)
    key = hashlib.md5(query.lower().encode()).hexdigest()
    try:
        response = table.get_item(Key={"query_hash": key})
        item = response.get("Item")
        if item and item.get("ttl", 0) > time.time():
            return json.loads(item["result"])
    except Exception:
        pass
    return None


def set_cached(query: str, result: dict) -> None:
    """Write answer to DynamoDB with TTL."""
    table = dynamodb.Table(DYNAMODB_TABLE)
    key = hashlib.md5(query.lower().encode()).hexdigest()
    try:
        table.put_item(Item={
            "query_hash": key,
            "result": json.dumps(result),
            "ttl": int(time.time()) + CACHE_TTL_SECONDS,
        })
    except Exception as e:
        print(f"[cache] write failed: {e}")


def lambda_handler(event, context):
    """
    API Gateway HTTP API event structure:
    {
      "body": "{\"query\": \"What is the voltage of TPS62902?\"}",
      "requestContext": {"http": {"sourceIp": "1.2.3.4"}}
    }
    """
    # Parse body
    body = json.loads(event.get("body") or "{}")
    query = (body.get("query") or "").strip()

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "query cannot be empty"}),
        }

    # Cache check (DynamoDB)
    cached = get_cached(query)
    if cached:
        return {
            "statusCode": 200,
            "body": json.dumps({**cached, "cached": True}),
            "headers": {"Content-Type": "application/json"},
        }

    # Import agent here (lazy import keeps Lambda cold start fast if cache hits)
    from aws.agent_aws import run_query
    result = run_query(query)

    response_data = {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "intent": result.get("intent", ""),
    }
    set_cached(query, response_data)

    return {
        "statusCode": 200,
        "body": json.dumps({**response_data, "cached": False}),
        "headers": {"Content-Type": "application/json"},
    }
