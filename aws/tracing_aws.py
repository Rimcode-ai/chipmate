"""
AWS Phase 8 — Distributed tracing with AWS X-Ray.
Mirrors local/tracing.py (Langfuse) but uses X-Ray for the AWS track.

X-Ray captures:
- Service map: visual graph of Lambda -> Bedrock -> Aurora dependencies
- Subsegments: one per agent node (router, retrieval, analysis, grounding)
- Custom annotations: intent, confidence, method (primary/fallback)

Interview talking point:
  "X-Ray is what Amazon uses internally for distributed tracing. It auto-instruments
   boto3 calls via patch_all(), so every Bedrock and Aurora call shows up as a
   subsegment without any extra code. I added custom subsegments for my agent nodes
   so I could see the breakdown: how much time is routing vs. retrieval vs. LLM."

Requires:
    pip install aws-xray-sdk
    Lambda must have tracing enabled (Mode=Active)
    See PLAN.md AWS Phase 8 for the aws lambda update-function-configuration command.

Usage in Lambda handler (lambda_api.py):
    from aws.tracing_aws import setup_xray, trace_node
    setup_xray()  # call once at module load time

    with trace_node("router_node", {"query": query}) as seg:
        result = router_node(state)
"""

from aws_xray_sdk.core import xray_recorder, patch_all
from aws_xray_sdk.core.models.subsegment import Subsegment
from contextlib import contextmanager
import boto3
from dotenv import load_dotenv
import os

load_dotenv()

CLOUDWATCH_REGION = os.getenv("AWS_REGION", "us-east-1")
CLOUDWATCH_NAMESPACE = "ChipMate"


def setup_xray() -> None:
    """
    patch_all() auto-instruments all boto3 calls (Bedrock, Aurora via psycopg2 is manual).
    Call once at Lambda module load time — not inside the handler.
    """
    patch_all()
    print("[xray] Instrumentation enabled")


@contextmanager
def trace_node(node_name: str, metadata: dict = None):
    """
    Context manager that creates an X-Ray subsegment for an agent node.
    Use in agent nodes:
        with trace_node("analysis_node", {"model": "claude-3-haiku"}):
            result = call_bedrock(...)
    """
    with xray_recorder.in_subsegment(node_name) as subsegment:
        if metadata:
            for key, value in metadata.items():
                # X-Ray annotations are indexed (searchable); metadata is not
                subsegment.put_annotation(key, str(value))
        yield subsegment


def emit_grounding_metric(confidence: float) -> None:
    """
    Push custom metric to CloudWatch.
    Viewable in CloudWatch Metrics under namespace "ChipMate".
    """
    cw = boto3.client("cloudwatch", region_name=CLOUDWATCH_REGION)
    cw.put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[{
            "MetricName": "GroundingScore",
            "Value": confidence,
            "Unit": "None",
        }],
    )


def emit_llm_method_metric(method: str) -> None:
    """Track primary vs fallback usage over time."""
    cw = boto3.client("cloudwatch", region_name=CLOUDWATCH_REGION)
    cw.put_metric_data(
        Namespace=CLOUDWATCH_NAMESPACE,
        MetricData=[{
            "MetricName": "LLMMethod",
            "Value": 1,
            "Unit": "Count",
            "Dimensions": [{"Name": "Method", "Value": method}],
        }],
    )
