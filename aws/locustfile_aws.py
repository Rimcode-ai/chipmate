"""
AWS Phase 9 — Locust load test against API Gateway.
Same structure as locustfile.py but pointing at the AWS endpoint.

IMPORTANT: Bedrock charges per token. Keep the test small:
  --users 10 --run-time 2m  (~60-120 Bedrock calls, a few cents at Haiku rates)
  Check Cost Explorer after: Billing > Cost Explorer > service=Bedrock

Run:
    locust -f aws/locustfile_aws.py \
           --host https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com \
           --users 10 --spawn-rate 2 --run-time 2m --headless \
           --csv results/load_test_aws

Note the latency difference vs local:
  Local: bottleneck is Ollama inference (~2-5s on CPU)
  AWS:   bottleneck is Bedrock + Aurora roundtrip (~0.5-2s on serverless)
  The AWS numbers should be faster because Bedrock uses GPU-backed inference.
"""

import random
from locust import HttpUser, task, between

SPEC_QUERIES = [
    "What is the input voltage range of TPS62902?",
    "What is the output current of TPS62902?",
    "What is the switching frequency of TPS62902?",
    "What is the supply voltage of LM358?",
]

RELATIONSHIP_QUERIES = [
    "What replaces the TPS62902?",
    "Is LM358 pin compatible with TLV2372?",
]


class ChipMateAWSUser(HttpUser):
    wait_time = between(2, 4)  # wider gap than local to keep Bedrock costs down

    @task(7)
    def spec_query(self):
        self.client.post(
            "/query",
            json={"query": random.choice(SPEC_QUERIES)},
            name="/query [spec]",
        )

    @task(3)
    def relationship_query(self):
        self.client.post(
            "/query",
            json={"query": random.choice(RELATIONSHIP_QUERIES)},
            name="/query [relationship]",
        )
