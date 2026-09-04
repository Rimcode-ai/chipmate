"""
Phase 9 — Locust load test against the local FastAPI service.

Run:
    # Start the API first:
    uvicorn local.main:app --port 8000

    # Then in another terminal:
    locust -f locustfile.py --host http://localhost:8000 \
           --users 20 --spawn-rate 5 --run-time 2m --headless \
           --csv results/load_test_local

Key metrics to record for interviews:
    - RPS (requests/sec): how many queries per second
    - Median latency: typical response time
    - p95 latency: 95th percentile — what 95% of users experience or better
    - Failure rate: % of requests that returned non-2xx

Expected bottleneck on local: Ollama inference time (mistral:7b ~2-5s/query on CPU).
The API and Redis layers add <5ms; the LLM is the wall.

When you run this on the AWS version:
    locust -f aws/locustfile_aws.py --host https://YOUR_API_GATEWAY_URL ...
    (keep users low: 10, run-time: 2m — Bedrock charges per token)
"""

import random
from locust import HttpUser, task, between

SPEC_QUERIES = [
    "What is the input voltage range of TPS62902?",
    "What is the output current of TPS62902?",
    "What is the switching frequency of TPS62902?",
    "What is the supply voltage range of LM358?",
    "How many channels does LM358 have?",
    "What is the resolution of ADS1115?",
    "What interface does ADS1115 use?",
    "What is the maximum current measured by INA219?",
]

RELATIONSHIP_QUERIES = [
    "What replaces the TPS62902?",
    "What replaces the LM358?",
    "Is the LM358 pin compatible with TLV2372?",
    "What can substitute the TPS62902?",
]


class ChipMateUser(HttpUser):
    # wait_time: pause between tasks per user (simulates think time)
    # between(1, 3) means wait 1-3 seconds between requests
    wait_time = between(1, 3)

    @task(7)   # 7 out of 10 requests will be spec queries (more common in real usage)
    def spec_query(self):
        self.client.post(
            "/query",
            json={"query": random.choice(SPEC_QUERIES)},
            name="/query [spec]",  # name groups results in Locust report
        )

    @task(3)   # 3 out of 10 requests will be relationship queries
    def relationship_query(self):
        self.client.post(
            "/query",
            json={"query": random.choice(RELATIONSHIP_QUERIES)},
            name="/query [relationship]",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")

    @task(1)
    def rate_limit_test(self):
        """Send the same query 3 times rapidly — tests cache hit on 2nd and 3rd."""
        query = "What is the input voltage range of TPS62902?"
        for _ in range(3):
            self.client.post(
                "/query",
                json={"query": query},
                name="/query [cache test]",
            )
