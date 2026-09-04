# ChipMate — TODO List

Track progress here. Check off each item as you complete it. Do not skip the check-in gates.

---

## Pre-Phase 0: Decisions Required

Answer these before writing any code. They will affect Phase 1 and Phase 2 implementation.

- [ ] **Decision 1:** BM25 implementation — pick one:
  - [ ] Option A: `tsvector` (Postgres native FTS, simple, portable)
  - [ ] Option B: `pg_bm25` / ParadeDB (true BM25, extra Docker image)
  - [ ] Option C: `rank_bm25` Python library (in-process, no DB dependency)
  - *Recommended: Option A for Phase 2, revisit in Phase 2 if interviewer specifically asks about BM25 vs FTS*

- [ ] **Decision 2:** Chunking strategy for datasheets — pick one:
  - [ ] Option A: Header regex split + table grouping (recommended)
  - [ ] Option B: Fixed token windows with overlap
  - *Recommended: Option A — gives you the "structure-aware" talking point*

- [ ] **Decision 3:** Claude CLI operating mode — pick one:
  - [ ] Option A: Claude writes full implementation, you run and test
  - [ ] Option B: Claude writes skeleton, you fill logic, Claude reviews
  - *Read EVALUATION.md Section 7 for trade-offs*

---

## Phase 0 — Local Environment

- [ ] Verify `python3 --version` >= 3.11
- [ ] Create `.venv` inside `chipmate/`: `python3 -m venv .venv`
- [ ] Activate venv: `source .venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify key imports: `python -c "import pdfplumber, langgraph, neo4j, fastapi, pybreaker, langfuse; print('OK')"`
- [ ] Create `docker-compose.yml` with Postgres+pgvector, Neo4j, Redis, Langfuse
- [ ] Run Docker services: `docker compose up -d`
- [ ] Verify all containers running: `docker compose ps`
- [ ] Install Ollama: `brew install ollama` or download from ollama.com
- [ ] Pull mistral:7b: `ollama pull mistral:7b` (one-time, ~4GB download)
- [ ] Pull nomic-embed-text: `ollama pull nomic-embed-text`
- [ ] Test Ollama: `curl http://localhost:11434/api/generate -d '{"model":"mistral:7b","prompt":"hello","stream":false}'`
- [ ] Create `.env` file with local connection strings (see PLAN.md Phase 0.7)
- [ ] Configure IntelliJ SDK to use `.venv/bin/python`
- [ ] Install IntelliJ plugins: AWS Toolkit, .env files support
- [ ] Initialize git repo and make initial commit
- [ ] **CHECK-IN GATE 0:** Every service starts cleanly, Ollama responds, venv imports pass

---

## Phase 1 — PDF Chunking

- [ ] Download 3-5 real datasheets (TPS62902, LM358, ADS1115, INA219) into `data/datasheets/`
- [ ] Write `local/ingest.py`:
  - [ ] Load PDF with pdfplumber
  - [ ] Detect section headers (regex on all-caps lines)
  - [ ] Group table rows to prevent cross-chunk splits
  - [ ] Assign metadata: `{chunk_id, component, section, page, text}`
  - [ ] Output to `data/chunks/<component>.json`
- [ ] Test: `python local/ingest.py --input data/datasheets/TPS62902.pdf`
- [ ] Inspect output: verify chunks have component/section/page metadata
- [ ] Verify no chunk is just whitespace or <50 characters
- [ ] **CHECK-IN GATE 1:** Run chunker on all 3-5 datasheets, inspect output manually, chunks look sane

---

## Phase 2 — Vector Store + Hybrid Search

- [ ] Connect to Postgres: `docker compose exec postgres psql -U chipmate -d chipmate`
- [ ] Install pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector;`
- [ ] Create `chunks` table with `embedding VECTOR(768)` and `fts_vector TSVECTOR`
- [ ] Create `ivfflat` index on embedding column
- [ ] Create `GIN` index on fts_vector column
- [ ] Write `local/embed.py`:
  - [ ] Generate embeddings via Ollama nomic-embed-text
  - [ ] Insert chunks with embeddings into Postgres
- [ ] Run: `python local/embed.py --input data/chunks/TPS62902.json`
- [ ] Verify rows: `SELECT count(*) FROM chunks;`
- [ ] Write `local/retrieve.py`:
  - [ ] Vector search: `ORDER BY embedding <-> $query_embedding LIMIT k`
  - [ ] Keyword search: `WHERE fts_vector @@ to_tsquery($terms)`
  - [ ] Reciprocal Rank Fusion to merge both result sets
- [ ] Test: `python local/retrieve.py --query "operating voltage range TPS62902"`
- [ ] **CHECK-IN GATE 2:** Hybrid search returns relevant chunks for 3 test queries, printed with scores and metadata

---

## Phase 3 — Component Graph (Neo4j)

- [ ] Verify Neo4j at http://localhost:7474 (login: neo4j / chipmate)
- [ ] Write `local/graph.py`:
  - [ ] `add_component(name, attrs)` — CREATE node
  - [ ] `add_relationship(a, rel_type, b)` — CREATE edge
  - [ ] `find_replacements(part)` — MATCH REPLACED_BY traversal
  - [ ] `find_pin_compatible(part)` — MATCH PIN_COMPATIBLE traversal
- [ ] Load test data: add TPS62902, TPS62903, LM358, TLV2372 with relationships:
  - [ ] TPS62902 REPLACED_BY TPS62903
  - [ ] LM358 PIN_COMPATIBLE TLV2372
- [ ] Test: `python local/graph.py --query "TPS62902"` → returns TPS62903
- [ ] Test graph traversal in Neo4j Browser (Cypher commands in PLAN.md Phase 3)
- [ ] **CHECK-IN GATE 3:** Graph query for TPS62902 returns correct replacement, pin-compat query works

---

## Phase 4 — LangGraph Multi-Agent

- [ ] Write `local/agent.py`:
  - [ ] Define `AgentState` TypedDict with all state fields
  - [ ] `router_node`: regex-based intent detection (no LLM call)
    - [ ] Patterns: `REPLACED_BY|replacement|substitute` → relationship_lookup
    - [ ] Patterns: `voltage|current|power|pin|spec` → spec_lookup
    - [ ] Default → general
  - [ ] `retrieval_node`: calls retrieve.py
  - [ ] `graph_node`: calls graph.py
  - [ ] `analysis_node`: calls Ollama mistral:7b via LangChain Ollama wrapper
  - [ ] `grounding_node`: regex extraction + source text comparison (see Phase 5 for detail)
  - [ ] `human_approval_node`: prints question, reads stdin, continues or escalates
  - [ ] Wire nodes into StateGraph with conditional edges
  - [ ] Compile and expose as `app = graph.compile()`
- [ ] Test spec query: `python local/agent.py --query "What is the operating voltage of TPS62902?"`
- [ ] Test graph query: `python local/agent.py --query "What replaces the TPS62902?"`
- [ ] Test low-confidence trigger: use a query about a component not in the dataset
- [ ] **CHECK-IN GATE 4:** Both routing paths work, human-approval node fires for low-confidence query

---

## Phase 5 — Evaluation Harness

- [ ] Write `golden_dataset.json` with 15-30 test cases:
  - [ ] At least 5 spec lookup cases (voltage, current, pin count)
  - [ ] At least 5 relationship lookup cases (replacements, pin-compat)
  - [ ] At least 3 edge cases (missing data, ambiguous query, multi-hop graph)
- [ ] Write `local/eval.py`:
  - [ ] Load golden dataset
  - [ ] For each test case: run agent, compare output to expected
  - [ ] For spec cases: regex-extract value+unit from answer, compare
  - [ ] For graph cases: check if expected part appears in answer
  - [ ] Report: per-case PASS/FAIL and overall accuracy %
- [ ] Run: `python local/eval.py --dataset golden_dataset.json`
- [ ] Record baseline accuracy (target: >80% on 15 cases)
- [ ] **CHECK-IN GATE 5:** Eval script runs clean, produces accuracy report, no unhandled exceptions

---

## Phase 6 — FastAPI Async Service + Redis

- [ ] Write `local/main.py`:
  - [ ] POST `/query` endpoint — takes `{"query": str}`, returns `{"answer": str, "confidence": float, "source_chunks": [...]}`
  - [ ] GET `/health` endpoint — returns `{"status": "ok"}`
  - [ ] Redis rate limiter: 10 requests/minute per IP
  - [ ] Redis semantic cache: hash query → cache answer (TTL 5 minutes)
- [ ] Start: `uvicorn local.main:app --reload --port 8000`
- [ ] Test happy path: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What is the voltage of TPS62902?"}'`
- [ ] Test rate limit: fire 11 requests, verify 11th returns HTTP 429
- [ ] Test cache: run same query twice, verify second is faster (log or time it)
- [ ] Test empty query: `{"query": ""}` → HTTP 400
- [ ] **CHECK-IN GATE 6:** API responds end-to-end, rate limit and cache both demonstrable

---

## Phase 7 — Circuit Breaker

- [ ] Write `local/resilience.py`:
  - [ ] `bedrock_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)`
  - [ ] Decorate primary LLM call with `@bedrock_breaker`
  - [ ] Fallback function: simpler prompt or cached response
  - [ ] `call_llm_with_fallback(prompt)`: tries primary, catches CircuitBreakerError, calls fallback
- [ ] Integrate into `analysis_node` in agent.py
- [ ] Demonstrate breaker tripping (kill Ollama, fire 4 requests, see open circuit)
- [ ] Demonstrate fallback responding after breaker opens
- [ ] **CHECK-IN GATE 7:** Can demonstrate breaker tripping and fallback in live terminal session

---

## Phase 8 — Langfuse Tracing

- [ ] Verify Langfuse UI at http://localhost:3000, create account, get API keys
- [ ] Add keys to `.env`: `LANGFUSE_PUBLIC_KEY=...` and `LANGFUSE_SECRET_KEY=...`
- [ ] Write `local/tracing.py`:
  - [ ] Initialize Langfuse client
  - [ ] Helper to create a trace per query
  - [ ] Helper to create a span per agent node
- [ ] Wrap each agent node with span creation (router, retrieval, graph, analysis, grounding)
- [ ] Include span metadata: node input/output, latency, confidence, chunk_count
- [ ] Run a query: `python local/agent.py --query "What is the voltage of TPS62902?"`
- [ ] Open Langfuse at http://localhost:3000 → Traces → verify trace appears with all spans
- [ ] **CHECK-IN GATE 8:** Can see full router → retrieval → analysis → grounding trace in Langfuse UI

---

## Phase 9 — Locust Load Test

- [ ] Write `locustfile.py`:
  - [ ] Mix of spec queries and relationship queries
  - [ ] `wait_time = between(1, 3)` to simulate realistic pacing
- [ ] Create `results/` directory
- [ ] Run: `locust -f locustfile.py --host http://localhost:8000 --users 20 --spawn-rate 5 --run-time 2m --headless --csv results/load_test_local`
- [ ] Record results: RPS, median latency, p95 latency, failure rate
- [ ] Identify bottleneck (likely: Ollama inference time)
- [ ] **CHECK-IN GATE 9:** Real numbers recorded. State the bottleneck.

---

## LOCAL TRACK COMPLETE

Verify all 12 success criteria from the spec (Section 5) before starting AWS track.

- [ ] SC-1: "What replaces TPS62902?" traverses graph correctly
- [ ] SC-2: "What is the operating voltage of TPS62902?" is deterministically verified
- [ ] SC-3: Router uses deterministic rules, not an LLM call
- [ ] SC-4: One low-confidence query triggers human-approval
- [ ] SC-5: Rate limit enforced and tested
- [ ] SC-6: Circuit breaker trips on 3 failures and fallback activates
- [ ] SC-7: Full trace visible in Langfuse
- [ ] SC-8: Golden dataset accuracy script runs and reports pass/fail
- [ ] SC-9: Load test numbers recorded
- [ ] SC-10: Same agent logic runs against both Ollama and Bedrock (verified in AWS Phase 4)
- [ ] SC-11: All AWS resources torn down after validation
- [ ] SC-12: Can explain every design decision verbally without notes

---

## AWS Phase 0 — Account Setup

- [ ] Refresh STS credentials if expired: re-run `aws configure` or SSO login
- [ ] Set profile: `export AWS_PROFILE=644921559645_Developer_FTE`
- [ ] Verify: `aws sts get-caller-identity` → returns account 644921559645
- [ ] Create $20 budget alarm (exact command in PLAN.md AWS Phase 0)
- [ ] Enable Bedrock model access in console (manual step):
  - [ ] claude-3-haiku-20240307-v1:0
  - [ ] amazon.titan-embed-text-v2:0
- [ ] Verify Bedrock: `aws bedrock-runtime invoke-model` test (command in PLAN.md)
- [ ] **CHECK-IN GATE A0:** `aws sts get-caller-identity` works, budget alarm set, Bedrock test returns text

---

## AWS Phase 1 — S3 + Lambda Ingestion

- [ ] Create S3 bucket with unique name (command in PLAN.md AWS Phase 1)
- [ ] Upload datasheets to S3: `aws s3 cp data/datasheets/ s3://$BUCKET_NAME/datasheets/ --recursive`
- [ ] Create IAM role `chipmate-lambda-role` with S3 + CloudWatch permissions
- [ ] Write `aws/lambda_chunk.py` (mirrors local/ingest.py but reads from S3)
- [ ] Package Lambda: `pip install pdfplumber -t lambda_package/ && zip -r lambda_chunk.zip lambda_package/`
- [ ] Deploy Lambda: `aws lambda create-function ...` (command in PLAN.md)
- [ ] Add S3 trigger (command in PLAN.md)
- [ ] Test: upload a PDF, check CloudWatch logs for Lambda execution
- [ ] **CHECK-IN GATE A1:** CloudWatch shows successful Lambda execution, chunks/ JSON appears in S3

---

## AWS Phase 2 — Aurora PostgreSQL

- [ ] Create Aurora Serverless v2 cluster (command in PLAN.md, MinCapacity=0.5)
- [ ] Wait for cluster available: `aws rds wait db-cluster-available --db-cluster-identifier chipmate-cluster`
- [ ] Get endpoint: `aws rds describe-db-clusters ...`
- [ ] Connect via psql or IntelliJ DB tool
- [ ] Run schema SQL: CREATE EXTENSION vector, CREATE TABLE chunks, CREATE indexes
- [ ] Write `aws/embed_aws.py` using Titan Embeddings via Bedrock
- [ ] Insert 5-10 chunks with real Titan embeddings
- [ ] Test vector query: `ORDER BY embedding <-> $query_vec LIMIT 5`
- [ ] Stop cluster after testing: `aws rds stop-db-cluster --db-cluster-identifier chipmate-cluster`
- [ ] **CHECK-IN GATE A2:** Vector query returns relevant chunks; cluster stopped to pause billing

---

## AWS Phase 3 — Neptune (time-boxed: complete same day, delete same day)

- [ ] Check current AWS spend before starting: AWS Console > Billing Dashboard
- [ ] Create Neptune cluster (command in PLAN.md) — start timer
- [ ] Wait for instance available: `aws neptune wait db-instance-available ...`
- [ ] Write `aws/graph_neptune.py` using gremlinpython
- [ ] Add 3-5 test relationships via Gremlin
- [ ] Test: `find_replacements("TPS62902")` returns correct result
- [ ] **DELETE Neptune immediately:** run both delete commands in PLAN.md
- [ ] Verify deletion: `aws neptune describe-db-clusters --db-cluster-identifier chipmate-graph` → error
- [ ] **CHECK-IN GATE A3:** Neptune worked, Neptune deleted, verified gone

---

## AWS Phase 4 — Bedrock + LangGraph

- [ ] Write `aws/agent_aws.py`:
  - [ ] `call_claude(prompt, system)` via boto3 bedrock-runtime
  - [ ] Same LangGraph structure as local/agent.py
  - [ ] Only `analysis_node` changes (calls Bedrock instead of Ollama)
- [ ] Start Aurora (if stopped): `aws rds start-db-cluster --db-cluster-identifier chipmate-cluster`
- [ ] Test: `python aws/agent_aws.py --query "What is the voltage of TPS62902?"`
- [ ] Verify real Bedrock response in terminal
- [ ] Check cost in Cost Explorer after 5-10 test calls (should be cents)
- [ ] Stop Aurora after: `aws rds stop-db-cluster --db-cluster-identifier chipmate-cluster`
- [ ] **CHECK-IN GATE A4:** Real Bedrock response received, cost verified in Cost Explorer

---

## AWS Phase 5 — Eval Harness (no AWS changes)

- [ ] Restart Aurora: `aws rds start-db-cluster ...`
- [ ] Run: `python local/eval.py --dataset golden_dataset.json` (same script, pointing at Aurora)
- [ ] Compare accuracy: local Ollama score vs. AWS Bedrock score — record both
- [ ] Stop Aurora after
- [ ] **CHECK-IN GATE A5:** Both scores recorded side by side

---

## AWS Phase 6 — Lambda API + API Gateway

- [ ] Write `aws/lambda_api.py`
- [ ] Add SQS queue (optional, for async pattern): `aws sqs create-queue --queue-name chipmate-queue`
- [ ] Package and deploy Lambda: `aws lambda create-function chipmate-api ...`
- [ ] Create API Gateway: `aws apigatewayv2 create-api --name chipmate-api-gateway ...`
- [ ] Test end-to-end: `curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/query ...`
- [ ] Note latency from CloudWatch logs
- [ ] **CHECK-IN GATE A6:** API Gateway + Lambda returns real answer end-to-end

---

## AWS Phase 7 — Circuit Breaker Around Bedrock

- [ ] Write `aws/resilience_aws.py` (same pybreaker pattern, wraps boto3 call)
- [ ] Fire 20 rapid Bedrock requests to trigger real throttling OR simulate via mock
- [ ] Verify CircuitBreakerError fires and fallback responds
- [ ] **CHECK-IN GATE A7:** Breaker opens, fallback works, logged in CloudWatch

---

## AWS Phase 8 — CloudWatch + X-Ray

- [ ] Write `aws/tracing_aws.py` with aws-xray-sdk
- [ ] Enable X-Ray on Lambda: `aws lambda update-function-configuration --function-name chipmate-api --tracing-config Mode=Active`
- [ ] Add custom CloudWatch metric: GroundingScore per query
- [ ] Run 5-10 queries through Lambda
- [ ] Open AWS Console > X-Ray > Service Map
- [ ] Verify trace shows router → retrieval → analysis spans
- [ ] **CHECK-IN GATE A8:** X-Ray Service Map shows trace breakdown with latency per step

---

## AWS Phase 9 — Locust Load Test vs API Gateway

- [ ] Write `aws/locustfile_aws.py` pointing at API Gateway URL
- [ ] Keep test small: 10 users, 2 minutes, to control Bedrock cost
- [ ] Run: `locust -f aws/locustfile_aws.py --users 10 --spawn-rate 2 --run-time 2m --headless --csv results/load_test_aws`
- [ ] Record: RPS, p50, p95, failure rate, cost from Cost Explorer for that window
- [ ] **CHECK-IN GATE A9:** Real numbers recorded, cost verified

---

## Teardown

- [ ] Run full teardown script from PLAN.md
- [ ] Verify $0 new charges after teardown in Billing Dashboard
- [ ] Keep Aurora stopped (not deleted) if you want to retest Phase 4-9 later at low cost
- [ ] Final deletion when fully done:
  - [ ] `aws rds delete-db-instance --db-instance-identifier chipmate-instance --skip-final-snapshot`
  - [ ] `aws rds delete-db-cluster --db-cluster-identifier chipmate-cluster --skip-final-snapshot`

---

## Interview Prep (do this after all phases)

- [ ] Can explain BM25 vs. vector search trade-off without notes
- [ ] Can explain why router is deterministic (not an LLM call) without notes
- [ ] Can explain circuit breaker pattern: CLOSED → OPEN → HALF-OPEN without notes
- [ ] Can explain why answers are grounded against retrieved text, not just trusted from LLM
- [ ] Can walk through LangGraph state flow verbally: state shape, node types, conditional edges
- [ ] Can state your load test numbers: X req/sec, Y ms p95, Z% failure rate
- [ ] Can name the one function that changes when swapping Ollama for Bedrock
- [ ] Can explain why Neptune was used for relationships vs. vector store
