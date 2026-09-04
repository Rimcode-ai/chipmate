# ChipMate — Implementation Plan

All commands are written to run in your terminal. Every command is annotated with what it does and why.

---

## AWS Profile Setup

Your credentials are STS session tokens (temporary). Add them to your AWS config so all CLI commands use the right account.

```bash
# Open or create ~/.aws/credentials and add:
# [644921559645_Developer_FTE]
# aws_access_key_id=ASIAZMKCSBJOVYQ5OR4S
# aws_secret_access_key=x3xJzfA4sUJDC/e2fIyu6ZH34iMZHdOHasWL+2pP
# aws_session_token=IQoJb3JpZ2...

# These are STS tokens — they expire (usually in 1-12 hours).
# You will need to refresh them each session.

# Verify the profile resolves:
aws sts get-caller-identity --profile 644921559645_Developer_FTE
# Expected output: {"UserId": "...", "Account": "644921559645", "Arn": "arn:aws:..."}

# Set as default for this terminal session so you don't have to --profile every command:
export AWS_PROFILE=644921559645_Developer_FTE
# Verify:
aws sts get-caller-identity
```

---

## Phase 0 — Local Environment Setup

### 0.1 Directory structure

```bash
# cd into your working directory
cd /Users/rima.modak/firstup/chipmate

# Verify structure exists
ls
# Expected: EVALUATION.md  PLAN.md  TODOS.md  data/  local/  aws/
```

### 0.2 Python venv

```bash
# python3 --version must be 3.11+
python3 --version

# Create a virtual environment inside the project
# -m venv: use Python's built-in venv module
# .venv: the directory name (conventional; IntelliJ looks for this name by default)
python3 -m venv .venv

# Activate the venv (must do this in every new terminal session)
source .venv/bin/activate

# Verify: your prompt should now show (.venv) prefix
# Also verify which python is being used:
which python
# Expected: /Users/rima.modak/firstup/chipmate/.venv/bin/python
```

### 0.3 Python dependencies

```bash
# Create requirements.txt with pinned versions
cat > requirements.txt << 'EOF'
# PDF processing
pdfplumber==0.11.4

# Vector/embedding
sentence-transformers==3.0.1
pgvector==0.3.2
psycopg2-binary==2.9.9

# Graph
neo4j==5.20.0
gremlinpython==3.7.1

# Agent orchestration
langgraph==0.2.28
langchain-core==0.2.38
langchain-community==0.2.17

# Resilience
pybreaker==1.1.0

# API layer
fastapi==0.115.0
uvicorn==0.30.6
redis==5.0.8

# Eval + load testing
locust==2.31.4

# AWS SDK (needed even in local phase for later port)
boto3==1.35.0
aws-xray-sdk==2.14.0

# Tracing
langfuse==2.40.0

# Utilities
python-dotenv==1.0.1
httpx==0.27.2
EOF

# Install all dependencies
pip install -r requirements.txt

# Verify key packages installed:
python -c "import pdfplumber, langgraph, neo4j, fastapi, pybreaker, langfuse; print('all imports OK')"
```

### 0.4 Docker services

```bash
# Verify Docker Desktop is running
docker info
# Expected: shows server version, storage driver, etc. If this fails, open Docker Desktop app.

# Create docker-compose.yml in the chipmate root (see file in repo)
# Then start all services:
docker compose up -d

# -d: detached mode (runs in background)
# Verify containers are running:
docker compose ps
# Expected: postgres, neo4j, redis all show state "running"

# Check Postgres logs specifically (useful when debugging connection issues):
docker compose logs postgres

# Check if pgvector extension is available in Postgres:
docker compose exec postgres psql -U chipmate -d chipmate -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
# Expected: returns a row with "vector" — if empty, the extension isn't installed yet (see Phase 2)
```

### 0.5 Ollama

```bash
# Install Ollama (if not already installed)
# Option A: via brew
brew install ollama
# Option B: download from ollama.com/download

# Start Ollama server (runs on http://localhost:11434)
ollama serve &
# The & runs it in the background. To stop: kill %1 or pkill ollama

# Pull a model — mistral is a good balance of size and quality for this use case
# This is a one-time download (~4GB for mistral:7b)
ollama pull mistral:7b

# Pull nomic-embed-text for local embeddings (much smaller, ~274MB)
ollama pull nomic-embed-text

# Verify both models are available:
ollama list
# Expected: shows mistral:7b and nomic-embed-text in the list

# Test Ollama is responding:
curl http://localhost:11434/api/generate \
  -d '{"model":"mistral:7b","prompt":"What is pgvector?","stream":false}'
# Expected: JSON response with a "response" field containing text
```

### 0.6 Langfuse (local tracing)

```bash
# Langfuse self-hosted runs via Docker. Add it to docker-compose.yml (already included in the compose file in this repo).
# After compose up, verify:
curl http://localhost:3000
# Expected: 200 or redirects to Langfuse UI
# Open http://localhost:3000 in browser, create an account, note your PUBLIC_KEY and SECRET_KEY
```

### 0.7 IntelliJ setup

Steps (not scriptable — do these in the IDE):
1. `File > Open` → select `/Users/rima.modak/firstup/chipmate`
2. `File > Project Structure > SDKs > + > Python SDK > Existing environment`
3. Navigate to `.venv/bin/python`, select it
4. Install plugins: `Preferences > Plugins` → search and install:
   - "AWS Toolkit" (official JetBrains plugin for Lambda, S3, etc.)
   - "Docker" (if not already installed — comes with IntelliJ Ultimate)
   - ".env files support" (for `.env` file syntax highlighting)
5. Create a `.env` file in `chipmate/` for local dev secrets (never commit this):
   ```
   POSTGRES_URL=postgresql://chipmate:chipmate@localhost:5432/chipmate
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=chipmate
   REDIS_URL=redis://localhost:6379
   OLLAMA_BASE_URL=http://localhost:11434
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   ```

---

## Phase 1 — Structure-Aware PDF Chunking

### 1.1 Get sample datasheets

```bash
# Download a real publicly available datasheet
# TPS62902 is the component mentioned in the spec — it's a Texas Instruments buck converter
# TI provides datasheets publicly via their website
# Place PDFs in data/datasheets/:
ls data/datasheets/
# Target: 3-5 PDFs before starting Phase 1

# Suggested components to download (search "<part number> datasheet PDF" on ti.com, onsemi.com, or digikey.com):
# TPS62902  — TI buck converter
# LM358     — TI op-amp (classic, well-documented)
# ADS1115   — TI 16-bit ADC
# INA219    — TI current sensor
```

### 1.2 Run the chunker

```bash
# After writing local/ingest.py (Phase 1 code):
python local/ingest.py --input data/datasheets/TPS62902.pdf --output data/chunks/

# Verify output:
ls data/chunks/
cat data/chunks/TPS62902.json | python -m json.tool | head -50
# Expected: JSON array of chunks, each with chunk_id, text, component, section, page
```

---

## Phase 2 — Vector Store + Hybrid Search

### 2.1 Set up pgvector schema

```bash
# Connect to the running Postgres container
docker compose exec postgres psql -U chipmate -d chipmate

# Inside psql:
# Install the pgvector extension (only needed once):
CREATE EXTENSION IF NOT EXISTS vector;

# Verify:
\dx
# Expected: shows "vector" in the extensions list

# Create the chunks table:
CREATE TABLE chunks (
    chunk_id    TEXT PRIMARY KEY,
    component   TEXT NOT NULL,
    section     TEXT,
    page        INTEGER,
    text        TEXT NOT NULL,
    embedding   VECTOR(768),  -- nomic-embed-text output dimension is 768
    fts_vector  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

# Create indexes for fast retrieval:
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
# ivfflat: approximate nearest neighbor index — much faster than exact scan at scale
# vector_cosine_ops: use cosine similarity (standard for text embeddings)
# lists=100: number of inverted lists — tune up if you have >1M rows; fine for our 10k-row dataset

CREATE INDEX ON chunks USING GIN (fts_vector);
# GIN: Generalized Inverted Index — optimized for full-text search

# Exit psql:
\q
```

### 2.2 Embed and store chunks

```bash
# After writing local/embed.py:
python local/embed.py --input data/chunks/TPS62902.json

# Verify rows were inserted:
docker compose exec postgres psql -U chipmate -d chipmate \
  -c "SELECT chunk_id, component, section, page, LEFT(text, 80) FROM chunks LIMIT 5;"
```

### 2.3 Test hybrid search

```bash
# After writing local/retrieve.py:
python local/retrieve.py --query "operating voltage range TPS62902"
# Expected: returns top 5 chunks ranked by hybrid score, with source metadata
```

---

## Phase 3 — Component Relationship Graph (Neo4j)

### 3.1 Connect to Neo4j

```bash
# Verify Neo4j is running:
docker compose exec neo4j neo4j status
# Or check the browser UI: http://localhost:7474 (default credentials: neo4j / chipmate)

# Test connection from Python:
python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','chipmate'))
d.verify_connectivity()
print('Neo4j connected')
d.close()
"
```

### 3.2 Load relationships

```bash
# After writing local/graph.py:
python local/graph.py --load  # loads hardcoded test relationships

# Verify:
python local/graph.py --query "TPS62902"
# Expected: prints list of components that replace TPS62902
```

### 3.3 Cypher commands to know

```cypher
-- Add a node (run in Neo4j Browser at http://localhost:7474):
CREATE (:Component {name: 'TPS62902', type: 'buck_converter', vendor: 'TI'})

-- Add a relationship:
MATCH (a:Component {name: 'TPS62902'}), (b:Component {name: 'TPS62903'})
CREATE (a)-[:REPLACED_BY]->(b)

-- Query replacements:
MATCH (a:Component {name: 'TPS62902'})-[:REPLACED_BY]->(b)
RETURN b.name

-- Find all components within 2 hops:
MATCH (a:Component {name: 'TPS62902'})-[*1..2]->(b)
RETURN b.name, type(last(relationships(path)))

-- Check all nodes:
MATCH (n) RETURN n LIMIT 25
```

---

## Phase 4 — LangGraph Multi-Agent Orchestration

### 4.1 Graph structure mental model

```
State = {query, intent, retrieved_chunks, graph_results, answer, confidence, requires_human_review}

Nodes:
  router_node       -- deterministic, regex-based intent detection
  retrieval_node    -- calls retrieve.py (vector + BM25)
  graph_node        -- calls graph.py (Cypher traversal)
  analysis_node     -- calls Ollama/Bedrock
  grounding_node    -- regex extraction + source comparison
  human_approval_node -- blocks and waits for input

Edges:
  router -> retrieval   (if intent == 'spec_lookup')
  router -> graph       (if intent == 'relationship_lookup')
  router -> retrieval   (if intent == 'general', hits both)
  retrieval -> analysis
  graph -> analysis
  analysis -> grounding
  grounding -> END      (if confidence >= threshold)
  grounding -> human_approval_node  (if confidence < threshold)
  human_approval_node -> END
```

### 4.2 Run the agent

```bash
# After writing local/agent.py:
python local/agent.py --query "What replaces the TPS62902?"
# Expected: routes to graph_node, traverses REPLACED_BY, returns grounded answer

python local/agent.py --query "What is the operating voltage of TPS62902?"
# Expected: routes to retrieval_node, returns answer with grounding check output

python local/agent.py --query "What is the pin configuration of LM358?"
# Expected: if low-confidence, triggers human_approval_node and pauses
```

---

## Phase 5 — Deterministic Evaluation Harness

### 5.1 Golden dataset format

```json
[
  {
    "id": "tc-001",
    "query": "What is the operating voltage of TPS62902?",
    "expected_value": "3.3",
    "expected_unit": "V",
    "expected_source_contains": "Input Voltage Range",
    "intent": "spec_lookup"
  },
  {
    "id": "tc-002",
    "query": "What replaces the TPS62902?",
    "expected_graph_result": ["TPS62903"],
    "intent": "relationship_lookup"
  }
]
```

### 5.2 Run eval

```bash
python local/eval.py --dataset golden_dataset.json

# Expected output:
# tc-001: PASS (extracted 3.3V, matched source)
# tc-002: PASS (graph returned TPS62903)
# ...
# Overall accuracy: 14/15 (93.3%)
```

---

## Phase 6 — FastAPI Async Service + Redis Rate Limiting

### 6.1 Start the API

```bash
# After writing local/main.py:
uvicorn local.main:app --reload --port 8000
# --reload: auto-restarts when files change (development only)

# Test the API:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the voltage of TPS62902?"}'

# Test rate limiting — send 11 requests (limit is 10/minute by default):
for i in $(seq 1 11); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "test query"}';
done
# Expected: first 10 return 200, 11th returns 429

# Test cache — same query twice should return same answer faster:
time curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the voltage of TPS62902?"}'
# Run this twice, second call should be significantly faster (cache hit)
```

---

## Phase 7 — Circuit Breaker

### 7.1 Demonstrate the circuit breaker tripping

```bash
# After writing local/resilience.py:
# The breaker wraps the Ollama call. Trigger it by stopping Ollama:

# Kill Ollama:
pkill ollama

# Now fire 4 requests:
python -c "
from local.resilience import call_llm_with_fallback
for i in range(4):
    try:
        result = call_llm_with_fallback('What is voltage?')
        print(f'Request {i+1}: {result[:50]}')
    except Exception as e:
        print(f'Request {i+1}: FAILED — {type(e).__name__}: {e}')
"
# Expected:
# Request 1: FAILED — ConnectionError: ...
# Request 2: FAILED — ConnectionError: ...
# Request 3: FAILED — ConnectionError: ...
# Request 4: FAILED — CircuitBreakerError: (breaker is now OPEN, rejects immediately)

# Restart Ollama:
ollama serve &
# After 30 seconds the breaker resets to HALF-OPEN and retries
```

---

## Phase 8 — Langfuse Tracing

### 8.1 Verify traces appear

```bash
# After writing local/tracing.py and integrating with agent.py:
python local/agent.py --query "What is the voltage of TPS62902?"

# Open http://localhost:3000 in browser
# Navigate to Traces
# Expected: a trace with spans for router_node, retrieval_node, analysis_node, grounding_node
# Each span shows: input, output, latency, and any metadata you attach (confidence, chunk_count, etc.)
```

---

## Phase 9 — Locust Load Test

### 9.1 Run the load test

```bash
# After writing locustfile.py:
locust -f locustfile.py \
  --host http://localhost:8000 \
  --users 20 \
  --spawn-rate 5 \
  --run-time 2m \
  --headless \
  --csv results/load_test_local

# --headless: no browser UI, output to terminal and CSV
# --csv: saves results to results/load_test_local_stats.csv and _failures.csv

# View results:
cat results/load_test_local_stats.csv

# Key metrics to note for interviews:
# RPS (requests/sec), median latency, p95 latency, failure rate
```

---

## AWS Phase 0 — Account Setup

```bash
# 1. Verify credentials are set (STS tokens expire — re-run aws configure if this fails)
export AWS_PROFILE=644921559645_Developer_FTE
aws sts get-caller-identity
# Expected: {"Account": "644921559645", ...}

# 2. Set up billing alarm — DO THIS BEFORE ANYTHING ELSE
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account ID: $ACCOUNT_ID"

aws budgets create-budget \
  --account-id "$ACCOUNT_ID" \
  --budget '{
    "BudgetName": "ChipMate-Budget",
    "BudgetLimit": {"Amount": "20", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[{
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "rima.modak@firstup.io"}]
  }]'
# This creates a budget that emails you when you hit 80% of $20 = $16

# 3. Enable Bedrock model access (manual step — cannot be done via CLI)
# Console: AWS Console > Bedrock > Model Access > Request access to:
#   - anthropic.claude-3-haiku-20240307-v1:0  (cheap, use for all dev)
#   - amazon.titan-embed-text-v2:0             (for embeddings in Phase A2)
# Usually approved instantly (within seconds).

# 4. Verify Bedrock access (after console approval):
aws bedrock list-foundation-models \
  --query 'modelSummaries[?modelId==`anthropic.claude-3-haiku-20240307-v1:0`].[modelId,modelLifecycle]' \
  --output table
# Expected: shows the model ID with lifecycle status "ACTIVE"
```

---

## AWS Phase 1 — S3 + Lambda Ingestion

```bash
# 5. Create the S3 bucket
# Bucket names must be globally unique — append your initials or a number
BUCKET_NAME="chipmate-datasheets-rm-$(date +%s)"
aws s3 mb s3://$BUCKET_NAME --region us-east-1
# mb = make bucket
# Region: us-east-1 recommended — Bedrock is available there

# Verify bucket exists:
aws s3 ls | grep chipmate

# 6. Upload datasheets:
aws s3 cp data/datasheets/ s3://$BUCKET_NAME/datasheets/ --recursive
# --recursive: upload the entire directory
# Verify:
aws s3 ls s3://$BUCKET_NAME/datasheets/

# 7. Create IAM role for Lambda (Lambda needs permission to read S3 and write back)
aws iam create-role \
  --role-name chipmate-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'
# This creates a role that Lambda can assume. "Principal: lambda.amazonaws.com" means only Lambda can use it.

aws iam attach-role-policy \
  --role-name chipmate-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
# Attach managed policy. For a real production system you'd scope this down to the specific bucket.
# For this learning project, full S3 access is fine.

aws iam attach-role-policy \
  --role-name chipmate-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
# Allows Lambda to write logs to CloudWatch — needed for debugging.

# Get the role ARN for use in next step:
ROLE_ARN=$(aws iam get-role --role-name chipmate-lambda-role --query Role.Arn --output text)
echo "Role ARN: $ROLE_ARN"

# 8. Package and deploy the Lambda:
mkdir -p lambda_package
pip install pdfplumber -t lambda_package/
cp aws/lambda_chunk.py lambda_package/
cd lambda_package && zip -r ../lambda_chunk.zip . && cd ..
# zip -r: recursively zip everything (all installed packages + your code)
# This is the Lambda deployment package

aws lambda create-function \
  --function-name chipmate-chunker \
  --runtime python3.11 \
  --role "$ROLE_ARN" \
  --handler lambda_chunk.lambda_handler \
  --zip-file fileb://lambda_chunk.zip \
  --timeout 60 \
  --memory-size 512
# handler: file_name.function_name — must match exactly
# timeout: max seconds Lambda will run before killing (PDF parsing can be slow)
# memory-size: MB of RAM. More RAM = faster CPU too (Lambda CPU scales with memory)

# Verify Lambda created:
aws lambda get-function --function-name chipmate-chunker --query 'Configuration.[FunctionName,State,Runtime]'

# 9. Add S3 trigger:
LAMBDA_ARN=$(aws lambda get-function --function-name chipmate-chunker --query Configuration.FunctionArn --output text)

# First give S3 permission to invoke the Lambda:
aws lambda add-permission \
  --function-name chipmate-chunker \
  --statement-id s3-trigger \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::$BUCKET_NAME

# Then configure the S3 bucket to send events to Lambda:
aws s3api put-bucket-notification-configuration \
  --bucket $BUCKET_NAME \
  --notification-configuration "{
    \"LambdaFunctionConfigurations\": [{
      \"LambdaFunctionArn\": \"$LAMBDA_ARN\",
      \"Events\": [\"s3:ObjectCreated:*\"],
      \"Filter\": {
        \"Key\": {
          \"FilterRules\": [{\"Name\": \"suffix\", \"Value\": \".pdf\"}]
        }
      }
    }]
  }"
# This triggers chipmate-chunker whenever any .pdf file is uploaded to the bucket

# 10. Test: upload a PDF and verify Lambda ran:
aws s3 cp data/datasheets/TPS62902.pdf s3://$BUCKET_NAME/

# Wait ~10 seconds, then check CloudWatch logs:
aws logs tail /aws/lambda/chipmate-chunker --since 5m
# Expected: shows the Lambda execution log with chunk_count in the output
```

---

## AWS Phase 2 — Aurora PostgreSQL + pgvector

```bash
# 11. Create Aurora Serverless v2 cluster:
aws rds create-db-cluster \
  --db-cluster-identifier chipmate-cluster \
  --engine aurora-postgresql \
  --engine-version 15.4 \
  --serverless-v2-scaling-configuration MinCapacity=0.5,MaxCapacity=2 \
  --master-username chipmate_admin \
  --master-user-password "ChipMate2024!" \
  --db-subnet-group-name default \
  --region us-east-1
# MinCapacity=0.5: scales down to 0.5 ACU (~$0.06/hr) when idle — important for cost
# MaxCapacity=2: caps at 2 ACU to prevent cost surprises

# Create the DB instance within the cluster:
aws rds create-db-instance \
  --db-instance-identifier chipmate-instance \
  --db-cluster-identifier chipmate-cluster \
  --engine aurora-postgresql \
  --db-instance-class db.serverless
# db.serverless: uses the Serverless v2 scaling configured on the cluster

# Wait for cluster to become available (takes 5-10 minutes):
aws rds wait db-cluster-available --db-cluster-identifier chipmate-cluster
echo "Cluster is ready"

# Get the endpoint:
AURORA_ENDPOINT=$(aws rds describe-db-clusters \
  --db-cluster-identifier chipmate-cluster \
  --query 'DBClusters[0].Endpoint' \
  --output text)
echo "Aurora endpoint: $AURORA_ENDPOINT"

# 12. Connect and create schema (requires psql or a DB client):
# Option A: from IntelliJ's Database tool (Preferences > Database > + > PostgreSQL, enter endpoint)
# Option B: from terminal if psql is installed:
psql -h $AURORA_ENDPOINT -U chipmate_admin -d postgres
# Then run the schema creation SQL from Phase 2 above

# 13. IMPORTANT — stop the cluster when not in use to avoid charges:
aws rds stop-db-cluster --db-cluster-identifier chipmate-cluster
# Restart when you need it:
aws rds start-db-cluster --db-cluster-identifier chipmate-cluster
```

---

## AWS Phase 3 — Neptune (Create and IMMEDIATELY Delete)

```bash
# 14. Create Neptune cluster:
aws neptune create-db-cluster \
  --db-cluster-identifier chipmate-graph \
  --engine neptune \
  --db-subnet-group-name default \
  --region us-east-1

aws neptune create-db-instance \
  --db-instance-identifier chipmate-graph-instance \
  --db-cluster-identifier chipmate-graph \
  --engine neptune \
  --db-instance-class db.t3.medium

# Wait for it to be available (~10 minutes):
aws neptune wait db-instance-available --db-instance-identifier chipmate-graph-instance
echo "Neptune ready"

NEPTUNE_ENDPOINT=$(aws neptune describe-db-clusters \
  --db-cluster-identifier chipmate-graph \
  --query 'DBClusters[0].Endpoint' \
  --output text)
echo "Neptune endpoint: $NEPTUNE_ENDPOINT"

# [Run your graph tests here — add relationships, query, verify results]

# 15. DELETE NEPTUNE IMMEDIATELY AFTER TESTING — no free tier, costs money per hour:
aws neptune delete-db-instance \
  --db-instance-identifier chipmate-graph-instance \
  --skip-final-snapshot
# Wait for instance to delete:
aws neptune wait db-instance-deleted --db-instance-identifier chipmate-graph-instance

aws neptune delete-db-cluster \
  --db-cluster-identifier chipmate-graph \
  --skip-final-snapshot

# Verify deletion:
aws neptune describe-db-clusters --db-cluster-identifier chipmate-graph 2>&1
# Expected: error "DBClusterNotFoundFault" — that means it's gone
```

---

## AWS Phase 4 — Bedrock Agent

```bash
# 16. Test Bedrock access:
aws bedrock-runtime invoke-model \
  --model-id anthropic.claude-3-haiku-20240307-v1:0 \
  --body '{"anthropic_version":"bedrock-2023-06-01","max_tokens":100,"messages":[{"role":"user","content":"Say hello in one word"}]}' \
  --content-type application/json \
  --accept application/json \
  /tmp/bedrock_test.json && cat /tmp/bedrock_test.json
# Expected: JSON with {"content":[{"type":"text","text":"Hello"}], ...}
# If you get "AccessDeniedException": you haven't enabled model access in the console yet

# 17. Test Titan embeddings:
aws bedrock-runtime invoke-model \
  --model-id amazon.titan-embed-text-v2:0 \
  --body '{"inputText":"What is the voltage range?"}' \
  --content-type application/json \
  --accept application/json \
  /tmp/titan_test.json && python -c "
import json
r = json.load(open('/tmp/titan_test.json'))
print(f'Embedding dimension: {len(r[\"embedding\"])}')
print(f'First 5 values: {r[\"embedding\"][:5]}')
"
# Expected: Embedding dimension: 1024 (Titan V2 produces 1024-dim vectors, not 1536)
```

---

## Full AWS Teardown

```bash
# Run this when done with the entire AWS track to avoid ongoing charges

# Delete Lambda functions:
aws lambda delete-function --function-name chipmate-chunker
aws lambda delete-function --function-name chipmate-api 2>/dev/null

# Delete API Gateway:
API_ID=$(aws apigatewayv2 get-apis --query 'Items[?Name==`chipmate-api-gateway`].ApiId' --output text)
aws apigatewayv2 delete-api --api-id $API_ID 2>/dev/null

# Delete Aurora:
aws rds delete-db-instance --db-instance-identifier chipmate-instance --skip-final-snapshot 2>/dev/null
aws rds delete-db-cluster --db-cluster-identifier chipmate-cluster --skip-final-snapshot 2>/dev/null

# Delete Neptune (if not already done):
aws neptune delete-db-instance --db-instance-identifier chipmate-graph-instance --skip-final-snapshot 2>/dev/null
aws neptune delete-db-cluster --db-cluster-identifier chipmate-graph --skip-final-snapshot 2>/dev/null

# Empty and delete S3 bucket:
aws s3 rm s3://$BUCKET_NAME --recursive
aws s3 rb s3://$BUCKET_NAME

# Delete SQS queue (if created):
aws sqs delete-queue --queue-url "https://sqs.us-east-1.amazonaws.com/644921559645/chipmate-queue" 2>/dev/null

# Delete IAM role (must detach policies first):
aws iam detach-role-policy --role-name chipmate-lambda-role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam detach-role-policy --role-name chipmate-lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name chipmate-lambda-role

# Final check — verify billing:
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-7d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --query 'ResultsByTime[0].Total.UnblendedCost'
# Shows total spend in the past 7 days
```

---

## Git Workflow

```bash
# Initialize the repo:
cd /Users/rima.modak/firstup/chipmate
git init
git branch -m main  # rename default branch to main

# Create .gitignore before first commit:
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.env
data/datasheets/*.pdf    # don't commit large PDFs
data/chunks/
lambda_package/
*.zip
results/
.DS_Store
EOF

# Initial commit:
git add .gitignore README.md EVALUATION.md PLAN.md TODOS.md requirements.txt docker-compose.yml
git commit -m "Phase 0: project scaffold, evaluation, plan, and local environment setup"
# Branch names for each phase start with phase prefix — no Jira ticket needed for this learning project
# Convention: phase-N-description

# Phase branch workflow:
git checkout -b phase-1-chunking
# ... work ...
git add local/ingest.py
git commit -m "Phase 1: structure-aware PDF chunking with section header detection"
git checkout main
git merge phase-1-chunking --no-ff  # --no-ff preserves the branch history
git branch -d phase-1-chunking

# View phase-by-phase history:
git log --oneline --graph
```
