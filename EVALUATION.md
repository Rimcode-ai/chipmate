# ChipMate — Project Evaluation

**Date:** 2026-09-02  
**Scope:** Feasibility, risk, and gap assessment before Phase 0 begins

---

## 1. Feasibility Assessment

### Overall verdict: Achievable in ~4-6 weeks part-time, with one hard cost trap

The spec is well-written and self-scoping. Every phase has a clear deliverable and a check-in gate. The risk profile is low for the local track and medium for the AWS track — the main danger is surprise cost from Neptune and Aurora, not technical complexity.

---

## 2. Phase-by-Phase Risk Rating

| Phase | Deliverable | Risk | Notes |
|---|---|---|---|
| 0 | Docker + Ollama + venv | Low | Standard setup; Ollama model pull is slow on first run |
| 1 | PDF chunking pipeline | Low | pdfplumber is stable; structure-aware chunking needs design decision |
| 2 | pgvector hybrid search | Low-Med | BM25 via pg_bm25/ParadeDB or a separate in-process rank; design choice needed |
| 3 | Neo4j graph | Low | Simple Gremlin or Cypher; the data model (vertices/edges) needs explicit design |
| 4 | LangGraph multi-agent | Med | LangGraph state graph API changed in 0.2.x; pin version |
| 5 | Eval harness | Low | Pure Python regex; most stable phase |
| 6 | FastAPI async API + Redis | Low | Straightforward; rate-limiter design choice |
| 7 | Circuit breaker | Low | pybreaker is simple; triggering it deterministically for demo needs a plan |
| 8 | Langfuse tracing | Low-Med | Self-hosted Langfuse Docker compose adds infra; callback integration is simple |
| 9 | Locust load test | Low | Straightforward |
| A2 | AWS port - Aurora | Med | Free tier only applies to new accounts; $0.06/hr if not |
| A3 | AWS port - Neptune | HIGH | No free tier. $0.10-0.20/hr minimum. Tear down within hours |
| A4 | AWS port - Bedrock | Low-Med | Haiku is cheap; must enable model access manually in console first |
| A6 | Lambda + API Gateway | Low | Packaging deps (pdfplumber etc.) needs Lambda layers or container image |

---

## 3. Gaps and Design Decisions That Need Answers Before Writing Code

These are not blockers — they are choices you need to make consciously so you can defend them in an interview.

### Gap 1: BM25 implementation for hybrid search
The spec says "vector + keyword/BM25" but does not specify how BM25 runs.

**Options:**
- **pg_bm25 (ParadeDB extension):** Runs inside Postgres. Closest to production. Requires installing a non-standard Postgres extension; the Docker image must include it.
- **rank_bm25 (Python in-process):** Python library, runs at query time, no extra DB dependency. Simpler to set up, easier to understand, but not a DB-native solution.
- **tsvector (built-in Postgres FTS):** Native Postgres full-text search. Not true BM25 but very defensible as "keyword retrieval using Postgres native FTS." Most portable.

**Recommendation:** Use `tsvector` for local Phase 2, note it as "keyword component of hybrid search using Postgres FTS," and upgrade to pg_bm25 only if the interviewer probes it. The hybrid weighting (Reciprocal Rank Fusion) is the interesting part, not the specific keyword backend.

**Decision needed from you before Phase 2 starts.**

### Gap 2: Structure-aware chunking strategy
"Structure-aware" is underspecified. For electrical datasheets, this typically means:
- Splitting on section headers (Electrical Characteristics, Pin Description, Absolute Maximum Ratings)
- Keeping table rows grouped (do not split a table across chunks)
- Tagging chunks with metadata: component name, section name, page number

**Recommendation:** Use pdfplumber's `.extract_tables()` to detect table regions, split on uppercase-heavy lines as section headers, and tag every chunk with `{component, section, page}`. This is defensible and directly mirrors the "chunk with metadata" pattern Amazon asked about.

### Gap 3: Graph data model
Neo4j (local) and Neptune (AWS) both support Gremlin and Cypher. The spec shows Gremlin examples for Neptune. You need to decide:
- Use Cypher locally (Neo4j native, more readable) and Gremlin on Neptune (required for Neptune)
- Use Gremlin for both (consistent, but Gremlin is more verbose)

**Recommendation:** Use **Cypher locally, Gremlin on Neptune.** Write a thin `graph.py` interface with `add_relationship(a, rel, b)` and `find_replacements(part)` — the caller never sees which query language is used. This is a good design pattern to discuss in interviews.

### Gap 4: AWS credentials are temporary (STS session)
The credentials you provided (`ASIAZMK...`) are short-lived STS session credentials. They will expire (typically in 1-12 hours). For sustained work across sessions, you need either:
- A named profile in `~/.aws/credentials` pointing at long-lived IAM credentials
- Or re-run `aws configure` / your SSO login flow each session

**Action before Phase 0 AWS starts:** Run `aws sts get-caller-identity --profile 644921559645_Developer_FTE` to confirm the profile resolves. If it returns your account ID, you are set. If it fails, you need to re-export the credentials.

### Gap 5: LangGraph version pinning
LangGraph changed its graph compilation API significantly between 0.1.x and 0.2.x. The `compile()` and `invoke()` patterns differ. Pin to a specific version in `requirements.txt` from the start to avoid wasting time on API mismatch errors.

**Recommendation:** Pin `langgraph==0.2.x` (latest stable as of mid-2025) and check the changelog before upgrading.

---

## 4. What This Project Proves in an Interview

This is the actual value of the project — each phase maps to a concrete interview talking point:

| Phase | Talking point |
|---|---|
| 1 | "I implemented structure-aware chunking — I split on section headers and kept table rows intact, then tagged every chunk with component, section, and page metadata" |
| 2 | "I used hybrid retrieval: vector similarity via pgvector plus keyword search via Postgres FTS, fused with Reciprocal Rank Fusion" |
| 3 | "Component relationships live in a graph. A query for 'what replaces X' traverses the REPLACED_BY edge, not a vector lookup — because that's a structural fact, not a semantic one" |
| 4 | "My router is deterministic — it does not call the LLM to decide intent. It uses regex patterns on the query. That keeps latency low and the decision auditable" |
| 4 | "The orchestration layer is provider-agnostic. Swapping Bedrock for Ollama touches one function" |
| 5 | "I validate answers deterministically — I extract the claimed value with regex and compare it against the retrieved source text. If the LLM says 3.3V and the source says 5V, the grounding check flags it regardless of how confident the LLM sounds" |
| 7 | "I wrapped the LLM call in a circuit breaker. If it fails 3 times in 30 seconds, the breaker trips and I fall back to a simpler model. I demonstrated this by triggering real Bedrock throttling under load" |
| 8 | "I can show you the trace for any query — router step, retrieval step, analysis step, with latency per step. On AWS that's X-Ray; locally it's Langfuse" |
| 9 | "I ran a load test. My numbers were X req/sec at Y ms p99. Here is where the bottleneck was" |

---

## 5. Cost Risk Summary

| Item | Risk | Mitigation |
|---|---|---|
| Neptune (Phase A3) | High — no free tier, $0.10-0.20/hr | Create, test, delete same session. Budget <$2 |
| Aurora (Phase A2) | Medium — free tier may not apply | Use Serverless v2 with MinCapacity=0.5; stop after testing |
| Bedrock | Low — Haiku is cheap | Use Haiku for all dev; switch to Sonnet only for final demo |
| Lambda/API Gateway | Low | Free tier covers testing volume |
| S3 | Negligible | |

**Non-negotiable before any AWS resource creation:** Set the $20 budget alarm in Phase A0.

---

## 6. IntelliJ Setup Notes

For Python in IntelliJ (PyCharm features apply if using IntelliJ Ultimate with the Python plugin):

- Create the Python venv inside `chipmate/` at `chipmate/.venv`
- Set the project SDK in IntelliJ: `File > Project Structure > SDKs > Add > Python SDK > Existing environment > .venv/bin/python`
- Install the `.env` plugin for IntelliJ to manage env vars (Docker connection strings, AWS keys) without hardcoding
- For Docker: IntelliJ Ultimate has a built-in Docker plugin that shows container logs inline — use it instead of running `docker logs` manually during dev
- For AWS: install the AWS Toolkit IntelliJ plugin. It shows Lambda functions, S3 buckets, and lets you invoke Lambda directly from the IDE

---

## 7. One Clarifying Question Before Starting

The spec says "prefer writing the actual logic myself with Claude's help debugging/reviewing." Do you want me to:

**Option A:** Write the full implementation per phase, explain every design decision in comments, and you run/test it.  
**Option B:** Write a skeleton with the key design decisions documented, then you fill in the logic, and I review/debug.

Option B is slower but produces stronger interview recall. Option A gets you a working artifact faster. Tell me which, and I will match that pace throughout.
