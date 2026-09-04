"""
AWS Phase 2 — Hybrid retrieval against Aurora PostgreSQL (pgvector + tsvector + RRF).
Mirrors local/retrieve.py but uses Titan embeddings for query encoding.

The retrieval logic is identical — only the embedding call changes (Bedrock vs Ollama).
This is the same point made in agent_aws.py: only the LLM/model call changes.
"""

import json
from typing import NamedTuple

import boto3
import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

AURORA_URL = os.getenv("AURORA_URL", "")
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
RRF_K = 60
DEFAULT_TOP_K = 5

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


class Chunk(NamedTuple):
    chunk_id: str
    component: str
    section: str
    page: int
    text: str
    rrf_score: float


def embed_query_titan(query: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": query, "dimensions": EMBED_DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(response["body"].read())["embedding"]


def vector_search(query_embedding: list[float], top_k: int, conn) -> list[tuple[str, int]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY embedding <-> %s::vector) AS rank
        FROM chunks ORDER BY embedding <-> %s::vector LIMIT %s
        """,
        (query_embedding, query_embedding, top_k * 2),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def keyword_search(query: str, top_k: int, conn) -> list[tuple[str, int]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', %s)) DESC) AS rank
        FROM chunks WHERE fts_vector @@ plainto_tsquery('english', %s) LIMIT %s
        """,
        (query, query, top_k * 2),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def reciprocal_rank_fusion(a, b, k=RRF_K):
    scores: dict[str, float] = {}
    for chunk_id, rank in a + b:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fetch_chunks_by_ids(ids: list[str], conn) -> dict[str, tuple]:
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, component, section, page, text FROM chunks WHERE chunk_id = ANY(%s)",
        (ids,),
    )
    return {r[0]: r for r in cur.fetchall()}


def hybrid_search_aws(query: str, top_k: int = DEFAULT_TOP_K, conn=None) -> list[Chunk]:
    close = conn is None
    if conn is None:
        conn = psycopg.connect(AURORA_URL)
    try:
        emb = embed_query_titan(query)
        vec = vector_search(emb, top_k, conn)
        kw = keyword_search(query, top_k, conn)
        merged = reciprocal_rank_fusion(vec, kw)[:top_k]
        ids = [cid for cid, _ in merged]
        scores = {cid: s for cid, s in merged}
        data = fetch_chunks_by_ids(ids, conn)
        return [
            Chunk(chunk_id=r[0], component=r[1], section=r[2], page=r[3],
                  text=r[4], rrf_score=scores[r[0]])
            for cid in ids if (r := data.get(cid))
        ]
    finally:
        if close:
            conn.close()
