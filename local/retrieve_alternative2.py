"""
Phase 2 — Hybrid retrieval with pg_bm25 / ParadeDB (ALTERNATIVE 2).
NOT used by the main pipeline. For learning/comparison only.

pg_bm25 is a Postgres extension by ParadeDB that adds a native BM25 index type.
This is the "production-correct" BM25 approach — runs entirely in the DB.

When you'd prefer this over retrieve.py:
- You want true BM25 (not Postgres FTS approximation)
- You have a large corpus where BM25 ranking quality measurably improves recall
- You are comfortable using a non-standard Postgres extension

Trade-offs vs tsvector (retrieve.py):
- REQUIRES a different Docker image: paradedb/paradedb:latest (not pgvector/pgvector)
- pg_bm25 is maintained by a startup — API can change between versions
- Adds a dependency that may not be available in managed DB offerings (Aurora, Cloud SQL)
- The actual ranking difference on a 10-20 doc corpus is negligible

To use this, update docker-compose.yml to use paradedb/paradedb image which bundles
both pgvector and pg_bm25, then:
    CREATE INDEX ON chunks USING bm25 (chunk_id, text) WITH (key_field='chunk_id');

Then the BM25 query uses:
    SELECT chunk_id, paradedb.score(chunk_id) AS score
    FROM chunks
    WHERE text @@@ 'operating voltage'  -- @@@ is the pg_bm25 operator
    ORDER BY score DESC
    LIMIT 10;

Run (requires paradedb Docker image):
    python -m local.retrieve_alternative2 --query "operating voltage TPS62902"
"""

import argparse
from typing import NamedTuple

import psycopg
import httpx
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_URL = os.getenv(
    "POSTGRES_URL", "postgresql://chipmate:chipmate@localhost:5432/chipmate"
)
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
RRF_K = 60
DEFAULT_TOP_K = 5


class Chunk(NamedTuple):
    chunk_id: str
    component: str
    section: str
    page: int
    text: str
    rrf_score: float


def embed_query(query: str) -> list[float]:
    r = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def vector_search(query_embedding: list[float], top_k: int, conn) -> list[tuple[str, int]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY embedding <-> %s::vector) AS rank
        FROM chunks
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k * 2),
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def pg_bm25_search(query: str, top_k: int, conn) -> list[tuple[str, int]]:
    """
    Use pg_bm25's @@@ operator and paradedb.score() function.
    Requires: CREATE INDEX ON chunks USING bm25 (chunk_id, text) WITH (key_field='chunk_id');
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY paradedb.score(chunk_id) DESC) AS rank
        FROM chunks
        WHERE text @@@ %s
        ORDER BY paradedb.score(chunk_id) DESC
        LIMIT %s
        """,
        (query, top_k * 2),
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def reciprocal_rank_fusion(
    vector_ranks: list[tuple[str, int]],
    bm25_ranks: list[tuple[str, int]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for chunk_id, rank in vector_ranks:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, rank in bm25_ranks:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fetch_chunks_by_ids(chunk_ids: list[str], conn) -> dict[str, tuple]:
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, component, section, page, text FROM chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    return {row[0]: row for row in cur.fetchall()}


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K, conn=None) -> list[Chunk]:
    close_conn = conn is None
    if conn is None:
        conn = psycopg.connect(POSTGRES_URL)

    try:
        query_embedding = embed_query(query)
        vector_ranks = vector_search(query_embedding, top_k, conn)
        bm25_ranks = pg_bm25_search(query, top_k, conn)
        merged = reciprocal_rank_fusion(vector_ranks, bm25_ranks)[:top_k]

        ranked_ids = [chunk_id for chunk_id, _ in merged]
        rrf_scores = {chunk_id: score for chunk_id, score in merged}
        chunk_data = fetch_chunks_by_ids(ranked_ids, conn)

        results = []
        for chunk_id in ranked_ids:
            if chunk_id in chunk_data:
                row = chunk_data[chunk_id]
                results.append(Chunk(
                    chunk_id=row[0], component=row[1], section=row[2],
                    page=row[3], text=row[4], rrf_score=rrf_scores[chunk_id],
                ))
        return results
    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid retrieval with pg_bm25 (ParadeDB)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    chunks = hybrid_search(args.query, args.top_k)
    print(f"\nTop {len(chunks)} results (pg_bm25) for: {args.query!r}\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c.chunk_id}  rrf={c.rrf_score:.4f}  section={c.section}")
        print(f"    {c.text[:120]}...")
        print()
