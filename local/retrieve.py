"""
Phase 2 — Hybrid retrieval: pgvector + Postgres FTS + Reciprocal Rank Fusion (RECOMMENDED).
Used by the full ChipMate pipeline.

Hybrid search combines two independent rankings and merges them using RRF.

Why RRF over simple weighted sum:
- Weighted sum (0.7 * vector_score + 0.3 * bm25_score) requires score normalization;
  cosine similarity and BM25 scores are on completely different scales.
- RRF only cares about rank position, not raw score — more stable across different queries.
- Formula: RRF(d) = sum over rankers of 1 / (k + rank_i(d)), where k=60 is standard.

Why Postgres FTS (tsvector) over pg_bm25/ParadeDB (alternative2):
- tsvector is built into every Postgres install — no extra Docker image or extension needed
- The retrieval quality difference is small for short keyword queries in a 10-20 doc corpus
- Use pg_bm25 if you have a large corpus (>100k chunks) where true BM25 ranking matters

Why this over rank_bm25 Python library (alternative1):
- tsvector search happens at the database level — only relevant rows cross the network
- rank_bm25 requires fetching ALL chunks from the DB to score them in Python, which
  is O(n) per query instead of O(log n) with a GIN index

Run:
    python -m local.retrieve --query "operating voltage TPS62902"
    python -m local.retrieve --query "What replaces TPS62902" --top-k 5
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
RRF_K = 60       # standard RRF smoothing constant; higher K = smoother ranking
DEFAULT_TOP_K = 5


class Chunk(NamedTuple):
    chunk_id: str
    component: str
    section: str
    page: int
    text: str
    rrf_score: float


def embed_query(query: str) -> list[float]:
    response = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def vector_search(query_embedding: list[float], top_k: int, conn) -> list[tuple[str, int]]:
    """
    Return list of (chunk_id, rank) ordered by cosine similarity.
    <-> is the pgvector cosine distance operator (lower = more similar).
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY embedding <-> %s::vector) AS rank
        FROM chunks
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k * 2),  # fetch 2x to give RRF more to merge
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def keyword_search(query: str, top_k: int, conn) -> list[tuple[str, int]]:
    """
    Return list of (chunk_id, rank) ordered by Postgres FTS relevance (ts_rank).
    plainto_tsquery converts plain text to a tsquery — no special syntax needed.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT chunk_id,
               ROW_NUMBER() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', %s)) DESC) AS rank
        FROM chunks
        WHERE fts_vector @@ plainto_tsquery('english', %s)
        LIMIT %s
        """,
        (query, query, top_k * 2),
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def reciprocal_rank_fusion(
    vector_ranks: list[tuple[str, int]],
    keyword_ranks: list[tuple[str, int]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists using RRF.
    Each document gets a score = sum of 1/(k + rank) across all lists it appears in.
    Documents only in one list still get a score (from that one list).
    """
    scores: dict[str, float] = {}
    for chunk_id, rank in vector_ranks:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, rank in keyword_ranks:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    # Sort descending by RRF score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fetch_chunks_by_ids(chunk_ids: list[str], conn) -> dict[str, dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, component, section, page, text FROM chunks WHERE chunk_id = ANY(%s)",
        (chunk_ids,),
    )
    return {row[0]: row for row in cur.fetchall()}


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K, conn=None) -> list[Chunk]:
    """
    Main entry point. Returns top_k Chunk objects ranked by RRF score.
    If conn is None, opens a new connection (for standalone use).
    """
    close_conn = conn is None
    if conn is None:
        conn = psycopg.connect(POSTGRES_URL)

    try:
        query_embedding = embed_query(query)
        vector_ranks = vector_search(query_embedding, top_k, conn)
        keyword_ranks = keyword_search(query, top_k, conn)
        merged = reciprocal_rank_fusion(vector_ranks, keyword_ranks)[:top_k]

        ranked_ids = [chunk_id for chunk_id, _ in merged]
        rrf_scores = {chunk_id: score for chunk_id, score in merged}
        chunk_data = fetch_chunks_by_ids(ranked_ids, conn)

        results = []
        for chunk_id in ranked_ids:
            if chunk_id in chunk_data:
                row = chunk_data[chunk_id]
                results.append(Chunk(
                    chunk_id=row[0],
                    component=row[1],
                    section=row[2],
                    page=row[3],
                    text=row[4],
                    rrf_score=rrf_scores[chunk_id],
                ))
        return results
    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid retrieval (pgvector + FTS + RRF)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    chunks = hybrid_search(args.query, args.top_k)
    print(f"\nTop {len(chunks)} results for: {args.query!r}\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c.chunk_id}  rrf={c.rrf_score:.4f}  component={c.component}  section={c.section}")
        print(f"    {c.text[:120]}...")
        print()
