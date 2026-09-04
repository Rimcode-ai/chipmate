"""
Phase 2 — Hybrid retrieval with rank_bm25 Python library (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

Strategy: run vector search in Postgres, then score ALL chunks with BM25 in Python.

When you'd prefer this over retrieve.py:
- You cannot add extensions to Postgres (managed DB with locked config)
- You want to experiment with BM25 parameters (b, k1) without touching the DB

Trade-offs vs tsvector (retrieve.py):
- REQUIRES loading all chunk texts from the DB into Python memory
- BM25 scoring is then done in-process — correct BM25, but at O(n) memory cost
- For our 10k-chunk dataset this is fine; for 1M chunks this would be a problem
- rank_bm25 supports two variants: BM25 and BM25Okapi. Okapi is the more standard one.

Install: pip install rank_bm25

Run:
    python -m local.retrieve_alternative1 --query "operating voltage TPS62902"
"""

import argparse
from typing import NamedTuple

import psycopg
import httpx
from rank_bm25 import BM25Okapi
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


def load_all_chunks(conn) -> list[tuple]:
    """Fetch ALL chunks from Postgres. This is the O(n) cost of this approach."""
    cur = conn.cursor()
    cur.execute("SELECT chunk_id, component, section, page, text FROM chunks")
    return cur.fetchall()


def bm25_search(
    query: str,
    all_chunks: list[tuple],
    top_k: int,
) -> list[tuple[str, int]]:
    """
    Build a BM25Okapi index over all chunk texts, score the query, return (chunk_id, rank).
    BM25Okapi tokenizes on whitespace. For production you'd use a proper tokenizer.
    """
    tokenized_corpus = [row[4].lower().split() for row in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)  # array of scores, one per doc

    # Build (chunk_id, rank) sorted by score descending
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(all_chunks[i][0], rank + 1) for rank, (i, _) in enumerate(indexed[:top_k * 2])]


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


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K, conn=None) -> list[Chunk]:
    close_conn = conn is None
    if conn is None:
        conn = psycopg.connect(POSTGRES_URL)

    try:
        query_embedding = embed_query(query)
        all_chunks = load_all_chunks(conn)

        vector_ranks = vector_search(query_embedding, top_k, conn)
        bm25_ranks = bm25_search(query, all_chunks, top_k)
        merged = reciprocal_rank_fusion(vector_ranks, bm25_ranks)[:top_k]

        ranked_ids = {chunk_id for chunk_id, _ in merged}
        rrf_scores = {chunk_id: score for chunk_id, score in merged}
        chunk_map = {row[0]: row for row in all_chunks if row[0] in ranked_ids}

        results = []
        for chunk_id, score in merged:
            if chunk_id in chunk_map:
                row = chunk_map[chunk_id]
                results.append(Chunk(
                    chunk_id=row[0], component=row[1], section=row[2],
                    page=row[3], text=row[4], rrf_score=score,
                ))
        return results
    finally:
        if close_conn:
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid retrieval with rank_bm25")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    chunks = hybrid_search(args.query, args.top_k)
    print(f"\nTop {len(chunks)} results (rank_bm25) for: {args.query!r}\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] {c.chunk_id}  rrf={c.rrf_score:.4f}  section={c.section}")
        print(f"    {c.text[:120]}...")
        print()
