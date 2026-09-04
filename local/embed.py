"""
Phase 2 — Generate embeddings and store chunks in pgvector (RECOMMENDED).
Used by the full ChipMate pipeline.

Model: Ollama nomic-embed-text (768-dim, runs fully locally, no API key)

Why this over sentence-transformers (alternative1):
- Zero extra Python model weight to download into the venv (~274MB pulled once into Ollama)
- Same Ollama server already running for the LLM — one less process to manage
- nomic-embed-text is designed for retrieval; quality is comparable to all-mpnet-base-v2

Schema (must be created first — see PLAN.md Phase 2):
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE chunks (
        chunk_id    TEXT PRIMARY KEY,
        component   TEXT NOT NULL,
        section     TEXT,
        page        INTEGER,
        text        TEXT NOT NULL,
        embedding   VECTOR(768),
        fts_vector  TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
    );
    CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX ON chunks USING GIN (fts_vector);

Run:
    python -m local.embed --input data/chunks/TPS62902.json
    python -m local.embed --input data/chunks/  # embed all JSON files in directory
"""

import json
import argparse
from pathlib import Path

import httpx
import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL", "postgresql://chipmate:chipmate@localhost:5432/chipmate"
)
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
BATCH_SIZE = 32  # embed this many chunks per Ollama call batch


def get_embedding(text: str) -> list[float]:
    """Call Ollama embeddings endpoint for a single text."""
    response = httpx.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Ollama does not have native batching, so we loop."""
    return [get_embedding(t) for t in texts]


def store_chunks(chunks: list[dict], conn) -> int:
    """Upsert chunks with embeddings into Postgres. Returns count inserted."""
    rows = []
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = get_embeddings_batch(texts)
        for chunk, emb in zip(batch, embeddings):
            rows.append((
                chunk["chunk_id"],
                chunk["component"],
                chunk.get("section", ""),
                chunk.get("page", -1),
                chunk["text"],
                emb,
            ))
        print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total} chunks...")

    cur = conn.cursor()
    # executemany is the psycopg3 equivalent of psycopg2's execute_values.
    # Each row is inserted individually; psycopg3 batches them efficiently.
    cur.executemany(
        """
        INSERT INTO chunks (chunk_id, component, section, page, text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def embed_file(json_path: str, conn) -> int:
    with open(json_path) as f:
        chunks = json.load(f)
    print(f"Embedding {len(chunks)} chunks from {json_path}")
    return store_chunks(chunks, conn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and store in pgvector")
    parser.add_argument("--input", required=True, help="Path to JSON file or directory of JSON files")
    args = parser.parse_args()

    conn = psycopg.connect(POSTGRES_URL)

    input_path = Path(args.input)
    if input_path.is_dir():
        json_files = list(input_path.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {input_path}")
        total = 0
        for jf in json_files:
            total += embed_file(str(jf), conn)
        print(f"\nTotal chunks stored: {total}")
    else:
        count = embed_file(str(input_path), conn)
        print(f"\nStored {count} chunks")

    conn.close()
