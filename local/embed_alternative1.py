"""
Phase 2 — Embed with sentence-transformers (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

Model: all-mpnet-base-v2 (768-dim, runs in-process, no Ollama needed)

When you'd prefer this over embed.py:
- Ollama is not available or you want zero network calls for embedding
- You want fine-grained control over batching (sentence-transformers has native batch_size)
- You're comparing embedding quality across models directly in Python

Trade-offs vs Ollama (embed.py):
- REQUIRES the model to be downloaded into the Python process (~420MB for all-mpnet-base-v2)
- FASTER in practice once loaded — no HTTP overhead per call
- sentence-transformers.encode() handles batching natively (pass a list, get a list)
- Same 768-dim output so the Postgres schema is identical

Install extra dep: pip install sentence-transformers

Run:
    python -m local.embed_alternative1 --input data/chunks/TPS62902.json
"""

import json
import argparse
from pathlib import Path

import psycopg
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_URL = os.getenv(
    "POSTGRES_URL", "postgresql://chipmate:chipmate@localhost:5432/chipmate"
)
MODEL_NAME = "all-mpnet-base-v2"
EMBED_DIM = 768
BATCH_SIZE = 64  # sentence-transformers handles large batches efficiently


def load_model() -> SentenceTransformer:
    print(f"Loading {MODEL_NAME} (first run downloads ~420MB)...")
    return SentenceTransformer(MODEL_NAME)


def store_chunks(chunks: list[dict], model: SentenceTransformer, conn) -> int:
    texts = [c["text"] for c in chunks]

    # encode() is the sentence-transformers batching API:
    # pass all texts at once, it splits into batches internally
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

    rows = [
        (
            c["chunk_id"],
            c["component"],
            c.get("section", ""),
            c.get("page", -1),
            c["text"],
            emb.tolist(),  # numpy array -> Python list for psycopg3
        )
        for c, emb in zip(chunks, embeddings)
    ]

    cur = conn.cursor()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed with sentence-transformers")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    model = load_model()
    conn = psycopg.connect(POSTGRES_URL)

    with open(args.input) as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks...")
    count = store_chunks(chunks, model, conn)
    print(f"Stored {count} chunks")
    conn.close()
