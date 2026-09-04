"""
AWS Phase 2 — Embed with Amazon Titan Embeddings via Bedrock.
Mirrors local/embed.py but uses Bedrock instead of Ollama.

Key difference: Titan V2 produces 1024-dim vectors (not 768 like nomic-embed-text).
The Aurora schema uses VECTOR(1024) for the AWS track.

Interview talking point:
  "I used Amazon Titan Embeddings V2 via Bedrock for the AWS version. The embedding
   model is the same vendor as the inference model (Amazon), which simplifies IAM:
   one policy grants access to both. The dimension difference (768 vs 1024) means
   the two vector stores are not cross-compatible — intentionally, since they're
   separate deployment targets."

Run (requires Aurora to be running):
    python -m aws.embed_aws --input data/chunks/TPS62902.json
"""

import json
import argparse
from pathlib import Path

import boto3
import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

AURORA_URL = os.getenv(
    "AURORA_URL",
    "postgresql://chipmate_admin:ChipMate2024!@<your-aurora-endpoint>:5432/postgres"
)
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024  # Titan V2 dimension (different from local nomic-embed-text 768-dim)

bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


def get_titan_embedding(text: str) -> list[float]:
    """Call Titan Embeddings V2 via Bedrock."""
    response = bedrock.invoke_model(
        modelId=TITAN_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def store_chunks_aws(chunks: list[dict], conn) -> int:
    rows = []
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        embedding = get_titan_embedding(chunk["text"])
        rows.append((
            chunk["chunk_id"],
            chunk["component"],
            chunk.get("section", ""),
            chunk.get("page", -1),
            chunk["text"],
            embedding,
        ))
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i+1}/{total}...")

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
    parser = argparse.ArgumentParser(description="Embed with Titan (AWS)")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    conn = psycopg.connect(AURORA_URL)
    with open(args.input) as f:
        chunks = json.load(f)

    print(f"Embedding {len(chunks)} chunks with Titan V2...")
    count = store_chunks_aws(chunks, conn)
    print(f"Stored {count} chunks in Aurora")
    conn.close()
