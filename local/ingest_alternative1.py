"""
Phase 1 — Fixed-token chunking with overlap (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

Strategy: split text into windows of N tokens with M-token overlap between windows.

When you'd prefer this over ingest.py:
- Your documents lack reliable structural markers (e.g., raw text, emails, blog posts)
- You need strict chunk size guarantees for embedding models with hard token limits
- You want the simplest possible implementation to start

Trade-offs vs structure-aware (ingest.py):
- LOSES context at boundaries: a table row can split across two chunks
- GAINS simplicity: no regex, no page-object manipulation
- The overlap (e.g., 50 tokens) partially compensates for split context but is not a perfect fix

Run:
    python -m local.ingest_alternative1 --input data/datasheets/TPS62902.pdf
"""

import re
import json
import argparse
from pathlib import Path

import pdfplumber

DEFAULT_CHUNK_TOKENS = 300   # target tokens per chunk
DEFAULT_OVERLAP_TOKENS = 50  # tokens repeated at the start of the next chunk
APPROX_CHARS_PER_TOKEN = 4   # rough approximation; good enough for this use case


def tokenize_approx(text: str) -> list[str]:
    """Split on whitespace. Not a real tokenizer — sufficient for size estimation."""
    return text.split()


def chunk_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = tokenize_approx(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = " ".join(tokens[start:end])
        if chunk.strip():
            chunks.append(chunk)
        # Move forward by chunk_size minus overlap so the next chunk repeats the tail
        start += chunk_size - overlap
    return chunks


def extract_full_text(pdf_path: str) -> str:
    """Extract all text from all pages as one string."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n".join(pages)


def ingest_pdf(
    pdf_path: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[dict]:
    path = Path(pdf_path)
    component = path.stem.upper()

    full_text = extract_full_text(pdf_path)
    raw_chunks = chunk_by_tokens(full_text, chunk_tokens, overlap_tokens)

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": f"{component}_chunk_{i:04d}",
            "component": component,
            "section": "UNKNOWN",  # fixed-token chunking has no section awareness
            "page": -1,            # page info is lost when we flatten all pages to one string
            "text": text,
            "source_file": path.name,
        })
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-token chunker with overlap")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/chunks")
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    chunks = ingest_pdf(args.input, args.chunk_tokens, args.overlap_tokens)

    component = Path(args.input).stem.upper()
    out_path = Path(args.output) / f"{component}_fixed_token.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Wrote {len(chunks)} chunks -> {out_path}")
    print(f"Settings: chunk_tokens={args.chunk_tokens}, overlap_tokens={args.overlap_tokens}")
    print("Sample chunks:")
    for c in chunks[:3]:
        print(f"  {c['chunk_id']}  len={len(c['text'])} chars")
