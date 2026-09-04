"""
Phase 1 — Sentence-boundary chunking (ALTERNATIVE 2).
NOT used by the main pipeline. For learning/comparison only.

Strategy: tokenize text into sentences, group sentences until a token budget is reached,
then flush as one chunk. No overlap; each sentence belongs to exactly one chunk.

When you'd prefer this over ingest.py:
- Documents are prose-heavy (research papers, manuals) where sentences are the natural unit
- You want no repeated content between chunks (vs. the overlap in alternative1)
- NLP pipelines downstream operate on sentence-granularity

Trade-offs vs structure-aware (ingest.py):
- WORKS POORLY on datasheets: table cells, unit abbreviations (V, mA, kΩ) confuse sentence
  tokenizers. "Min. 3.3V. Max. 5V." gets split into multiple sentences incorrectly.
- USEFUL to know: in research/paper RAG this is often the preferred approach.

Requires: pip install nltk
First run: python -c "import nltk; nltk.download('punkt_tab')"

Run:
    python -m local.ingest_alternative2 --input data/datasheets/TPS62902.pdf
"""

import json
import argparse
from pathlib import Path

import pdfplumber
import nltk

# Download punkt sentence tokenizer on first use
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

DEFAULT_CHUNK_SENTENCES = 5   # group this many sentences per chunk
APPROX_TOKENS_PER_SENTENCE = 20


def extract_sentences_per_page(pdf_path: str) -> list[tuple[int, str]]:
    """Return list of (page_num, sentence) tuples."""
    result = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Replace newlines with spaces so cross-line sentences are joined
            text = text.replace("\n", " ")
            sentences = nltk.sent_tokenize(text)
            for s in sentences:
                s = s.strip()
                if len(s) > 20:
                    result.append((page_num, s))
    return result


def group_sentences(
    sentence_tuples: list[tuple[int, str]],
    sentences_per_chunk: int,
) -> list[dict]:
    """Group consecutive sentences into chunks, tracking the page of the first sentence."""
    chunks = []
    for i in range(0, len(sentence_tuples), sentences_per_chunk):
        group = sentence_tuples[i : i + sentences_per_chunk]
        page_num = group[0][0]  # page of the first sentence in the group
        text = " ".join(s for _, s in group)
        if text.strip():
            chunks.append({"text": text, "page": page_num})
    return chunks


def ingest_pdf(pdf_path: str, sentences_per_chunk: int = DEFAULT_CHUNK_SENTENCES) -> list[dict]:
    path = Path(pdf_path)
    component = path.stem.upper()

    sentence_tuples = extract_sentences_per_page(pdf_path)
    raw_chunks = group_sentences(sentence_tuples, sentences_per_chunk)

    chunks = []
    for i, c in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": f"{component}_sent_{i:04d}",
            "component": component,
            "section": "UNKNOWN",
            "page": c["page"],
            "text": c["text"],
            "source_file": path.name,
        })
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentence-boundary chunker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/chunks")
    parser.add_argument("--sentences-per-chunk", type=int, default=DEFAULT_CHUNK_SENTENCES)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    chunks = ingest_pdf(args.input, args.sentences_per_chunk)

    component = Path(args.input).stem.upper()
    out_path = Path(args.output) / f"{component}_sentence.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Wrote {len(chunks)} chunks -> {out_path}")
    print("Sample chunks:")
    for c in chunks[:3]:
        print(f"  {c['chunk_id']}  page={c['page']}  len={len(c['text'])}")
