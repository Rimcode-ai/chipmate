"""
Phase 1 — Structure-aware PDF chunking (RECOMMENDED).
Used by the full ChipMate pipeline.

Strategy: split on section headers, keep table rows grouped as a single chunk.

Why this over fixed-token (alternative1):
- Datasheets have explicit sections (Electrical Characteristics, Pin Description, etc.)
- A section split preserves semantic coherence — all voltage specs stay together
- Tables split across chunks cause retrieval failures; this keeps each table intact

Why this over sentence-boundary (alternative2):
- Sentence tokenizers misfire on datasheet prose (abbreviated units, table notation)
- Section boundaries are more reliable signal than sentence boundaries in technical PDFs

Run:
    python -m local.ingest --input data/datasheets/TPS62902.pdf
    python -m local.ingest --input data/datasheets/TPS62902.pdf --output data/chunks
"""

import re
import json
import argparse
from pathlib import Path

import pdfplumber

# Matches lines that look like section headers:
# - All-uppercase, optionally with spaces/hyphens/slashes
# - At least 4 characters (filters out single-word noise)
HEADER_RE = re.compile(r'^[A-Z][A-Z0-9\s\-/]{3,}$')
MIN_CHUNK_CHARS = 50


def is_section_header(line: str) -> bool:
    return bool(HEADER_RE.match(line.strip()))


def chunk_text_by_section(text: str, page_num: int, current_section: str) -> tuple[list[dict], str]:
    """
    Split raw page text into section-bounded chunks.
    Returns (list of partial chunks, updated current_section).
    """
    lines = text.split("\n")
    chunks = []
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if is_section_header(stripped) and buffer:
            body = " ".join(buffer).strip()
            if len(body) >= MIN_CHUNK_CHARS:
                chunks.append({"text": body, "section": current_section, "page": page_num})
            current_section = stripped
            buffer = []
        else:
            buffer.append(stripped)

    # flush remaining buffer
    if buffer:
        body = " ".join(buffer).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append({"text": body, "section": current_section, "page": page_num})

    return chunks, current_section


def extract_tables_as_chunks(page, page_num: int, current_section: str) -> list[dict]:
    """
    Extract each table on the page as one chunk.
    Rows are joined with ' | ' so the table structure is readable as text.
    """
    table_chunks = []
    for table in page.extract_tables() or []:
        rows = []
        for row in table:
            row_text = " | ".join(str(cell or "").strip() for cell in row)
            if row_text.replace("|", "").strip():
                rows.append(row_text)
        if rows:
            text = "\n".join(rows)
            if len(text) >= MIN_CHUNK_CHARS:
                table_chunks.append({"text": text, "section": current_section, "page": page_num})
    return table_chunks


def ingest_pdf(pdf_path: str) -> list[dict]:
    path = Path(pdf_path)
    # Use the filename stem as the component name (TPS62902.pdf -> TPS62902)
    component = path.stem.upper()
    all_chunks = []
    current_section = "GENERAL"
    chunk_index = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""

            text_chunks, current_section = chunk_text_by_section(page_text, page_num, current_section)
            table_chunks = extract_tables_as_chunks(page, page_num, current_section)

            for chunk in text_chunks + table_chunks:
                chunk["chunk_id"] = f"{component}_p{page_num:03d}_c{chunk_index:03d}"
                chunk["component"] = component
                chunk["source_file"] = path.name
                chunk_index += 1
                all_chunks.append(chunk)

    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structure-aware PDF chunker")
    parser.add_argument("--input", required=True, help="Path to PDF file")
    parser.add_argument("--output", default="data/chunks", help="Output directory for JSON")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    chunks = ingest_pdf(args.input)

    component = Path(args.input).stem.upper()
    out_path = Path(args.output) / f"{component}.json"
    with open(out_path, "w") as f:
        json.dump(chunks, f, indent=2)

    print(f"Wrote {len(chunks)} chunks -> {out_path}")
    print("Sample chunks:")
    for c in chunks[:3]:
        print(f"  {c['chunk_id']}  section={c['section']!r}  page={c['page']}  len={len(c['text'])}")
