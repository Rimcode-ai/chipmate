"""
AWS Phase 1 — Lambda function for S3-triggered PDF chunking.
Mirrors local/ingest.py but triggered by S3 upload event instead of CLI.

Event-driven pattern: PDF upload to S3 -> S3 sends event -> Lambda runs chunker
-> Lambda writes chunks JSON back to S3 (chunks/<key>.json)

Interview talking point:
  "I used S3 event notifications to trigger chunking automatically on upload.
   The Lambda reads the PDF bytes directly from S3 into memory — no EFS, no tmp disk.
   Chunks are written back to the same bucket under a chunks/ prefix."

Deploy:
    See PLAN.md AWS Phase 1 for full packaging and deploy commands.
"""

import re
import json
import io
import boto3
import pdfplumber

s3 = boto3.client("s3")

HEADER_RE = re.compile(r'^[A-Z][A-Z0-9\s\-/]{3,}$')
MIN_CHUNK_CHARS = 50


def is_section_header(line: str) -> bool:
    return bool(HEADER_RE.match(line.strip()))


def chunk_text(text: str, page_num: int, current_section: str):
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
    if buffer:
        body = " ".join(buffer).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append({"text": body, "section": current_section, "page": page_num})
    return chunks, current_section


def lambda_handler(event, context):
    """
    S3 event structure:
    {
      "Records": [{
        "s3": {
          "bucket": {"name": "chipmate-datasheets-xxx"},
          "object": {"key": "datasheets/TPS62902.pdf"}
        }
      }]
    }
    """
    record = event["Records"][0]["s3"]
    bucket = record["bucket"]["name"]
    key = record["object"]["key"]

    print(f"Processing: s3://{bucket}/{key}")

    # Read PDF bytes from S3 into memory
    obj = s3.get_object(Bucket=bucket, Key=key)
    pdf_bytes = io.BytesIO(obj["Body"].read())

    component = key.split("/")[-1].replace(".pdf", "").upper()
    all_chunks = []
    current_section = "GENERAL"
    chunk_index = 0

    with pdfplumber.open(pdf_bytes) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            page_chunks, current_section = chunk_text(text, page_num, current_section)
            for chunk in page_chunks:
                chunk["chunk_id"] = f"{component}_p{page_num:03d}_c{chunk_index:03d}"
                chunk["component"] = component
                chunk["source_file"] = key
                chunk_index += 1
                all_chunks.append(chunk)

    # Write chunks JSON back to S3
    output_key = f"chunks/{key.split('/')[-1]}.json"
    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(all_chunks),
        ContentType="application/json",
    )

    print(f"Wrote {len(all_chunks)} chunks to s3://{bucket}/{output_key}")
    return {"statusCode": 200, "chunks_created": len(all_chunks)}
