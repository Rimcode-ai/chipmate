"""
Phase 0 verification script.
Run from chipmate/ root: python local/setup_verify.py
All checks must pass before starting Phase 1.
"""

import sys
import subprocess

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(label, fn):
    try:
        fn()
        print(f"  {PASS} {label}")
        results.append((label, True))
    except Exception as e:
        print(f"  {FAIL} {label} — {e}")
        results.append((label, False))


print("\n=== Phase 0 Verification ===\n")

# Python version
print("Python")
check("version >= 3.11", lambda: (
    None if sys.version_info >= (3, 11)
    else (_ for _ in ()).throw(RuntimeError(f"got {sys.version_info[:2]}"))
))

# Core imports
print("\nPython packages")
check("pdfplumber", lambda: __import__("pdfplumber"))
check("langgraph", lambda: __import__("langgraph"))
check("neo4j", lambda: __import__("neo4j"))
check("fastapi", lambda: __import__("fastapi"))
check("pybreaker", lambda: __import__("pybreaker"))
check("langfuse", lambda: __import__("langfuse"))
check("pgvector", lambda: __import__("pgvector"))
check("psycopg (psycopg3)", lambda: __import__("psycopg"))
check("redis", lambda: __import__("redis"))
check("locust", lambda: __import__("locust"))
check("boto3", lambda: __import__("boto3"))
check("aws_xray_sdk", lambda: __import__("aws_xray_sdk"))

# Docker services
print("\nDocker services")

def check_postgres():
    import psycopg
    conn = psycopg.connect(
        host="localhost", port=5432,
        dbname="chipmate", user="chipmate", password="chipmate",
        connect_timeout=3
    )
    conn.close()

def check_pgvector():
    import psycopg
    conn = psycopg.connect(
        host="localhost", port=5432,
        dbname="chipmate", user="chipmate", password="chipmate",
    )
    cur = conn.cursor()
    cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError("pgvector extension not installed — run: CREATE EXTENSION IF NOT EXISTS vector;")

def check_neo4j():
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "chipmate"))
    driver.verify_connectivity()
    driver.close()

def check_redis():
    import redis
    r = redis.Redis(host="localhost", port=6379)
    r.ping()

def check_langfuse():
    import httpx
    r = httpx.get("http://localhost:3000", timeout=3, follow_redirects=True)
    # Langfuse returns 200 on its landing/login page
    assert r.status_code in (200, 302, 307), f"status {r.status_code}"

check("postgres reachable", check_postgres)
check("pgvector extension installed", check_pgvector)
check("neo4j reachable (bolt://localhost:7687)", check_neo4j)
check("redis reachable", check_redis)
check("langfuse UI reachable (http://localhost:3000)", check_langfuse)

# Ollama
print("\nOllama")

def check_ollama():
    import httpx
    r = httpx.get("http://localhost:11434", timeout=3)
    assert r.status_code == 200

def check_mistral():
    import httpx, json
    r = httpx.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    assert any("mistral" in m for m in models), f"mistral not found in {models}"

def check_nomic():
    import httpx
    r = httpx.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    assert any("nomic" in m for m in models), f"nomic-embed-text not found in {models}"

check("ollama server responding", check_ollama)
check("mistral:7b pulled", check_mistral)
check("nomic-embed-text pulled", check_nomic)

# AWS profile
print("\nAWS (optional — skip if STS tokens expired)")

def check_aws():
    import boto3
    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    print(f"         Account: {identity['Account']}, ARN: {identity['Arn']}")

check("aws sts get-caller-identity", check_aws)

# Summary
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n=== {passed}/{total} checks passed ===")

if passed < total:
    print("\nFailed checks:")
    for label, ok in results:
        if not ok:
            print(f"  - {label}")
    print("\nFix failing checks before starting Phase 1.")
    sys.exit(1)
else:
    print("\nAll checks passed. Ready for Phase 1.\n")
