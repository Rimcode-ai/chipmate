"""
Phase 3 — Component relationship graph using Gremlin (ALTERNATIVE 1).
NOT used by the main pipeline. For learning/comparison only.

Why learn this alongside graph.py (Cypher):
- Amazon Neptune does NOT support Cypher natively (it supports Gremlin and openCypher,
  but the Python Gremlin driver is the standard path for Neptune)
- This file is the direct model for aws/graph_neptune.py
- Reading both side-by-side makes the syntax difference visible

Gremlin vs Cypher comparison (same operation):
  Cypher:  MATCH (a {name:'TPS62902'})-[:REPLACED_BY]->(b) RETURN b.name
  Gremlin: g.V().has('name','TPS62902').out('REPLACED_BY').values('name')

Gremlin traversal API:
  g.V()           — start from all vertices
  .has(k,v)       — filter vertices where property k == v
  .out('EDGE')    — follow outgoing edges of type EDGE
  .in_('EDGE')    — follow incoming edges
  .both('EDGE')   — follow either direction
  .values('name') — extract property 'name' from each result vertex
  .fold()         — collect all results into a list
  .coalesce(a,b)  — return first non-empty result of a or b (like COALESCE in SQL)
  .addV('label')  — create a new vertex with a label
  .addE('label')  — create a new edge
  .property(k,v)  — set a property on the current element

Install: pip install gremlinpython (already in requirements.txt)

Run (requires local Neo4j with Gremlin bolt support, or Neptune endpoint):
    python -m local.graph_alternative1 --load
    python -m local.graph_alternative1 --query TPS62902
"""

import argparse
from dotenv import load_dotenv
import os

from gremlin_python.driver import client as gremlin_client
from gremlin_python.driver import serializer

load_dotenv()

# Local Neo4j does support Gremlin via the bolt+ws protocol if configured.
# For a pure Gremlin endpoint (Neptune), the URL format is:
#   wss://<cluster>.neptune.amazonaws.com:8182/gremlin
# For local testing you can use TinkerGraph or a Gremlin-enabled Neo4j.
GREMLIN_URL = os.getenv("GREMLIN_URL", "ws://localhost:8182/gremlin")


def get_client():
    return gremlin_client.Client(
        GREMLIN_URL,
        "g",
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


def submit(c, query: str, bindings: dict = None) -> list:
    """Execute a Gremlin query and return all results."""
    result = c.submit(query, bindings or {})
    return result.all().result()


def add_component(c, name: str, comp_type: str, vendor: str) -> None:
    """
    coalesce(unfold(), addV(...)) is the Gremlin MERGE pattern:
    - fold() collects existing vertices with this name into a list (possibly empty)
    - coalesce tries to unfold() (return existing) first
    - if the list is empty, unfold fails, so coalesce falls through to addV (create new)
    """
    submit(
        c,
        """
        g.V().has('component', 'name', name).fold()
         .coalesce(
             unfold(),
             addV('component').property('name', name)
         )
         .property('type', type).property('vendor', vendor)
        """,
        {"name": name, "type": comp_type, "vendor": vendor},
    )


def add_relationship(c, from_name: str, rel_type: str, to_name: str) -> None:
    """
    from_('a') in Gremlin uses from_ with underscore to avoid Python keyword clash.
    """
    submit(
        c,
        f"""
        g.V().has('component','name', fromName).as('a')
         .V().has('component','name', toName).as('b')
         .coalesce(
             __.select('a').outE('{rel_type}').where(inV().as('b')),
             __.addE('{rel_type}').from_('a').to('b')
         )
        """,
        {"fromName": from_name, "toName": to_name},
    )


def find_replacements(c, part_name: str) -> list[str]:
    return submit(
        c,
        "g.V().has('component','name', name).out('REPLACED_BY').values('name')",
        {"name": part_name},
    )


def find_pin_compatible(c, part_name: str) -> list[str]:
    return submit(
        c,
        "g.V().has('component','name', name).out('PIN_COMPATIBLE').values('name')",
        {"name": part_name},
    )


def find_all_related(c, part_name: str, max_hops: int = 2) -> list:
    """Gremlin repeat().times() is the multi-hop traversal pattern."""
    return submit(
        c,
        f"g.V().has('component','name', name).repeat(out()).times({max_hops}).values('name').dedup()",
        {"name": part_name},
    )


TEST_COMPONENTS = [
    ("TPS62902", "buck_converter", "TI"),
    ("TPS62903", "buck_converter", "TI"),
    ("LM358",    "op_amp",         "TI"),
    ("TLV2372",  "op_amp",         "TI"),
]

TEST_RELATIONSHIPS = [
    ("TPS62902", "REPLACED_BY",    "TPS62903"),
    ("LM358",    "REPLACED_BY",    "TLV2372"),
    ("LM358",    "PIN_COMPATIBLE", "TLV2372"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component graph (Gremlin)")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--query")
    args = parser.parse_args()

    c = get_client()

    if args.load:
        for name, t, v in TEST_COMPONENTS:
            add_component(c, name, t, v)
            print(f"  Added: {name}")
        for a, rel, b in TEST_RELATIONSHIPS:
            add_relationship(c, a, rel, b)
            print(f"  Added: ({a})-[{rel}]->({b})")

    if args.query:
        replacements = find_replacements(c, args.query)
        pin_compat = find_pin_compatible(c, args.query)
        print(f"\nReplacements for {args.query}: {replacements}")
        print(f"Pin-compatible: {pin_compat}")

    c.close()
