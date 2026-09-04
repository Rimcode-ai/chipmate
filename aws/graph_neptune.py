"""
AWS Phase 3 — Component relationship graph using Amazon Neptune + Gremlin.
Mirrors local/graph.py (Cypher) but uses Gremlin for Neptune.

Neptune requires access from within the VPC. Run this from:
  - An EC2 instance in the same VPC, OR
  - A Lambda function with VPC config pointing to the Neptune subnet

Interview talking point:
  "Neptune doesn't support direct internet access — it's VPC-secured, which mirrors
   production security. I ran traversals from a Lambda in the same VPC. The Gremlin
   API is the same as what I used locally with TinkerGraph — only the endpoint URL changes."

IMPORTANT: Neptune costs ~$0.10-0.20/hr. Create, test, DELETE same session.
See PLAN.md AWS Phase 3 for create/delete commands.

Run (from within VPC):
    python -m aws.graph_neptune --load
    python -m aws.graph_neptune --query TPS62902
"""

import argparse
import os
from dotenv import load_dotenv
from gremlin_python.driver import client as gremlin_client
from gremlin_python.driver import serializer

load_dotenv()

# Neptune Gremlin endpoint format: wss://<cluster>.neptune.amazonaws.com:8182/gremlin
NEPTUNE_ENDPOINT = os.getenv(
    "NEPTUNE_ENDPOINT",
    "wss://your-cluster.cluster-xxxxx.us-east-1.neptune.amazonaws.com:8182/gremlin"
)


def get_client():
    return gremlin_client.Client(
        NEPTUNE_ENDPOINT,
        "g",
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )


def submit(c, query: str, bindings: dict = None) -> list:
    return c.submit(query, bindings or {}).all().result()


def add_component(c, name: str, comp_type: str, vendor: str) -> None:
    submit(
        c,
        """g.V().has('component', 'name', name).fold()
           .coalesce(unfold(), addV('component').property('name', name))
           .property('type', type).property('vendor', vendor)""",
        {"name": name, "type": comp_type, "vendor": vendor},
    )


def add_relationship(c, from_name: str, rel_type: str, to_name: str) -> None:
    submit(
        c,
        f"""g.V().has('component','name', fromName).as('a')
            .V().has('component','name', toName).as('b')
            .coalesce(
                __.select('a').outE('{rel_type}').where(inV().as('b')),
                __.addE('{rel_type}').from_('a').to('b')
            )""",
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


TEST_DATA = [
    ("TPS62902", "buck_converter", "TI"),
    ("TPS62903", "buck_converter", "TI"),
    ("LM358",    "op_amp",         "TI"),
    ("TLV2372",  "op_amp",         "TI"),
]
TEST_EDGES = [
    ("TPS62902", "REPLACED_BY",    "TPS62903"),
    ("LM358",    "REPLACED_BY",    "TLV2372"),
    ("LM358",    "PIN_COMPATIBLE", "TLV2372"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neptune Gremlin graph")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--query")
    args = parser.parse_args()

    c = get_client()

    if args.load:
        for name, t, v in TEST_DATA:
            add_component(c, name, t, v)
            print(f"  Added: {name}")
        for a, rel, b in TEST_EDGES:
            add_relationship(c, a, rel, b)
            print(f"  Added: ({a})-[{rel}]->({b})")

    if args.query:
        replacements = find_replacements(c, args.query)
        pin_compat = find_pin_compatible(c, args.query)
        print(f"Replacements: {replacements}")
        print(f"Pin-compatible: {pin_compat}")

    c.close()
