"""
Phase 3 — Component relationship graph using Neo4j + Cypher (RECOMMENDED).
Used by the full ChipMate pipeline.

Why Cypher over Gremlin (alternative1):
- Cypher is Neo4j's native language — cleaner syntax, better tooling (Neo4j Browser)
- MATCH/CREATE/MERGE patterns are more readable than Gremlin's fluent chain API
- The AWS port needs Gremlin (Neptune doesn't support Cypher natively), so alternative1
  is the direct bridge to that; this file is the easier starting point

Node model:
    (:Component {name: str, type: str, vendor: str, datasheet_url: str})

Edge types used:
    REPLACED_BY     — part A is a newer replacement for part B
    PIN_COMPATIBLE  — parts share pinout, can substitute on board
    FUNCTIONALLY_SIMILAR — same function, may differ in specs/footprint

Why graph over vector search for relationships:
- "What replaces X?" is a structural fact, not a semantic similarity question.
  A vector search for "replacement for TPS62902" might return datasheet prose about
  the TPS62903 if its text happens to mention TPS62902, but that is coincidence.
  The graph gives a guaranteed, traversable answer.

Run:
    python -m local.graph --load          # load test data
    python -m local.graph --query TPS62902
    python -m local.graph --all-nodes
"""

import argparse
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "chipmate")

# Test data — the relationships we want to demo in the pipeline
TEST_COMPONENTS = [
    {"name": "TPS62902", "type": "buck_converter", "vendor": "TI"},
    {"name": "TPS62903", "type": "buck_converter", "vendor": "TI"},
    {"name": "LM358",    "type": "op_amp",          "vendor": "TI"},
    {"name": "TLV2372",  "type": "op_amp",          "vendor": "TI"},
    {"name": "ADS1115",  "type": "adc",              "vendor": "TI"},
    {"name": "INA219",   "type": "current_sensor",   "vendor": "TI"},
]

TEST_RELATIONSHIPS = [
    ("TPS62902", "REPLACED_BY",         "TPS62903"),
    ("LM358",    "REPLACED_BY",         "TLV2372"),
    ("LM358",    "PIN_COMPATIBLE",      "TLV2372"),
    ("TPS62902", "FUNCTIONALLY_SIMILAR","TPS62903"),
]


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def add_component(driver, name: str, type: str, vendor: str) -> None:
    """
    MERGE ensures we don't create duplicate nodes if this is run twice.
    SET updates properties if the node already exists.
    """
    with driver.session() as session:
        session.run(
            """
            MERGE (c:Component {name: $name})
            SET c.type = $type, c.vendor = $vendor
            """,
            name=name, type=type, vendor=vendor,
        )


def add_relationship(driver, from_name: str, rel_type: str, to_name: str) -> None:
    """
    MERGE both endpoints (idempotent) then MERGE the edge.
    Using MERGE on the edge prevents duplicate relationships.
    """
    with driver.session() as session:
        session.run(
            f"""
            MERGE (a:Component {{name: $from_name}})
            MERGE (b:Component {{name: $to_name}})
            MERGE (a)-[:{rel_type}]->(b)
            """,
            from_name=from_name, to_name=to_name,
        )


def find_replacements(driver, part_name: str) -> list[str]:
    """Return names of components that directly replace part_name."""
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Component {name: $name})-[:REPLACED_BY]->(b) RETURN b.name AS name",
            name=part_name,
        )
        return [record["name"] for record in result]


def find_pin_compatible(driver, part_name: str) -> list[str]:
    """Return names of components pin-compatible with part_name."""
    with driver.session() as session:
        result = session.run(
            "MATCH (a:Component {name: $name})-[:PIN_COMPATIBLE]->(b) RETURN b.name AS name",
            name=part_name,
        )
        return [record["name"] for record in result]


def find_all_related(driver, part_name: str, max_hops: int = 2) -> list[dict]:
    """
    Find all components reachable within max_hops hops via any relationship.
    Returns list of {name, relationship, distance}.
    Useful for the analysis node to provide full context.
    """
    with driver.session() as session:
        result = session.run(
            f"""
            MATCH path = (a:Component {{name: $name}})-[*1..{max_hops}]->(b)
            RETURN b.name AS name,
                   type(last(relationships(path))) AS relationship,
                   length(path) AS distance
            """,
            name=part_name,
        )
        return [
            {"name": r["name"], "relationship": r["relationship"], "distance": r["distance"]}
            for r in result
        ]


def get_all_nodes(driver) -> list[dict]:
    with driver.session() as session:
        result = session.run("MATCH (c:Component) RETURN c.name AS name, c.type AS type, c.vendor AS vendor")
        return [dict(r) for r in result]


def load_test_data(driver) -> None:
    print("Loading test components...")
    for c in TEST_COMPONENTS:
        add_component(driver, c["name"], c["type"], c["vendor"])
        print(f"  Added: {c['name']}")

    print("Loading test relationships...")
    for a, rel, b in TEST_RELATIONSHIPS:
        add_relationship(driver, a, rel, b)
        print(f"  Added: ({a})-[{rel}]->({b})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Component graph (Neo4j/Cypher)")
    parser.add_argument("--load", action="store_true", help="Load test data")
    parser.add_argument("--query", help="Query replacements for a component name")
    parser.add_argument("--all-nodes", action="store_true", help="List all nodes")
    args = parser.parse_args()

    driver = get_driver()
    driver.verify_connectivity()

    if args.load:
        load_test_data(driver)

    if args.query:
        replacements = find_replacements(driver, args.query)
        pin_compat = find_pin_compatible(driver, args.query)
        related = find_all_related(driver, args.query)
        print(f"\nReplacements for {args.query}: {replacements}")
        print(f"Pin-compatible with {args.query}: {pin_compat}")
        print(f"All related (2 hops): {related}")

    if args.all_nodes:
        nodes = get_all_nodes(driver)
        print(f"\nAll nodes ({len(nodes)}):")
        for n in nodes:
            print(f"  {n['name']}  type={n['type']}  vendor={n['vendor']}")

    driver.close()
