"""
noia/knowledge_graph/kg_builder.py
────────────────────────────────────
Lightweight operational knowledge graph builder for NOIA.

Extracts typed entities and relations from ingested telecom operational
documents and constructs a NetworkX directed graph that augments retrieval
context for root cause analysis (RCA) queries with topological reasoning.

Entity types:
  - NetworkNode   (PE-01, P-03, etc.)
  - FaultType     (BGP session drop, fiber cut, etc.)
  - Protocol      (BGP, OSPF, MPLS, etc.)
  - Region        (Lagos, Accra, etc.)
  - Runbook       (RB-0042, etc.)
  - Incident      (INC-000123, etc.)

Relation types:
  - CAUSED_BY     (Incident → FaultType)
  - RESOLVED_BY   (Incident → Runbook)
  - AFFECTS       (FaultType → NetworkNode)
  - LOCATED_IN    (NetworkNode → Region)
  - CO_OCCURS     (FaultType ↔ FaultType in same incident)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# ── Entity & relation definitions ─────────────────────────────────────────────

ENTITY_PATTERNS: dict[str, re.Pattern] = {
    "NetworkNode": re.compile(r"\b(P[E]?-\d{2}|CE-\d{2}|P-\d{2})\b"),
    "Incident":    re.compile(r"\b(INC-\d{4,10})\b"),
    "Runbook":     re.compile(r"\b(RB-\d{4})\b"),
    "Region":      re.compile(
        r"\b(Lagos|Accra|Nairobi|Cairo|Dakar|Abidjan|Kigali|"
        r"Dar es Salaam|Johannesburg|Addis Ababa)\b",
        re.IGNORECASE,
    ),
    "Protocol":    re.compile(
        r"\b(BGP|OSPF|IS-IS|MPLS|LDP|RSVP-TE|BFD|VRRP|OTN|DWDM)\b"
    ),
    "FaultType":   re.compile(
        r"(route flap|fiber cut|optical power degradation|hardware failure|"
        r"CPU overload|memory exhaustion|interface CRC errors|BGP session drop|"
        r"OSPF adjacency loss|MPLS label exhaustion|power supply failure|"
        r"packet loss|latency anomaly)",
        re.IGNORECASE,
    ),
}

MAINTENANCE_ID_RE = re.compile(r"\b(CHG-\d{5})\b")
SLA_ID_RE         = re.compile(r"\b(SLA-\d{4}|SCH-\d{4})\b")


class GraphEntity(NamedTuple):
    entity_id:   str
    entity_type: str
    source_doc:  str


class GraphRelation(NamedTuple):
    source:        str
    relation_type: str
    target:        str
    source_doc:    str


# ── KG builder ────────────────────────────────────────────────────────────────

class KnowledgeGraphBuilder:
    """
    Builds and queries a telecom operational knowledge graph.

    The graph is backed by NetworkX and serialised to JSON for persistence.

    Parameters
    ----------
    graph_path : str | Path
        Path to persist the graph as JSON.
    """

    def __init__(self, graph_path: str | Path = "data/kg.json") -> None:
        self.graph_path = Path(graph_path)
        try:
            import networkx as nx  # noqa: PLC0415
            self._nx = nx
            self._G: "nx.DiGraph" = nx.DiGraph()
        except ImportError:
            logger.warning("networkx not installed — KG features disabled.")
            self._nx = None
            self._G  = None

    # ── Entity extraction ─────────────────────────────────────────────────────

    def _extract_entities(self, text: str, doc_id: str) -> list[GraphEntity]:
        entities: list[GraphEntity] = []
        for entity_type, pattern in ENTITY_PATTERNS.items():
            for match in pattern.finditer(text):
                eid = match.group(0).strip().upper()
                entities.append(GraphEntity(eid, entity_type, doc_id))
        return entities

    def _extract_relations(
        self,
        text: str,
        doc_id: str,
        entities: list[GraphEntity],
    ) -> list[GraphRelation]:
        """Infer relations from co-occurrence within the same document."""
        relations: list[GraphRelation] = []
        by_type: dict[str, list[str]] = {}
        for ent in entities:
            by_type.setdefault(ent.entity_type, []).append(ent.entity_id)

        # Incident CAUSED_BY FaultType
        for inc in by_type.get("Incident", []):
            for fault in by_type.get("FaultType", []):
                relations.append(GraphRelation(inc, "CAUSED_BY", fault, doc_id))

        # Incident RESOLVED_BY Runbook
        for inc in by_type.get("Incident", []):
            for rb in by_type.get("Runbook", []):
                relations.append(GraphRelation(inc, "RESOLVED_BY", rb, doc_id))

        # FaultType AFFECTS NetworkNode
        for fault in by_type.get("FaultType", []):
            for node in by_type.get("NetworkNode", []):
                relations.append(GraphRelation(fault, "AFFECTS", node, doc_id))

        # NetworkNode LOCATED_IN Region
        for node in by_type.get("NetworkNode", []):
            for region in by_type.get("Region", []):
                relations.append(GraphRelation(node, "LOCATED_IN", region, doc_id))

        # FaultType CO_OCCURS FaultType (within same doc)
        faults = by_type.get("FaultType", [])
        for i, f1 in enumerate(faults):
            for f2 in faults[i + 1:]:
                if f1 != f2:
                    relations.append(GraphRelation(f1, "CO_OCCURS", f2, doc_id))

        return relations

    # ── Graph population ──────────────────────────────────────────────────────

    def add_document(self, text: str, doc_id: str, source_type: str) -> int:
        """
        Extract entities and relations from a document and add to the graph.

        Returns the number of new nodes added.
        """
        if self._G is None:
            return 0

        entities  = self._extract_entities(text, doc_id)
        relations = self._extract_relations(text, doc_id, entities)

        added = 0
        for ent in entities:
            if not self._G.has_node(ent.entity_id):
                self._G.add_node(
                    ent.entity_id,
                    entity_type=ent.entity_type,
                    source_docs=[doc_id],
                )
                added += 1
            else:
                self._G.nodes[ent.entity_id]["source_docs"].append(doc_id)

        for rel in relations:
            self._G.add_edge(
                rel.source, rel.target,
                relation=rel.relation_type,
                source_doc=rel.source_doc,
            )

        logger.debug(
            "Added doc %s: %d entities, %d relations.", doc_id, len(entities), len(relations)
        )
        return added

    # ── Graph queries ─────────────────────────────────────────────────────────

    def get_related_incidents(self, fault_type: str, max_results: int = 5) -> list[str]:
        """
        Find incidents related to a given fault type via CAUSED_BY edges.
        """
        if self._G is None:
            return []
        fault_id = fault_type.upper()
        related  = []
        for pred in self._G.predecessors(fault_id):
            edge_data = self._G[pred][fault_id]
            if edge_data.get("relation") == "CAUSED_BY":
                related.append(pred)
        return related[:max_results]

    def get_affected_nodes(self, fault_type: str, max_results: int = 10) -> list[str]:
        """Return NetworkNode entities affected by a given fault type."""
        if self._G is None:
            return []
        fault_id = fault_type.upper()
        affected = []
        if fault_id in self._G:
            for successor in self._G.successors(fault_id):
                edge = self._G[fault_id][successor]
                if edge.get("relation") == "AFFECTS":
                    affected.append(successor)
        return affected[:max_results]

    def get_co_occurring_faults(self, fault_type: str) -> list[str]:
        """Return fault types that frequently co-occur with the given fault."""
        if self._G is None:
            return []
        fault_id = fault_type.upper()
        co_faults = []
        if fault_id in self._G:
            for neighbour in list(self._G.successors(fault_id)) + list(self._G.predecessors(fault_id)):
                ndata = self._G.nodes.get(neighbour, {})
                if ndata.get("entity_type") == "FaultType" and neighbour != fault_id:
                    co_faults.append(neighbour)
        return list(set(co_faults))

    def context_for_rca(self, fault_description: str) -> dict:
        """
        Build a structured KG context snippet for RCA augmentation.

        Returns a dict with related incidents, affected nodes, and co-occurring
        faults extracted from the graph, to be merged into the RCA prompt.
        """
        if self._G is None:
            return {}

        # Extract fault types from description
        fault_matches = ENTITY_PATTERNS["FaultType"].findall(fault_description)
        node_matches  = ENTITY_PATTERNS["NetworkNode"].findall(fault_description)

        ctx: dict = {
            "graph_related_incidents": [],
            "graph_affected_nodes":    [],
            "graph_co_occurring_faults": [],
        }

        for fault in fault_matches:
            ctx["graph_related_incidents"].extend(self.get_related_incidents(fault))
            ctx["graph_affected_nodes"].extend(self.get_affected_nodes(fault))
            ctx["graph_co_occurring_faults"].extend(self.get_co_occurring_faults(fault))

        for node in node_matches:
            # Find faults that affect this node
            preds = [
                p for p in self._G.predecessors(node.upper())
                if self._G[p][node.upper()].get("relation") == "AFFECTS"
            ]
            ctx["graph_affected_nodes"].extend(preds)

        # Deduplicate
        for key in ctx:
            ctx[key] = list(dict.fromkeys(ctx[key]))

        return ctx

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist the graph to JSON."""
        if self._G is None:
            return
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": [
                {"id": n, **self._G.nodes[n]} for n in self._G.nodes
            ],
            "edges": [
                {"source": u, "target": v, **self._G[u][v]}
                for u, v in self._G.edges
            ],
        }
        with self.graph_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info("KG saved: %d nodes, %d edges → %s",
                    self._G.number_of_nodes(), self._G.number_of_edges(), self.graph_path)

    def load(self) -> None:
        """Load a persisted graph from JSON."""
        if self._G is None or not self.graph_path.exists():
            return
        with self.graph_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        for node in data.get("nodes", []):
            nid = node.pop("id")
            self._G.add_node(nid, **node)
        for edge in data.get("edges", []):
            src = edge.pop("source")
            tgt = edge.pop("target")
            self._G.add_edge(src, tgt, **edge)
        logger.info("KG loaded: %d nodes, %d edges from %s",
                    self._G.number_of_nodes(), self._G.number_of_edges(), self.graph_path)

    def stats(self) -> dict:
        if self._G is None:
            return {"nodes": 0, "edges": 0}
        return {"nodes": self._G.number_of_nodes(), "edges": self._G.number_of_edges()}
