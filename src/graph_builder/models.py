"""
Data models for graph building
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    """Represents a node to be created in the graph"""
    node_id: str  # Unique identifier
    label: str  # Node label (e.g., "Customer", "Product")
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source_table: Optional[str] = None
    source_pk: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "properties": self.properties,
            "has_embedding": self.embedding is not None,
            "source_table": self.source_table,
            "source_pk": self.source_pk,
        }


@dataclass
class GraphEdge:
    """Represents an edge/relationship to be created in the graph"""
    from_node_id: str
    to_node_id: str
    relationship_type: str  # Edge label (e.g., "PURCHASED", "BELONGS_TO")
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "relationship_type": self.relationship_type,
            "properties": self.properties,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class GraphData:
    """Container for all graph data to be loaded"""
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        self.edges.append(edge)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph data"""
        node_labels = {}
        edge_types = {}

        for node in self.nodes:
            node_labels[node.label] = node_labels.get(node.label, 0) + 1

        for edge in self.edges:
            edge_types[edge.relationship_type] = edge_types.get(edge.relationship_type, 0) + 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_labels": node_labels,
            "edge_types": edge_types,
            "nodes_with_embeddings": sum(1 for n in self.nodes if n.embedding),
            "edges_with_embeddings": sum(1 for e in self.edges if e.embedding),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "statistics": self.get_statistics(),
        }


@dataclass
class TransformationConfig:
    """Configuration for data transformation"""
    batch_size: int = 1000
    generate_embeddings: bool = True
    embedding_fields: List[str] = field(default_factory=list)  # Which fields to embed
    max_embedding_length: int = 8000  # Max characters for embedding
    skip_empty_nodes: bool = True
    deduplicate_nodes: bool = True
