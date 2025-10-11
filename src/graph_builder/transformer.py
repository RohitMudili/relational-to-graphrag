"""
Data transformer - convert relational data to graph nodes and edges
"""
from typing import List, Dict, Any, Set
from .models import GraphNode, GraphEdge, GraphData, TransformationConfig
from .embeddings.embedding_service import EmbeddingService
from ..schema_analyzer.models import GraphOntology, NodeType, EdgeType


class DataTransformer:
    """Transform relational data into graph nodes and edges"""

    def __init__(
        self,
        ontology: GraphOntology,
        embedding_service: EmbeddingService,
        config: TransformationConfig
    ):
        """
        Initialize transformer

        Args:
            ontology: Graph ontology from Phase 1
            embedding_service: Service for generating embeddings
            config: Transformation configuration
        """
        self.ontology = ontology
        self.embedding_service = embedding_service
        self.config = config
        self.node_id_map: Dict[str, str] = {}  # Maps source PK to graph node ID

    def transform_nodes(
        self,
        node_type: NodeType,
        rows: List[Dict[str, Any]],
        primary_key_columns: List[str]
    ) -> List[GraphNode]:
        """
        Transform rows into graph nodes

        Args:
            node_type: Node type definition
            rows: Data rows from database
            primary_key_columns: Primary key column names

        Returns:
            List of GraphNode objects
        """
        nodes = []

        # Prepare texts for batch embedding
        texts_to_embed = []
        if self.config.generate_embeddings:
            for row in rows:
                text = self.embedding_service.create_node_text(
                    row,
                    fields=self.config.embedding_fields or None
                )
                texts_to_embed.append(text)

            print(f"   Generating embeddings for {len(texts_to_embed)} {node_type.label} nodes...")
            embeddings = self.embedding_service.generate_embeddings_batch(texts_to_embed)
        else:
            embeddings = [None] * len(rows)

        # Create nodes
        for i, row in enumerate(rows):
            # Generate unique node ID
            pk_values = [str(row.get(col, '')) for col in primary_key_columns]
            pk_key = ':'.join(pk_values)
            node_id = f"{node_type.source_table}:{pk_key}"

            # Store mapping for edge creation
            self.node_id_map[f"{node_type.source_table}:{pk_key}"] = node_id

            # Filter properties based on ontology
            properties = {}
            for prop in node_type.properties:
                if prop in row and row[prop] is not None:
                    properties[prop] = self._convert_value(row[prop])

            # Create node
            node = GraphNode(
                node_id=node_id,
                label=node_type.label,
                properties=properties,
                embedding=embeddings[i] if embeddings[i] else None,
                source_table=node_type.source_table,
                source_pk=pk_key
            )

            nodes.append(node)

        return nodes

    def transform_edges(
        self,
        edge_type: EdgeType,
        rows: List[Dict[str, Any]],
        from_pk_columns: List[str],
        to_pk_columns: List[str]
    ) -> List[GraphEdge]:
        """
        Transform rows into graph edges

        Args:
            edge_type: Edge type definition
            rows: Data rows (from junction table or FK relationships)
            from_pk_columns: PK columns for source node
            to_pk_columns: PK columns for target node

        Returns:
            List of GraphEdge objects
        """
        edges = []

        # Find source tables for from/to nodes
        from_table = self._find_source_table(edge_type.from_node)
        to_table = self._find_source_table(edge_type.to_node)

        for row in rows:
            # Build node IDs
            from_pk_values = [str(row.get(col, '')) for col in from_pk_columns]
            to_pk_values = [str(row.get(col, '')) for col in to_pk_columns]

            from_node_id = f"{from_table}:{':'.join(from_pk_values)}"
            to_node_id = f"{to_table}:{':'.join(to_pk_values)}"

            # Check if nodes exist in our mapping
            if from_node_id not in self.node_id_map or to_node_id not in self.node_id_map:
                continue  # Skip orphan edges

            # Get edge properties
            properties = {}
            for prop in edge_type.properties:
                if prop in row and row[prop] is not None:
                    properties[prop] = self._convert_value(row[prop])

            # Create edge
            edge = GraphEdge(
                from_node_id=self.node_id_map[from_node_id],
                to_node_id=self.node_id_map[to_node_id],
                relationship_type=edge_type.label,
                properties=properties,
                embedding=None  # Edge embeddings can be added later if needed
            )

            edges.append(edge)

        return edges

    def _find_source_table(self, node_label: str) -> str:
        """Find source table for a node label"""
        for node_type in self.ontology.node_types:
            if node_type.label == node_label:
                return node_type.source_table
        return node_label.lower()  # Fallback

    def _convert_value(self, value: Any) -> Any:
        """Convert database value to JSON-serializable format"""
        # Handle dates, decimals, etc.
        if value is None:
            return None

        # Convert datetime to string
        if hasattr(value, 'isoformat'):
            return value.isoformat()

        # Convert decimal to float
        if hasattr(value, '__float__'):
            try:
                return float(value)
            except:
                return str(value)

        # Default: return as-is or convert to string
        if isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    def deduplicate_nodes(self, nodes: List[GraphNode]) -> List[GraphNode]:
        """Remove duplicate nodes based on node_id"""
        seen: Set[str] = set()
        unique_nodes = []

        for node in nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_nodes.append(node)

        removed = len(nodes) - len(unique_nodes)
        if removed > 0:
            print(f"   Removed {removed} duplicate nodes")

        return unique_nodes

    def deduplicate_edges(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Remove duplicate edges"""
        seen: Set[tuple] = set()
        unique_edges = []

        for edge in edges:
            key = (edge.from_node_id, edge.to_node_id, edge.relationship_type)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)

        removed = len(edges) - len(unique_edges)
        if removed > 0:
            print(f"   Removed {removed} duplicate edges")

        return unique_edges
