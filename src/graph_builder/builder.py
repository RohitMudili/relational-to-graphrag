"""
Main graph builder - orchestrates the entire ETL pipeline
"""
import json
from pathlib import Path
from typing import Optional
from .extractor import DataExtractor
from .transformer import DataTransformer
from .embeddings.embedding_service import EmbeddingService
from .loaders.neo4j_loader import Neo4jLoader
from .models import GraphData, TransformationConfig
from ..schema_analyzer.models import GraphOntology


class GraphBuilder:
    """
    Main graph builder that orchestrates:
    1. Data extraction from PostgreSQL
    2. Transformation to graph format
    3. Embedding generation
    4. Loading into Neo4j
    """

    def __init__(
        self,
        postgres_conn_string: str,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize graph builder

        Args:
            postgres_conn_string: PostgreSQL connection string
            neo4j_uri: Neo4j URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key
            embedding_model: OpenAI embedding model
        """
        self.extractor = DataExtractor(postgres_conn_string)
        self.embedding_service = EmbeddingService(openai_api_key, embedding_model)
        self.loader = Neo4jLoader(neo4j_uri, neo4j_user, neo4j_password)

    def build_graph(
        self,
        ontology: GraphOntology,
        config: Optional[TransformationConfig] = None,
        clear_existing: bool = True
    ) -> GraphData:
        """
        Build complete knowledge graph

        Args:
            ontology: Graph ontology from Phase 1
            config: Transformation configuration
            clear_existing: Whether to clear existing graph data

        Returns:
            GraphData with all nodes and edges
        """
        if config is None:
            config = TransformationConfig()

        # Initialize transformer
        transformer = DataTransformer(ontology, self.embedding_service, config)

        print("=" * 80)
        print("GRAPH BUILDER - PHASE 2: ETL PIPELINE")
        print("=" * 80)

        # Step 1: Clear existing data if requested
        if clear_existing:
            print("\n🗑️  Step 1: Clearing existing graph data...")
            self.loader.clear_database()

        # Step 2: Extract and transform nodes
        print(f"\n📦 Step 2: Extracting and transforming nodes...")
        all_nodes = []

        for node_type in ontology.node_types:
            print(f"\n   Processing {node_type.label} (from {node_type.source_table})...")

            # Get primary key columns
            pk_columns = self.extractor.get_primary_key_columns(node_type.source_table)

            # Extract data in batches
            node_count = 0
            for batch in self.extractor.extract_nodes_data(node_type, config.batch_size):
                nodes = transformer.transform_nodes(node_type, batch, pk_columns)
                all_nodes.extend(nodes)
                node_count += len(nodes)

            print(f"   ✓ Transformed {node_count} {node_type.label} nodes")

        # Deduplicate
        if config.deduplicate_nodes:
            print(f"\n   Deduplicating nodes...")
            all_nodes = transformer.deduplicate_nodes(all_nodes)

        print(f"\n   ✓ Total nodes: {len(all_nodes)}")

        # Step 3: Load nodes into Neo4j
        print(f"\n💾 Step 3: Loading nodes into Neo4j...")
        self.loader.load_nodes(all_nodes, batch_size=config.batch_size)

        # Step 4: Create indexes
        print(f"\n🔍 Step 4: Creating indexes...")
        node_labels = list(set(node.label for node in all_nodes))
        self.loader.create_indexes(node_labels)

        # Create vector indexes for embeddings
        if config.generate_embeddings:
            for label in node_labels:
                self.loader.create_vector_index(label)

        # Step 5: Extract and transform edges
        print(f"\n🔗 Step 5: Extracting and transforming relationships...")
        all_edges = []

        for edge_type in ontology.edge_types:
            print(f"\n   Processing {edge_type.from_node} -{edge_type.label}-> {edge_type.to_node}...")

            # Extract edge data
            edge_count = 0
            try:
                for batch_data, from_pk_cols, to_pk_cols in self.extractor.extract_edge_data(
                    edge_type, ontology, config.batch_size
                ):
                    edges = transformer.transform_edges(edge_type, batch_data, from_pk_cols, to_pk_cols)
                    all_edges.extend(edges)
                    edge_count += len(edges)

                print(f"   ✓ Extracted {edge_count} {edge_type.label} relationships")

            except Exception as e:
                print(f"   ⚠️  Error extracting edges: {e}")

        # Deduplicate edges
        if all_edges and config.deduplicate_nodes:  # Reuse config flag
            print(f"\n   Deduplicating edges...")
            all_edges = transformer.deduplicate_edges(all_edges)

        print(f"\n   ✓ Total edges: {len(all_edges)}")

        # Step 6: Load edges (if any)
        if all_edges:
            print(f"\n💾 Step 6: Loading edges into Neo4j...")
            self.loader.load_edges(all_edges, batch_size=config.batch_size)

        # Step 7: Get final statistics
        print(f"\n📊 Step 7: Gathering statistics...")
        stats = self.loader.get_statistics()

        print("\n" + "=" * 80)
        print("✅ GRAPH BUILD COMPLETE!")
        print("=" * 80)
        print(f"\n📈 Final Statistics:")
        print(f"   Total Nodes: {stats['total_nodes']}")
        print(f"   Total Edges: {stats['total_edges']}")

        print(f"\n   Nodes by Label:")
        for label, count in sorted(stats['nodes_by_label'].items(), key=lambda x: x[1], reverse=True):
            print(f"      {label}: {count}")

        if stats['edges_by_type']:
            print(f"\n   Edges by Type:")
            for rel_type, count in sorted(stats['edges_by_type'].items(), key=lambda x: x[1], reverse=True):
                print(f"      {rel_type}: {count}")

        # Create graph data object
        graph_data = GraphData(nodes=all_nodes, edges=all_edges)

        return graph_data

    def close(self):
        """Close all connections"""
        self.loader.close()


def build_from_ontology_file(
    ontology_file: str,
    postgres_conn_string: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    openai_api_key: str,
    generate_embeddings: bool = True,
    clear_existing: bool = True
) -> GraphData:
    """
    Convenience function to build graph from ontology file

    Args:
        ontology_file: Path to graph_ontology.json from Phase 1
        postgres_conn_string: PostgreSQL connection
        neo4j_uri: Neo4j URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        openai_api_key: OpenAI API key
        generate_embeddings: Whether to generate embeddings
        clear_existing: Whether to clear existing data

    Returns:
        GraphData object
    """
    # Load ontology
    with open(ontology_file, 'r') as f:
        ontology_dict = json.load(f)

    # Reconstruct ontology object (simplified)
    from ..schema_analyzer.models import NodeType, EdgeType

    node_types = [
        NodeType(
            label=nt['label'],
            source_table=nt['source_table'],
            properties=nt['properties'],
            description=nt.get('description')
        )
        for nt in ontology_dict['node_types']
    ]

    edge_types = [
        EdgeType(
            label=et['label'],
            from_node=et['from_node'],
            to_node=et['to_node'],
            source_relationship=et.get('source_relationship'),
            properties=et.get('properties', []),
            description=et.get('description')
        )
        for et in ontology_dict['edge_types']
    ]

    ontology = GraphOntology(node_types=node_types, edge_types=edge_types)

    # Build graph
    builder = GraphBuilder(
        postgres_conn_string,
        neo4j_uri,
        neo4j_user,
        neo4j_password,
        openai_api_key
    )

    config = TransformationConfig(generate_embeddings=generate_embeddings)

    try:
        graph_data = builder.build_graph(ontology, config, clear_existing)
        return graph_data
    finally:
        builder.close()
