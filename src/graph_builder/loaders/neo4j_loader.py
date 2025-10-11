"""
Neo4j loader - bulk load graph data into Neo4j
"""
from typing import List
from neo4j import GraphDatabase
from ..models import GraphNode, GraphEdge, GraphData


class Neo4jLoader:
    """Load graph data into Neo4j database"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j loader

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the database connection"""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        with self.driver.session() as session:
            print("   Clearing existing graph data...")
            session.run("MATCH (n) DETACH DELETE n")
            print("   ✓ Database cleared")

    def create_indexes(self, node_labels: List[str]):
        """
        Create indexes for better performance

        Args:
            node_labels: List of node labels to index
        """
        with self.driver.session() as session:
            print("   Creating indexes...")

            for label in node_labels:
                # Create index on node_id for faster lookups
                try:
                    session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.node_id)")
                    print(f"   ✓ Index created for {label}.node_id")
                except Exception as e:
                    print(f"   ⚠️  Index creation warning for {label}: {e}")

    def load_nodes(self, nodes: List[GraphNode], batch_size: int = 1000):
        """
        Load nodes into Neo4j using batch operations

        Args:
            nodes: List of GraphNode objects
            batch_size: Number of nodes per batch
        """
        with self.driver.session() as session:
            print(f"   Loading {len(nodes)} nodes in batches of {batch_size}...")

            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                self._load_node_batch(session, batch)

                if (i + batch_size) % 5000 == 0:
                    print(f"   Progress: {min(i + batch_size, len(nodes))}/{len(nodes)} nodes loaded")

            print(f"   ✓ All {len(nodes)} nodes loaded")

    def _load_node_batch(self, session, nodes: List[GraphNode]):
        """Load a batch of nodes"""
        # Prepare node data
        nodes_data = []
        for node in nodes:
            node_data = {
                "node_id": node.node_id,
                "label": node.label,
                "properties": node.properties,
                "source_table": node.source_table,
                "source_pk": node.source_pk,
            }

            # Add embedding if present
            if node.embedding:
                node_data["embedding"] = node.embedding

            nodes_data.append(node_data)

        # Use UNWIND for batch insert
        query = """
        UNWIND $nodes AS nodeData
        CALL apoc.create.node([nodeData.label],
            apoc.map.merge(nodeData.properties, {
                node_id: nodeData.node_id,
                source_table: nodeData.source_table,
                source_pk: nodeData.source_pk,
                embedding: CASE WHEN nodeData.embedding IS NOT NULL
                           THEN nodeData.embedding ELSE null END
            })
        ) YIELD node
        RETURN count(node) as created
        """

        try:
            result = session.run(query, nodes=nodes_data)
            result.single()
        except Exception as e:
            # Fallback to simple CREATE if APOC not available
            print(f"   APOC not available, using simple CREATE: {e}")
            self._load_node_batch_simple(session, nodes)

    def _load_node_batch_simple(self, session, nodes: List[GraphNode]):
        """Load nodes without APOC (slower but always works)"""
        for node in nodes:
            # Build properties string
            props = dict(node.properties)
            props['node_id'] = node.node_id
            props['source_table'] = node.source_table
            props['source_pk'] = node.source_pk

            if node.embedding:
                props['embedding'] = node.embedding

            # Create node with dynamic label
            query = f"""
            CREATE (n:{node.label})
            SET n = $props
            """

            session.run(query, props=props)

    def load_edges(self, edges: List[GraphEdge], batch_size: int = 1000):
        """
        Load edges into Neo4j using batch operations

        Args:
            edges: List of GraphEdge objects
            batch_size: Number of edges per batch
        """
        with self.driver.session() as session:
            print(f"   Loading {len(edges)} edges in batches of {batch_size}...")

            for i in range(0, len(edges), batch_size):
                batch = edges[i:i + batch_size]
                self._load_edge_batch(session, batch)

                if (i + batch_size) % 5000 == 0:
                    print(f"   Progress: {min(i + batch_size, len(edges))}/{len(edges)} edges loaded")

            print(f"   ✓ All {len(edges)} edges loaded")

    def _load_edge_batch(self, session, edges: List[GraphEdge]):
        """Load a batch of edges"""
        edges_data = []
        for edge in edges:
            edge_data = {
                "from_node_id": edge.from_node_id,
                "to_node_id": edge.to_node_id,
                "rel_type": edge.relationship_type,
                "properties": edge.properties,
            }

            if edge.embedding:
                edge_data["properties"]["embedding"] = edge.embedding

            edges_data.append(edge_data)

        # Use UNWIND for batch edge creation
        query = """
        UNWIND $edges AS edgeData
        MATCH (from {node_id: edgeData.from_node_id})
        MATCH (to {node_id: edgeData.to_node_id})
        CALL apoc.create.relationship(from, edgeData.rel_type, edgeData.properties, to)
        YIELD rel
        RETURN count(rel) as created
        """

        try:
            result = session.run(query, edges=edges_data)
            result.single()
        except Exception as e:
            # Fallback without APOC
            print(f"   APOC not available for edges, using MERGE: {e}")
            self._load_edge_batch_simple(session, edges)

    def _load_edge_batch_simple(self, session, edges: List[GraphEdge]):
        """Load edges without APOC"""
        for edge in edges:
            # Note: This won't support dynamic relationship types well
            # For production, you'd need APOC or generate type-specific queries
            query = f"""
            MATCH (from {{node_id: $from_id}})
            MATCH (to {{node_id: $to_id}})
            CREATE (from)-[r:{edge.relationship_type}]->(to)
            SET r = $props
            """

            session.run(
                query,
                from_id=edge.from_node_id,
                to_id=edge.to_node_id,
                props=edge.properties
            )

    def create_vector_index(self, label: str, property_name: str = "embedding"):
        """
        Create vector index for similarity search

        Args:
            label: Node label to index
            property_name: Property containing the embedding vector
        """
        with self.driver.session() as session:
            try:
                # Neo4j 5.x vector index syntax
                query = f"""
                CREATE VECTOR INDEX {label}_embedding IF NOT EXISTS
                FOR (n:{label})
                ON n.{property_name}
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
                session.run(query)
                print(f"   ✓ Vector index created for {label}.{property_name}")
            except Exception as e:
                print(f"   ⚠️  Vector index creation failed for {label}: {e}")
                print(f"      (This is normal if using Neo4j < 5.11)")

    def get_statistics(self) -> dict:
        """Get database statistics"""
        with self.driver.session() as session:
            # Count nodes by label
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as label, count(n) as count
                ORDER BY count DESC
            """)
            node_counts = {row["label"]: row["count"] for row in result}

            # Count relationships by type
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
            """)
            edge_counts = {row["rel_type"]: row["count"] for row in result}

            # Total counts
            result = session.run("MATCH (n) RETURN count(n) as total")
            total_nodes = result.single()["total"]

            result = session.run("MATCH ()-[r]->() RETURN count(r) as total")
            total_edges = result.single()["total"]

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "nodes_by_label": node_counts,
                "edges_by_type": edge_counts,
            }
