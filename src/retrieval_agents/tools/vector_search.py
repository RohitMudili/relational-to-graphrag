"""
Vector search tool - semantic similarity search using embeddings
"""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase


class VectorSearchTool:
    """Search for nodes using vector similarity (embeddings)"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize vector search tool

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close database connection"""
        self.driver.close()

    def search_similar_nodes(
        self,
        query_embedding: List[float],
        node_label: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for nodes similar to the query embedding

        Args:
            query_embedding: Query vector (1536-dimensional)
            node_label: Optional node label to filter (e.g., "Customer", "Product")
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)

        Returns:
            List of similar nodes with scores
        """
        with self.driver.session() as session:
            # Build query based on whether we filter by label
            if node_label:
                # Use vector index for specific label
                index_name = f"{node_label}_embedding"

                try:
                    query = f"""
                    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                    YIELD node, score
                    WHERE score >= $min_score
                    RETURN node, score
                    ORDER BY score DESC
                    """

                    result = session.run(
                        query,
                        index_name=index_name,
                        top_k=top_k,
                        embedding=query_embedding,
                        min_score=min_score
                    )
                except Exception as e:
                    # Fallback to manual cosine similarity if vector index not available
                    return self._manual_vector_search(session, query_embedding, node_label, top_k, min_score)
            else:
                # Search across all nodes (slower)
                return self._manual_vector_search(session, query_embedding, None, top_k, min_score)

            # Parse results
            nodes = []
            for record in result:
                node = record["node"]
                score = record["score"]

                nodes.append({
                    "node_id": node.get("node_id"),
                    "label": list(node.labels)[0] if node.labels else "Unknown",
                    "properties": dict(node),
                    "score": score
                })

            return nodes

    def _manual_vector_search(
        self,
        session,
        query_embedding: List[float],
        node_label: Optional[str],
        top_k: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """Manual vector search using cosine similarity (fallback)"""

        # Build MATCH clause
        if node_label:
            match_clause = f"MATCH (n:{node_label})"
        else:
            match_clause = "MATCH (n)"

        query = f"""
        {match_clause}
        WHERE n.embedding IS NOT NULL
        WITH n,
             gds.similarity.cosine(n.embedding, $embedding) AS score
        WHERE score >= $min_score
        RETURN n as node, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        try:
            result = session.run(
                query,
                embedding=query_embedding,
                min_score=min_score,
                top_k=top_k
            )

            nodes = []
            for record in result:
                node = record["node"]
                score = record["score"]

                nodes.append({
                    "node_id": node.get("node_id"),
                    "label": list(node.labels)[0] if node.labels else "Unknown",
                    "properties": dict(node),
                    "score": score
                })

            return nodes

        except Exception as e:
            print(f"Manual vector search failed: {e}")
            return []

    def search_by_text(
        self,
        query_text: str,
        embedding_service,
        node_label: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search using text query (generates embedding first)

        Args:
            query_text: Natural language query
            embedding_service: EmbeddingService instance to generate query embedding
            node_label: Optional node label filter
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of similar nodes
        """
        # Generate embedding for query
        query_embedding = embedding_service.generate_embedding(query_text)

        # Search using embedding
        return self.search_similar_nodes(
            query_embedding,
            node_label=node_label,
            top_k=top_k,
            min_score=min_score
        )

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific node by its ID (supports multiple ID formats)"""
        with self.driver.session() as session:
            # Try multiple ways to find the node
            queries = [
                # Try exact node_id match
                ("MATCH (n {node_id: $node_id}) RETURN n, labels(n) as labels", {"node_id": node_id}),
                # Try source_pk match
                ("MATCH (n {source_pk: $source_pk}) RETURN n, labels(n) as labels", {"source_pk": node_id}),
                # Try with table prefix (e.g., customers:ALFKI)
                ("MATCH (n) WHERE n.node_id CONTAINS $partial_id RETURN n, labels(n) as labels LIMIT 1", {"partial_id": node_id})
            ]

            for query, params in queries:
                result = session.run(query, params)
                record = result.single()

                if record:
                    node = record["n"]
                    return {
                        "node_id": node.get("node_id"),
                        "label": record["labels"][0] if record["labels"] else "Unknown",
                        "properties": dict(node)
                    }

            return None
