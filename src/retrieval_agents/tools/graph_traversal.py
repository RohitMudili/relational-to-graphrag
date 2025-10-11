"""
Graph traversal tool - explore relationships and paths
"""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase


class GraphTraversalTool:
    """Traverse graph relationships and find paths"""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize graph traversal tool

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close database connection"""
        self.driver.close()

    def get_neighbors(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "BOTH",  # "OUTGOING", "INCOMING", "BOTH"
        depth: int = 1
    ) -> Dict[str, Any]:
        """
        Get neighboring nodes connected to a given node

        Args:
            node_id: Source node ID
            relationship_types: Optional list of relationship types to follow
            direction: Direction to traverse ("OUTGOING", "INCOMING", "BOTH")
            depth: How many hops to traverse

        Returns:
            Dict with nodes and relationships
        """
        with self.driver.session() as session:
            # Build relationship pattern
            if relationship_types:
                rel_pattern = "|".join(relationship_types)
                rel_pattern = f"[r:{rel_pattern}]"
            else:
                rel_pattern = "[r]"

            # Build direction pattern
            if direction == "OUTGOING":
                pattern = f"-{rel_pattern}->"
            elif direction == "INCOMING":
                pattern = f"<-{rel_pattern}-"
            else:  # BOTH
                pattern = f"-{rel_pattern}-"

            # Build depth pattern
            if depth > 1:
                rel_pattern = rel_pattern[:-1] + f"*1..{depth}]"
                pattern = pattern.replace("[r]", rel_pattern).replace("[r:", f"[r:{rel_pattern[3:]}")

            query = f"""
            MATCH (start {{node_id: $node_id}}){pattern}(end)
            RETURN start, r, end, labels(end) as end_labels
            LIMIT 50
            """

            result = session.run(query, node_id=node_id)

            nodes = {}
            relationships = []

            for record in result:
                start_node = record["start"]
                end_node = record["end"]
                rel = record["r"]

                # Add start node
                start_id = start_node.get("node_id")
                if start_id not in nodes:
                    nodes[start_id] = {
                        "node_id": start_id,
                        "label": list(start_node.labels)[0] if start_node.labels else "Unknown",
                        "properties": dict(start_node)
                    }

                # Add end node
                end_id = end_node.get("node_id")
                if end_id not in nodes:
                    nodes[end_id] = {
                        "node_id": end_id,
                        "label": record["end_labels"][0] if record["end_labels"] else "Unknown",
                        "properties": dict(end_node)
                    }

                # Add relationship
                relationships.append({
                    "from": start_id,
                    "to": end_id,
                    "type": type(rel).__name__ if hasattr(rel, '__name__') else str(rel.type) if hasattr(rel, 'type') else "RELATED_TO",
                    "properties": dict(rel) if hasattr(rel, 'items') else {}
                })

            return {
                "nodes": list(nodes.values()),
                "relationships": relationships,
                "count": len(nodes)
            }

    def find_path(
        self,
        start_node_id: str,
        end_node_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes

        Args:
            start_node_id: Starting node ID
            end_node_id: Target node ID
            max_depth: Maximum path length
            relationship_types: Optional relationship types to follow

        Returns:
            Path information or None if no path found
        """
        with self.driver.session() as session:
            # Build relationship pattern
            if relationship_types:
                rel_pattern = "|".join(relationship_types)
                rel_pattern = f"[r:{rel_pattern}*1..{max_depth}]"
            else:
                rel_pattern = f"[r*1..{max_depth}]"

            query = f"""
            MATCH path = shortestPath(
                (start {{node_id: $start_id}})-{rel_pattern}-(end {{node_id: $end_id}})
            )
            RETURN path, length(path) as path_length
            """

            result = session.run(
                query,
                start_id=start_node_id,
                end_id=end_node_id
            )

            record = result.single()
            if not record:
                return None

            path = record["path"]
            path_length = record["path_length"]

            # Extract nodes and relationships from path
            nodes = []
            relationships = []

            for node in path.nodes:
                nodes.append({
                    "node_id": node.get("node_id"),
                    "label": list(node.labels)[0] if node.labels else "Unknown",
                    "properties": dict(node)
                })

            for rel in path.relationships:
                relationships.append({
                    "from": rel.start_node.get("node_id"),
                    "to": rel.end_node.get("node_id"),
                    "type": rel.type,
                    "properties": dict(rel)
                })

            return {
                "nodes": nodes,
                "relationships": relationships,
                "path_length": path_length
            }

    def get_relationship_summary(self, node_id: str) -> Dict[str, Any]:
        """
        Get summary of relationships for a node

        Args:
            node_id: Node ID

        Returns:
            Summary of relationship types and counts
        """
        with self.driver.session() as session:
            query = """
            MATCH (n {node_id: $node_id})-[r]-(connected)
            RETURN type(r) as rel_type,
                   count(r) as count,
                   collect(DISTINCT labels(connected)[0]) as connected_labels
            """

            result = session.run(query, node_id=node_id)

            summary = []
            for record in result:
                summary.append({
                    "relationship_type": record["rel_type"],
                    "count": record["count"],
                    "connected_to": record["connected_labels"]
                })

            return {
                "node_id": node_id,
                "relationships": summary,
                "total_connections": sum(s["count"] for s in summary)
            }

    def expand_subgraph(
        self,
        node_ids: List[str],
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Expand a subgraph from multiple starting nodes

        Args:
            node_ids: List of starting node IDs
            max_depth: Maximum depth to expand

        Returns:
            Subgraph with all nodes and relationships
        """
        with self.driver.session() as session:
            query = f"""
            MATCH path = (start)-[*1..{max_depth}]-(connected)
            WHERE start.node_id IN $node_ids
            WITH nodes(path) as path_nodes, relationships(path) as path_rels
            UNWIND path_nodes as n
            WITH collect(DISTINCT n) as all_nodes, path_rels
            UNWIND path_rels as r
            WITH all_nodes, collect(DISTINCT r) as all_rels
            RETURN all_nodes, all_rels
            """

            result = session.run(query, node_ids=node_ids)
            record = result.single()

            if not record:
                return {"nodes": [], "relationships": [], "count": 0}

            # Process nodes
            nodes = {}
            for node in record["all_nodes"]:
                node_id = node.get("node_id")
                nodes[node_id] = {
                    "node_id": node_id,
                    "label": list(node.labels)[0] if node.labels else "Unknown",
                    "properties": dict(node)
                }

            # Process relationships
            relationships = []
            for rel in record["all_rels"]:
                relationships.append({
                    "from": rel.start_node.get("node_id"),
                    "to": rel.end_node.get("node_id"),
                    "type": rel.type,
                    "properties": dict(rel)
                })

            return {
                "nodes": list(nodes.values()),
                "relationships": relationships,
                "count": len(nodes)
            }
