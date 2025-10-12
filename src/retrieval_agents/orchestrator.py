"""
Agent orchestrator - coordinate retrieval tools and execute queries
"""
from typing import Dict, Any, List, Optional, Iterator
from openai import OpenAI

from .tools import VectorSearchTool, GraphTraversalTool, CypherGeneratorTool
from .query_planner import QueryPlanner, QueryStrategy, QueryType
from ..graph_builder.embeddings.embedding_service import EmbeddingService


class RetrievalResult:
    """Result from a retrieval operation"""

    def __init__(
        self,
        query: str,
        strategy: QueryStrategy,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.query = query
        self.strategy = strategy
        self.results = results
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "strategy": {
                "type": self.strategy.query_type.value,
                "primary_tool": self.strategy.primary_tool,
                "reasoning": self.strategy.reasoning
            },
            "results": self.results,
            "result_count": len(self.results),
            "metadata": self.metadata
        }


class AgentOrchestrator:
    """Coordinate retrieval agents and execute queries"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: str,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize agent orchestrator

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            openai_api_key: OpenAI API key
            model: LLM model for reasoning
            embedding_model: Embedding model
        """
        # Initialize tools
        self.vector_search = VectorSearchTool(neo4j_uri, neo4j_user, neo4j_password)
        self.graph_traversal = GraphTraversalTool(neo4j_uri, neo4j_user, neo4j_password)
        self.cypher_generator = CypherGeneratorTool(
            neo4j_uri, neo4j_user, neo4j_password, openai_api_key, model
        )

        # Initialize supporting services
        self.query_planner = QueryPlanner(openai_api_key, model)
        self.embedding_service = EmbeddingService(openai_api_key, embedding_model)
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def close(self):
        """Close all connections"""
        self.vector_search.close()
        self.graph_traversal.close()
        self.cypher_generator.close()

    def query(
        self,
        natural_language_query: str,
        top_k: int = 5,
        min_score: float = 0.7,
        max_depth: int = 2
    ) -> RetrievalResult:
        """
        Execute a natural language query

        Args:
            natural_language_query: User's question
            top_k: Number of results for vector search
            min_score: Minimum similarity score
            max_depth: Maximum depth for graph traversal

        Returns:
            RetrievalResult with query results
        """
        print(f"\n🔍 Query: {natural_language_query}")

        # Plan the query strategy
        schema_context = self.cypher_generator.get_schema_info()
        strategy = self.query_planner.plan_query(natural_language_query, schema_context)

        print(f"📋 Strategy: {strategy.query_type.value}")
        print(f"   Primary Tool: {strategy.primary_tool}")
        print(f"   Reasoning: {strategy.reasoning}\n")

        # Execute based on strategy
        if strategy.query_type == QueryType.VECTOR_SEARCH:
            results = self._execute_vector_search(
                natural_language_query, strategy, top_k, min_score
            )
        elif strategy.query_type == QueryType.GRAPH_TRAVERSAL:
            results = self._execute_graph_traversal(
                natural_language_query, strategy, max_depth
            )
        elif strategy.query_type == QueryType.CYPHER_QUERY:
            results = self._execute_cypher_query(natural_language_query, strategy)
        else:  # HYBRID
            results = self._execute_hybrid(
                natural_language_query, strategy, top_k, min_score, max_depth
            )

        return RetrievalResult(
            query=natural_language_query,
            strategy=strategy,
            results=results
        )

    def _execute_vector_search(
        self,
        query: str,
        strategy: QueryStrategy,
        top_k: int,
        min_score: float
    ) -> List[Dict[str, Any]]:
        """Execute vector similarity search"""
        print("🔎 Executing vector search...")

        # Get node label filter if specified
        node_label = strategy.parameters.get("node_label")
        top_k = strategy.parameters.get("top_k", top_k)
        min_score = strategy.parameters.get("min_score", min_score)

        # Check if this is a "similar to X" query
        if "similar to" in query.lower() or "like" in query.lower():
            # Extract reference node ID
            node_info = self._extract_node_identifiers(query)
            if node_info.get("node_ids"):
                reference_id = node_info["node_ids"][0]
                print(f"   Finding nodes similar to: {reference_id}")

                # Get the reference node
                reference_node = self.vector_search.get_node_by_id(reference_id)
                if reference_node and reference_node.get("properties", {}).get("embedding"):
                    # Use the reference node's embedding to find similar nodes
                    results = self.vector_search.search_similar_nodes(
                        reference_node["properties"]["embedding"],
                        node_label=node_label,
                        top_k=top_k + 1,  # +1 to exclude the reference node itself
                        min_score=min_score
                    )
                    # Filter out the reference node from results
                    results = [r for r in results if r.get("node_id") != reference_id][:top_k]
                    print(f"   Found {len(results)} similar nodes")
                    return results
                else:
                    print(f"   Warning: Could not find reference node {reference_id}")

        # Default: Search using text query
        results = self.vector_search.search_by_text(
            query,
            self.embedding_service,
            node_label=node_label,
            top_k=top_k,
            min_score=min_score
        )

        print(f"   Found {len(results)} similar nodes")
        return results

    def _execute_graph_traversal(
        self,
        query: str,
        strategy: QueryStrategy,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Execute graph traversal"""
        print("🕸️  Executing graph traversal...")

        # Use LLM to extract node identifiers from query
        node_info = self._extract_node_identifiers(query)

        if not node_info.get("node_ids"):
            # If no specific nodes, try vector search first to find starting point
            print("   No specific node IDs found, using vector search to find starting point...")
            vector_results = self._execute_vector_search(query, strategy, top_k=3, min_score=0.7)

            if not vector_results:
                return [{"error": "Could not find starting nodes for traversal"}]

            node_ids = [n["node_id"] for n in vector_results if n.get("node_id")]
        else:
            node_ids = node_info["node_ids"]

        # Determine if this is a path query or neighbor query
        if "path" in query.lower() or "between" in query.lower():
            # Path finding
            if len(node_ids) >= 2:
                result = self.graph_traversal.find_path(
                    node_ids[0], node_ids[1], max_depth=max_depth
                )
                return [result] if result else []
            else:
                return [{"error": "Path queries require two node identifiers"}]
        else:
            # Neighbor exploration or subgraph expansion
            if len(node_ids) == 1:
                result = self.graph_traversal.get_neighbors(
                    node_ids[0],
                    depth=strategy.parameters.get("depth", 1)
                )
                return [result]
            else:
                # Multiple nodes - expand subgraph
                result = self.graph_traversal.expand_subgraph(node_ids, max_depth=max_depth)
                return [result]

    def _execute_cypher_query(
        self,
        query: str,
        strategy: QueryStrategy
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query generation and execution"""
        print("⚙️  Executing Cypher query...")

        # Generate and execute query
        result = self.cypher_generator.query_with_nl(query)

        print(f"   Generated Cypher: {result['cypher'][:100]}...")
        print(f"   Found {result['count']} results")

        return result["results"]

    def _execute_hybrid(
        self,
        query: str,
        strategy: QueryStrategy,
        top_k: int,
        min_score: float,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Execute hybrid strategy combining multiple approaches"""
        print("🔀 Executing hybrid strategy...")

        results = []

        # Start with vector search to find relevant nodes
        vector_results = self._execute_vector_search(query, strategy, top_k, min_score)
        results.extend(vector_results)

        # If we have node IDs, expand with graph traversal
        if vector_results and "traversal" in strategy.secondary_tools:
            print("   Expanding with graph traversal...")
            node_ids = [n["node_id"] for n in vector_results[:3] if n.get("node_id")]
            if node_ids:
                subgraph = self.graph_traversal.expand_subgraph(node_ids, max_depth=1)
                results.append({"subgraph": subgraph})

        return results

    def _extract_node_identifiers(self, query: str) -> Dict[str, Any]:
        """Use LLM to extract node identifiers from query"""

        prompt = f"""Extract node identifiers from this query.

Query: {query}

Look for:
- Specific IDs (like "customer ALFKI", "employee 5")
- Names that might be node IDs

Respond with JSON:
{{
    "node_ids": ["id1", "id2"],
    "node_types": ["Customer", "Employee"]
}}

If no specific identifiers found, return empty arrays."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract structured information from queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )

            import json
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"   Warning: Could not extract node identifiers: {e}")
            return {"node_ids": [], "node_types": []}

    def explain_results(self, result: RetrievalResult) -> str:
        """
        Generate natural language explanation of results

        Args:
            result: Retrieval result to explain

        Returns:
            Natural language explanation
        """
        if not result.results:
            return "No results found for your query."

        # Build context for LLM
        results_summary = {
            "query": result.query,
            "strategy": result.strategy.query_type.value,
            "result_count": len(result.results),
            "sample_results": result.results[:3]  # First 3 results
        }

        prompt = f"""Explain these query results in natural language.

Query: {result.query}
Strategy Used: {result.strategy.query_type.value}
Results: {results_summary}

Provide a concise, helpful explanation of what was found. Focus on the most relevant information."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You explain database query results in clear, natural language."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Found {len(result.results)} results. Error generating explanation: {e}"

    def streaming_query(
        self,
        natural_language_query: str,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Execute query with streaming updates

        Args:
            natural_language_query: User's question
            **kwargs: Additional parameters for query execution

        Yields:
            Status updates and results
        """
        yield {"status": "planning", "message": "Analyzing query..."}

        schema_context = self.cypher_generator.get_schema_info()
        strategy = self.query_planner.plan_query(natural_language_query, schema_context)

        yield {
            "status": "strategy",
            "strategy": strategy.query_type.value,
            "reasoning": strategy.reasoning
        }

        yield {"status": "executing", "message": f"Executing {strategy.primary_tool}..."}

        # Execute query (non-streaming for now)
        result = self.query(natural_language_query, **kwargs)

        yield {
            "status": "results",
            "data": result.to_dict()
        }

        # Generate explanation
        yield {"status": "explaining", "message": "Generating explanation..."}

        explanation = self.explain_results(result)

        yield {
            "status": "complete",
            "explanation": explanation
        }
