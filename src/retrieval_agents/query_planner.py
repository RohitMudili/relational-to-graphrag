"""
Query planner - analyze queries and determine optimal retrieval strategy
"""
from typing import Dict, Any, List, Optional
from enum import Enum
from openai import OpenAI


class QueryType(Enum):
    """Types of queries the system can handle"""
    VECTOR_SEARCH = "vector_search"  # Semantic similarity
    GRAPH_TRAVERSAL = "graph_traversal"  # Relationship exploration
    CYPHER_QUERY = "cypher_query"  # Complex logical queries
    HYBRID = "hybrid"  # Combination of multiple strategies


class QueryStrategy:
    """Query execution strategy"""

    def __init__(
        self,
        query_type: QueryType,
        primary_tool: str,
        secondary_tools: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        reasoning: str = ""
    ):
        self.query_type = query_type
        self.primary_tool = primary_tool
        self.secondary_tools = secondary_tools or []
        self.parameters = parameters or {}
        self.reasoning = reasoning


class QueryPlanner:
    """Analyze queries and plan retrieval strategy"""

    def __init__(self, openai_api_key: str, model: str = "gpt-4o"):
        """
        Initialize query planner

        Args:
            openai_api_key: OpenAI API key
            model: LLM model to use for query analysis
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # Query pattern keywords for heuristic classification
        self.vector_keywords = [
            "similar", "like", "find", "search", "related",
            "comparable", "close to", "resembles"
        ]
        self.graph_keywords = [
            "connected", "relationship", "path", "between",
            "neighbors", "links", "reports to", "works with",
            "hierarchy", "structure", "network"
        ]
        self.cypher_keywords = [
            "all", "count", "total", "average", "sum",
            "where", "filter", "group by", "order by",
            "top", "most", "least", "aggregate"
        ]

    def plan_query(
        self,
        natural_language_query: str,
        schema_context: Optional[str] = None
    ) -> QueryStrategy:
        """
        Analyze query and determine optimal strategy

        Args:
            natural_language_query: User's question
            schema_context: Optional graph schema information

        Returns:
            QueryStrategy with execution plan
        """
        # First try heuristic classification
        heuristic_strategy = self._heuristic_classification(natural_language_query)

        # For complex queries, use LLM for better classification
        if self._is_complex_query(natural_language_query):
            return self._llm_classification(natural_language_query, schema_context)

        return heuristic_strategy

    def _heuristic_classification(self, query: str) -> QueryStrategy:
        """Fast heuristic-based classification"""
        query_lower = query.lower()

        # Count keyword matches
        vector_score = sum(1 for kw in self.vector_keywords if kw in query_lower)
        graph_score = sum(1 for kw in self.graph_keywords if kw in query_lower)
        cypher_score = sum(1 for kw in self.cypher_keywords if kw in query_lower)

        # Determine primary strategy
        if vector_score > graph_score and vector_score > cypher_score:
            return QueryStrategy(
                query_type=QueryType.VECTOR_SEARCH,
                primary_tool="vector_search",
                reasoning="Query suggests semantic similarity search"
            )
        elif graph_score > cypher_score:
            return QueryStrategy(
                query_type=QueryType.GRAPH_TRAVERSAL,
                primary_tool="graph_traversal",
                reasoning="Query involves relationship exploration"
            )
        else:
            return QueryStrategy(
                query_type=QueryType.CYPHER_QUERY,
                primary_tool="cypher_generator",
                reasoning="Query requires complex filtering or aggregation"
            )

    def _is_complex_query(self, query: str) -> bool:
        """Determine if query needs LLM analysis"""
        # Complex if it has multiple clauses or conditions
        complexity_markers = [
            " and ", " or ", " but ", " then ",
            "?", "how many", "which", "what are all"
        ]
        return any(marker in query.lower() for marker in complexity_markers)

    def _llm_classification(
        self,
        query: str,
        schema_context: Optional[str]
    ) -> QueryStrategy:
        """LLM-based query classification for complex queries"""

        schema_info = schema_context if schema_context else "Graph database with nodes and relationships"

        prompt = f"""Analyze this natural language query and determine the best retrieval strategy.

Database Schema:
{schema_info}

Query: {query}

Available strategies:
1. VECTOR_SEARCH: Semantic similarity search (best for "find similar", "like this", "related to")
2. GRAPH_TRAVERSAL: Relationship exploration (best for "connected", "path between", "neighbors", "hierarchy")
3. CYPHER_QUERY: Complex logical queries (best for filtering, aggregation, counting, "all X where Y")
4. HYBRID: Combination of strategies (when query needs multiple approaches)

Respond with JSON:
{{
    "strategy": "VECTOR_SEARCH|GRAPH_TRAVERSAL|CYPHER_QUERY|HYBRID",
    "primary_tool": "vector_search|graph_traversal|cypher_generator",
    "secondary_tools": [],
    "reasoning": "brief explanation",
    "parameters": {{
        "node_label": "optional node type to focus on",
        "top_k": 5,
        "min_score": 0.7
    }}
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query planner for a graph database. Analyze queries and suggest optimal retrieval strategies."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            import json
            result = json.loads(response.choices[0].message.content)

            return QueryStrategy(
                query_type=QueryType[result["strategy"]],
                primary_tool=result["primary_tool"],
                secondary_tools=result.get("secondary_tools", []),
                parameters=result.get("parameters", {}),
                reasoning=result["reasoning"]
            )

        except Exception as e:
            print(f"LLM classification failed: {e}, falling back to heuristic")
            return self._heuristic_classification(query)

    def explain_strategy(self, strategy: QueryStrategy) -> str:
        """
        Generate human-readable explanation of the strategy

        Args:
            strategy: Query strategy to explain

        Returns:
            Explanation string
        """
        explanation = f"Query Type: {strategy.query_type.value}\n"
        explanation += f"Primary Tool: {strategy.primary_tool}\n"

        if strategy.secondary_tools:
            explanation += f"Secondary Tools: {', '.join(strategy.secondary_tools)}\n"

        explanation += f"Reasoning: {strategy.reasoning}\n"

        if strategy.parameters:
            explanation += f"Parameters: {strategy.parameters}\n"

        return explanation
