"""
Cypher query generator - use LLM to generate Cypher from natural language
"""
from typing import Dict, Any, List, Optional
from openai import OpenAI
from neo4j import GraphDatabase


class CypherGeneratorTool:
    """Generate and execute Cypher queries from natural language"""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        openai_api_key: str,
        model: str = "gpt-4o"
    ):
        """
        Initialize Cypher generator

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Database username
            neo4j_password: Database password
            openai_api_key: OpenAI API key
            model: LLM model to use
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def close(self):
        """Close database connection"""
        self.driver.close()

    def get_schema_info(self) -> str:
        """Get database schema information for context"""
        with self.driver.session() as session:
            # Get node labels
            result = session.run("CALL db.labels()")
            labels = [record[0] for record in result]

            # Get relationship types
            result = session.run("CALL db.relationshipTypes()")
            rel_types = [record[0] for record in result]

            # Get sample properties for each label
            schema_info = ["Node Labels:"]
            for label in labels[:10]:  # Limit to first 10
                result = session.run(f"MATCH (n:{label}) RETURN n LIMIT 1")
                record = result.single()
                if record:
                    props = list(dict(record["n"]).keys())[:5]  # First 5 properties
                    schema_info.append(f"  {label}: {', '.join(props)}")

            schema_info.append("\nRelationship Types:")
            schema_info.extend([f"  {rt}" for rt in rel_types[:15]])

            return "\n".join(schema_info)

    def generate_cypher(
        self,
        natural_language_query: str,
        schema_context: Optional[str] = None
    ) -> str:
        """
        Generate Cypher query from natural language

        Args:
            natural_language_query: User's question in natural language
            schema_context: Optional schema information

        Returns:
            Cypher query string
        """
        if not schema_context:
            schema_context = self.get_schema_info()

        prompt = f"""You are a Neo4j Cypher query expert. Convert the natural language question to a Cypher query.

Database Schema:
{schema_context}

Natural Language Question:
{natural_language_query}

Generate a Cypher query that answers this question. Return ONLY the Cypher query, no explanation.
Make sure to:
- Use MATCH clauses appropriately
- Use WHERE for filtering
- Use RETURN to specify what to return
- Limit results to a reasonable number (e.g., LIMIT 10 unless user specifies otherwise)
- Handle potential NULL values

Cypher Query:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a Neo4j Cypher expert. Generate only the Cypher query, nothing else."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500
        )

        cypher_query = response.choices[0].message.content.strip()

        # Clean up markdown code blocks if present
        if "```" in cypher_query:
            # Extract code from markdown
            lines = cypher_query.split("\n")
            cypher_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block or not line.strip().startswith("```"):
                    cypher_lines.append(line)

            cypher_query = "\n".join(cypher_lines).strip()

        return cypher_query

    def execute_cypher(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query

        Args:
            cypher_query: Cypher query to execute
            parameters: Optional query parameters

        Returns:
            List of result records
        """
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, parameters or {})

                records = []
                for record in result:
                    # Convert record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]

                        # Handle Neo4j node/relationship objects
                        if hasattr(value, 'labels'):  # Node
                            record_dict[key] = {
                                "type": "node",
                                "labels": list(value.labels),
                                "properties": dict(value)
                            }
                        elif hasattr(value, 'type'):  # Relationship
                            record_dict[key] = {
                                "type": "relationship",
                                "rel_type": value.type,
                                "properties": dict(value)
                            }
                        else:
                            record_dict[key] = value

                    records.append(record_dict)

                return records

            except Exception as e:
                return [{"error": str(e), "query": cypher_query}]

    def query_with_nl(
        self,
        natural_language_query: str
    ) -> Dict[str, Any]:
        """
        Execute natural language query (generate Cypher and run it)

        Args:
            natural_language_query: User's question

        Returns:
            Results with query and data
        """
        # Generate Cypher
        cypher_query = self.generate_cypher(natural_language_query)

        # Execute query
        results = self.execute_cypher(cypher_query)

        return {
            "query": natural_language_query,
            "cypher": cypher_query,
            "results": results,
            "count": len(results)
        }

    def explain_query(self, cypher_query: str) -> str:
        """
        Get natural language explanation of a Cypher query

        Args:
            cypher_query: Cypher query to explain

        Returns:
            Natural language explanation
        """
        prompt = f"""Explain this Cypher query in simple terms:

{cypher_query}

Explanation:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You explain database queries in simple, non-technical language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        return response.choices[0].message.content.strip()
