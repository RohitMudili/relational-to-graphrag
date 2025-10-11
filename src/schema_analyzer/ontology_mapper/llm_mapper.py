"""
LLM-based ontology mapping - use LLMs to generate semantic graph ontology
"""
import json
from typing import List, Dict, Any
from openai import OpenAI
from ..models import DatabaseSchema, GraphOntology, NodeType, EdgeType, Table, Relationship


class LLMOntologyMapper:
    """Use LLM to map relational schema to graph ontology"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize LLM mapper

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o, supports JSON mode)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

        # Check if model supports JSON mode
        self.supports_json_mode = model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview"]

    def generate_ontology(self, schema: DatabaseSchema) -> GraphOntology:
        """
        Generate graph ontology from database schema using LLM

        Args:
            schema: DatabaseSchema object

        Returns:
            GraphOntology with node and edge types
        """
        # Step 1: Determine which tables become nodes vs edges
        node_edge_mapping = self._classify_tables(schema)

        # Step 2: Generate semantic labels for nodes
        node_types = self._generate_node_types(schema, node_edge_mapping["nodes"])

        # Step 3: Generate semantic labels for relationships
        edge_types = self._generate_edge_types(schema, node_edge_mapping["edges"])

        return GraphOntology(
            node_types=node_types,
            edge_types=edge_types
        )

    def _classify_tables(self, schema: DatabaseSchema) -> Dict[str, List[str]]:
        """
        Use LLM to classify which tables should become nodes vs edges

        Args:
            schema: DatabaseSchema

        Returns:
            Dict with "nodes" and "edges" lists
        """
        # Prepare schema summary for LLM
        schema_summary = self._prepare_schema_summary(schema)

        prompt = f"""You are a database-to-graph expert. Analyze this relational database schema and classify each table as either:
- NODE: Tables representing entities (customers, products, orders, etc.)
- EDGE: Tables representing relationships/junctions between entities (order_details, employee_territories, etc.)

Junction/join tables typically:
- Have 2+ foreign keys
- Have few columns besides the foreign keys
- Represent many-to-many relationships

Database Schema:
{schema_summary}

Respond in JSON format:
{{
    "nodes": ["table1", "table2", ...],
    "edges": ["junction_table1", ...]
}}"""

        # Build request parameters
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a database and graph modeling expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
        }

        # Add JSON mode if supported
        if self.supports_json_mode:
            params["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**params)

        content = response.choices[0].message.content

        # Try to extract JSON from response
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON block from markdown
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # Last resort: look for any JSON-like structure
                json_match = re.search(r'(\{.*?\})', content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    raise ValueError(f"Could not parse JSON from response: {content}")

        return result

    def _generate_node_types(self, schema: DatabaseSchema, node_tables: List[str]) -> List[NodeType]:
        """
        Generate semantic node types with proper labels and descriptions

        Args:
            schema: DatabaseSchema
            node_tables: List of table names that should become nodes

        Returns:
            List of NodeType objects
        """
        node_types = []

        for table_name in node_tables:
            table = schema.get_table(table_name)
            if not table:
                continue

            # Use LLM to generate semantic label
            label = self._generate_semantic_label(table_name, table)

            # Determine which columns become properties
            properties = [
                col.name for col in table.columns
                if not col.is_primary_key  # Exclude PK as it becomes node ID
            ]

            node_types.append(NodeType(
                label=label,
                source_table=table_name,
                properties=properties,
                description=f"Node type representing {label} entities from {table_name} table"
            ))

        return node_types

    def _generate_edge_types(self, schema: DatabaseSchema, edge_tables: List[str]) -> List[EdgeType]:
        """
        Generate semantic edge types from relationships

        Args:
            schema: DatabaseSchema
            edge_tables: List of junction table names

        Returns:
            List of EdgeType objects
        """
        edge_types = []

        # 1. Edges from explicit foreign keys
        for rel in schema.relationships:
            if rel.from_table in edge_tables:
                # Skip junction tables, handle them separately
                continue

            # Generate semantic relationship label
            label = self._generate_relationship_label(rel.from_table, rel.to_table, rel.from_column)

            edge_types.append(EdgeType(
                label=label,
                from_node=rel.from_table,
                to_node=rel.to_table,
                source_relationship=f"{rel.from_table}.{rel.from_column}",
                properties=[],
                description=f"Relationship from {rel.from_table} to {rel.to_table}"
            ))

        # 2. Edges from junction tables
        for junction_table_name in edge_tables:
            junction = schema.get_table(junction_table_name)
            if not junction or len(junction.foreign_keys) < 2:
                continue

            fk1, fk2 = junction.foreign_keys[0], junction.foreign_keys[1]

            # Generate semantic label for many-to-many
            label = self._generate_relationship_label(fk1.to_table, fk2.to_table, junction_table_name)

            # Get additional properties (non-FK columns)
            fk_columns = {fk.from_column for fk in junction.foreign_keys}
            properties = [col.name for col in junction.columns if col.name not in fk_columns]

            edge_types.append(EdgeType(
                label=label,
                from_node=fk1.to_table,
                to_node=fk2.to_table,
                source_relationship=junction_table_name,
                properties=properties,
                description=f"Many-to-many relationship via {junction_table_name}"
            ))

        return edge_types

    def _generate_semantic_label(self, table_name: str, table: Table) -> str:
        """
        Use LLM to generate a semantic label for a node type

        Args:
            table_name: Original table name
            table: Table object with columns

        Returns:
            Semantic label (e.g., "Customer", "Product")
        """
        columns_str = ", ".join([col.name for col in table.columns[:5]])  # First 5 columns

        prompt = f"""Given a database table named '{table_name}' with columns: {columns_str}

What is the best semantic label for this entity in a knowledge graph?
Respond with a single word or short phrase in PascalCase (e.g., "Customer", "OrderDetail", "ProductCategory").

Respond with just the label, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        label = response.choices[0].message.content.strip()
        return label

    def _generate_relationship_label(self, from_table: str, to_table: str, context: str) -> str:
        """
        Use LLM to generate semantic relationship label

        Args:
            from_table: Source table name
            to_table: Target table name
            context: Additional context (column name or junction table)

        Returns:
            Semantic relationship label (e.g., "PURCHASED", "BELONGS_TO")
        """
        prompt = f"""Given a relationship from '{from_table}' to '{to_table}' (context: {context}),
what is the best semantic label for this relationship in a knowledge graph?

Use UPPER_SNAKE_CASE and make it a verb (e.g., "PURCHASED", "BELONGS_TO", "EMPLOYS", "CONTAINS").

Respond with just the label, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20
        )

        label = response.choices[0].message.content.strip().upper()
        return label

    def _prepare_schema_summary(self, schema: DatabaseSchema) -> str:
        """Prepare a concise schema summary for LLM"""
        lines = []

        for table in schema.tables:
            fk_count = len(table.foreign_keys)
            col_count = len(table.columns)
            pk_count = len(table.primary_keys)

            lines.append(
                f"- {table.name}: {col_count} columns, {pk_count} PK(s), {fk_count} FK(s), "
                f"{table.row_count} rows{' [junction]' if table.is_junction_table() else ''}"
            )

            # Show columns for context
            cols = ", ".join([f"{c.name}({c.data_type})" for c in table.columns[:5]])
            lines.append(f"  Columns: {cols}...")

        return "\n".join(lines)
