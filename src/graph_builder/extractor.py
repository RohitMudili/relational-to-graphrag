"""
Data extractor - extract data from PostgreSQL based on ontology
"""
import psycopg2
from typing import List, Dict, Any, Iterator, Tuple
from ..schema_analyzer.models import GraphOntology, NodeType, EdgeType


class DataExtractor:
    """Extract data from relational database based on graph ontology"""

    def __init__(self, connection_string: str):
        """
        Initialize data extractor

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        self._fk_cache = {}  # Cache foreign key information

    def extract_nodes_data(
        self,
        node_type: NodeType,
        batch_size: int = 1000
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Extract data for a specific node type

        Args:
            node_type: NodeType definition from ontology
            batch_size: Number of rows per batch

        Yields:
            Batches of row data as dictionaries
        """
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()

            # Build SELECT query
            query = f"SELECT * FROM {node_type.source_table}"
            cursor.execute(query)

            # Get column names
            col_names = [desc[0] for desc in cursor.description]

            # Fetch in batches
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                # Convert to dictionaries
                batch = []
                for row in rows:
                    row_dict = dict(zip(col_names, row))
                    batch.append(row_dict)

                yield batch

            cursor.close()

    def extract_edge_data(
        self,
        edge_type: EdgeType,
        ontology: GraphOntology,
        batch_size: int = 1000
    ) -> Iterator[Tuple[List[Dict[str, Any]], List[str], List[str]]]:
        """
        Extract data for edges (relationships)

        Args:
            edge_type: EdgeType definition from ontology
            ontology: Full ontology for lookups
            batch_size: Number of rows per batch

        Yields:
            Tuple of (batch_data, from_pk_columns, to_pk_columns)
        """
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()

            # Determine edge source
            if edge_type.source_relationship:
                if '.' in edge_type.source_relationship:
                    # Foreign key: "table.column"
                    yield from self._extract_fk_edges(cursor, edge_type, ontology, batch_size)
                elif edge_type.source_relationship.startswith('via_'):
                    # Junction table: "via_order_details"
                    junction_table = edge_type.source_relationship.replace('via_', '')
                    yield from self._extract_junction_edges(cursor, edge_type, junction_table, ontology, batch_size)
                else:
                    # Direct table reference (junction table)
                    yield from self._extract_junction_edges(cursor, edge_type, edge_type.source_relationship, ontology, batch_size)

            cursor.close()

    def _extract_fk_edges(
        self,
        cursor,
        edge_type: EdgeType,
        ontology: GraphOntology,
        batch_size: int
    ) -> Iterator[Tuple[List[Dict[str, Any]], List[str], List[str]]]:
        """Extract edges from foreign key relationships"""
        # Parse: "table.column" -> table has FK column pointing to target
        from_table, fk_column = edge_type.source_relationship.split('.')

        # Find source tables for nodes
        to_table = self._find_source_table(edge_type.to_node, ontology)

        # Get primary keys
        from_pk_cols = self.get_primary_key_columns(from_table)
        to_pk_cols = self.get_primary_key_columns(to_table)

        # Build column lists for SELECT
        from_cols = ', '.join([f'f.{col} as from_{col}' for col in from_pk_cols])
        to_cols = ', '.join([f't.{col} as to_{col}' for col in to_pk_cols])
        fk_col_select = f'f.{fk_column} as fk_value'

        # Query to get FK relationships
        query = f"""
            SELECT {from_cols}, {to_cols}, {fk_col_select}
            FROM {from_table} f
            INNER JOIN {to_table} t ON f.{fk_column} = t.{to_pk_cols[0]}
            WHERE f.{fk_column} IS NOT NULL
        """

        try:
            cursor.execute(query)
            col_names = [desc[0] for desc in cursor.description]

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                batch = []
                for row in rows:
                    row_dict = dict(zip(col_names, row))
                    batch.append(row_dict)

                # Return batch with column info for ID construction
                yield (batch, [f'from_{pk}' for pk in from_pk_cols], [f'to_{pk}' for pk in to_pk_cols])

        except Exception as e:
            print(f"      Error extracting FK edges: {e}")
            return

    def _extract_junction_edges(
        self,
        cursor,
        edge_type: EdgeType,
        junction_table: str,
        ontology: GraphOntology,
        batch_size: int
    ) -> Iterator[Tuple[List[Dict[str, Any]], List[str], List[str]]]:
        """Extract edges from junction tables (many-to-many)"""
        # Get the junction table's foreign keys
        fk_info = self._get_table_foreign_keys(cursor, junction_table)

        if len(fk_info) < 2:
            print(f"      Warning: Junction table {junction_table} has fewer than 2 FKs")
            return

        # Find source tables
        from_table = self._find_source_table(edge_type.from_node, ontology)
        to_table = self._find_source_table(edge_type.to_node, ontology)

        # Match FKs to our from/to tables
        from_fk = None
        to_fk = None

        for fk in fk_info:
            if fk['referenced_table'] == from_table:
                from_fk = fk
            elif fk['referenced_table'] == to_table:
                to_fk = fk

        if not from_fk or not to_fk:
            print(f"      Warning: Could not match FKs for {junction_table}")
            return

        # Get all columns from junction table
        query = f"SELECT * FROM {junction_table}"

        try:
            cursor.execute(query)
            col_names = [desc[0] for desc in cursor.description]

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                batch = []
                for row in rows:
                    row_dict = dict(zip(col_names, row))
                    batch.append(row_dict)

                # Return with FK column names
                yield (batch, [from_fk['column']], [to_fk['column']])

        except Exception as e:
            print(f"      Error extracting junction edges: {e}")
            return

    def _get_table_foreign_keys(self, cursor, table_name: str) -> List[Dict[str, str]]:
        """Get foreign key information for a table"""
        if table_name in self._fk_cache:
            return self._fk_cache[table_name]

        cursor.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name as referenced_table,
                ccu.column_name as referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name = %s
        """, (table_name,))

        fk_info = []
        for row in cursor.fetchall():
            fk_info.append({
                'column': row[0],
                'referenced_table': row[1],
                'referenced_column': row[2]
            })

        self._fk_cache[table_name] = fk_info
        return fk_info

    def _find_source_table(self, node_label: str, ontology: GraphOntology) -> str:
        """Find the source table for a node label"""
        for node_type in ontology.node_types:
            if node_type.label == node_label:
                return node_type.source_table

        # Fallback: return as-is (should not happen if ontology is correct)
        print(f"      Warning: Could not find source table for '{node_label}', using lowercase")
        return node_label.lower()

    def get_primary_key_columns(self, table_name: str) -> List[str]:
        """Get primary key columns for a table"""
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT ku.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_name = %s
                ORDER BY ku.ordinal_position
            """, (table_name,))

            pk_columns = [row[0] for row in cursor.fetchall()]
            cursor.close()

            return pk_columns or ['id']  # Default to 'id' if not found
