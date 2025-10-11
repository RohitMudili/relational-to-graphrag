"""
PostgreSQL schema extraction
"""
import psycopg2
from typing import List, Dict, Any
from ..models import Table, Column, ForeignKey, DatabaseSchema, Relationship, RelationshipType, RelationshipSource


class PostgresSchemaExtractor:
    """Extract schema information from PostgreSQL database"""

    def __init__(self, connection_string: str):
        """
        Initialize extractor with database connection

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string

    def extract_schema(self, schema_name: str = "public") -> DatabaseSchema:
        """
        Extract complete database schema

        Args:
            schema_name: Schema to extract (default: public)

        Returns:
            DatabaseSchema object with all tables and relationships
        """
        with psycopg2.connect(self.connection_string) as conn:
            cursor = conn.cursor()

            # Get database name
            cursor.execute("SELECT current_database()")
            db_name = cursor.fetchone()[0]

            # Extract tables
            tables = self._extract_tables(cursor, schema_name)

            # Extract foreign keys and build explicit relationships
            relationships = []
            for table in tables:
                table.foreign_keys = self._extract_foreign_keys(cursor, table.name, schema_name)
                # Convert foreign keys to relationships
                for fk in table.foreign_keys:
                    relationships.append(Relationship(
                        from_table=fk.from_table,
                        to_table=fk.to_table,
                        relationship_type=RelationshipType.MANY_TO_ONE,
                        source=RelationshipSource.FOREIGN_KEY,
                        from_column=fk.from_column,
                        to_column=fk.to_column,
                        constraint_name=fk.constraint_name,
                        confidence=1.0
                    ))

            cursor.close()

            return DatabaseSchema(
                database_name=db_name,
                tables=tables,
                relationships=relationships
            )

    def _extract_tables(self, cursor, schema_name: str) -> List[Table]:
        """Extract all tables from schema"""
        # Get all tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """, (schema_name,))

        table_names = [row[0] for row in cursor.fetchall()]
        tables = []

        for table_name in table_names:
            table = self._extract_table_details(cursor, table_name, schema_name)
            tables.append(table)

        return tables

    def _extract_table_details(self, cursor, table_name: str, schema_name: str) -> Table:
        """Extract detailed information about a single table"""
        # Get columns
        cursor.execute("""
            SELECT
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                CASE WHEN pk.column_name IS NOT NULL THEN TRUE ELSE FALSE END as is_pk
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name, ku.table_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                    ON tc.constraint_name = ku.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = %s
            ) pk ON c.column_name = pk.column_name AND c.table_name = pk.table_name
            WHERE c.table_schema = %s
              AND c.table_name = %s
            ORDER BY c.ordinal_position
        """, (schema_name, schema_name, table_name))

        columns = []
        primary_keys = []

        for row in cursor.fetchall():
            col_name, data_type, is_nullable, default_val, max_len, is_pk = row

            column = Column(
                name=col_name,
                data_type=data_type,
                is_nullable=(is_nullable == 'YES'),
                is_primary_key=is_pk,
                default_value=default_val,
                max_length=max_len
            )
            columns.append(column)

            if is_pk:
                primary_keys.append(col_name)

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
        row_count = cursor.fetchone()[0]

        return Table(
            name=table_name,
            columns=columns,
            primary_keys=primary_keys,
            row_count=row_count
        )

    def _extract_foreign_keys(self, cursor, table_name: str, schema_name: str) -> List[ForeignKey]:
        """Extract foreign keys for a table"""
        cursor.execute("""
            SELECT
                tc.constraint_name,
                kcu.column_name as from_column,
                ccu.table_name as to_table,
                ccu.column_name as to_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = %s
              AND tc.table_name = %s
            ORDER BY tc.constraint_name
        """, (schema_name, table_name))

        foreign_keys = []
        for row in cursor.fetchall():
            constraint_name, from_col, to_table, to_col = row
            fk = ForeignKey(
                constraint_name=constraint_name,
                from_table=table_name,
                from_column=from_col,
                to_table=to_table,
                to_column=to_col
            )
            foreign_keys.append(fk)

            # Mark column as foreign key
            # (This is a bit hacky but works in the current flow)

        return foreign_keys
