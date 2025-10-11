"""
Relationship inference engine - detect implicit relationships
"""
import re
from typing import List, Set
from ..models import DatabaseSchema, Relationship, RelationshipType, RelationshipSource, Table


class RelationshipInferenceEngine:
    """Infer implicit relationships from naming conventions and patterns"""

    def __init__(self):
        # Common foreign key patterns
        self.fk_patterns = [
            r"^(.+)_id$",  # customer_id -> customer
            r"^(.+)id$",   # customerid -> customer
            r"^id_(.+)$",  # id_customer -> customer
            r"^fk_(.+)$",  # fk_customer -> customer
        ]

    def infer_relationships(self, schema: DatabaseSchema) -> List[Relationship]:
        """
        Infer additional relationships not captured by foreign keys

        Args:
            schema: DatabaseSchema object

        Returns:
            List of inferred relationships
        """
        inferred = []

        # 1. Naming convention based inference
        inferred.extend(self._infer_from_naming(schema))

        # 2. Many-to-many through junction tables
        inferred.extend(self._infer_many_to_many(schema))

        # 3. Hierarchical relationships (self-referencing)
        inferred.extend(self._infer_hierarchical(schema))

        return inferred

    def _infer_from_naming(self, schema: DatabaseSchema) -> List[Relationship]:
        """Infer relationships from column naming conventions"""
        inferred = []
        table_names = {t.name.lower() for t in schema.tables}

        for table in schema.tables:
            for column in table.columns:
                # Skip if already a known foreign key
                if column.is_foreign_key:
                    continue

                # Try to match patterns
                for pattern in self.fk_patterns:
                    match = re.match(pattern, column.name.lower())
                    if match:
                        potential_table = match.group(1)

                        # Handle plural/singular
                        candidates = [
                            potential_table,
                            potential_table + 's',  # customer -> customers
                            potential_table[:-1] if potential_table.endswith('s') else None,  # customers -> customer
                        ]

                        for candidate in candidates:
                            if candidate and candidate in table_names:
                                # Found a match!
                                inferred.append(Relationship(
                                    from_table=table.name,
                                    to_table=candidate,
                                    relationship_type=RelationshipType.MANY_TO_ONE,
                                    source=RelationshipSource.NAMING_CONVENTION,
                                    from_column=column.name,
                                    to_column=None,  # Unknown
                                    confidence=0.8  # Lower confidence than explicit FK
                                ))
                                break

        return inferred

    def _infer_many_to_many(self, schema: DatabaseSchema) -> List[Relationship]:
        """Infer many-to-many relationships through junction tables"""
        inferred = []

        # Find junction tables
        junction_tables = [t for t in schema.tables if t.is_junction_table()]

        for junction in junction_tables:
            if len(junction.foreign_keys) >= 2:
                # Get the two main foreign keys
                fk1 = junction.foreign_keys[0]
                fk2 = junction.foreign_keys[1]

                # Create bidirectional many-to-many relationships
                inferred.append(Relationship(
                    from_table=fk1.to_table,
                    to_table=fk2.to_table,
                    relationship_type=RelationshipType.MANY_TO_MANY,
                    source=RelationshipSource.JUNCTION_TABLE,
                    from_column=None,
                    to_column=None,
                    constraint_name=f"via_{junction.name}",
                    confidence=1.0
                ))

                inferred.append(Relationship(
                    from_table=fk2.to_table,
                    to_table=fk1.to_table,
                    relationship_type=RelationshipType.MANY_TO_MANY,
                    source=RelationshipSource.JUNCTION_TABLE,
                    from_column=None,
                    to_column=None,
                    constraint_name=f"via_{junction.name}",
                    confidence=1.0
                ))

        return inferred

    def _infer_hierarchical(self, schema: DatabaseSchema) -> List[Relationship]:
        """Infer hierarchical (self-referencing) relationships"""
        inferred = []

        for table in schema.tables:
            for fk in table.foreign_keys:
                # Self-referencing foreign key
                if fk.from_table == fk.to_table:
                    inferred.append(Relationship(
                        from_table=table.name,
                        to_table=table.name,
                        relationship_type=RelationshipType.ONE_TO_MANY,
                        source=RelationshipSource.FOREIGN_KEY,
                        from_column=fk.from_column,
                        to_column=fk.to_column,
                        constraint_name=fk.constraint_name,
                        confidence=1.0
                    ))

        return inferred

    def deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships, keeping higher confidence ones"""
        seen = {}

        for rel in relationships:
            key = (rel.from_table, rel.to_table, rel.relationship_type)

            if key not in seen or seen[key].confidence < rel.confidence:
                seen[key] = rel

        return list(seen.values())
