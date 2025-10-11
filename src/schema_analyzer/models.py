"""
Data models for schema analysis
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class RelationshipType(Enum):
    """Types of relationships in the schema"""
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"
    ONE_TO_ONE = "one_to_one"


class RelationshipSource(Enum):
    """How the relationship was discovered"""
    FOREIGN_KEY = "foreign_key"
    NAMING_CONVENTION = "naming_convention"
    JUNCTION_TABLE = "junction_table"
    LLM_INFERRED = "llm_inferred"


@dataclass
class Column:
    """Represents a database column"""
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool = False
    is_foreign_key: bool = False
    default_value: Optional[str] = None
    max_length: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "is_nullable": self.is_nullable,
            "is_primary_key": self.is_primary_key,
            "is_foreign_key": self.is_foreign_key,
            "default_value": self.default_value,
            "max_length": self.max_length,
        }


@dataclass
class ForeignKey:
    """Represents a foreign key constraint"""
    constraint_name: str
    from_table: str
    from_column: str
    to_table: str
    to_column: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_name": self.constraint_name,
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
        }


@dataclass
class Table:
    """Represents a database table"""
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    row_count: int = 0

    def get_column(self, name: str) -> Optional[Column]:
        """Get column by name"""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def is_junction_table(self) -> bool:
        """Check if this table is likely a junction/join table"""
        # Junction tables typically have:
        # - 2+ foreign keys
        # - Small number of columns (mostly FKs)
        # - Composite primary key or no primary key besides FKs
        if len(self.foreign_keys) < 2:
            return False

        fk_columns = {fk.from_column for fk in self.foreign_keys}
        non_fk_columns = [col for col in self.columns if col.name not in fk_columns]

        # If most columns are foreign keys, likely a junction table
        return len(non_fk_columns) <= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [col.to_dict() for col in self.columns],
            "primary_keys": self.primary_keys,
            "foreign_keys": [fk.to_dict() for fk in self.foreign_keys],
            "row_count": self.row_count,
            "is_junction_table": self.is_junction_table(),
        }


@dataclass
class Relationship:
    """Represents a relationship between tables"""
    from_table: str
    to_table: str
    relationship_type: RelationshipType
    source: RelationshipSource
    from_column: Optional[str] = None
    to_column: Optional[str] = None
    constraint_name: Optional[str] = None
    confidence: float = 1.0  # Confidence score for inferred relationships

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_table": self.from_table,
            "to_table": self.to_table,
            "relationship_type": self.relationship_type.value,
            "source": self.source.value,
            "from_column": self.from_column,
            "to_column": self.to_column,
            "constraint_name": self.constraint_name,
            "confidence": self.confidence,
        }


@dataclass
class DatabaseSchema:
    """Complete database schema representation"""
    database_name: str
    tables: List[Table] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)

    def get_table(self, name: str) -> Optional[Table]:
        """Get table by name"""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_name": self.database_name,
            "tables": [table.to_dict() for table in self.tables],
            "relationships": [rel.to_dict() for rel in self.relationships],
            "statistics": {
                "total_tables": len(self.tables),
                "total_columns": sum(len(t.columns) for t in self.tables),
                "total_relationships": len(self.relationships),
                "junction_tables": len([t for t in self.tables if t.is_junction_table()]),
            }
        }


@dataclass
class NodeType:
    """Represents a node type in the graph ontology"""
    label: str
    source_table: str
    properties: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "source_table": self.source_table,
            "properties": self.properties,
            "description": self.description,
        }


@dataclass
class EdgeType:
    """Represents an edge/relationship type in the graph ontology"""
    label: str
    from_node: str
    to_node: str
    source_relationship: Optional[str] = None
    properties: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "source_relationship": self.source_relationship,
            "properties": self.properties,
            "description": self.description,
        }


@dataclass
class GraphOntology:
    """Graph ontology/schema definition"""
    node_types: List[NodeType] = field(default_factory=list)
    edge_types: List[EdgeType] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_types": [node.to_dict() for node in self.node_types],
            "edge_types": [edge.to_dict() for edge in self.edge_types],
            "statistics": {
                "total_node_types": len(self.node_types),
                "total_edge_types": len(self.edge_types),
            }
        }
