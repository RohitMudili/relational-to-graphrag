"""
Main schema analyzer - orchestrates the entire schema analysis process
"""
import json
from pathlib import Path
from typing import Optional
from .extractors.postgres_extractor import PostgresSchemaExtractor
from .relationship_inference.inference_engine import RelationshipInferenceEngine
from .ontology_mapper.llm_mapper import LLMOntologyMapper
from .models import DatabaseSchema, GraphOntology


class SchemaAnalyzer:
    """
    Main schema analyzer that orchestrates:
    1. Schema extraction from database
    2. Relationship inference
    3. Ontology generation using LLM
    """

    def __init__(
        self,
        db_connection_string: str,
        openai_api_key: str,
        model: str = "gpt-4"
    ):
        """
        Initialize schema analyzer

        Args:
            db_connection_string: Database connection string
            openai_api_key: OpenAI API key for LLM
            model: LLM model to use (default: gpt-4)
        """
        self.extractor = PostgresSchemaExtractor(db_connection_string)
        self.inference_engine = RelationshipInferenceEngine()
        self.llm_mapper = LLMOntologyMapper(openai_api_key, model)

    def analyze(self, schema_name: str = "public") -> tuple[DatabaseSchema, GraphOntology]:
        """
        Perform complete schema analysis

        Args:
            schema_name: Database schema to analyze (default: public)

        Returns:
            Tuple of (DatabaseSchema, GraphOntology)
        """
        print("🔍 Step 1: Extracting database schema...")
        schema = self.extractor.extract_schema(schema_name)
        print(f"   ✓ Found {len(schema.tables)} tables, {len(schema.relationships)} explicit relationships")

        print("\n🧠 Step 2: Inferring implicit relationships...")
        inferred_relationships = self.inference_engine.infer_relationships(schema)
        print(f"   ✓ Inferred {len(inferred_relationships)} additional relationships")

        # Add inferred relationships to schema
        all_relationships = schema.relationships + inferred_relationships
        schema.relationships = self.inference_engine.deduplicate_relationships(all_relationships)
        print(f"   ✓ Total relationships after deduplication: {len(schema.relationships)}")

        print("\n🤖 Step 3: Generating graph ontology with LLM...")
        ontology = self.llm_mapper.generate_ontology(schema)
        print(f"   ✓ Generated {len(ontology.node_types)} node types, {len(ontology.edge_types)} edge types")

        print("\n✅ Schema analysis complete!")
        return schema, ontology

    def save_results(
        self,
        schema: DatabaseSchema,
        ontology: GraphOntology,
        output_dir: str = "output"
    ):
        """
        Save analysis results to JSON files

        Args:
            schema: DatabaseSchema to save
            ontology: GraphOntology to save
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save database schema
        schema_file = output_path / "database_schema.json"
        with open(schema_file, 'w') as f:
            json.dump(schema.to_dict(), f, indent=2)
        print(f"\n💾 Saved database schema to: {schema_file}")

        # Save graph ontology
        ontology_file = output_path / "graph_ontology.json"
        with open(ontology_file, 'w') as f:
            json.dump(ontology.to_dict(), f, indent=2)
        print(f"💾 Saved graph ontology to: {ontology_file}")

        # Save summary report
        summary_file = output_path / "analysis_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(self._generate_summary_report(schema, ontology))
        print(f"💾 Saved analysis summary to: {summary_file}")

    def _generate_summary_report(self, schema: DatabaseSchema, ontology: GraphOntology) -> str:
        """Generate human-readable summary report"""
        lines = []

        lines.append("=" * 80)
        lines.append("SCHEMA ANALYSIS SUMMARY")
        lines.append("=" * 80)

        lines.append(f"\nDatabase: {schema.database_name}")
        lines.append(f"\nDatabase Statistics:")
        lines.append(f"  • Total tables: {len(schema.tables)}")
        lines.append(f"  • Total columns: {sum(len(t.columns) for t in schema.tables)}")
        lines.append(f"  • Total relationships: {len(schema.relationships)}")
        lines.append(f"  • Junction tables: {len([t for t in schema.tables if t.is_junction_table()])}")

        lines.append(f"\nGraph Ontology:")
        lines.append(f"  • Node types: {len(ontology.node_types)}")
        lines.append(f"  • Edge types: {len(ontology.edge_types)}")

        lines.append(f"\nNode Types:")
        for node in ontology.node_types:
            lines.append(f"  • {node.label} (from {node.source_table})")
            lines.append(f"    Properties: {len(node.properties)}")

        lines.append(f"\nEdge Types:")
        for edge in ontology.edge_types:
            lines.append(f"  • {edge.from_node} -{edge.label}-> {edge.to_node}")

        lines.append(f"\nRelationship Breakdown:")
        fk_rels = [r for r in schema.relationships if r.source.value == 'foreign_key']
        inferred_rels = [r for r in schema.relationships if r.source.value != 'foreign_key']
        lines.append(f"  • Foreign key relationships: {len(fk_rels)}")
        lines.append(f"  • Inferred relationships: {len(inferred_rels)}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def analyze_northwind(
    db_connection_string: str,
    openai_api_key: str,
    output_dir: str = "output"
) -> tuple[DatabaseSchema, GraphOntology]:
    """
    Convenience function to analyze Northwind database

    Args:
        db_connection_string: PostgreSQL connection string
        openai_api_key: OpenAI API key
        output_dir: Output directory for results

    Returns:
        Tuple of (DatabaseSchema, GraphOntology)
    """
    analyzer = SchemaAnalyzer(db_connection_string, openai_api_key)
    schema, ontology = analyzer.analyze()
    analyzer.save_results(schema, ontology, output_dir)

    return schema, ontology
