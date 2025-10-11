"""
Run graph builder to load data into Neo4j
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder.builder import build_from_ontology_file
from config.config import settings


def main():
    """Run the graph builder"""
    print("=" * 80)
    print("RELATIONAL DB TO GRAPHRAG - PHASE 2: GRAPH BUILDER")
    print("=" * 80)

    # Check for ontology file
    project_root = Path(__file__).parent.parent
    ontology_file = project_root / "output" / "graph_ontology.json"

    if not ontology_file.exists():
        print(f"\n❌ Error: Ontology file not found at {ontology_file}")
        print("   Please run Phase 1 (schema analyzer) first:")
        print("   python scripts/run_schema_analyzer.py")
        return 1

    # Validate OpenAI API key
    if not settings.validate_openai():
        print("\n❌ Error: OPENAI_API_KEY not found in .env file")
        return 1

    print(f"\n📊 Configuration:")
    print(f"   Source DB: {settings.postgres.db} @ {settings.postgres.host}:{settings.postgres.port}")
    print(f"   Target: Neo4j @ {settings.neo4j.uri}")
    print(f"   Ontology: {ontology_file}")
    print(f"   Embeddings: {settings.openai.embedding_model}")

    # Ask for confirmation
    print(f"\n⚠️  WARNING: This will clear all existing data in Neo4j!")
    response = input("   Continue? (yes/no): ").strip().lower()

    if response != 'yes':
        print("\n   Cancelled.")
        return 0

    try:
        # Run graph builder
        graph_data = build_from_ontology_file(
            ontology_file=str(ontology_file),
            postgres_conn_string=settings.postgres.connection_string,
            neo4j_uri=settings.neo4j.uri,
            neo4j_user=settings.neo4j.user,
            neo4j_password=settings.neo4j.password,
            openai_api_key=settings.openai.api_key,
            generate_embeddings=True,
            clear_existing=True
        )

        print("\n" + "=" * 80)
        print("✅ PHASE 2 COMPLETE - Knowledge Graph Built!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Open Neo4j Browser: http://localhost:7474")
        print("  2. Explore the graph with Cypher queries")
        print("  3. Proceed to Phase 3: Build Retrieval Agents")
        print("\nExample Cypher queries:")
        print("  MATCH (n) RETURN n LIMIT 25")
        print("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")

        return 0

    except Exception as e:
        print(f"\n❌ Error during graph build: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
