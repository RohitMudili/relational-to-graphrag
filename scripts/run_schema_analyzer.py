"""
Run schema analyzer on Northwind database
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema_analyzer.analyzer import SchemaAnalyzer
from config.config import settings


def main():
    """Run the schema analyzer"""
    print("=" * 80)
    print("RELATIONAL DB TO GRAPHRAG - PHASE 1: SCHEMA ANALYZER")
    print("=" * 80)

    # Validate OpenAI API key
    if not settings.validate_openai():
        print("\n❌ Error: OPENAI_API_KEY not found in .env file")
        print("   Please add your OpenAI API key to the .env file")
        return 1

    # Initialize analyzer
    print(f"\n📊 Analyzing database: {settings.postgres.db}")
    print(f"   Host: {settings.postgres.host}:{settings.postgres.port}")
    print(f"   Using LLM: {settings.openai.model}\n")

    try:
        analyzer = SchemaAnalyzer(
            db_connection_string=settings.postgres.connection_string,
            openai_api_key=settings.openai.api_key,
            model=settings.openai.model
        )

        # Run analysis
        schema, ontology = analyzer.analyze()

        # Save results to project root output directory
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "output"
        analyzer.save_results(schema, ontology, output_dir=str(output_dir))

        print("\n" + "=" * 80)
        print("✅ PHASE 1 COMPLETE - Schema Analysis Finished!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Review output files in 'output/' directory")
        print("  2. Validate the generated ontology")
        print("  3. Proceed to Phase 2: Graph Builder")

        return 0

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
