"""
Run retrieval agent - interactive natural language query interface
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval_agents.orchestrator import AgentOrchestrator
from config.config import settings


def print_results(result_dict: dict):
    """Pretty print query results"""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Print strategy info
    strategy = result_dict.get("strategy", {})
    print(f"\n📊 Strategy: {strategy.get('type', 'unknown')}")
    print(f"   Tool: {strategy.get('primary_tool', 'unknown')}")
    print(f"   Reasoning: {strategy.get('reasoning', 'N/A')}")

    # Print results
    results = result_dict.get("results", [])
    count = result_dict.get("result_count", 0)

    print(f"\n✅ Found {count} results\n")

    if not results:
        print("No results found.")
        return

    # Display first few results
    display_limit = min(5, len(results))

    for i, item in enumerate(results[:display_limit], 1):
        print(f"--- Result {i} ---")

        # Handle different result types
        if "error" in item:
            print(f"❌ Error: {item['error']}")
        elif "node_id" in item:
            # Node result
            print(f"Node ID: {item.get('node_id')}")
            print(f"Label: {item.get('label', 'Unknown')}")
            if "score" in item:
                print(f"Similarity: {item['score']:.3f}")
            if "properties" in item:
                props = item["properties"]
                # Display key properties
                for key, value in list(props.items())[:5]:
                    if key != "embedding" and key != "node_id":
                        print(f"  {key}: {value}")
        elif "nodes" in item and "relationships" in item:
            # Graph structure result
            print(f"Nodes: {len(item['nodes'])}")
            print(f"Relationships: {len(item['relationships'])}")
            if "path_length" in item:
                print(f"Path Length: {item['path_length']}")
        elif "query" in item and "cypher" in item:
            # Cypher query result
            print(f"Query: {item['query']}")
            print(f"Cypher: {item['cypher']}")
        else:
            # Generic result
            print(json.dumps(item, indent=2, default=str)[:300])

        print()

    if len(results) > display_limit:
        print(f"... and {len(results) - display_limit} more results")

    print("=" * 80)


def interactive_mode(orchestrator: AgentOrchestrator):
    """Run interactive query mode"""
    print("\n" + "=" * 80)
    print("INTERACTIVE RETRIEVAL MODE")
    print("=" * 80)
    print("\nEnter natural language queries to search the knowledge graph.")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - 'explain' to get explanation of last results")
    print("  - 'quit' or 'exit' to quit")
    print("=" * 80 + "\n")

    last_result = None

    while True:
        try:
            query = input("\n🔍 Query> ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if query.lower() == "explain":
                if last_result:
                    print("\n💡 Explanation:")
                    explanation = orchestrator.explain_results(last_result)
                    print(explanation)
                else:
                    print("No previous results to explain.")
                continue

            # Execute query
            result = orchestrator.query(query)
            last_result = result

            # Display results
            print_results(result.to_dict())

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def single_query_mode(orchestrator: AgentOrchestrator, query: str):
    """Run a single query and exit"""
    print(f"\n🔍 Query: {query}\n")

    try:
        # Execute query
        result = orchestrator.query(query)

        # Display results
        print_results(result.to_dict())

        # Generate explanation
        print("\n💡 Explanation:")
        explanation = orchestrator.explain_results(result)
        print(explanation)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def streaming_mode(orchestrator: AgentOrchestrator, query: str):
    """Run query with streaming updates"""
    print(f"\n🔍 Query: {query}\n")

    try:
        for update in orchestrator.streaming_query(query):
            status = update.get("status")

            if status == "planning":
                print(f"📋 {update['message']}")
            elif status == "strategy":
                print(f"🎯 Strategy: {update['strategy']}")
                print(f"   {update['reasoning']}")
            elif status == "executing":
                print(f"⚙️  {update['message']}")
            elif status == "results":
                print_results(update["data"])
            elif status == "explaining":
                print(f"\n💡 {update['message']}")
            elif status == "complete":
                print(update["explanation"])

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Natural language query interface for GraphRAG system"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to execute (non-interactive mode)"
    )
    parser.add_argument(
        "--stream", "-s",
        action="store_true",
        help="Enable streaming mode for query execution"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Number of results for vector search (default: 5)"
    )
    parser.add_argument(
        "--min-score", "-m",
        type=float,
        default=0.7,
        help="Minimum similarity score (default: 0.7)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RELATIONAL DB TO GRAPHRAG - PHASE 3: RETRIEVAL SYSTEM")
    print("=" * 80)

    # Validate configuration
    if not settings.validate_openai():
        print("\n❌ Error: OPENAI_API_KEY not found in .env file")
        return 1

    # Basic Neo4j validation
    if not settings.neo4j.uri or not settings.neo4j.password:
        print("\n❌ Error: Neo4j configuration incomplete in .env file")
        print("   Required: NEO4J_URI, NEO4J_PASSWORD")
        return 1

    # Initialize orchestrator
    print(f"\n🔗 Connecting to Neo4j: {settings.neo4j.uri}")
    print(f"🤖 Using LLM: {settings.openai.model}")
    print(f"🔢 Embedding Model: {settings.openai.embedding_model}\n")

    try:
        orchestrator = AgentOrchestrator(
            neo4j_uri=settings.neo4j.uri,
            neo4j_user=settings.neo4j.user,
            neo4j_password=settings.neo4j.password,
            openai_api_key=settings.openai.api_key,
            model=settings.openai.model,
            embedding_model=settings.openai.embedding_model
        )

        # Run appropriate mode
        if args.query:
            if args.stream:
                exit_code = streaming_mode(orchestrator, args.query)
            else:
                exit_code = single_query_mode(orchestrator, args.query)
        else:
            # Interactive mode
            interactive_mode(orchestrator)
            exit_code = 0

        # Cleanup
        orchestrator.close()

        if exit_code == 0:
            print("\n" + "=" * 80)
            print("✅ RETRIEVAL COMPLETE")
            print("=" * 80)

        return exit_code

    except Exception as e:
        print(f"\n❌ Error initializing retrieval system: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
