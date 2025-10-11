"""
Streamlit Frontend for Relational-to-GraphRAG System
A professional web interface for natural language graph queries
"""
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import time
import streamlit as st

# Add parent directory to path before importing local modules
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval_agents.orchestrator import AgentOrchestrator, RetrievalResult
from config.config import settings

# Page configuration
st.set_page_config(
    page_title="GraphRAG Query System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .query-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'connected' not in st.session_state:
    st.session_state.connected = False


def initialize_orchestrator():
    """Initialize the agent orchestrator"""
    try:
        with st.spinner("🔗 Connecting to databases..."):
            orchestrator = AgentOrchestrator(
                neo4j_uri=settings.neo4j.uri,
                neo4j_user=settings.neo4j.user,
                neo4j_password=settings.neo4j.password,
                openai_api_key=settings.openai.api_key,
                model=settings.openai.model,
                embedding_model=settings.openai.embedding_model
            )
            st.session_state.orchestrator = orchestrator
            st.session_state.connected = True
            return True
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")
        st.session_state.connected = False
        return False


def format_result_display(result: RetrievalResult):
    """Format and display query results"""
    result_dict = result.to_dict()

    # Strategy info
    st.markdown("### 📊 Query Strategy")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Strategy Type", result_dict['strategy']['type'].replace('_', ' ').title())
    with col2:
        st.metric("Primary Tool", result_dict['strategy']['primary_tool'].replace('_', ' ').title())
    with col3:
        st.metric("Results Found", result_dict['result_count'])

    st.info(f"**Reasoning:** {result_dict['strategy']['reasoning']}")

    # Results display
    st.markdown("### 🎯 Results")

    if result_dict['result_count'] == 0:
        st.warning("No results found for your query.")
        return

    results = result_dict['results']

    # Display based on result type
    for i, item in enumerate(results[:10], 1):  # Limit to first 10
        with st.expander(f"Result {i}", expanded=(i <= 3)):
            if 'error' in item:
                st.error(f"❌ Error: {item['error']}")

            elif 'node_id' in item:
                # Node result (from vector search)
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.markdown(f"**ID:** `{item.get('node_id', 'N/A')}`")
                    st.markdown(f"**Label:** {item.get('label', 'Unknown')}")
                    if 'score' in item:
                        score = item['score']
                        st.metric("Similarity Score", f"{score:.3f}")
                        st.progress(score)

                with col2:
                    if 'properties' in item:
                        props = item['properties']
                        # Filter out embedding and display key properties
                        display_props = {k: v for k, v in props.items()
                                       if k not in ['embedding', 'node_id', 'source_table', 'source_pk']}

                        if display_props:
                            st.markdown("**Properties:**")
                            # Create a nice table
                            props_df = pd.DataFrame([
                                {"Property": k, "Value": str(v)[:100]}
                                for k, v in list(display_props.items())[:10]
                            ])
                            st.dataframe(props_df, hide_index=True, use_container_width=True)

            elif 'nodes' in item and 'relationships' in item:
                # Graph structure result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nodes", len(item['nodes']))
                with col2:
                    st.metric("Relationships", len(item['relationships']))
                with col3:
                    if 'path_length' in item:
                        st.metric("Path Length", item['path_length'])

                # Display nodes
                if item['nodes']:
                    st.markdown("**Nodes:**")
                    nodes_data = []
                    for node in item['nodes'][:20]:  # Limit to 20
                        nodes_data.append({
                            "ID": node.get('node_id', 'N/A'),
                            "Label": node.get('label', 'Unknown'),
                            "Properties": str(len(node.get('properties', {})))
                        })
                    st.dataframe(pd.DataFrame(nodes_data), hide_index=True, use_container_width=True)

                # Display relationships
                if item['relationships']:
                    st.markdown("**Relationships:**")
                    rels_data = []
                    for rel in item['relationships'][:20]:
                        rels_data.append({
                            "From": rel.get('from', 'N/A'),
                            "Type": rel.get('type', 'RELATED'),
                            "To": rel.get('to', 'N/A')
                        })
                    st.dataframe(pd.DataFrame(rels_data), hide_index=True, use_container_width=True)

            else:
                # Generic result
                st.json(item)

    if len(results) > 10:
        st.info(f"📝 Showing 10 of {len(results)} results. Download full results below.")


def main():
    """Main application"""

    # Header
    st.markdown('<h1 class="main-header">🔍 GraphRAG Query System</h1>', unsafe_allow_html=True)
    st.markdown("**Natural Language Interface for Knowledge Graph Exploration**")

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/graph.png", width=80)
        st.title("Navigation")

        page = st.radio(
            "Select Page",
            ["🏠 Home", "🔍 Query Interface", "📊 Statistics", "⚙️ Configuration"],
            label_visibility="collapsed"
        )

        st.divider()

        # Connection status
        st.subheader("Connection Status")
        if st.session_state.connected:
            st.success("✅ Connected")
        else:
            st.warning("⚠️ Not Connected")
            if st.button("Connect Now", use_container_width=True):
                initialize_orchestrator()

        st.divider()

        # Settings
        st.subheader("Query Settings")
        top_k = st.slider("Results (Top-K)", 1, 20, 5)
        min_score = st.slider("Min Similarity", 0.0, 1.0, 0.7, 0.05)
        max_depth = st.slider("Max Traversal Depth", 1, 5, 2)

        st.divider()

        # About
        st.subheader("About")
        st.info("""
        **GraphRAG System**

        Transform relational databases into intelligent knowledge graphs with AI-powered retrieval.

        - 🔍 Vector Search
        - 🕸️ Graph Traversal
        - ⚙️ Cypher Queries
        - 🤖 LLM-Powered
        """)

    # Main content based on page
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔍 Query Interface":
        show_query_page(top_k, min_score, max_depth)
    elif page == "📊 Statistics":
        show_statistics_page()
    elif page == "⚙️ Configuration":
        show_config_page()


def show_home_page():
    """Home page with welcome and quick start"""
    st.markdown("## Welcome to GraphRAG Query System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>🔍 Vector Search</h3>
            <p>Find semantically similar entities using AI embeddings</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3>🕸️ Graph Traversal</h3>
            <p>Explore relationships and find paths between entities</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3>⚙️ Cypher Queries</h3>
            <p>Generate complex queries from natural language</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick Start
    st.markdown("## 🚀 Quick Start")

    st.markdown("""
    ### How to use this system:

    1. **Go to Query Interface** - Navigate using the sidebar
    2. **Ask Your Question** - Type in natural language (e.g., "Find customers similar to ALFKI")
    3. **View Results** - Get intelligent results with explanations

    ### Example Queries:

    - "Find customers similar to ALFKI"
    - "Show all products in the Beverages category"
    - "What is the reporting structure for employees?"
    - "How many orders were placed in 1997?"
    - "Find the path between customer ALFKI and product Chai"
    """)

    # Recent activity
    if st.session_state.query_history:
        st.divider()
        st.markdown("## 📜 Recent Queries")

        for query_item in reversed(st.session_state.query_history[-5:]):
            with st.expander(f"🔍 {query_item['query'][:60]}...", expanded=False):
                st.markdown(f"**Time:** {query_item['timestamp']}")
                st.markdown(f"**Strategy:** {query_item['strategy']}")
                st.markdown(f"**Results:** {query_item['result_count']}")


def show_query_page(top_k, min_score, max_depth):
    """Query interface page"""
    st.markdown("## 🔍 Query Interface")

    # Check connection
    if not st.session_state.connected:
        st.warning("⚠️ Please connect to the database first (click 'Connect Now' in sidebar)")
        if st.button("Initialize Connection"):
            initialize_orchestrator()
        return

    # Query input
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    query = st.text_area(
        "Enter your natural language query:",
        placeholder="e.g., Find customers similar to ALFKI...",
        height=100,
        key="query_input"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        execute_button = st.button("🚀 Execute Query", type="primary", use_container_width=True)
    with col2:
        explain_button = st.button("💡 Explain Last Result", use_container_width=True)
    with col3:
        clear_button = st.button("🗑️ Clear History", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Execute query
    if execute_button and query.strip():
        with st.spinner("🔄 Processing your query..."):
            try:
                # Execute query
                start_time = time.time()
                result = st.session_state.orchestrator.query(
                    query,
                    top_k=top_k,
                    min_score=min_score,
                    max_depth=max_depth
                )
                execution_time = time.time() - start_time

                # Store result
                st.session_state.last_result = result

                # Add to history
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'strategy': result.strategy.query_type.value,
                    'result_count': len(result.results),
                    'execution_time': execution_time
                })

                # Display results
                st.success(f"✅ Query executed in {execution_time:.2f}s")
                format_result_display(result)

                # Generate explanation
                with st.spinner("💭 Generating explanation..."):
                    explanation = st.session_state.orchestrator.explain_results(result)

                st.markdown("### 💡 AI Explanation")
                st.info(explanation)

                # Download option
                result_json = json.dumps(result.to_dict(), indent=2, default=str)
                st.download_button(
                    "📥 Download Results (JSON)",
                    data=result_json,
                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"❌ Error executing query: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    # Explain last result
    if explain_button:
        if st.session_state.last_result:
            with st.spinner("💭 Generating explanation..."):
                explanation = st.session_state.orchestrator.explain_results(st.session_state.last_result)
            st.markdown("### 💡 AI Explanation")
            st.info(explanation)
        else:
            st.warning("No previous results to explain.")

    # Clear history
    if clear_button:
        st.session_state.query_history = []
        st.session_state.last_result = None
        st.success("✅ History cleared!")
        st.rerun()

    # Query history
    if st.session_state.query_history:
        st.divider()
        st.markdown("## 📜 Query History")

        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(
            history_df[['timestamp', 'query', 'strategy', 'result_count', 'execution_time']],
            hide_index=True,
            use_container_width=True
        )


def show_statistics_page():
    """Statistics dashboard"""
    st.markdown("## 📊 Database Statistics")

    if not st.session_state.connected:
        st.warning("⚠️ Please connect to the database first")
        return

    try:
        with st.spinner("📊 Loading statistics..."):
            # Get Neo4j statistics
            from src.graph_builder.loaders.neo4j_loader import Neo4jLoader

            loader = Neo4jLoader(
                settings.neo4j.uri,
                settings.neo4j.user,
                settings.neo4j.password
            )

            stats = loader.get_statistics()
            loader.close()

        # Display statistics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Nodes", f"{stats['total_nodes']:,}")
        with col2:
            st.metric("Total Edges", f"{stats['total_edges']:,}")

        st.divider()

        # Nodes by label
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🏷️ Nodes by Label")
            if stats['nodes_by_label']:
                nodes_df = pd.DataFrame([
                    {"Label": k, "Count": v}
                    for k, v in sorted(stats['nodes_by_label'].items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(nodes_df, hide_index=True, use_container_width=True)

                # Bar chart
                st.bar_chart(nodes_df.set_index('Label')['Count'])

        with col2:
            st.markdown("### 🔗 Edges by Type")
            if stats['edges_by_type']:
                edges_df = pd.DataFrame([
                    {"Type": k, "Count": v}
                    for k, v in sorted(stats['edges_by_type'].items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(edges_df, hide_index=True, use_container_width=True)

                # Bar chart
                st.bar_chart(edges_df.set_index('Type')['Count'])

        # Query performance
        if st.session_state.query_history:
            st.divider()
            st.markdown("### ⏱️ Query Performance")

            history_df = pd.DataFrame(st.session_state.query_history)

            col1, col2, col3 = st.columns(3)
            with col1:
                avg_time = history_df['execution_time'].mean()
                st.metric("Avg Query Time", f"{avg_time:.2f}s")
            with col2:
                total_queries = len(history_df)
                st.metric("Total Queries", total_queries)
            with col3:
                if 'strategy' in history_df.columns:
                    most_common = history_df['strategy'].mode()[0] if len(history_df) > 0 else "N/A"
                    st.metric("Most Used Strategy", most_common.replace('_', ' ').title())

            # Strategy distribution
            if 'strategy' in history_df.columns:
                st.markdown("**Strategy Distribution:**")
                strategy_counts = history_df['strategy'].value_counts()
                st.bar_chart(strategy_counts)

    except Exception as e:
        st.error(f"❌ Error loading statistics: {e}")


def show_config_page():
    """Configuration page"""
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🗄️ Database Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### PostgreSQL (Source)")
        st.code(f"""
Host: {settings.postgres.host}
Port: {settings.postgres.port}
Database: {settings.postgres.db}
User: {settings.postgres.user}
        """)

    with col2:
        st.markdown("#### Neo4j (Graph)")
        st.code(f"""
URI: {settings.neo4j.uri}
User: {settings.neo4j.user}
        """)

    st.divider()

    st.markdown("### 🤖 AI Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### OpenAI Settings")
        st.code(f"""
Model: {settings.openai.model}
Embedding Model: {settings.openai.embedding_model}
API Key: {"✅ Configured" if settings.validate_openai() else "❌ Not configured"}
        """)

    with col2:
        st.markdown("#### Application Settings")
        st.code(f"""
Batch Size: {settings.app.batch_size}
Max Workers: {settings.app.max_workers}
Agent Max Iterations: {settings.app.agent_max_iterations}
Streaming: {"Enabled" if settings.app.streaming_enabled else "Disabled"}
        """)

    st.divider()

    # Test connections
    st.markdown("### 🧪 Test Connections")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Test PostgreSQL", use_container_width=True):
            with st.spinner("Testing PostgreSQL..."):
                try:
                    import psycopg2
                    conn = psycopg2.connect(settings.postgres.connection_string)
                    conn.close()
                    st.success("✅ PostgreSQL connection successful")
                except Exception as e:
                    st.error(f"❌ PostgreSQL connection failed: {e}")

    with col2:
        if st.button("Test Neo4j", use_container_width=True):
            with st.spinner("Testing Neo4j..."):
                try:
                    from neo4j import GraphDatabase
                    driver = GraphDatabase.driver(
                        settings.neo4j.uri,
                        auth=(settings.neo4j.user, settings.neo4j.password)
                    )
                    driver.verify_connectivity()
                    driver.close()
                    st.success("✅ Neo4j connection successful")
                except Exception as e:
                    st.error(f"❌ Neo4j connection failed: {e}")

    with col3:
        if st.button("Test OpenAI", use_container_width=True):
            with st.spinner("Testing OpenAI..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=settings.openai.api_key)
                    # Simple test
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    st.success("✅ OpenAI API connection successful")
                except Exception as e:
                    st.error(f"❌ OpenAI API connection failed: {e}")

    st.divider()

    # System info
    st.markdown("### 📋 System Information")

    import platform
    st.code(f"""
Python Version: {platform.python_version()}
Platform: {platform.system()} {platform.release()}
Machine: {platform.machine()}
    """)


if __name__ == "__main__":
    main()
