"""
Test database connectivity for PostgreSQL and Neo4j
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from neo4j import GraphDatabase
from rich.console import Console
from rich.table import Table
from config.config import settings

console = Console()


def test_postgres_connection():
    """Test PostgreSQL connection and query Northwind database"""
    console.print("\n[bold blue]Testing PostgreSQL Connection...[/bold blue]")

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=settings.postgres.host,
            port=settings.postgres.port,
            database=settings.postgres.database,
            user=settings.postgres.user,
            password=settings.postgres.password
        )
        cursor = conn.cursor()

        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        console.print(f"[green]✓ Connected to PostgreSQL[/green]")
        console.print(f"  Version: {version[:50]}...")

        # Check Northwind tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        if tables:
            console.print(f"[green]✓ Northwind database loaded with {len(tables)} tables[/green]")

            # Display tables in a nice format
            table = Table(title="Northwind Tables")
            table.add_column("Table Name", style="cyan")
            table.add_column("Row Count", style="magenta")

            for (table_name,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                table.add_row(table_name, str(count))

            console.print(table)
        else:
            console.print("[yellow]⚠ No tables found. Database may not be initialized.[/yellow]")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        console.print(f"[red]✗ PostgreSQL connection failed: {e}[/red]")
        return False


def test_neo4j_connection():
    """Test Neo4j connection"""
    console.print("\n[bold blue]Testing Neo4j Connection...[/bold blue]")

    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            settings.neo4j.uri,
            auth=(settings.neo4j.user, settings.neo4j.password)
        )

        # Test query
        with driver.session() as session:
            result = session.run("CALL dbms.components() YIELD name, versions, edition")
            record = result.single()

            console.print(f"[green]✓ Connected to Neo4j[/green]")
            console.print(f"  Edition: {record['edition']}")
            console.print(f"  Version: {record['versions'][0]}")

            # Check if database is empty
            result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = result.single()["count"]
            console.print(f"  Current nodes: {node_count}")

            if node_count == 0:
                console.print("[yellow]  Database is empty (ready for import)[/yellow]")

        driver.close()
        return True

    except Exception as e:
        console.print(f"[red]✗ Neo4j connection failed: {e}[/red]")
        console.print("[yellow]  Make sure Neo4j is running: docker-compose up -d[/yellow]")
        return False


def main():
    """Run all connection tests"""
    console.print("[bold]Database Connectivity Test[/bold]")
    console.print("=" * 60)

    postgres_ok = test_postgres_connection()
    neo4j_ok = test_neo4j_connection()

    console.print("\n" + "=" * 60)
    if postgres_ok and neo4j_ok:
        console.print("[bold green]✓ All connections successful![/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("  1. Explore the schema: python scripts/explore_schema.py")
        console.print("  2. Start building the schema analyzer")
        return 0
    else:
        console.print("[bold red]✗ Some connections failed[/bold red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  • Make sure Docker is running")
        console.print("  • Start services: docker-compose up -d")
        console.print("  • Check logs: docker-compose logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
