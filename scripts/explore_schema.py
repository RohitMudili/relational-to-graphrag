"""
Explore the Northwind database schema
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from config.config import settings

console = Console()


def get_tables_info(cursor):
    """Get all tables with their columns"""
    cursor.execute("""
        SELECT
            t.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable,
            CASE
                WHEN pk.column_name IS NOT NULL THEN 'PK'
                ELSE ''
            END as is_primary_key
        FROM information_schema.tables t
        LEFT JOIN information_schema.columns c
            ON t.table_name = c.table_name
        LEFT JOIN (
            SELECT ku.table_name, ku.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage ku
                ON tc.constraint_name = ku.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
        ) pk ON c.table_name = pk.table_name AND c.column_name = pk.column_name
        WHERE t.table_schema = 'public'
        ORDER BY t.table_name, c.ordinal_position;
    """)
    return cursor.fetchall()


def get_foreign_keys(cursor):
    """Get all foreign key relationships"""
    cursor.execute("""
        SELECT
            tc.table_name as from_table,
            kcu.column_name as from_column,
            ccu.table_name as to_table,
            ccu.column_name as to_column,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        ORDER BY tc.table_name, kcu.column_name;
    """)
    return cursor.fetchall()


def display_schema_tree(tables_info, foreign_keys):
    """Display schema as a tree structure"""
    tree = Tree("[bold cyan]Northwind Database Schema[/bold cyan]")

    # Group columns by table
    tables = {}
    for table_name, col_name, data_type, nullable, is_pk in tables_info:
        if table_name not in tables:
            tables[table_name] = []
        tables[table_name].append({
            'name': col_name,
            'type': data_type,
            'nullable': nullable,
            'is_pk': is_pk
        })

    # Build tree
    for table_name, columns in sorted(tables.items()):
        table_node = tree.add(f"[bold yellow]{table_name}[/bold yellow]")

        for col in columns:
            pk_marker = "[red]🔑 [/red]" if col['is_pk'] else "  "
            nullable_marker = "NULL" if col['nullable'] == 'YES' else "NOT NULL"
            col_str = f"{pk_marker}[green]{col['name']}[/green]: {col['type']} ({nullable_marker})"
            table_node.add(col_str)

    console.print(tree)


def display_relationships(foreign_keys):
    """Display foreign key relationships"""
    console.print("\n[bold cyan]Relationships (Foreign Keys)[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("From Table", style="yellow")
    table.add_column("Column", style="green")
    table.add_column("→", justify="center")
    table.add_column("To Table", style="yellow")
    table.add_column("Column", style="green")

    for from_table, from_col, to_table, to_col, constraint in foreign_keys:
        table.add_row(from_table, from_col, "→", to_table, to_col)

    console.print(table)


def analyze_schema_for_graph():
    """Analyze schema and suggest graph structure"""
    console.print("\n[bold cyan]Suggested Graph Structure[/bold cyan]")

    suggestions = {
        "customers": "Customer nodes",
        "employees": "Employee nodes",
        "orders": "Order nodes",
        "products": "Product nodes",
        "suppliers": "Supplier nodes",
        "categories": "Category nodes",
        "shippers": "Shipper nodes",
        "order_details": "Relationships between Orders and Products (CONTAINS)",
    }

    tree = Tree("[bold]Graph Ontology Suggestions[/bold]")

    nodes_tree = tree.add("[yellow]Node Types[/yellow]")
    relationships_tree = tree.add("[green]Relationship Types[/green]")

    for table, suggestion in suggestions.items():
        if "nodes" in suggestion.lower():
            nodes_tree.add(f"{table} → {suggestion}")
        else:
            relationships_tree.add(f"{table} → {suggestion}")

    # Add common relationship patterns
    relationships_tree.add("customers → orders: [cyan]PLACED[/cyan]")
    relationships_tree.add("orders → products: [cyan]CONTAINS[/cyan]")
    relationships_tree.add("products → categories: [cyan]BELONGS_TO[/cyan]")
    relationships_tree.add("products → suppliers: [cyan]SUPPLIED_BY[/cyan]")
    relationships_tree.add("orders → employees: [cyan]PROCESSED_BY[/cyan]")
    relationships_tree.add("orders → shippers: [cyan]SHIPPED_BY[/cyan]")

    console.print(tree)


def get_sample_data(cursor, table_name, limit=3):
    """Get sample data from a table"""
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    return cursor.fetchall(), [desc[0] for desc in cursor.description]


def main():
    """Main exploration function"""
    console.print("[bold]Northwind Database Schema Explorer[/bold]")
    console.print("=" * 80)

    try:
        conn = psycopg2.connect(
            host=settings.postgres.host,
            port=settings.postgres.port,
            database=settings.postgres.database,
            user=settings.postgres.user,
            password=settings.postgres.password
        )
        cursor = conn.cursor()

        # Get schema information
        tables_info = get_tables_info(cursor)
        foreign_keys = get_foreign_keys(cursor)

        # Display schema
        display_schema_tree(tables_info, foreign_keys)
        display_relationships(foreign_keys)
        analyze_schema_for_graph()

        # Show sample data from a table
        console.print("\n[bold cyan]Sample Data from 'customers' table:[/bold cyan]")
        rows, columns = get_sample_data(cursor, "customers", limit=2)

        if rows:
            table = Table(show_header=True, header_style="bold magenta")
            for col in columns[:5]:  # Show first 5 columns
                table.add_column(col, style="cyan")

            for row in rows:
                table.add_row(*[str(val)[:30] for val in row[:5]])

            console.print(table)

        console.print("\n[bold green]✓ Schema exploration complete![/bold green]")
        console.print("\n[cyan]Key Insights:[/cyan]")
        console.print(f"  • Total tables: {len(set(t[0] for t in tables_info))}")
        console.print(f"  • Foreign key relationships: {len(foreign_keys)}")
        console.print("  • This schema is ideal for graph transformation!")

        cursor.close()
        conn.close()

    except Exception as e:
        console.print(f"[red]Error exploring schema: {e}[/red]")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
