"""
Load Northwind SQL file to Neon.tech PostgreSQL
No psql required - uses psycopg2 directly
"""
import psycopg2
from pathlib import Path
import sys

def load_sql_file(connection_string: str, sql_file_path: str):
    """
    Load SQL file into PostgreSQL database

    Args:
        connection_string: PostgreSQL connection string
        sql_file_path: Path to SQL file
    """
    print(f"🔗 Connecting to Neon database...")

    try:
        # Connect to database (add SSL mode for Neon)
        # If connection string doesn't have sslmode, add it
        if 'sslmode' not in connection_string:
            connection_string += '?sslmode=require' if '?' not in connection_string else '&sslmode=require'

        conn = psycopg2.connect(connection_string)
        conn.autocommit = False
        cursor = conn.cursor()

        print(f"✅ Connected successfully!")
        print(f"📁 Reading SQL file: {sql_file_path}")

        # Read SQL file
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        print(f"📊 Executing SQL statements...")
        print(f"   (This may take 30-60 seconds...)")

        # Execute SQL
        cursor.execute(sql_content)
        conn.commit()

        print(f"✅ SQL file loaded successfully!")

        # Verify data
        print(f"\n🔍 Verifying data...")

        # Count tables
        cursor.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        table_count = cursor.fetchone()[0]
        print(f"   ✅ Tables created: {table_count}")

        # Count customers
        try:
            cursor.execute("SELECT COUNT(*) FROM customers")
            customer_count = cursor.fetchone()[0]
            print(f"   ✅ Customers: {customer_count}")
        except:
            print(f"   ⚠️  Could not count customers")

        # Count orders
        try:
            cursor.execute("SELECT COUNT(*) FROM orders")
            order_count = cursor.fetchone()[0]
            print(f"   ✅ Orders: {order_count}")
        except:
            print(f"   ⚠️  Could not count orders")

        # Count products
        try:
            cursor.execute("SELECT COUNT(*) FROM products")
            product_count = cursor.fetchone()[0]
            print(f"   ✅ Products: {product_count}")
        except:
            print(f"   ⚠️  Could not count products")

        cursor.close()
        conn.close()

        print(f"\n🎉 Database setup complete!")
        print(f"✅ Your Neon database is ready to use")

        return True

    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ SQL file not found: {sql_file_path}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 NEON.TECH DATABASE LOADER")
    print("=" * 60)

    # Get connection string
    if len(sys.argv) > 1:
        connection_string = sys.argv[1]
    else:
        print("\n📝 Please enter your Neon connection string:")
        print("   (Format: postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb)")
        connection_string = input("\nConnection string: ").strip()

    if not connection_string:
        print("❌ Connection string is required!")
        sys.exit(1)

    # Find SQL file
    project_root = Path(__file__).parent.parent
    sql_file = project_root / "data" / "northwind.sql"

    if not sql_file.exists():
        print(f"❌ SQL file not found at: {sql_file}")
        print(f"   Please ensure the file exists")
        sys.exit(1)

    print(f"\n📂 SQL file location: {sql_file}")
    print(f"💾 File size: {sql_file.stat().st_size / 1024:.1f} KB")

    # Load the data
    print(f"\n" + "=" * 60)
    success = load_sql_file(connection_string, str(sql_file))
    print("=" * 60)

    if success:
        print(f"\n✅ SUCCESS! Your database is ready.")
        print(f"\n📝 Next steps:")
        print(f"   1. Update your .env file with these credentials")
        print(f"   2. Run: python scripts/test_connections.py")
        print(f"   3. Proceed to Phase 3 (Neo4j setup)")
        sys.exit(0)
    else:
        print(f"\n❌ FAILED! Please check the errors above.")
        sys.exit(1)
