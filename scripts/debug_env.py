"""
Debug script to check .env loading
"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("DEBUG: Environment Variable Loading")
print("=" * 60)

# Check current directory
print(f"\n1. Current working directory: {os.getcwd()}")

# Check where script is located
script_path = Path(__file__).resolve()
print(f"2. Script location: {script_path}")
print(f"3. Script parent: {script_path.parent}")
print(f"4. Project root: {script_path.parent.parent}")

# Check .env file location
env_path = script_path.parent.parent / ".env"
print(f"\n5. Looking for .env at: {env_path}")
print(f"6. .env exists: {env_path.exists()}")

if env_path.exists():
    print(f"7. .env file size: {env_path.stat().st_size} bytes")

    # Try reading .env manually
    print("\n8. .env file contents (first 10 lines):")
    with open(env_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            if 'PASSWORD' in line or 'KEY' in line:
                # Mask sensitive data
                parts = line.split('=')
                if len(parts) == 2:
                    print(f"   {parts[0]}=***MASKED***")
            else:
                print(f"   {line.rstrip()}")

# Check environment variables BEFORE loading
print("\n9. Environment variables BEFORE dotenv:")
print(f"   POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'NOT SET')}")
print(f"   POSTGRES_DB: {os.getenv('POSTGRES_DB', 'NOT SET')}")
print(f"   NEO4J_URI: {os.getenv('NEO4J_URI', 'NOT SET')}")

# Now try loading with dotenv
print("\n10. Loading .env with python-dotenv...")
try:
    from dotenv import load_dotenv
    result = load_dotenv(env_path, override=True)
    print(f"    load_dotenv() returned: {result}")
except Exception as e:
    print(f"    ❌ Error: {e}")

# Check environment variables AFTER loading
print("\n11. Environment variables AFTER dotenv:")
print(f"    POSTGRES_HOST: {os.getenv('POSTGRES_HOST', 'NOT SET')}")
print(f"    POSTGRES_DB: {os.getenv('POSTGRES_DB', 'NOT SET')}")
print(f"    NEO4J_URI: {os.getenv('NEO4J_URI', 'NOT SET')}")

# Try loading config
print("\n12. Testing config loading...")
sys.path.insert(0, str(script_path.parent.parent))
try:
    from config.config import settings
    print(f"    settings.postgres.host: {settings.postgres.host}")
    print(f"    settings.postgres.db: {settings.postgres.db}")
    print(f"    settings.neo4j.uri: {settings.neo4j.uri}")
except Exception as e:
    print(f"    ❌ Error loading config: {e}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
