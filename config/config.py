"""
Configuration management for the GraphRAG system
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env file FIRST before any settings classes are instantiated
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    # Try loading from current directory as fallback
    load_dotenv(override=True)

# Check if running in Streamlit Cloud and load secrets
try:
    import streamlit as st
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        # Load Streamlit secrets into environment variables
        for key, value in st.secrets.items():
            if key not in os.environ:  # Don't override existing env vars
                os.environ[key] = str(value)
except (ImportError, FileNotFoundError, RuntimeError):
    # Not running in Streamlit or secrets not available
    pass


class PostgresConfig(BaseSettings):
    """PostgreSQL database configuration"""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    db: str = Field(default="neondb", alias="database")  # Changed default to neondb
    user: str = Field(default="postgres")
    password: str = Field(default="postgres123")

    class Config:
        case_sensitive = False
        env_prefix = "POSTGRES_"
        populate_by_name = True

    @property
    def database(self) -> str:
        """Alias for db field"""
        return self.db

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class Neo4jConfig(BaseSettings):
    """Neo4j database configuration"""
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="neo4jpassword")

    class Config:
        env_prefix = "NEO4J_"
        case_sensitive = False


class NeptuneConfig(BaseSettings):
    """AWS Neptune configuration"""
    endpoint: Optional[str] = Field(default=None, env="NEPTUNE_ENDPOINT")
    port: int = Field(default=8182, env="NEPTUNE_PORT")
    region: str = Field(default="us-east-1", env="NEPTUNE_REGION")

    @property
    def connection_string(self) -> Optional[str]:
        if self.endpoint:
            return f"wss://{self.endpoint}:{self.port}/gremlin"
        return None


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration"""
    api_key: str = Field(default="")
    model: str = Field(default="gpt-4")
    embedding_model: str = Field(default="text-embedding-3-small")

    class Config:
        case_sensitive = False
        env_prefix = "OPENAI_"


class AppConfig(BaseSettings):
    """Application configuration"""
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")

    # Agent configuration
    agent_max_iterations: int = Field(default=5, env="AGENT_MAX_ITERATIONS")
    streaming_enabled: bool = Field(default=True, env="STREAMING_ENABLED")


class Settings:
    """Main settings container"""

    def __init__(self):
        # Debug: print what's in os.environ
        import sys
        if 'debug' in sys.argv or os.getenv('DEBUG_CONFIG'):
            print(f"DEBUG: os.environ['POSTGRES_HOST'] = {os.getenv('POSTGRES_HOST', 'NOT SET')}")

        self.postgres = PostgresConfig()
        self.neo4j = Neo4jConfig()
        self.neptune = NeptuneConfig()
        self.openai = OpenAIConfig()
        self.app = AppConfig()

    def validate_openai(self) -> bool:
        """Check if OpenAI API key is configured"""
        return bool(self.openai.api_key and self.openai.api_key != "")


# Global settings instance
settings = Settings()
