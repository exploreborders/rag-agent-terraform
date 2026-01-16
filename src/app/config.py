"""Application configuration management."""

import os

from pydantic import ConfigDict, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    environment: str = "development"
    debug: bool = True  # Development mode has debug enabled
    secret_key: str = "your-secret-key-here-change-in-production"
    version: str = "0.1.0"

    # API Settings
    api_host: str = "0.0.0.0"  # Bind to all interfaces for containerized deployment
    api_port: int = 8000
    api_reload: bool = True

    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_db: str = "rag_db"
    database_url: str = "postgresql://rag_user:rag_password@localhost:5432/rag_db"

    @field_validator("database_url", mode="before")
    @classmethod
    def assemble_database_url(cls, v, info):
        """Assemble database URL from components."""
        if (
            isinstance(v, str)
            and v != "postgresql://rag_user:rag_password@localhost:5432/rag_db"
        ):
            return v
        values = info.data
        return f"postgresql://{values.get('postgres_user', 'rag_user')}:{values.get('postgres_password', 'rag_password')}@{values.get('postgres_host', 'localhost')}:{values.get('postgres_port', 5432)}/{values.get('postgres_db', 'rag_db')}"

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str = "redis://localhost:6379"

    @field_validator("redis_url", mode="before")
    @classmethod
    def assemble_redis_url(cls, v, info):
        """Assemble redis URL from components."""
        if isinstance(v, str) and v != "redis://localhost:6379":
            return v
        values = info.data
        return f"redis://{values.get('redis_host', 'localhost')}:{values.get('redis_port', 6379)}"

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_embed_model: str = "embeddinggemma:latest"
    ollama_vision_model: str = "devstral-small-2:latest"

    # MCP Coordinator Configuration
    mcp_coordinator_host: str = "localhost"
    mcp_coordinator_port: int = 8001
    mcp_coordinator_url: str = "http://localhost:8001"

    @field_validator("mcp_coordinator_url", mode="before")
    @classmethod
    def assemble_mcp_coordinator_url(cls, v, info):
        """Assemble MCP coordinator URL from components."""
        if isinstance(v, str) and v != "http://localhost:8001":
            return v
        values = info.data
        host = values.get("mcp_coordinator_host", "localhost")
        port = values.get("mcp_coordinator_port", 8001)
        return f"http://{host}:{port}"

    # File Handling
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: str = ".pdf,.txt,.jpg,.jpeg,.png"
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Get allowed extensions as a list."""
        extensions = [
            ext.strip() for ext in self.allowed_extensions.split(",") if ext.strip()
        ]
        return [f".{ext}" if not ext.startswith(".") else ext for ext in extensions]

    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_document: int = 1000

    # Vector Search
    vector_dimension: int = 768
    similarity_threshold: float = 0.7
    top_k_results: int = 5
    max_documents_per_query: int = 10

    # Performance
    max_workers: int = 4
    batch_size: int = 10
    cache_ttl: int = 3600  # 1 hour

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Health Check
    health_check_interval: int = 30

    model_config = ConfigDict(env_file=".env", case_sensitive=False)

    def get_upload_path(self) -> str:
        """Get the full upload directory path."""
        return os.path.join(os.getcwd(), self.upload_dir)

    def get_processed_path(self) -> str:
        """Get the full processed directory path."""
        return os.path.join(os.getcwd(), self.processed_dir)


# Global settings instance
settings = Settings()
