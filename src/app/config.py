"""Application configuration management."""

import os
from typing import Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    environment: str = "development"
    debug: bool = False
    secret_key: str
    version: str = "0.1.0"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Database Configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_db: str = "rag_db"
    database_url: Optional[str] = None

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: Optional[str] = None

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:latest"
    ollama_embed_model: str = "embeddinggemma:latest"
    ollama_vision_model: str = "devstral-small-2:latest"

    # File Handling
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list[str] = [".pdf", ".txt", ".jpg", ".jpeg", ".png"]
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"

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

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("database_url", pre=True, always=True)
    def assemble_database_url(cls, v, values):
        """Assemble database URL from components if not provided."""
        if v:
            return v
        return (
            f"postgresql://{values.get('postgres_user')}:"
            f"{values.get('postgres_password')}@"
            f"{values.get('postgres_host')}:"
            f"{values.get('postgres_port')}/"
            f"{values.get('postgres_db')}"
        )

    @validator("redis_url", pre=True, always=True)
    def assemble_redis_url(cls, v, values):
        """Assemble Redis URL from components if not provided."""
        if v:
            return v
        return f"redis://{values.get('redis_host')}:{values.get('redis_port')}"

    @validator("secret_key", pre=True, always=True)
    def generate_secret_key(cls, v):
        """Generate a random secret key if not provided."""
        if v:
            return v
        import secrets

        return secrets.token_hex(32)

    def get_upload_path(self) -> str:
        """Get the full upload directory path."""
        return os.path.join(os.getcwd(), self.upload_dir)

    def get_processed_path(self) -> str:
        """Get the full processed directory path."""
        return os.path.join(os.getcwd(), self.processed_dir)


# Global settings instance
settings = Settings()
