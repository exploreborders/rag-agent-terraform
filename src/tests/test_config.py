"""Unit tests for application configuration."""

import os
from unittest.mock import patch

from app.config import Settings


class TestSettings:
    """Test cases for application settings."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = Settings()
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.vector_dimension == 768

    def test_database_url_default(self):
        """Test default database URL."""
        settings = Settings()
        expected_url = "postgresql://rag_user:rag_password@localhost:5432/rag_db"
        assert settings.database_url == expected_url

    def test_redis_url_default(self):
        """Test default Redis URL."""
        settings = Settings()
        expected_url = "redis://localhost:6379"
        assert settings.redis_url == expected_url

    def test_custom_database_url(self):
        """Test that custom DATABASE_URL is preserved."""
        custom_url = "postgresql://custom:custom@localhost:5432/custom"
        settings = Settings(database_url=custom_url)
        assert settings.database_url == custom_url

    def test_custom_redis_url(self):
        """Test that custom REDIS_URL is preserved."""
        custom_url = "redis://custom:6379"
        settings = Settings(redis_url=custom_url)
        assert settings.redis_url == custom_url

    @patch.dict(
        os.environ,
        {
            "ENVIRONMENT": "production",
            "DEBUG": "true",
            "API_PORT": "9000",
            "OLLAMA_MODEL": "llama3.2:latest",
        },
    )
    def test_environment_variables(self):
        """Test loading settings from environment variables."""
        settings = Settings()
        assert settings.environment == "production"
        assert settings.debug is True
        assert settings.api_port == 9000
        assert settings.ollama_model == "llama3.2:latest"

    def test_upload_paths(self):
        """Test upload and processed path generation."""
        with patch("os.getcwd", return_value="/app"):
            settings = Settings()
            assert settings.get_upload_path() == "/app/data/uploads"
            assert settings.get_processed_path() == "/app/data/processed"

    def test_secret_key_generation(self):
        """Test secret key generation when not provided."""
        settings = Settings()
        # Now uses empty string default since validator was removed
        assert settings.secret_key == ""

    def test_custom_secret_key(self):
        """Test that custom secret key is preserved."""
        custom_key = "my-custom-secret-key"
        settings = Settings(secret_key=custom_key)
        assert settings.secret_key == custom_key

    def test_ollama_configuration(self):
        """Test Ollama model configuration."""
        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "llama3.2:latest"
        assert settings.ollama_embed_model == "embeddinggemma:latest"
        assert settings.ollama_vision_model == "devstral-small-2:latest"

    def test_file_handling_limits(self):
        """Test file handling configuration."""
        settings = Settings()
        assert settings.max_upload_size == 50 * 1024 * 1024  # 50MB
        assert ".pdf" in settings.allowed_extensions_list
        assert ".txt" in settings.allowed_extensions_list
        assert ".jpg" in settings.allowed_extensions_list
        assert ".png" in settings.allowed_extensions_list

    def test_vector_search_config(self):
        """Test vector search configuration."""
        settings = Settings()
        assert settings.vector_dimension == 768
        assert settings.similarity_threshold == 0.4
        assert settings.top_k_results == 5
        assert settings.max_documents_per_query == 10
