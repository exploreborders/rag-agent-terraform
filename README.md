# 🚀 RAG Agent Terraform

**Terraform-managed local infrastructure for an agentic RAG (Retrieval-Augmented Generation) system** combining LangChain orchestration with LlamaIndex document indexing.

## 📋 Overview

This project provides a complete local RAG system that can process PDF, text, and image documents using local Ollama AI models. The system features:

- **Hybrid RAG Architecture**: LangChain agents orchestrating LlamaIndex document indexing
- **Local AI Models**: Ollama integration with `llama3.2:latest` and `embeddinggemma:latest`
- **Vector Database**: PostgreSQL with pgvector extension for embeddings
- **Memory Management**: Redis-backed conversation memory and caching
- **Document Processing**: Multi-format support (PDF, text, images with OCR)
- **Infrastructure as Code**: Complete Terraform-managed local container orchestration
- **FastAPI Backend**: RESTful API with comprehensive endpoints
- **Comprehensive Testing**: Unit and integration test suites

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangChain     │    │   LlamaIndex    │
│   Application   │◄──►│   Agents        │◄──►│   Indexing      │
│                 │    │                 │    │                 │
│ • REST API      │    │ • Tool calling  │    │ • Vector search │
│ • Health checks │    │ • Memory        │    │ • Document proc │
│ • Error handling│    │ • Orchestration │    │ • Chunking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Ollama        │
                    │   Local Models  │
                    │                 │
                    │ • llama3.2       │
                    │ • embeddinggemma│
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Infrastructure  │
                    │                 │
                    │ • PostgreSQL    │
                    │ • Redis         │
                    │ • Terraform     │
                    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker 24.0+
- Terraform 1.5.0+
- Ollama with models: `llama3.2:latest`, `embeddinggemma:latest`

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd rag-agent-terraform
   make setup
   ```

2. **Start infrastructure:**
   ```bash
   make deploy
   ```

3. **Start development server:**
   ```bash
   make dev
   ```

4. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📁 Project Structure

```
rag-agent-terraform/
├── 📁 terraform/          # Infrastructure as Code
├── 📁 docker/            # Container configurations
├── 📁 src/               # Python application
│   ├── app/             # FastAPI application
│   ├── scripts/         # Utility scripts
│   └── tests/           # Test suite
├── 📁 docs/             # Documentation
├── 📁 scripts/          # Shell scripts
├── 📁 data/             # Sample data
├── AGENTS.md            # Agent coding guidelines
├── IMPLEMENTATION_PLAN.md # Project roadmap
└── Makefile            # Build automation
```

## 🛠️ Development

### Available Commands

```bash
# Setup and installation
make setup              # Set up development environment
make install            # Install dependencies

# Development
make dev                # Start development server
make lint               # Run code quality checks
make format             # Format code

# Testing
make test               # Run all tests
make test-unit          # Run unit tests only
make test-integration   # Run integration tests only

# Infrastructure
make infra-init         # Initialize Terraform
make infra-plan         # Plan infrastructure changes
make infra-apply        # Apply infrastructure changes
make infra-destroy      # Destroy infrastructure

# Deployment
make deploy             # Full deployment
make destroy            # Full teardown
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## 📚 API Documentation

### Core Endpoints

- `GET /health` - System health check
- `POST /documents/upload` - Upload documents for processing
- `POST /query` - Query the RAG system
- `GET /documents` - List processed documents
- `GET /documents/{id}` - Get document details

### Document Processing

Supported formats:
- **PDF**: Text extraction and layout analysis
- **Text**: Direct processing with encoding detection
- **Images**: OCR processing with CLIP embeddings

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `development` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `MAX_UPLOAD_SIZE` | Maximum file size (bytes) | `52428800` |

### Ollama Models

Required models:
- `llama3.2:latest` - Primary generation model (latest Llama 3.2)
- `embeddinggemma:latest` - Text embeddings (Google's EmbeddingGemma, 768 dimensions)

Install with:
```bash
ollama pull llama3.2:latest
ollama pull embeddinggemma:latest
```

## 🧪 Testing

### Test Structure

```
tests/
├── test_rag_agent.py      # Agent functionality
├── test_document_loader.py # Document processing
├── test_vector_store.py    # Database operations
└── test_api.py            # API endpoints
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# With coverage
make test-cov
```

## 📊 Performance & Monitoring

### Health Checks

- **Application**: `/health` endpoint
- **Database**: PostgreSQL connection
- **Redis**: Cache connectivity
- **Ollama**: Model availability

### Logging

Structured JSON logging with configurable levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Warning conditions
- `ERROR`: Error conditions

## 🔒 Security

### Best Practices

- Environment-based configuration
- Input validation and sanitization
- Secure file upload handling
- No hardcoded secrets
- Container security scanning

### File Upload Security

- File type validation
- Size limits enforcement
- Path traversal protection
- Content scanning

## 🚀 Deployment

### Local Development

```bash
make workflow-dev  # Complete setup
```

### Production Deployment

```bash
export ENVIRONMENT=production
export SECRET_KEY=$(openssl rand -hex 32)
make deploy
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make ci` to validate
5. Submit a pull request

### Code Quality

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

## 📝 Documentation

- **API Docs**: `/docs` endpoint (Swagger UI)
- **Project Docs**: `docs/` directory
- **Code Docs**: Inline documentation following Google style

## 🐛 Troubleshooting

### Common Issues

1. **Ollama connection failed**
   - Ensure Ollama is running: `ollama serve`
   - Check model availability: `ollama list`

2. **Database connection error**
   - Verify PostgreSQL container is running
   - Check connection string in `.env`

3. **Redis connection failed**
   - Ensure Redis container is running
   - Verify Redis URL configuration

### Logs and Debugging

```bash
# View application logs
make docker-logs

# Check infrastructure status
docker ps

# Terraform state
cd terraform && terraform show
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Agent orchestration
- [LlamaIndex](https://github.com/run-llm/llamaindex) - Document indexing
- [Ollama](https://github.com/jmorganca/ollama) - Local AI models
- [pgvector](https://github.com/pgvector/pgvector) - Vector database
- [FastAPI](https://github.com/tiangolo/fastapi) - Web framework

---

**Built with ❤️ for local AI-powered document processing**