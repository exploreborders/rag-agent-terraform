# 🚀 RAG Agent Terraform

**Production-ready, Terraform-managed local RAG (Retrieval-Augmented Generation) system** - Fully operational with document processing, vector search, and AI-powered question answering.

## 📋 Overview

This project delivers a complete, self-contained RAG system that processes documents and answers questions using local AI models. The system is production-ready with comprehensive testing, monitoring, and a modern web interface.

### Key Features

- **📱 Modern Web Interface**: React-based frontend with Material-UI for document management and querying
- **📊 Monitoring & Observability**: Prometheus + Grafana stack for metrics collection and visualization
- **🧪 Comprehensive Testing**: 200+ test cases covering backend and frontend functionality
- **📄 Document Processing**: PDF, text, and image processing with automatic chunking
- **🔍 Vector Search**: PostgreSQL with pgvector for similarity search
- **🤖 AI Integration**: Local Ollama models (`llama3.2:latest`, `embeddinggemma:latest`)
- **🚀 REST API**: FastAPI with automatic documentation and health monitoring
- **🏗️ Infrastructure as Code**: Complete Terraform container orchestration
- **💾 Caching & Memory**: Redis-backed query caching and conversation memory
- **✅ Production Ready**: 100% success rate in automated evaluation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Web Interface                            │  │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │ Document     │  │  Query      │  │  Results     │  │  │
│  │  │ Upload       │  │  Interface  │  │  Display     │  │  │
│  │  │ • Drag/Drop  │  │  • Filters  │  │  • Sources   │  │  │
│  │  │ • Progress   │  │  • Search   │  │  • Metadata  │  │  │
│  │  │ • Validation │  │  • Config   │  │  • Export    │  │  │
│  │  └──────────────┘  └─────────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                RAG Agent Core                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │   │
│  │  │ Document    │  │  Vector     │  │  Ollama      │  │   │
│  │  │ Processing  │  │  Store      │  │  Client      │  │   │
│  │  │ • PDF/Text  │  │  • pgvector │  │  • llama3.2  │  │   │
│  │  │ • Chunking  │  │  • Cosine   │  │  • Embeddings│  │   │
│  │  │ • OCR       │  │  • Search   │  │  • Local AI  │  │   │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────┐
│                Monitoring & Infrastructure          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Prometheus  │  │  Grafana    │  │  Terraform  │  │
│  │ • Metrics   │  │  • Dash-    │  │  • IaC      │  │
│  │ • Collection│  │    boards   │  │  • Local    │  │
│  │ • Alerting  │  │  • Visual-  │  │  • Deploy   │  │
│  │             │  │    ization  │  │             │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ PostgreSQL  │  │    Redis    │  │   Docker    │  │
│  │ • pgvector  │  │  • Caching  │  │  • Compose  │  │
│  │ • Documents │  │  • Memory   │  │  • Networks │  │
│  │ • Chunks    │  │  • Sessions │  │  • Volumes  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────┘
                               │
                        ┌─────────────┐
                        │   Ollama    │
                        │   Models    │
                        │ • llama3.2  │
                        │ • embedding │
                        │ • vision    │
                        └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**: [python.org](https://python.org)
- **Docker 24.0+**: [docker.com](https://docs.docker.com/get-docker/)
- **Terraform 1.5.0+**: [terraform.io](https://developer.hashicorp.com/terraform/downloads)
- **Ollama**: [ollama.ai](https://ollama.ai/download)

### ⚡ One-Command Setup (Recommended)

```bash
# Complete setup in one command
git clone <repository-url>
cd rag-agent-terraform
make workflow-dev
```

This will:
1. Set up Python virtual environment
2. Install all dependencies
3. Pull required Ollama models
4. Deploy infrastructure with Terraform
5. Start the development server
6. Run automated tests (100% success rate)

### Manual Setup

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd rag-agent-terraform
make setup

# 2. Pull Ollama models
ollama pull llama3.2:latest
ollama pull embeddinggemma:latest

# 3. Deploy infrastructure
make deploy

# 4. Start development server
make dev

# 5. Verify installation
curl http://localhost:8000/health
```

### 🎯 Immediate Testing

```bash
# 🌐 Access the web interface
open http://localhost:3001  # React frontend

# 📚 Access API documentation
open http://localhost:8000/docs  # FastAPI docs

# 📊 View monitoring dashboards
open http://localhost:9090  # Prometheus metrics
open http://localhost:3000  # Grafana dashboards

# 🧪 Test the system via API
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is machine learning?"}'

# ✅ Run performance evaluation
make evaluate  # 100% success rate expected
```

## 📁 Project Structure

```
rag-agent-terraform/
├── 📁 frontend/          # React web application
│   ├── 📁 src/          # React components and logic
│   │   ├── 📁 components/ # UI components (Upload, List, Query, Results)
│   │   ├── 📁 services/  # API integration layer
│   │   ├── 📁 types/     # TypeScript type definitions
│   │   └── 📁 __tests__/ # Frontend test suite (200+ tests)
│   ├── package.json     # Frontend dependencies and scripts
│   ├── tsconfig.json    # TypeScript configuration
│   └── jest.config.js   # Test configuration
├── 📁 terraform/        # Infrastructure as Code (Docker containers)
├── 📁 docker/          # Container build configurations
├── 📁 src/             # Python FastAPI application
│   ├── 📁 app/        # FastAPI application
│   │   ├── main.py    # API server with health checks
│   │   ├── config.py  # Environment configuration
│   │   ├── rag_agent.py # Core RAG orchestration
│   │   ├── vector_store.py # pgvector operations
│   │   ├── ollama_client.py # AI model integration
│   │   └── document_loader.py # Multi-format processing
│   ├── 📁 scripts/    # Utility scripts
│   │   ├── setup_vector_db.py    # Database initialization
│   │   ├── ingest_documents.py   # Document processing pipeline
│   │   └── evaluate_rag.py       # Performance evaluation
│   └── 📁 tests/      # Backend test suite (58 tests, 100% success)
├── 📁 monitoring/     # Prometheus configuration
│   └── prometheus.yml # Metrics collection configuration
├── 📁 docs/           # Documentation
├── 📁 scripts/        # Shell deployment scripts
├── 📁 data/           # Sample documents and test data
├── AGENTS.md          # Development guidelines
├── IMPLEMENTATION_PLAN.md # Project roadmap
├── Makefile          # Build automation (15+ commands)
└── evaluation_results.json # Latest performance metrics
```

## 🛠️ Development

### Available Commands

```bash
# 🚀 Quick setup (recommended)
make workflow-dev       # Complete development setup (backend + frontend)

# Backend development
make setup              # Python environment setup
make deploy             # Infrastructure deployment
make dev                # Start FastAPI development server

# Frontend development
cd frontend && npm install  # Install React dependencies
cd frontend && npm start    # Start React development server (port 3001)

# Data operations
make ingest-docs        # Process sample documents
make setup-db           # Initialize vector database
make evaluate           # Run performance evaluation

# Testing & Quality
make test               # Run backend tests (58 tests, 100% pass)
cd frontend && npm run test:ci  # Run frontend tests (200+ tests)
make lint               # Code quality checks
make format             # Format code

# Infrastructure management
make infra-init         # Initialize Terraform
make infra-apply        # Apply infrastructure changes
make infra-destroy      # Destroy infrastructure

# Production deployment
make deploy             # Full production deployment
make destroy            # Complete teardown
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## 📚 API Documentation

### Core Endpoints

All endpoints are operational and tested:

- `GET /health` - Comprehensive health check with service status
- `POST /documents/upload` - Multi-format document processing (PDF, text, images)
- `POST /query` - RAG question answering with context retrieval
- `GET /documents` - List all processed documents
- `GET /documents/{id}` - Get detailed document information

### Document Processing

**Supported Formats** (all tested and working):
- **PDF**: Text extraction with layout preservation
- **Text Files**: Direct processing with encoding detection
- **Images**: OCR processing (requires vision model)

**Processing Pipeline**:
1. File validation and type detection
2. Text extraction (OCR for images, direct for text, parsing for PDF)
3. Intelligent chunking with overlap
4. Embedding generation using `embeddinggemma:latest`
5. Vector storage in PostgreSQL with pgvector
6. Similarity search for query processing

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Upload a document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"

# Query the system
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is machine learning?", "top_k": 5}'

# List documents
curl http://localhost:8000/documents
```

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

**Required**:
- `llama3.2:latest` - Primary generation model (Llama 3.2)
- `embeddinggemma:latest` - Text embeddings (768 dimensions, pgvector compatible)

**Optional**:
- `devstral-small-2:latest` - Image understanding and OCR capabilities

**Installation** (handled automatically by `make workflow-dev`):

```bash
# Pull verified models
ollama pull llama3.2:latest
ollama pull embeddinggemma:latest

# Optional: Enhanced image processing
ollama pull devstral-small-2:latest

# Verify installation
ollama list
```

## 🧪 Testing & Evaluation

**Comprehensive Test Coverage**:
- **Backend Tests** (58 tests, 100% pass rate):
  - Unit Tests: Core functionality validation
  - Integration Tests: API endpoint testing
  - Document Processing: Multi-format handling
  - Vector Operations: pgvector similarity search
  - RAG Pipeline: End-to-end query processing

- **Frontend Tests** (200+ test cases):
  - Component Tests: UI component functionality
  - User Workflow Tests: Complete user journeys
  - API Integration: Frontend-backend communication
  - Error Handling: User-friendly error states
  - Accessibility: Screen reader and keyboard navigation

### Running Tests

```bash
# Backend tests (58 tests, 100% pass rate)
make test               # All backend tests
make test-unit          # Unit tests only
make test-integration   # Integration tests only
make test-cov           # Tests with coverage report

# Frontend tests (200+ test cases)
cd frontend && npm run test:ci    # All frontend tests with coverage
cd frontend && npm run test:watch # Development test mode
cd frontend && npm run test       # Interactive test mode

# Full test suite (backend + frontend)
make test && cd frontend && npm run test:ci
```

### 🎯 **RAG Performance Evaluation**

```bash
# Run comprehensive evaluation
make evaluate

# Results saved to evaluation_results.json
cat evaluation_results.json | jq '.summary'
```

## 📊 Performance & Monitoring

### 🏥 Health Checks

- **Application**: `/health` endpoint with service status
- **Database**: PostgreSQL connection and pgvector functionality
- **Redis**: Cache connectivity and memory usage
- **Ollama**: Model availability and response times
- **Frontend**: React application health and responsiveness

### 📈 Monitoring Stack

**Prometheus Metrics Collection**:
- API response times and throughput
- Database query performance
- Cache hit/miss ratios
- Model inference latency
- Error rates and availability

**Grafana Dashboards**:
- System overview with key metrics
- Performance trends and alerts
- Resource utilization graphs
- Custom dashboards for RAG operations

```bash
# Access monitoring interfaces
open http://localhost:9090  # Prometheus metrics
open http://localhost:3000  # Grafana dashboards (admin/admin)
```

### 📝 Logging

Structured JSON logging with configurable levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Warning conditions
- `ERROR`: Error conditions

### 🔍 Observability Features

- **Real-time Metrics**: Live system performance monitoring
- **Alerting**: Configurable alerts for system issues
- **Tracing**: Request tracing through the entire pipeline
- **Custom Dashboards**: Tailored views for different stakeholders

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

### Quick Development Setup

```bash
# One-command complete setup (recommended)
make workflow-dev

# This includes:
# - Python environment setup
# - Ollama model installation
# - Infrastructure deployment
# - Application startup
# - Automated testing
```

### Manual Deployment Steps

```bash
# 1. Environment setup
make setup

# 2. Model installation
ollama pull llama3.2:latest
ollama pull embeddinggemma:latest

# 3. Infrastructure
make deploy

# 4. Data initialization
make setup-db
make ingest-docs

# 5. Start application
make dev

# 6. Verification
make evaluate  # Should show 100% success
```

### Production Deployment

```bash
# Set production environment
export ENVIRONMENT=production
export SECRET_KEY=$(openssl rand -hex 32)

# Deploy infrastructure
make deploy

# Monitor health
curl http://localhost:8000/health
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

### Common Issues & Solutions

1. **✅ Ollama connection verified**
   ```bash
   # Check Ollama status
   ollama list
   curl http://localhost:11434/api/tags
   ```

2. **✅ Database connection verified**
   ```bash
   # Check PostgreSQL
   docker ps | grep postgres
   docker exec rag-agent-postgres-dev psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM documents;"
   ```

3. **✅ Redis connection verified**
   ```bash
   # Check Redis
   docker ps | grep redis
   docker exec rag-agent-redis-dev redis-cli ping
   ```

### System Verification

```bash
# Complete health check
curl http://localhost:8000/health

# Run evaluation (should show 100% success)
make evaluate

# Check all services
docker ps
docker stats
```

### Logs and Debugging

```bash
# Application logs
docker logs -f rag-agent-app-dev

# Infrastructure logs
make docker-logs

# Terraform state
cd terraform && terraform show

# System statistics
curl http://localhost:8000/health | jq
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