# RAG Agent Terraform - API Documentation

This document provides comprehensive API documentation for the RAG Agent Terraform system.

## 📋 Overview

The RAG Agent API provides RESTful endpoints for document processing, question answering, and system management. All endpoints return JSON responses and use standard HTTP status codes.

**Base URL**: `http://localhost:8000`
**API Documentation**: `http://localhost:8000/docs` (Swagger UI)

## 🔗 Endpoints

### Health & Status

#### GET `/health`
Check system health and component status.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "0.1.0",
  "services": {
    "ollama": "healthy",
    "vector_store": "healthy",
    "redis": "healthy"
  }
}
```

**Response** (503 Service Unavailable):
```json
{
  "status": "degraded",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "ollama": "unhealthy",
    "vector_store": "healthy",
    "redis": "healthy"
  }
}
```

### Document Management

#### POST `/documents/upload`
Upload and process a document for the RAG system.

**Request** (multipart/form-data):
- `file`: Document file (PDF, text, or image)
- `content_type`: MIME type (optional, auto-detected)

**Response** (200 OK):
```json
{
  "id": "a1b2c3d4...",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "status": "processed",
  "chunks_count": 25,
  "embeddings_count": 25
}
```

**Error Responses**:
- `400`: Unsupported file type or invalid file
- `413`: File too large
- `422`: Document processing failed
- `500`: Internal server error

**Example**:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf"
```

#### GET `/documents`
List all processed documents.

**Query Parameters**:
- `limit` (int, optional): Maximum documents to return (default: 100, max: 1000)
- `offset` (int, optional): Number of documents to skip (default: 0)

**Response** (200 OK):
```json
[
  {
    "id": "a1b2c3d4...",
    "filename": "document.pdf",
    "content_type": "application/pdf",
    "size": 1024000,
    "upload_time": "2024-01-01T12:00:00Z",
    "page_count": 10,
    "word_count": 2500,
    "checksum": "sha256_hash..."
  }
]
```

#### GET `/documents/{document_id}`
Get details of a specific document.

**Path Parameters**:
- `document_id` (string): Document identifier

**Response** (200 OK):
```json
{
  "id": "a1b2c3d4...",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "size": 1024000,
  "upload_time": "2024-01-01T12:00:00Z",
  "page_count": 10,
  "word_count": 2500,
  "checksum": "sha256_hash..."
}
```

**Error Responses**:
- `404`: Document not found
- `500`: Internal server error

#### DELETE `/documents/{document_id}`
Delete a document and its associated chunks.

**Path Parameters**:
- `document_id` (string): Document identifier

**Response** (200 OK):
```json
{
  "message": "Document deleted successfully",
  "document_id": "a1b2c3d4..."
}
```

**Error Responses**:
- `404`: Document not found
- `500`: Internal server error

### Question Answering

#### POST `/query`
Query the RAG system for answers based on processed documents.

**Request Body**:
```json
{
  "query": "What is machine learning?",
  "document_ids": ["doc1", "doc2"],
  "top_k": 5,
  "session_id": "session_123"
}
```

**Parameters**:
- `query` (string, required): Question text (1-1000 characters)
- `document_ids` (array, optional): Specific document IDs to search
- `top_k` (int, optional): Number of results (1-20, default: 5)
- `session_id` (string, optional): Conversation session ID

**Response** (200 OK):
```json
{
  "query": "What is machine learning?",
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "document_id": "doc1",
      "filename": "ml_guide.pdf",
      "content_type": "application/pdf",
      "chunk_text": "Machine learning algorithms learn from data...",
      "similarity_score": 0.89,
      "metadata": {
        "chunk_index": 5,
        "word_count": 45
      }
    }
  ],
  "confidence_score": 0.89,
  "processing_time": 1.23,
  "total_sources": 1
}
```

**Error Responses**:
- `400`: Invalid query (empty or too long)
- `422`: Query processing failed
- `500`: Internal server error

**Example**:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is machine learning?",
       "top_k": 3
     }'
```

### System Management

#### GET `/stats`
Get system statistics and metrics.

**Response** (200 OK):
```json
{
  "total_documents": 5,
  "total_chunks": 1250,
  "cache_stats": {
    "connected_clients": 2,
    "used_memory": 1024000,
    "total_connections_received": 150,
    "uptime_in_seconds": 3600,
    "keyspace_hits": 95,
    "keyspace_misses": 15
  },
  "ollama_available": true,
  "vector_store_healthy": true,
  "memory_healthy": true
}
```

#### POST `/cache/clear`
Clear all system caches.

**Response** (200 OK):
```json
{
  "message": "All caches cleared successfully"
}
```

**Error Responses**:
- `500`: Cache clearing failed

## 📊 Data Models

### QueryRequest
```json
{
  "query": "string",
  "document_ids": ["string"],
  "top_k": 5,
  "filters": {}
}
```

### QueryResponse
```json
{
  "query": "string",
  "answer": "string",
  "sources": [
    {
      "document_id": "string",
      "filename": "string",
      "content_type": "string",
      "chunk_text": "string",
      "similarity_score": 0.85,
      "metadata": {}
    }
  ],
  "confidence_score": 0.85,
  "processing_time": 1.23,
  "total_sources": 1
}
```

### DocumentResponse
```json
{
  "id": "string",
  "filename": "string",
  "content_type": "string",
  "status": "processed",
  "chunks_count": 25,
  "embeddings_count": 25
}
```

### HealthStatus
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "0.1.0",
  "services": {
    "ollama": "healthy",
    "vector_store": "healthy",
    "redis": "healthy"
  }
}
```

### ErrorResponse
```json
{
  "error": "Error Type",
  "message": "Error description",
  "details": {},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🔒 Authentication

Currently, the API does not require authentication. For production deployments, consider adding:

- API key authentication
- JWT tokens
- OAuth integration

## 📏 Rate Limiting

The API includes built-in protections:

- File upload size limits (50MB default)
- Query length limits (1000 characters)
- Result count limits (20 max)
- Document count limits (10 per query)

## 🛠️ Error Handling

All errors follow a consistent format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {},
  "timestamp": "ISO 8601 timestamp"
}
```

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Not Found
- `413`: Payload Too Large
- `422`: Unprocessable Entity (processing error)
- `500`: Internal Server Error
- `503`: Service Unavailable (health check failure)

## 📊 Performance

### Response Times

- **Health Check**: <100ms
- **Document Upload**: 1-30 seconds (depends on file size)
- **Query**: 0.5-5 seconds (depends on complexity)
- **List Documents**: <500ms

### Caching

The system implements multiple caching layers:

- **Query Results**: 1 hour TTL
- **Embeddings**: 24 hour TTL
- **Document Chunks**: 24 hour TTL
- **Conversation Memory**: 24 hour TTL

## 🔄 WebSocket Support

Future versions may include WebSocket support for:

- Real-time query streaming
- Document processing progress
- System event notifications

## 📚 SDKs and Libraries

### Python Client

```python
import httpx

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    async def query(self, query, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/query",
                json={"query": query, **kwargs}
            )
            return response.json()

    async def upload_document(self, file_path):
        async with httpx.AsyncClient() as client:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = await client.post(
                    f"{self.base_url}/documents/upload",
                    files=files
                )
                return response.json()
```

### JavaScript/TypeScript Client

```javascript
class RAGClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async query(query, options = {}) {
        const response = await fetch(`${this.baseURL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, ...options }),
        });
        return response.json();
    }

    async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseURL}/documents/upload`, {
            method: 'POST',
            body: formData,
        });
        return response.json();
    }
}
```

## 🔗 Related Documentation

- [Setup Guide](../README.md) - Installation and configuration
- [RAG System Guide](./rag-system.md) - Usage and examples
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
- [Terraform Guide](./terraform.md) - Infrastructure documentation