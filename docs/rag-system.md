# RAG Agent Terraform - Usage Guide

This guide demonstrates how to use the RAG Agent system for document processing and question answering.

## 📚 Overview

The RAG Agent system allows you to:
- Upload documents (PDF, text, images)
- Ask questions about your documents
- Get AI-powered answers with source citations
- Maintain conversation context
- Process documents in multiple formats

## 🚀 Quick Examples

### 1. Upload a Document

```bash
# Upload a PDF document
curl -X POST "http://localhost:8000/documents/upload" \
     -H "accept: application/json" \
     -F "file=@research_paper.pdf"

# Response
{
  "id": "abc123...",
  "filename": "research_paper.pdf",
  "content_type": "application/pdf",
  "status": "processed",
  "chunks_count": 45,
  "embeddings_count": 45
}
```

### 2. Ask Questions

```bash
# Ask a question about the uploaded document
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What are the main findings of this research?",
       "top_k": 3
     }'

# Response
{
  "query": "What are the main findings of this research?",
  "answer": "The research found that...",
  "sources": [
    {
      "document_id": "abc123...",
      "filename": "research_paper.pdf",
      "chunk_text": "Our analysis revealed...",
      "similarity_score": 0.92
    }
  ],
  "confidence_score": 0.92,
  "processing_time": 1.45
}
```

### 3. List Documents

```bash
# Get all processed documents
curl "http://localhost:8000/documents"

# Get documents with pagination
curl "http://localhost:8000/documents?limit=10&offset=20"
```

## 📄 Document Processing

### Supported Formats

| Format | MIME Type | Processing Method |
|--------|-----------|-------------------|
| PDF | `application/pdf` | Text extraction, OCR fallback |
| Text | `text/plain` | Direct text processing |
| Markdown | `text/markdown` | Direct text processing |
| Python | `text/x-python` | Direct text processing |
| JPEG | `image/jpeg` | OCR processing |
| PNG | `image/png` | OCR processing |
| GIF | `image/gif` | OCR processing |
| WebP | `image/webp` | OCR processing |

### Processing Pipeline

1. **File Validation**: Check size, type, and content
2. **Text Extraction**: Extract text using format-specific methods
3. **Chunking**: Split text into semantic chunks with overlap
4. **Embedding Generation**: Create vector embeddings for each chunk
5. **Storage**: Store document metadata and vector embeddings
6. **Indexing**: Make document searchable for queries

### Chunking Strategy

- **Chunk Size**: 1000 characters (configurable)
- **Overlap**: 200 characters (configurable)
- **Semantic Boundaries**: Attempts to break at sentence boundaries
- **Metadata**: Each chunk includes position and content metadata

## ❓ Question Answering

### Basic Queries

```bash
# Simple question
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is artificial intelligence?"}'
```

### Advanced Queries

```bash
# Query specific documents
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Explain the algorithm",
       "document_ids": ["doc1", "doc2"],
       "top_k": 5
     }'
```

### Conversational Queries

```bash
# First question
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is machine learning?",
       "session_id": "conversation_123"
     }'

# Follow-up question (maintains context)
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How does it compare to deep learning?",
       "session_id": "conversation_123"
     }'
```

## 🔍 Search and Retrieval

### Similarity Search

The system uses vector similarity search to find relevant document chunks:

- **Embedding Model**: `embeddinggemma:latest` (768 dimensions)
- **Similarity Metric**: Cosine similarity
- **Default Threshold**: 0.7
- **Top-K Results**: Configurable (default: 5)

### Search Filters

```bash
# Search with filters
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "quantum physics",
       "filters": {
         "content_type": "application/pdf",
         "filename": "*.physics.*"
       }
     }'
```

## 💬 Conversation Management

### Session-Based Conversations

```python
import requests

class RAGConversation:
    def __init__(self, session_id, base_url="http://localhost:8000"):
        self.session_id = session_id
        self.base_url = base_url

    def ask(self, question):
        response = requests.post(f"{self.base_url}/query", json={
            "query": question,
            "session_id": self.session_id
        })
        return response.json()

# Start a conversation
conv = RAGConversation("my_session_123")

# Ask questions
answer1 = conv.ask("What is machine learning?")
answer2 = conv.ask("Can you explain neural networks?")
answer3 = conv.ask("How do they relate?")
```

### Conversation Memory

- **Storage**: Redis-backed conversation memory
- **TTL**: 24 hours for conversation history
- **Context Window**: Last 4 messages included in context
- **Session Isolation**: Each session maintains separate context

## 📊 System Monitoring

### Health Checks

```bash
# Overall system health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "ollama": "healthy",
    "vector_store": "healthy",
    "redis": "healthy"
  }
}
```

### System Statistics

```bash
# Get system metrics
curl http://localhost:8000/stats

# Response includes:
# - Total documents and chunks
# - Cache statistics
# - Service health status
# - Performance metrics
```

### Cache Management

```bash
# Clear all caches
curl -X POST "http://localhost:8000/cache/clear"

# Response:
{
  "message": "All caches cleared successfully"
}
```

## 🐍 Python SDK Examples

### Complete Python Client

```python
import requests
import json
from typing import List, Dict, Optional

class RAGClient:
    """Python client for RAG Agent API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def upload_document(self, file_path: str) -> Dict:
        """Upload a document."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.base_url}/documents/upload",
                files=files
            )
            response.raise_for_status()
            return response.json()

    def query(self, query: str, **kwargs) -> Dict:
        """Query the RAG system."""
        data = {"query": query, **kwargs}
        response = requests.post(
            f"{self.base_url}/query",
            json=data
        )
        response.raise_for_status()
        return response.json()

    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """List documents."""
        params = {"limit": limit, "offset": offset}
        response = requests.get(
            f"{self.base_url}/documents",
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_document(self, document_id: str) -> Dict:
        """Get document details."""
        response = requests.get(f"{self.base_url}/documents/{document_id}")
        response.raise_for_status()
        return response.json()

    def delete_document(self, document_id: str) -> Dict:
        """Delete a document."""
        response = requests.delete(f"{self.base_url}/documents/{document_id}")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict:
        """Check system health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

# Usage example
client = RAGClient()

# Upload document
result = client.upload_document("research_paper.pdf")
doc_id = result["id"]

# Ask questions
answer = client.query("What are the main conclusions?")
print(f"Answer: {answer['answer']}")

# List all documents
docs = client.list_documents()
print(f"Total documents: {len(docs)}")
```

### Batch Processing Example

```python
import asyncio
import aiohttp
from pathlib import Path

class AsyncRAGClient:
    """Async version of RAG client for batch processing."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    async def upload_documents(self, file_paths: List[str]) -> List[Dict]:
        """Upload multiple documents concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for file_path in file_paths:
                tasks.append(self._upload_single(session, file_path))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]

    async def _upload_single(self, session: aiohttp.ClientSession, file_path: str) -> Dict:
        """Upload a single document."""
        with open(file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=Path(file_path).name)

            async with session.post(f"{self.base_url}/documents/upload", data=data) as response:
                response.raise_for_status()
                return await response.json()

# Usage
async def main():
    client = AsyncRAGClient()

    # Upload multiple PDFs
    pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    results = await client.upload_documents(pdf_files)

    print(f"Uploaded {len(results)} documents")

asyncio.run(main())
```

## 📈 Advanced Usage

### Custom Embeddings

```python
# Query with custom embedding parameters
response = requests.post("http://localhost:8000/query", json={
    "query": "complex technical question",
    "top_k": 10,
    "filters": {
        "content_type": "application/pdf",
        "word_count": {"$gt": 100}  # Documents with >100 words
    }
})
```

### Performance Optimization

```python
# Batch queries for better performance
queries = [
    "What is AI?",
    "Explain machine learning",
    "What are neural networks?"
]

results = []
for query in queries:
    result = client.query(query, top_k=3)
    results.append(result)

# Process results
for i, result in enumerate(results):
    print(f"Query {i+1}: {result['processing_time']}s")
```

### Error Handling

```python
try:
    result = client.query("What is the meaning of life?")
    print(result["answer"])
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
except KeyError as e:
    print(f"Unexpected response format: {e}")
```

## 🔧 Configuration Tuning

### Performance Settings

```python
# Adjust chunking parameters
# These would be set in .env or terraform variables
CHUNK_SIZE=1500          # Larger chunks for more context
CHUNK_OVERLAP=300        # More overlap for better continuity
VECTOR_DIMENSION=768     # Embedding model dimension
SIMILARITY_THRESHOLD=0.8 # Higher threshold for more relevant results
```

### Memory Management

```python
# Cache TTL settings (in seconds)
QUERY_CACHE_TTL=3600     # 1 hour for query results
EMBEDDING_CACHE_TTL=86400 # 24 hours for embeddings
DOCUMENT_CACHE_TTL=86400  # 24 hours for document chunks
```

## 🎯 Best Practices

### Document Preparation

1. **File Formats**: Use PDF for complex documents, text for simple content
2. **File Size**: Keep under 50MB, split large documents if needed
3. **Content Quality**: Ensure text is OCR-readable for image documents
4. **Naming**: Use descriptive filenames for better organization

### Query Optimization

1. **Specific Questions**: Ask detailed, specific questions
2. **Context**: Provide context in multi-turn conversations
3. **Filters**: Use document filters to narrow search scope
4. **Follow-ups**: Use session IDs for conversational context

### Performance Tips

1. **Caching**: Leverage built-in caching for repeated queries
2. **Batch Upload**: Upload multiple documents at once when possible
3. **Pagination**: Use pagination for large document lists
4. **Health Monitoring**: Regularly check system health and stats

## 🔗 Integration Examples

### Web Application Integration

```javascript
// Frontend integration example
async function queryDocuments(question) {
    const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: question,
            top_k: 5,
            session_id: getCurrentSessionId()
        })
    });

    const result = await response.json();
    displayAnswer(result.answer);
    displaySources(result.sources);
}
```

### CLI Tool

```bash
#!/bin/bash
# rag_query.sh - Command line interface for RAG queries

if [ $# -eq 0 ]; then
    echo "Usage: $0 'your question here'"
    exit 1
fi

QUESTION="$1"
curl -s -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d "{\"query\": \"$QUESTION\"}" | jq '.answer'
```

This comprehensive guide covers the core functionality and advanced usage patterns for the RAG Agent system. Start with the basic examples and gradually explore advanced features as needed.