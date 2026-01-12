# Sample Data Directory

This directory contains sample documents for testing the RAG Agent system.

## 📁 Directory Structure

```
data/
├── documents/          # Sample documents for testing
│   ├── sample_text.txt
│   ├── sample_markdown.md
│   ├── sample_code.py
│   └── README.md       # Document descriptions
├── embeddings/         # Pre-computed embeddings (if any)
└── README.md          # This file
```

## 📄 Sample Documents

### Text Document (`sample_text.txt`)
A simple text file demonstrating basic text processing capabilities.

**Content Preview:**
```
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a broad field of computer science focused on creating
systems that can perform tasks that typically require human intelligence.

Machine Learning is a subset of AI that involves training algorithms to recognize
patterns in data and make predictions or decisions without being explicitly programmed.
```

### Markdown Document (`sample_markdown.md`)
A markdown file showing how structured text is processed.

**Features:**
- Headers and subheaders
- Code blocks
- Lists and formatting
- Links and references

### Python Code (`sample_code.py`)
A Python script demonstrating code document processing.

**Features:**
- Function definitions
- Comments and docstrings
- Import statements
- Class definitions

## 🚀 Usage Examples

### 1. Upload Sample Documents

```bash
# Upload text document
curl -X POST "http://localhost:8000/documents/upload" \
     -F "file=@data/documents/sample_text.txt"

# Upload markdown document
curl -X POST "http://localhost:8000/documents/upload" \
     -F "file=@data/documents/sample_markdown.md"

# Upload code document
curl -X POST "http://localhost:8000/documents/upload" \
     -F "file=@data/documents/sample_code.py"
```

### 2. Test Queries

```bash
# Query about AI concepts
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is artificial intelligence?"}'

# Query about machine learning
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "Explain machine learning"}'

# Query about Python code
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What functions are defined in this code?"}'
```

### 3. Batch Upload Script

```python
#!/usr/bin/env python3
"""Batch upload sample documents to RAG Agent."""

import requests
import os
from pathlib import Path

def upload_document(file_path: str, base_url: str = "http://localhost:8000") -> dict:
    """Upload a single document."""
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/documents/upload", files=files)
        response.raise_for_status()
        return response.json()

def main():
    """Upload all sample documents."""
    sample_dir = Path("data/documents")

    if not sample_dir.exists():
        print("Sample documents directory not found!")
        return

    uploaded = 0
    for file_path in sample_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.py']:
            try:
                result = upload_document(str(file_path))
                print(f"✅ Uploaded {file_path.name}: {result['chunks_count']} chunks")
                uploaded += 1
            except Exception as e:
                print(f"❌ Failed to upload {file_path.name}: {e}")

    print(f"\n📊 Successfully uploaded {uploaded} documents")

if __name__ == "__main__":
    main()
```

## 📊 Expected Results

After uploading the sample documents, you should see:

### System Statistics
```bash
curl http://localhost:8000/stats

# Expected output:
{
  "total_documents": 3,
  "total_chunks": "~10-15",
  "ollama_available": true,
  "vector_store_healthy": true,
  "memory_healthy": true
}
```

### Document List
```bash
curl http://localhost:8000/documents

# Expected output includes:
# - sample_text.txt
# - sample_markdown.md
# - sample_code.py
```

### Query Examples

**Query: "What is AI?"**
- Should return relevant chunks from `sample_text.txt`
- Answer should reference the AI definition

**Query: "Show me code examples"**
- Should return chunks from `sample_code.py`
- Answer should include function definitions

**Query: "Explain markdown formatting"**
- Should return chunks from `sample_markdown.md`
- Answer should explain markdown syntax

## 🔧 Customization

### Adding Your Own Documents

1. **Create documents** in the `data/documents/` directory
2. **Supported formats**: PDF, TXT, MD, PY, JPG, PNG
3. **File size limit**: 50MB per document
4. **Upload via API** or use the batch script

### Document Categories

Consider organizing documents by category:

```
data/documents/
├── technical/
│   ├── api_docs.pdf
│   └── code_examples.py
├── research/
│   ├── papers.pdf
│   └── articles.md
├── personal/
│   ├── notes.txt
│   └── journal.md
```

### Testing Scenarios

Use these document types for testing:

1. **Short documents** (< 1KB): Test minimal processing
2. **Large documents** (> 1MB): Test chunking and performance
3. **Mixed formats**: Test format detection and processing
4. **Special characters**: Test encoding handling
5. **Images with text**: Test OCR capabilities

## 📈 Performance Benchmarks

### Expected Processing Times

- **Text files (< 10KB)**: < 1 second
- **Code files (< 50KB)**: < 2 seconds
- **Markdown files (< 100KB)**: < 3 seconds
- **Images (< 1MB)**: < 5 seconds (with OCR)
- **PDFs (< 5MB)**: < 10 seconds

### Chunking Results

- **Chunk size**: 1000 characters
- **Overlap**: 200 characters
- **Average chunks per document**: 5-20 (depending on content)

## 🐛 Testing Edge Cases

### Error Scenarios to Test

1. **Unsupported file types**
   ```bash
   curl -X POST "http://localhost:8000/documents/upload" \
        -F "file=@data/documents/unsupported.xyz"
   # Should return 400 Bad Request
   ```

2. **Oversized files**
   ```bash
   # Create a large file for testing
   dd if=/dev/zero of=/tmp/large_file.txt bs=1M count=60
   curl -X POST "http://localhost:8000/documents/upload" \
        -F "file=@/tmp/large_file.txt"
   # Should return 413 Payload Too Large
   ```

3. **Corrupted files**
   ```bash
   echo -n "corrupted content" > /tmp/corrupted.pdf
   curl -X POST "http://localhost:8000/documents/upload" \
        -F "file=@/tmp/corrupted.pdf"
   # Should return 422 Unprocessable Entity
   ```

### Query Edge Cases

1. **Empty queries**
2. **Very long queries** (> 1000 characters)
3. **Special characters and emojis**
4. **Queries in different languages**
5. **Context-dependent follow-up questions**

## 🔗 Integration Testing

### API Integration Test

```bash
#!/bin/bash
# test_integration.sh - Full integration test

echo "🧪 Starting RAG Agent Integration Test"

# 1. Health check
echo "📊 Checking system health..."
if ! curl -f http://localhost:8000/health > /dev/null; then
    echo "❌ Health check failed"
    exit 1
fi

# 2. Upload documents
echo "📤 Uploading sample documents..."
for file in data/documents/*; do
    if [ -f "$file" ]; then
        echo "  Uploading $(basename "$file")..."
        curl -X POST "http://localhost:8000/documents/upload" \
             -F "file=@$file" > /dev/null 2>&1
    fi
done

# 3. Verify uploads
echo "✅ Verifying uploads..."
stats=$(curl -s http://localhost:8000/stats)
doc_count=$(echo "$stats" | grep -o '"total_documents":[0-9]*' | cut -d: -f2)

if [ "$doc_count" -gt 0 ]; then
    echo "✅ Successfully uploaded $doc_count documents"
else
    echo "❌ No documents found after upload"
    exit 1
fi

# 4. Test queries
echo "❓ Testing queries..."
queries=("What is AI?" "Show code examples" "Explain markdown")

for query in "${queries[@]}"; do
    echo "  Testing: '$query'"
    result=$(curl -s -X POST "http://localhost:8000/query" \
             -H "Content-Type: application/json" \
             -d "{\"query\": \"$query\"}")
    if echo "$result" | grep -q '"answer"'; then
        echo "    ✅ Query successful"
    else
        echo "    ❌ Query failed"
    fi
done

echo "🎉 Integration test completed!"
```

## 📚 Next Steps

After testing with sample data:

1. **Monitor Performance**: Check system stats and response times
2. **Add Real Documents**: Upload your actual documents
3. **Tune Parameters**: Adjust chunking and similarity settings
4. **Scale Up**: Test with larger document collections
5. **Production Deployment**: Move to production environment

This sample data provides a solid foundation for testing and understanding the RAG Agent system's capabilities.