#!/bin/bash
# Database Seeding Script
# Seeds the database with sample data for development and testing

set -e  # Exit on any error

echo "🌱 Starting database seeding process..."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Virtual environment not activated. Please run 'source venv/bin/activate' first."
    exit 1
fi

# Change to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
cd "$PROJECT_DIR"

echo "📁 Working directory: $(pwd)"

# Setup vector database schema
echo "🏗️  Setting up vector database schema..."
python src/scripts/setup_vector_db.py

if [ $? -ne 0 ]; then
    echo "❌ Failed to setup vector database schema"
    exit 1
fi

echo "✅ Vector database schema setup complete"

# Create sample data directory if it doesn't exist
echo "📂 Creating sample data directory..."
mkdir -p data/documents

# Create sample documents
echo "📄 Creating sample documents..."

# Sample text document
cat > data/documents/sample_machine_learning.txt << 'EOF'
Machine Learning Fundamentals

Machine learning is a method of data analysis that automates analytical model building.
It is a branch of artificial intelligence based on the idea that systems can learn from
data, identify patterns and make decisions with minimal human intervention.

Types of Machine Learning:
1. Supervised Learning - Uses labeled data to train models
2. Unsupervised Learning - Finds patterns in unlabeled data
3. Reinforcement Learning - Learns through trial and error

Key Concepts:
- Training Data: The dataset used to teach the algorithm
- Features: Input variables used for prediction
- Labels: Output variables we want to predict
- Model: The mathematical representation learned from data
- Prediction: The output of the model for new data

Applications:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Fraud detection
- Medical diagnosis
- Autonomous vehicles
EOF

# Sample technical document
cat > data/documents/sample_architecture.txt << 'EOF'
System Architecture Overview

This document describes the architecture of our RAG (Retrieval-Augmented Generation) system.

Core Components:

1. Document Loader
   - Processes PDF, text, and image files
   - Extracts text content using OCR for images
   - Splits content into manageable chunks with overlap

2. Vector Store
   - PostgreSQL database with pgvector extension
   - Stores document embeddings for similarity search
   - Provides efficient nearest neighbor queries

3. AI Models
   - Ollama for local model inference
   - Supports multiple model sizes and types
   - Handles both generation and embedding tasks

4. Caching Layer
   - Redis for query result caching
   - Conversation memory storage
   - Performance optimization

5. API Server
   - FastAPI-based REST API
   - Async processing for high concurrency
   - Comprehensive error handling and validation

Data Flow:
1. Document uploaded via API
2. Text extracted and chunked
3. Embeddings generated and stored
4. User query processed
5. Relevant documents retrieved
6. AI generates contextual response
EOF

echo "✅ Sample documents created"

# Ingest the sample documents
echo "📥 Ingesting sample documents..."
python src/scripts/ingest_documents.py --input data/documents/ --recursive

if [ $? -ne 0 ]; then
    echo "❌ Failed to ingest sample documents"
    exit 1
fi

echo "✅ Sample documents ingested successfully"

# Verify the setup
echo "🔍 Verifying setup..."

# Check if documents are in the database
python -c "
import asyncio
import sys
import os
sys.path.insert(0, 'src')
os.chdir('src')
import app.vector_store
import app.config

async def check():
    settings = app.config.Settings()
    store = app.vector_store.VectorStore()
    await store.connect()
    count = await store.get_chunk_count()
    print(f'📊 Document chunks in database: {count}')
    await store.disconnect()

asyncio.run(check())
"

echo ""
echo "🎉 Database seeding completed successfully!"
echo ""
echo "📋 Summary:"
echo "  - Vector database schema initialized"
echo "  - Sample documents created and ingested"
echo "  - System ready for queries"
echo ""
echo "🚀 You can now test the system by running:"
echo "  curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\": \"What is machine learning?\"}'"