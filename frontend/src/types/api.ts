// API Types for RAG Agent Frontend

export interface Document {
  id: string;
  filename: string;
  content_type: string;
  size: number;
  uploaded_at: string;
  status: 'processing' | 'completed' | 'failed';
  chunks_count?: number;
}

export interface DocumentUploadResponse {
  id: string;
  message: string;
  status: string;
}

export interface QueryRequest {
  query: string;
  document_ids?: string[];
  top_k?: number;
  filters?: Record<string, any>;
}

export interface QueryResponse {
  answer: string;
  sources: Array<{
    id: string;
    content: string;
    score: number;
    metadata?: Record<string, any>;
  }>;
  confidence?: number;
  processing_time?: number;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  services: {
    ollama: 'healthy' | 'unhealthy';
    vector_store: 'healthy' | 'unhealthy';
    redis: 'healthy' | 'unhealthy';
  };
}

export class ApiError extends Error {
  status_code?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'ApiError';
    this.status_code = statusCode;
  }
}

export interface StatsResponse {
  total_documents: number;
  total_chunks: number;
  total_queries: number;
  average_response_time: number;
  uptime_seconds: number;
}