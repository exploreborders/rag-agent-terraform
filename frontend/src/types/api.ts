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
  query: string; // Added for multi-agent response
  answer: string;
  sources: Array<{
    document_id: string;
    filename: string;
    content_type: string;
    chunk_text: string;
    similarity_score: number;
    metadata?: Record<string, any>;
  }>;
  confidence_score?: number; // Renamed from confidence for consistency
  processing_time?: number;
  total_sources?: number;
  // Multi-agent specific fields
  agent_metrics?: Record<string, any>;
  mcp_results?: Record<string, any>;
  phase?: string;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  services: {
    vector_store: 'healthy' | 'unhealthy';
    ollama_client: 'healthy' | 'unhealthy';
    rag_agent: 'healthy' | 'unhealthy';
    multi_agent_system: 'healthy' | 'unhealthy';
    mcp_coordinator: 'healthy' | 'unhealthy';
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

// Chat Interface Types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: QueryResponse['sources'];
  processing_time?: number;
  confidence?: number;
  error?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  created_at: Date;
  updated_at: Date;
  document_ids?: string[];
}

export interface CreateSessionRequest {
  title?: string;
  document_ids?: string[];
}

export interface SendMessageRequest {
  message: string;
  document_ids?: string[];
}

export interface ChatSessionResponse {
  session: ChatSession;
}

export interface MessageResponse {
  message: ChatMessage;
  session_id: string;
}