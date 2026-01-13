import axios from 'axios';
import {
  Document,
  DocumentUploadResponse,
  QueryRequest,
  QueryResponse,
  HealthStatus,
  StatsResponse,
  ApiError
} from '../types/api';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
});

// Request interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      const apiError: ApiError = {
        detail: error.response.data.detail || 'An error occurred',
        status_code: error.response.status,
      };
      throw apiError;
    } else if (error.request) {
      // Network error
      throw new Error('Network error - please check your connection');
    } else {
      // Something else happened
      throw new Error('An unexpected error occurred');
    }
  }
);

// API Service class
export class ApiService {
  // Health check
  static async getHealth(): Promise<HealthStatus> {
    const response = await api.get('/health');
    return response.data;
  }

  // Documents
  static async getDocuments(): Promise<Document[]> {
    const response = await api.get('/documents');
    return response.data;
  }

  static async uploadDocument(file: File): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  static async deleteDocument(documentId: string): Promise<void> {
    await api.delete(`/documents/${documentId}`);
  }

  // Query
  static async queryDocuments(queryData: QueryRequest): Promise<QueryResponse> {
    const response = await api.post('/query', queryData);
    return response.data;
  }

  // Statistics
  static async getStats(): Promise<StatsResponse> {
    const response = await api.get('/stats');
    return response.data;
  }
}

// Export default API instance for custom requests
export default api;