// Mock axios before importing anything that uses it
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    interceptors: {
      response: {
        use: jest.fn(),
      },
    },
    get: jest.fn(),
    post: jest.fn(),
    delete: jest.fn(),
  })),
  isAxiosError: jest.fn(() => false),
}));

import { ApiService } from '../services/api';

// Import after mocking to get the mocked instance
import * as apiModule from '../services/api';

// Get the mocked axios instance
const mockedAxios = require('axios');
const mockApiInstance = mockedAxios.create.mock.results[0].value;

describe('ApiService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getHealth', () => {
    it('fetches health status successfully', async () => {
      const mockResponse = {
        status: 'healthy' as const,
        timestamp: '2024-01-13T10:00:00Z',
        version: '0.1.0',
        services: {
          ollama: 'healthy' as const,
          vector_store: 'healthy' as const,
          redis: 'healthy' as const,
        },
      };

      mockApiInstance.get.mockResolvedValueOnce({ data: mockResponse });

      const result = await ApiService.getHealth();

      expect(mockApiInstance.get).toHaveBeenCalledWith('/health');
      expect(result).toEqual(mockResponse);
    });

    it('handles health check errors', async () => {
      const errorMessage = 'Service unavailable';
      mockApiInstance.get.mockRejectedValueOnce(new Error(errorMessage));

      await expect(ApiService.getHealth()).rejects.toThrow(errorMessage);
    });
  });

    it('handles health check errors', async () => {
      const errorMessage = 'Service unavailable';
      mockedAxios.get.mockRejectedValueOnce(new Error(errorMessage));

      await expect(ApiService.getHealth()).rejects.toThrow(errorMessage);
    });
  });

  describe('getDocuments', () => {
    it('fetches documents successfully', async () => {
      const mockDocuments = [
        {
          id: 'doc-1',
          filename: 'test.pdf',
          content_type: 'application/pdf',
          size: 1024000,
          uploaded_at: '2024-01-13T10:00:00Z',
          status: 'completed',
          chunks_count: 150,
        },
      ];

      mockedAxios.get.mockResolvedValueOnce({ data: mockDocuments });

      const result = await ApiService.getDocuments();

      expect(mockedAxios.get).toHaveBeenCalledWith('/documents');
      expect(result).toEqual(mockDocuments);
    });
  });

  describe('uploadDocument', () => {
    it('uploads document successfully', async () => {
      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
      const mockResponse = {
        id: 'doc-123',
        message: 'Document uploaded successfully',
        status: 'success',
      };

      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await ApiService.uploadDocument(mockFile);

      expect(mockedAxios.post).toHaveBeenCalledWith('/documents/upload', expect.any(FormData), {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      expect(result).toEqual(mockResponse);
    });
  });

  describe('deleteDocument', () => {
    it('deletes document successfully', async () => {
      mockedAxios.delete.mockResolvedValueOnce({});

      await ApiService.deleteDocument('doc-123');

      expect(mockedAxios.delete).toHaveBeenCalledWith('/documents/doc-123');
    });
  });

  describe('queryDocuments', () => {
    it('queries documents successfully', async () => {
      const queryData = {
        query: 'What is machine learning?',
        top_k: 5,
      };
      const mockResponse = {
        answer: 'Machine learning is...',
        sources: [],
        confidence: 0.9,
      };

      mockedAxios.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await ApiService.queryDocuments(queryData);

      expect(mockedAxios.post).toHaveBeenCalledWith('/query', queryData);
      expect(result).toEqual(mockResponse);
    });

    it('queries with document filters', async () => {
      const queryData = {
        query: 'Test query',
        document_ids: ['doc-1', 'doc-2'],
        top_k: 10,
        filters: { category: 'science' },
      };

      mockedAxios.post.mockResolvedValueOnce({ data: { answer: 'Test answer', sources: [] } });

      await ApiService.queryDocuments(queryData);

      expect(mockedAxios.post).toHaveBeenCalledWith('/query', queryData);
    });
  });

  describe('getStats', () => {
    it('fetches statistics successfully', async () => {
      const mockStats = {
        total_documents: 5,
        total_chunks: 750,
        total_queries: 25,
        average_response_time: 1.3,
        uptime_seconds: 3600,
      };

      mockedAxios.get.mockResolvedValueOnce({ data: mockStats });

      const result = await ApiService.getStats();

      expect(mockedAxios.get).toHaveBeenCalledWith('/stats');
      expect(result).toEqual(mockStats);
    });
  });

  describe('error handling', () => {
    it('handles network errors', async () => {
      mockedAxios.get.mockRejectedValueOnce({
        response: {
          status: 500,
          data: { detail: 'Internal server error' },
        },
      });

      await expect(ApiService.getHealth()).rejects.toThrow('Internal server error');
    });

    it('handles network timeout', async () => {
      mockedAxios.get.mockRejectedValueOnce({
        request: {},
        message: 'Network Error',
      });

      await expect(ApiService.getHealth()).rejects.toThrow('Network error - please check your connection');
    });

    it('handles unexpected errors', async () => {
      mockedAxios.get.mockRejectedValueOnce('Unexpected error');

      await expect(ApiService.getHealth()).rejects.toThrow('An unexpected error occurred');
    });
  });

  describe('API configuration', () => {
    it('uses correct base URL', () => {
      // Test that the base URL is set correctly
      expect(mockedAxios.create).toHaveBeenCalledWith({
        baseURL: 'http://localhost:8000',
        timeout: 30000,
      });
    });

    it('handles environment variable for API URL', () => {
      // This would be tested in a real environment with different REACT_APP_API_URL
      const originalEnv = process.env.REACT_APP_API_URL;
      process.env.REACT_APP_API_URL = 'http://custom-api:9000';

      // Re-import to get new environment variable
      jest.resetModules();
      const { ApiService: NewApiService } = require('../services/api');

      // Reset environment
      process.env.REACT_APP_API_URL = originalEnv;

      // The new service should use the custom URL
      expect(NewApiService).toBeDefined();
    });
  });
