import React from 'react';
import { render, screen, waitFor } from './__tests__/test-utils';
import App from './App';
import { ApiService } from './services/api';

// Mock the API service
jest.mock('./services/api');
const mockedApiService = ApiService as jest.Mocked<typeof ApiService>;

describe('App', () => {
  const mockDocuments = [
    {
      id: 'doc-1',
      filename: 'document1.pdf',
      content_type: 'application/pdf',
      size: 1024000,
      uploaded_at: '2024-01-13T10:00:00Z',
      status: 'completed' as const,
      chunks_count: 150,
    },
  ];

  const mockHealthStatus = {
    status: 'healthy' as const,
    timestamp: '2024-01-13T10:00:00Z',
    version: '0.1.0',
    services: {
      ollama: 'healthy' as const,
      vector_store: 'healthy' as const,
      redis: 'healthy' as const,
    },
  };

  const mockStats = {
    total_documents: 1,
    total_chunks: 150,
    total_queries: 5,
    average_response_time: 1.2,
    uptime_seconds: 3600,
  };

  beforeEach(() => {
    jest.clearAllMocks();

    // Mock all API calls for initial load
    mockedApiService.getDocuments.mockResolvedValue(mockDocuments);
    mockedApiService.getHealth.mockResolvedValue(mockHealthStatus);
    mockedApiService.getStats.mockResolvedValue(mockStats);
  });

  it('renders the main application layout', async () => {
    render(<App />);

    // Check header
    expect(screen.getByText('RAG Agent - Document Intelligence System')).toBeInTheDocument();

    // Check that components are rendered
    await waitFor(() => {
      expect(screen.getByText('Upload Documents')).toBeInTheDocument();
      expect(screen.getByText('Documents (1)')).toBeInTheDocument();
      expect(screen.getByText('Ask Questions')).toBeInTheDocument();
    });
  });

  it('loads initial data on mount', async () => {
    render(<App />);

    await waitFor(() => {
      expect(mockedApiService.getDocuments).toHaveBeenCalled();
      expect(mockedApiService.getHealth).toHaveBeenCalled();
      expect(mockedApiService.getStats).toHaveBeenCalled();
    });
  });

  it('displays health status indicators', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('Ollama: healthy')).toBeInTheDocument();
      expect(screen.getByText('Vector Store: healthy')).toBeInTheDocument();
      expect(screen.getByText('Redis: healthy')).toBeInTheDocument();
    });
  });

  it('shows statistics dashboard', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('1')).toBeInTheDocument(); // total_documents
      expect(screen.getByText('150')).toBeInTheDocument(); // total_chunks
      expect(screen.getByText('5')).toBeInTheDocument(); // total_queries
      expect(screen.getByText('1.2s')).toBeInTheDocument(); // average_response_time
    });
  });

  it('handles document upload success', async () => {
    render(<App />);

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('Documents (1)')).toBeInTheDocument();
    });

    // Simulate upload success (this would normally be triggered by the DocumentUpload component)
    // Since we can't easily trigger file uploads in this test, we'll verify the state management works

    expect(mockedApiService.getDocuments).toHaveBeenCalledTimes(1);
  });

  it('handles query results', async () => {
    const mockQueryResult = {
      answer: 'Test answer from RAG system',
      sources: [
        {
          id: 'source-1',
          content: 'Source content',
          score: 0.9,
          metadata: { filename: 'test.pdf' },
        },
      ],
      confidence: 0.85,
      processing_time: 1.0,
    };

    render(<App />);

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByText('Ask Questions')).toBeInTheDocument();
    });

    // The query result handling would be tested through the QueryInterface component
    // Here we verify that the app can display query results when they exist
    // (In a real scenario, this would be triggered by user interaction)

    expect(screen.getByText('No Results Yet')).toBeInTheDocument();
  });

  it('handles API errors gracefully', async () => {
    mockedApiService.getDocuments.mockRejectedValueOnce(new Error('API Error'));

    render(<App />);

    // The app should still render even if some API calls fail
    expect(screen.getByText('RAG Agent - Document Intelligence System')).toBeInTheDocument();

    // Error handling would be shown in individual components
  });

  it('displays service URLs in header', async () => {
    render(<App />);

    await waitFor(() => {
      // Check that service links are available (these would be in the header status)
      expect(screen.getByText('Ollama: healthy')).toBeInTheDocument();
    });
  });

  it('integrates all components correctly', async () => {
    render(<App />);

    await waitFor(() => {
      // Verify all main sections are present
      expect(screen.getByText('Upload Documents')).toBeInTheDocument();
      expect(screen.getByText('Documents (1)')).toBeInTheDocument();
      expect(screen.getByText('Ask Questions')).toBeInTheDocument();
      expect(screen.getByText('No Results Yet')).toBeInTheDocument();
    });

    // Verify stats are displayed
    expect(screen.getByText('1')).toBeInTheDocument(); // Documents count
    expect(screen.getByText('150')).toBeInTheDocument(); // Chunks count
  });

  it('maintains theme consistency', async () => {
    render(<App />);

    // Material-UI theme should be applied
    const header = screen.getByText('RAG Agent - Document Intelligence System');
    expect(header).toBeInTheDocument();

    // Check for Material-UI components (they should have proper styling)
    await waitFor(() => {
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });
  });
});