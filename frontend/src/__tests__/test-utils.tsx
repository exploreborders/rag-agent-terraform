import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Create Material-UI theme for tests
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

// Custom render function that includes providers
const AllTheProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options });

// Mock API service
export const mockApiService = {
  getHealth: jest.fn(),
  getDocuments: jest.fn(),
  uploadDocument: jest.fn(),
  deleteDocument: jest.fn(),
  queryDocuments: jest.fn(),
  getStats: jest.fn(),
};

// Mock file for testing
export const createMockFile = (name: string = 'test.pdf', size: number = 1024, type: string = 'application/pdf'): File => {
  const blob = new Blob(['test content'], { type });
  return new File([blob], name, { type, size });
};

// Mock document data
export const mockDocument = {
  id: 'test-doc-1',
  filename: 'test-document.pdf',
  content_type: 'application/pdf',
  size: 1024000,
  uploaded_at: '2024-01-13T10:00:00Z',
  status: 'completed' as const,
  chunks_count: 150,
};

export const mockQueryResponse = {
  answer: 'This is a test answer from the RAG system.',
  sources: [
    {
      id: 'source-1',
      content: 'This is the source content from the document.',
      score: 0.95,
      metadata: {
        filename: 'test-document.pdf',
        chunk_index: 1,
      },
    },
  ],
  confidence: 0.92,
  processing_time: 1.2,
};

export const mockHealthStatus = {
  status: 'healthy' as const,
  timestamp: '2024-01-13T10:00:00Z',
  version: '0.1.0',
  services: {
    ollama: 'healthy' as const,
    vector_store: 'healthy' as const,
    redis: 'healthy' as const,
  },
};

export const mockStatsResponse = {
  total_documents: 4,
  total_chunks: 7,
  total_queries: 42,
  average_response_time: 1.2,
  uptime_seconds: 3600,
  cache_stats: {
    connected_clients: 2,
    used_memory: 1162472,
    total_connections_received: 191,
    uptime_in_seconds: 5641,
    keyspace_hits: 0,
    keyspace_misses: 3,
  },
  ollama_available: true,
  vector_store_healthy: true,
  memory_healthy: true,
};

// Re-export everything
export * from '@testing-library/react';

// Override render method
export { customRender as render };

// Dummy test to satisfy Jest requirement
describe('test-utils', () => {
  it('exports test utilities', () => {
    expect(createMockFile).toBeDefined();
    expect(mockApiService).toBeDefined();
    expect(mockDocument).toBeDefined();
  });
});