import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../__tests__/test-utils';
import { ChatInterface } from '../ChatInterface';

// Mock the API service
jest.mock('../../services/api');

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

describe('ChatInterface', () => {
  const mockDocuments = [
    {
      id: 'doc-1',
      filename: 'sample_text.txt',
      content_type: 'text/plain',
      size: 1520,
      uploaded_at: '2024-01-14T10:00:00Z',
      status: 'completed' as const,
      chunks_count: 3,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    localStorageMock.setItem.mockImplementation(() => {});
  });

  it('renders chat interface with welcome message', () => {
    render(<ChatInterface documents={[]} onError={jest.fn()} />);

    expect(screen.getByText('Welcome to RAG Chat')).toBeInTheDocument();
    expect(screen.getByText('Start a conversation with your documents')).toBeInTheDocument();
  });

  it('displays document selection when documents are available', () => {
    render(<ChatInterface documents={mockDocuments} onError={jest.fn()} />);

    expect(screen.getByText('Document Context (1 available)')).toBeInTheDocument();
    expect(screen.getByText('sample_text.txt (3 chunks)')).toBeInTheDocument();
  });

  it('creates a new session automatically', () => {
    render(<ChatInterface documents={mockDocuments} onError={jest.fn()} />);

    // Should create a session and show chat interface
    expect(screen.getByText('New Chat')).toBeInTheDocument();
  });

  it('displays sources with proper naming and scores', () => {
    const mockSources = [
      {
        document_id: 'doc-1',
        filename: 'sample_text.txt',
        content_type: 'text/plain',
        chunk_text: 'This is sample content about AI.',
        similarity_score: 0.85,
      },
      {
        document_id: 'doc-2',
        filename: 'sample_code.py',
        content_type: 'text/x-python',
        chunk_text: 'def hello_world(): return "Hello"',
        similarity_score: 0.72,
      },
    ];

    const mockResponse = {
      answer: 'This is a test answer',
      sources: mockSources,
      processing_time: 1.5,
      confidence: 0.9,
    };

    // Mock the API call
    const { ApiService } = require('../../services/api');
    ApiService.queryDocuments = jest.fn().mockResolvedValue(mockResponse);

    render(<ChatInterface documents={mockDocuments} onError={jest.fn()} />);

    // Find the input and send button
    const input = screen.getByPlaceholderText('Ask a question about your documents...');
    const sendButton = screen.getByText('Send');

    // Type a question
    fireEvent.change(input, { target: { value: 'What is AI?' } });

    // Click send
    fireEvent.click(sendButton);

    // Wait for the response
    return waitFor(() => {
      expect(screen.getByText('This is a test answer')).toBeInTheDocument();

      // Check that sources are displayed with proper names and scores
      expect(screen.getByText('sample_text.txt • Score: 85.0%')).toBeInTheDocument();
      expect(screen.getByText('sample_code.py • Score: 72.0%')).toBeInTheDocument();

      // Check that source content is displayed
      expect(screen.getByText('This is sample content about AI.')).toBeInTheDocument();
      expect(screen.getByText('def hello_world(): return "Hello"')).toBeInTheDocument();
    });
  });

  it('handles NaN similarity scores gracefully', () => {
    const mockSourcesWithNaN = [
      {
        document_id: 'doc-1',
        filename: 'sample_text.txt',
        content_type: 'text/plain',
        chunk_text: 'Content without valid score',
        similarity_score: NaN,
      },
    ];

    const mockResponse = {
      answer: 'Answer with NaN score',
      sources: mockSourcesWithNaN,
    };

    const { ApiService } = require('../../services/api');
    ApiService.queryDocuments = jest.fn().mockResolvedValue(mockResponse);

    render(<ChatInterface documents={mockDocuments} onError={jest.fn()} />);

    const input = screen.getByPlaceholderText('Ask a question about your documents...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.click(sendButton);

    return waitFor(() => {
      expect(screen.getByText('sample_text.txt • Score: N/A')).toBeInTheDocument();
    });
  });
});