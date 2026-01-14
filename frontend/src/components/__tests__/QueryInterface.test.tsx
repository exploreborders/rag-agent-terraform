import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../__tests__/test-utils';
import { QueryInterface } from '../QueryInterface';
import { ApiService } from '../../services/api';

// Mock the API service
jest.mock('../../services/api');
const mockedApiService = ApiService as jest.Mocked<typeof ApiService>;

describe('QueryInterface', () => {
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
    {
      id: 'doc-2',
      filename: 'document2.pdf',
      content_type: 'application/pdf',
      size: 512000,
      uploaded_at: '2024-01-13T09:30:00Z',
      status: 'completed' as const,
      chunks_count: 75,
    },
    {
      id: 'doc-3',
      filename: 'document3.pdf',
      content_type: 'application/pdf',
      size: 2048000,
      uploaded_at: '2024-01-13T08:00:00Z',
      status: 'processing' as const,
    },
  ];

  const mockOnQueryResult = jest.fn();
  const mockOnQueryError = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders query interface correctly', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    expect(screen.getByText('Ask Questions')).toBeInTheDocument();
    expect(screen.getByLabelText('Enter your question')).toBeInTheDocument();
    expect(screen.getByText('Ask Question')).toBeInTheDocument();
  });

  it('shows available documents section', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    expect(screen.getByText('Available Documents (2):')).toBeInTheDocument();
    expect(screen.getByText('document1.pdf (150 chunks)')).toBeInTheDocument();
    expect(screen.getByText('document2.pdf (75 chunks)')).toBeInTheDocument();
    expect(screen.queryByText('document3.pdf')).not.toBeInTheDocument(); // Not completed
  });

  it('allows selecting specific documents', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Initially shows "All documents" as the selected value
    expect(screen.getByText('All documents')).toBeInTheDocument();

    // Change to selected documents mode - get the second combobox (Search scope)
    const comboboxes = screen.getAllByRole('combobox');
    const searchScopeSelect = comboboxes[1]; // Second combobox is Search scope
    fireEvent.mouseDown(searchScopeSelect);
    fireEvent.click(screen.getByText('Selected documents'));

    // After switching to "Selected documents" mode, the selected documents section should not appear yet (no documents selected)
    expect(screen.queryByText(/Selected Documents/)).not.toBeInTheDocument();
  });

  it('handles document selection', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Switch to selected documents mode - get the second combobox (Search scope)
    const comboboxes = screen.getAllByRole('combobox');
    const searchScopeSelect = comboboxes[1]; // Second combobox is Search scope
    fireEvent.mouseDown(searchScopeSelect);
    fireEvent.click(screen.getByText('Selected documents'));

    // Click on first document chip
    const docChip = screen.getByText('document1.pdf (150 chunks)');
    fireEvent.click(docChip);

    expect(screen.getByText('Selected Documents (1):')).toBeInTheDocument();
    expect(screen.getByText('document1.pdf')).toBeInTheDocument();
  });

  it('allows changing number of results', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    const comboboxes = screen.getAllByRole('combobox');
    const resultsSelect = comboboxes[0]; // First combobox is Results to show
    expect(screen.getByDisplayValue('5')).toBeInTheDocument();

    fireEvent.mouseDown(resultsSelect);
    fireEvent.click(screen.getByText('10'));

    expect(screen.getByDisplayValue('10')).toBeInTheDocument();
  });

  it('submits query successfully', async () => {
    const mockResponse = {
      answer: 'Test answer',
      sources: [],
      confidence: 0.9,
      processing_time: 1.2,
    };

    mockedApiService.queryDocuments.mockResolvedValue(mockResponse);

    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Enter query
    const queryInput = screen.getByLabelText('Enter your question');
    fireEvent.change(queryInput, { target: { value: 'What is machine learning?' } });

    // Submit query
    const submitButton = screen.getByText('Ask Question');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockedApiService.queryDocuments).toHaveBeenCalledWith({
        query: 'What is machine learning?',
        document_ids: undefined,
        top_k: 5,
      });
    });
    await waitFor(() => {
      expect(mockOnQueryResult).toHaveBeenCalledWith(mockResponse);
    });
  });

  it('submits query with selected documents', async () => {
    const mockResponse = {
      answer: 'Test answer',
      sources: [],
    };

    mockedApiService.queryDocuments.mockResolvedValue(mockResponse);

    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Switch to selected documents mode and select one
    const scopeSelect = screen.getAllByRole('combobox')[1];
    fireEvent.mouseDown(scopeSelect);
    fireEvent.click(screen.getByText('Selected documents'));

    const docChip = screen.getByText('document1.pdf (150 chunks)');
    fireEvent.click(docChip);

    // Enter and submit query
    const queryInput = screen.getByLabelText('Enter your question');
    fireEvent.change(queryInput, { target: { value: 'Test question' } });

    const submitButton = screen.getByText('Ask Question');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockedApiService.queryDocuments).toHaveBeenCalledWith({
        query: 'Test question',
        document_ids: ['doc-1'],
        top_k: 5,
      });
    });
  });

  it('handles query errors', async () => {
    mockedApiService.queryDocuments.mockRejectedValue(new Error('Query failed'));

    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Enter and submit query
    const queryInput = screen.getByLabelText('Enter your question');
    fireEvent.change(queryInput, { target: { value: 'Test question' } });

    const submitButton = screen.getByText('Ask Question');
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockOnQueryError).toHaveBeenCalledWith('Query failed');
    });
  });

  it('disables submit button when no query entered', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    const submitButton = screen.getByText('Ask Question');
    expect(submitButton).toBeDisabled();
  });

  it('disables submit button when no completed documents', () => {
    const processingDocs = mockDocuments.map(doc => ({ ...doc, status: 'processing' as const }));

    render(
      <QueryInterface
        documents={processingDocs}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    const queryInput = screen.getByLabelText('Enter your question');
    fireEvent.change(queryInput, { target: { value: 'Test question' } });

    const submitButton = screen.getByText('Ask Question');
    expect(submitButton).toBeDisabled();
  });

  it('shows loading state during query', async () => {
    mockedApiService.queryDocuments.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve({ answer: 'Test', sources: [] }), 100))
    );

    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Enter and submit query
    const queryInput = screen.getByLabelText('Enter your question');
    fireEvent.change(queryInput, { target: { value: 'Test question' } });

    const submitButton = screen.getByText('Ask Question');
    fireEvent.click(submitButton);

    // Should show loading state
    await waitFor(() => {
      expect(screen.getByText('Searching...')).toBeInTheDocument();
    });
  });

  it('shows warning when no completed documents', () => {
    const processingDocs = mockDocuments.map(doc => ({ ...doc, status: 'processing' as const }));

    render(
      <QueryInterface
        documents={processingDocs}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    expect(screen.getByText('No processed documents available. Please upload and wait for documents to be processed before querying.')).toBeInTheDocument();
  });

  it('allows clearing document selection', () => {
    render(
      <QueryInterface
        documents={mockDocuments}
        onQueryResult={mockOnQueryResult}
        onQueryError={mockOnQueryError}
      />
    );

    // Switch to selected documents mode and select one
    const scopeSelect = screen.getAllByRole('combobox')[1];
    fireEvent.mouseDown(scopeSelect);
    fireEvent.click(screen.getByText('Selected documents'));

    const docChip = screen.getByText('document1.pdf (150 chunks)');
    fireEvent.click(docChip);

    expect(screen.getByText('Selected Documents (1):')).toBeInTheDocument();

    // Clear selection
    const clearButton = screen.getByText('Clear All');
    fireEvent.click(clearButton);

    // After clearing all documents, the selected documents section should disappear
    expect(screen.queryByText(/Selected Documents/)).not.toBeInTheDocument();
  });
});