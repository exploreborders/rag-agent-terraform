import React from 'react';
import { render, screen, fireEvent, waitFor } from '../__tests__/test-utils';
import { DocumentList } from '../components/DocumentList';
import { ApiService } from '../services/api';

// Mock the API service
jest.mock('../services/api');
const mockedApiService = ApiService as jest.Mocked<typeof ApiService>;

describe('DocumentList', () => {
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
      filename: 'document2.txt',
      content_type: 'text/plain',
      size: 512000,
      uploaded_at: '2024-01-13T09:30:00Z',
      status: 'processing' as const,
    },
    {
      id: 'doc-3',
      filename: 'document3.pdf',
      content_type: 'application/pdf',
      size: 2048000,
      uploaded_at: '2024-01-13T08:00:00Z',
      status: 'failed' as const,
    },
  ];

  const mockOnDocumentDeleted = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    mockedApiService.getDocuments.mockResolvedValue(mockDocuments);
  });

  it('renders loading state initially', () => {
    mockedApiService.getDocuments.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve(mockDocuments), 100))
    );

    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    expect(screen.getByText('Loading documents...')).toBeInTheDocument();
  });

  it('displays documents correctly', async () => {
    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('Documents (3)')).toBeInTheDocument();
    });

    // Check document names
    expect(screen.getByText('document1.pdf')).toBeInTheDocument();
    expect(screen.getByText('document2.txt')).toBeInTheDocument();
    expect(screen.getByText('document3.pdf')).toBeInTheDocument();

    // Check file sizes
    expect(screen.getByText('976.6 KB')).toBeInTheDocument();
    expect(screen.getByText('500 KB')).toBeInTheDocument();
    expect(screen.getByText('1.95 MB')).toBeInTheDocument();

    // Check status chips
    expect(screen.getByText('completed')).toBeInTheDocument();
    expect(screen.getByText('processing')).toBeInTheDocument();
    expect(screen.getByText('failed')).toBeInTheDocument();
  });

  it('shows chunk counts for completed documents', async () => {
    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('150')).toBeInTheDocument();
      expect(screen.getByText('N/A')).toBeInTheDocument(); // For processing and failed docs
    });
  });

  it('displays file type icons correctly', async () => {
    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      // PDF icons should be present for PDF files
      const pdfIcons = screen.getAllByTestId('PictureAsPdfIcon');
      expect(pdfIcons).toHaveLength(2); // document1.pdf and document3.pdf

      // Description icon should be present for text file
      expect(screen.getByTestId('DescriptionIcon')).toBeInTheDocument();
    });
  });

  it('handles delete functionality', async () => {
    mockedApiService.deleteDocument.mockResolvedValue(undefined);

    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('document1.pdf')).toBeInTheDocument();
    });

    // Click delete button for first document
    const deleteButtons = screen.getAllByTestId('DeleteIcon');
    fireEvent.click(deleteButtons[0]);

    await waitFor(() => {
      expect(mockedApiService.deleteDocument).toHaveBeenCalledWith('doc-1');
      expect(mockOnDocumentDeleted).toHaveBeenCalled();
    });
  });

  it('shows delete loading state', async () => {
    mockedApiService.deleteDocument.mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 100))
    );

    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('document1.pdf')).toBeInTheDocument();
    });

    // Click delete button
    const deleteButtons = screen.getAllByTestId('DeleteIcon');
    fireEvent.click(deleteButtons[0]);

    // Should show loading spinner
    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    mockedApiService.getDocuments.mockRejectedValueOnce(new Error('API Error'));

    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('Failed to load documents')).toBeInTheDocument();
    });
  });

  it('handles delete errors', async () => {
    mockedApiService.deleteDocument.mockRejectedValueOnce(new Error('Delete failed'));

    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('document1.pdf')).toBeInTheDocument();
    });

    // Click delete button
    const deleteButtons = screen.getAllByTestId('DeleteIcon');
    fireEvent.click(deleteButtons[0]);

    await waitFor(() => {
      expect(screen.getByText('Failed to delete document')).toBeInTheDocument();
    });
  });

  it('shows empty state when no documents', async () => {
    mockedApiService.getDocuments.mockResolvedValue([]);

    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('No documents uploaded yet')).toBeInTheDocument();
      expect(screen.getByText('Upload some documents to get started with RAG queries')).toBeInTheDocument();
    });
  });

  it('refreshes data when refreshTrigger changes', async () => {
    const { rerender } = render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} refreshTrigger={1} />);

    await waitFor(() => {
      expect(mockedApiService.getDocuments).toHaveBeenCalledTimes(1);
    });

    // Change refresh trigger
    rerender(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} refreshTrigger={2} />);

    await waitFor(() => {
      expect(mockedApiService.getDocuments).toHaveBeenCalledTimes(2);
    });
  });

  it('allows manual refresh', async () => {
    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(mockedApiService.getDocuments).toHaveBeenCalledTimes(2);
    });
  });

  it('formats dates correctly', async () => {
    render(<DocumentList onDocumentDeleted={mockOnDocumentDeleted} />);

    await waitFor(() => {
      // Should show formatted dates
      expect(screen.getByText(/1\/13\/2024/)).toBeInTheDocument();
    });
  });
});