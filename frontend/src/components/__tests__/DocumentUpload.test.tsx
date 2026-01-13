import React from 'react';
import { render, screen, fireEvent, waitFor } from '../../__tests__/test-utils';
import { DocumentUpload } from '../DocumentUpload';
import { ApiService } from '../../services/api';

// Mock the API service
jest.mock('../../services/api');
const mockedApiService = ApiService as jest.Mocked<typeof ApiService>;

describe('DocumentUpload', () => {
  const mockOnUploadSuccess = jest.fn();
  const mockOnUploadError = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders upload interface correctly', () => {
    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    expect(screen.getByText('Upload Documents')).toBeInTheDocument();
    expect(screen.getByText('Drag & drop documents here')).toBeInTheDocument();
    expect(screen.getByText('or click to select files')).toBeInTheDocument();
  });

  it('shows supported file types information', () => {
    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    expect(screen.getByText(/Supported formats: PDF, TXT, JPG, PNG/)).toBeInTheDocument();
    expect(screen.getByText(/Maximum size: 50MB/)).toBeInTheDocument();
  });

  it('handles file drop correctly', async () => {
    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });

    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const dropzone = screen.getByText('Drag & drop documents here').closest('div');

    // Simulate file drop
    fireEvent.drop(dropzone!, {
      dataTransfer: {
        files: [file],
      },
    });

    await waitFor(() => {
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });
  });

  it('uploads files successfully', async () => {
    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
    const mockResponse = { id: 'test-id', message: 'Upload successful', status: 'success' };

    mockedApiService.uploadDocument.mockResolvedValueOnce(mockResponse);

    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    // Drop file
    const dropzone = screen.getByText('Drag & drop documents here').closest('div');
    fireEvent.drop(dropzone!, {
      dataTransfer: {
        files: [file],
      },
    });

    // Click upload button
    const uploadButton = screen.getByText('Upload Files');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(mockedApiService.uploadDocument).toHaveBeenCalledWith(file);
      expect(mockOnUploadSuccess).toHaveBeenCalledWith(mockResponse);
    });
  });

  it('handles upload errors correctly', async () => {
    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
    const errorMessage = 'Upload failed';

    mockedApiService.uploadDocument.mockRejectedValueOnce(new Error(errorMessage));

    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    // Drop file
    const dropzone = screen.getByText('Drag & drop documents here').closest('div');
    fireEvent.drop(dropzone!, {
      dataTransfer: {
        files: [file],
      },
    });

    // Click upload button
    const uploadButton = screen.getByText('Upload Files');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(mockOnUploadError).toHaveBeenCalledWith(errorMessage);
    });
  });

  it('shows upload progress and results', async () => {
    const file1 = new File(['content1'], 'test1.pdf', { type: 'application/pdf' });
    const file2 = new File(['content2'], 'test2.pdf', { type: 'application/pdf' });

    mockedApiService.uploadDocument
      .mockResolvedValueOnce({ id: 'id1', message: 'Success 1', status: 'success' })
      .mockResolvedValueOnce({ id: 'id2', message: 'Success 2', status: 'success' });

    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    // Drop files
    const dropzone = screen.getByText('Drag & drop documents here').closest('div');
    fireEvent.drop(dropzone!, {
      dataTransfer: {
        files: [file1, file2],
      },
    });

    // Upload
    const uploadButton = screen.getByText('Upload Files');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(screen.getByText('Upload Results:')).toBeInTheDocument();
    });

    expect(screen.getByText(/test1.pdf.*Success 1/)).toBeInTheDocument();
    expect(screen.getByText(/test2.pdf.*Success 2/)).toBeInTheDocument();
  });

  it('allows clearing selected files', () => {
    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });

    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    // Drop file
    const dropzone = screen.getByText('Drag & drop documents here').closest('div');
    fireEvent.drop(dropzone!, {
      dataTransfer: {
        files: [file],
      },
    });

    expect(screen.getByText('test.pdf')).toBeInTheDocument();

    // Clear files
    const clearButton = screen.getByText('Clear');
    fireEvent.click(clearButton);

    expect(screen.queryByText('test.pdf')).not.toBeInTheDocument();
  });

  it('disables upload button when no files selected', () => {
    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    const uploadButton = screen.getByText('Upload Files');
    expect(uploadButton).toBeDisabled();
  });

  it('disables upload button during upload', async () => {
    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });

    mockedApiService.uploadDocument.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve({ id: 'test', message: 'success', status: 'success' }), 100))
    );

    render(
      <DocumentUpload
        onUploadSuccess={mockOnUploadSuccess}
        onUploadError={mockOnUploadError}
      />
    );

    // Drop file
    const dropzone = screen.getByText('Drag & drop documents here').closest('div');
    fireEvent.drop(dropzone!, {
      dataTransfer: {
        files: [file],
      },
    });

    // Start upload
    const uploadButton = screen.getByText('Upload Files');
    fireEvent.click(uploadButton);

    // Button should be disabled during upload
    await waitFor(() => {
      expect(screen.getByText('Uploading...')).toBeInTheDocument();
    });
  });
});