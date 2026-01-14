import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Tooltip,
  Alert,
  CircularProgress,
  Button,
} from '@mui/material';
import {
  Delete,
  InsertDriveFile,
  PictureAsPdf,
  Image,
  Description,
  Refresh,
} from '@mui/icons-material';
import { ApiService } from '../services/api';
import { Document } from '../types/api';

interface DocumentListProps {
  refreshTrigger?: number; // Trigger to refresh the list
  onDocumentDeleted?: () => void;
}

export const DocumentList: React.FC<DocumentListProps> = ({
  refreshTrigger,
  onDocumentDeleted,
}) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const docs = await ApiService.getDocuments();
      console.log('Received documents:', docs?.length || 0, 'documents');
      if (docs && docs.length > 0) {
        console.log('Sample document:', docs[0]);
        console.log('First document uploaded_at:', `"${docs[0].uploaded_at}"`, 'Type:', typeof docs[0].uploaded_at);
      }
      setDocuments(docs || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [refreshTrigger]);

  const handleDeleteDocument = async (documentId: string) => {
    try {
      setDeletingId(documentId);
      await ApiService.deleteDocument(documentId);
      setDocuments(documents.filter(doc => doc.id !== documentId));
      onDocumentDeleted?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document');
    } finally {
      setDeletingId(null);
    }
  };

  const getFileIcon = (contentType: string) => {
    if (contentType.includes('pdf')) return <PictureAsPdf color="error" />;
    if (contentType.includes('image')) return <Image color="primary" />;
    return <Description color="action" />;
  };

  const getStatusColor = (status: Document['status']) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'processing':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (dateString: string | undefined | null) => {
    console.log('formatDate called with:', `"${dateString}"`, 'Type:', typeof dateString);
    // Handle empty/null/undefined values
    if (dateString === null || dateString === undefined || dateString === '') {
      console.log('Returning "Not uploaded" for null/undefined/empty value');
      return 'Not uploaded';
    }

    // Ensure it's a string
    const dateStr = String(dateString).trim();
    if (!dateStr) {
      return 'Not uploaded';
    }

    try {
      // Try different parsing strategies
      let date: Date | null = null;

      // Strategy 1: Direct Date constructor (handles ISO strings)
      date = new Date(dateStr);
      if (!isNaN(date.getTime())) {
        return date.toLocaleString();
      }

      // Strategy 2: Handle common backend formats
      const formats = [
        // Remove microseconds: "2024-01-14T16:11:53.123456" -> "2024-01-14T16:11:53"
        dateStr.split('.')[0],
        // Add Z if missing: "2024-01-14T16:11:53" -> "2024-01-14T16:11:53Z"
        dateStr + (dateStr.includes('Z') ? '' : 'Z'),
        // Handle ISO with timezone: "2024-01-14T16:11:53+00:00" -> "2024-01-14T16:11:53Z"
        dateStr.replace(/\+.*$/, 'Z'),
        // Replace space with T and add Z
        dateStr.replace(' ', 'T') + (dateStr.includes('Z') ? '' : 'Z'),
      ];

      for (const format of formats) {
        date = new Date(format);
        if (!isNaN(date.getTime())) {
          return date.toLocaleString();
        }
      }

      // Strategy 3: Manual parsing for common patterns
      const patterns = [
        /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z?$/,
        /^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$/,
      ];

      for (const pattern of patterns) {
        const match = dateStr.match(pattern);
        if (match) {
          const [, year, month, day, hour, minute, second] = match;
          date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day),
                         parseInt(hour), parseInt(minute), parseInt(second));
          if (!isNaN(date.getTime())) {
            return date.toLocaleString();
          }
        }
      }

    } catch (error) {
      console.error('Error parsing date:', dateStr, error);
    }

    return 'Invalid date format';
  };

  if (loading) {
    return (
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box display="flex" justifyContent="center" alignItems="center" p={4}>
          <CircularProgress />
          <Typography sx={{ ml: 2 }}>Loading documents...</Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          Documents ({documents.length})
        </Typography>
        <Button
          startIcon={<Refresh />}
          onClick={fetchDocuments}
          disabled={loading}
          size="small"
        >
          Refresh
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {documents.length === 0 ? (
        <Box textAlign="center" py={4}>
          <InsertDriveFile sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No documents uploaded yet
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Upload some documents to get started with RAG queries
          </Typography>
        </Box>
      ) : (
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>File</TableCell>
                <TableCell>Type</TableCell>
                <TableCell>Size</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Chunks</TableCell>
                <TableCell>Uploaded</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {documents.map((doc) => (
                <TableRow key={doc.id} hover>
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={1}>
                      {getFileIcon(doc.content_type)}
                      <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                        {doc.filename}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="caption">
                      {doc.content_type.split('/')[1].toUpperCase()}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatFileSize(doc.size)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={doc.status}
                      color={getStatusColor(doc.status)}
                      size="small"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {doc.chunks_count || 'N/A'}
                    </Typography>
                  </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatDate(doc.uploaded_at) || 'Not available'}
                        </Typography>
                      </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Delete document">
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDeleteDocument(doc.id)}
                        disabled={deletingId === doc.id}
                        data-testid={`delete-document-${doc.id}`}
                      >
                        {deletingId === doc.id ? (
                          <CircularProgress size={20} />
                        ) : (
                          <Delete />
                        )}
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
};