import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
  Alert,
  CircularProgress,
} from '@mui/material';
import { Send } from '@mui/icons-material';
import { ApiService } from '../services/api';
import { QueryRequest, QueryResponse, Document } from '../types/api';

interface QueryInterfaceProps {
  documents: Document[];
  // eslint-disable-next-line no-unused-vars
  onQueryResult: (result: QueryResponse) => void;
  // eslint-disable-next-line no-unused-vars
  onQueryError: (error: string) => void;
}

export const QueryInterface: React.FC<QueryInterfaceProps> = ({
  documents,
  onQueryResult,
  onQueryError,
}) => {
  const [query, setQuery] = useState('');
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);

  const completedDocuments = documents.filter(doc => doc.status === 'completed');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!query.trim()) {
      onQueryError('Please enter a question');
      return;
    }

    try {
      setLoading(true);
      const queryData: QueryRequest = {
        query: query.trim(),
        document_ids: selectedDocuments.length > 0 ? selectedDocuments : undefined,
        top_k: topK,
      };

      const result = await ApiService.queryDocuments(queryData);
      onQueryResult(result);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Query failed';
      onQueryError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleDocumentToggle = (documentId: string) => {
    setSelectedDocuments(prev =>
      prev.includes(documentId)
        ? prev.filter(id => id !== documentId)
        : [...prev, documentId]
    );
  };

  const clearSelection = () => {
    setSelectedDocuments([]);
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Ask Questions
      </Typography>

      <form onSubmit={handleSubmit}>
        <TextField
          fullWidth
          multiline
          rows={3}
          label="Enter your question"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., What are the main findings in the research paper?"
          sx={{ mb: 2 }}
          disabled={loading}
        />

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 2 }}>
          <FormControl sx={{ minWidth: 120 }}>
            <InputLabel>Results to show</InputLabel>
            <Select
              value={topK}
              label="Results to show"
              onChange={(e) => setTopK(Number(e.target.value))}
              disabled={loading}
            >
              <MenuItem value={3}>3</MenuItem>
              <MenuItem value={5}>5</MenuItem>
              <MenuItem value={10}>10</MenuItem>
              <MenuItem value={20}>20</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Search scope</InputLabel>
            <Select
              value={selectedDocuments.length === 0 ? 'all' : 'selected'}
              label="Search scope"
              onChange={(e) => {
                if (e.target.value === 'all') {
                  setSelectedDocuments([]);
                }
              }}
              disabled={loading || completedDocuments.length === 0}
            >
              <MenuItem value="all">All documents</MenuItem>
              <MenuItem value="selected">Selected documents</MenuItem>
            </Select>
          </FormControl>
        </Stack>

        {selectedDocuments.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <Typography variant="subtitle2">
                Selected Documents ({selectedDocuments.length}):
              </Typography>
              <Button size="small" onClick={clearSelection} disabled={loading}>
                Clear All
              </Button>
            </Box>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {selectedDocuments.map((docId) => {
                const doc = completedDocuments.find(d => d.id === docId);
                return (
                  <Chip
                    key={docId}
                    label={doc?.filename || docId}
                    onDelete={() => handleDocumentToggle(docId)}
                    size="small"
                    variant="outlined"
                  />
                );
              })}
            </Stack>
          </Box>
        )}

        {completedDocuments.length > 0 && selectedDocuments.length === 0 && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              Currently searching across all {completedDocuments.length} processed documents.
              You can select specific documents below for more targeted queries.
            </Typography>
          </Alert>
        )}

        {completedDocuments.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Available Documents ({completedDocuments.length}):
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {completedDocuments.map((doc) => (
                <Chip
                  key={doc.id}
                  label={`${doc.filename} (${doc.chunks_count || 0} chunks)`}
                  onClick={() => handleDocumentToggle(doc.id)}
                  color={selectedDocuments.includes(doc.id) ? 'primary' : 'default'}
                  variant={selectedDocuments.includes(doc.id) ? 'filled' : 'outlined'}
                  size="small"
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Stack>
          </Box>
        )}

        {completedDocuments.length === 0 && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            No processed documents available. Please upload and wait for documents to be processed before querying.
          </Alert>
        )}

        <Box display="flex" justifyContent="center">
          <Button
            type="submit"
            variant="contained"
            size="large"
            startIcon={loading ? <CircularProgress size={20} /> : <Send />}
            disabled={loading || !query.trim() || completedDocuments.length === 0}
            sx={{ minWidth: 200 }}
          >
            {loading ? 'Searching...' : 'Ask Question'}
          </Button>
        </Box>
      </form>
    </Paper>
  );
};