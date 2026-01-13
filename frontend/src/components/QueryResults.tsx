import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider,
  Alert,
} from '@mui/material';
import {
  ExpandMore,
  Info,
} from '@mui/icons-material';
import { QueryResponse } from '../types/api';

interface QueryResultsProps {
  result: QueryResponse | null;
  error: string | null;
}

export const QueryResults: React.FC<QueryResultsProps> = ({ result, error }) => {
  if (error) {
    return (
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Alert severity="error">
          <Typography variant="h6">Query Error</Typography>
          <Typography>{error}</Typography>
        </Alert>
      </Paper>
    );
  }

  if (!result) {
    return (
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box textAlign="center" py={4}>
          <Info sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No Results Yet
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Submit a question to see RAG-powered answers with source references
          </Typography>
        </Box>
      </Paper>
    );
  }

  const formatProcessingTime = (ms?: number) => {
    if (!ms) return 'N/A';
    return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Box display="flex" alignItems="center" gap={1} mb={2}>
        <CheckCircle color="success" />
        <Typography variant="h6">
          Answer Found
        </Typography>
        {result.processing_time && (
          <Chip
            label={`Processed in ${formatProcessingTime(result.processing_time)}`}
            size="small"
            variant="outlined"
            color="primary"
          />
        )}
        {result.confidence && (
          <Chip
            label={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
            size="small"
            variant="outlined"
            color="secondary"
          />
        )}
      </Box>

      <Typography variant="body1" sx={{ mb: 3, whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
        {result.answer}
      </Typography>

      <Divider sx={{ my: 2 }} />

      <Typography variant="h6" gutterBottom>
        Source References ({result.sources.length})
      </Typography>

      {result.sources.length === 0 ? (
        <Alert severity="info">
          No specific sources were found for this query.
        </Alert>
      ) : (
        <Box>
          {result.sources.map((source, index) => (
            <Accordion key={index} sx={{ mb: 1 }}>
              <AccordionSummary
                expandIcon={<ExpandMore />}
                sx={{
                  '& .MuiAccordionSummary-content': {
                    alignItems: 'center',
                    gap: 1,
                  },
                }}
              >
                <Typography variant="subtitle1" sx={{ flex: 1 }}>
                  Source {index + 1}
                </Typography>
                <Chip
                  label={`Score: ${(source.score * 100).toFixed(1)}%`}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
                {source.metadata?.filename && (
                  <Chip
                    label={source.metadata.filename}
                    size="small"
                    variant="outlined"
                  />
                )}
              </AccordionSummary>
              <AccordionDetails>
                <Typography
                  variant="body2"
                  sx={{
                    whiteSpace: 'pre-wrap',
                    lineHeight: 1.5,
                    bgcolor: 'grey.50',
                    p: 2,
                    borderRadius: 1,
                    fontFamily: 'monospace',
                  }}
                >
                  {source.content}
                </Typography>

                {source.metadata && Object.keys(source.metadata).length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      Metadata:
                    </Typography>
                    <Box sx={{ mt: 1 }}>
                      {Object.entries(source.metadata).map(([key, value]) => (
                        <Chip
                          key={key}
                          label={`${key}: ${String(value)}`}
                          size="small"
                          variant="outlined"
                          sx={{ mr: 1, mb: 1 }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      <Alert severity="info" sx={{ mt: 2 }}>
        <Typography variant="body2">
          This answer was generated using Retrieval-Augmented Generation (RAG) technology.
          The system retrieved relevant information from your documents and used AI to provide
          a comprehensive answer with source references.
        </Typography>
      </Alert>
    </Paper>
  );
};