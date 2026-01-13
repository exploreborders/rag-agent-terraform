import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Chip,
  Stack,
} from '@mui/material';
import { CloudUpload, InsertDriveFile, CheckCircle, Error as ErrorIcon } from '@mui/icons-material';
import { ApiService } from '../services/api';
import { DocumentUploadResponse } from '../types/api';

interface DocumentUploadProps {
  onUploadSuccess: (document: DocumentUploadResponse) => void;
  onUploadError: (error: string) => void;
}

export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  onUploadSuccess,
  onUploadError,
}) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [uploadResults, setUploadResults] = useState<Array<{
    file: File;
    status: 'success' | 'error';
    message?: string;
  }>>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(acceptedFiles);
    setUploadResults([]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: true,
  });

  const handleUpload = async () => {
    if (uploadedFiles.length === 0) return;

    setUploading(true);
    setUploadProgress(0);
    const results = [];

    for (let i = 0; i < uploadedFiles.length; i++) {
      const file = uploadedFiles[i];
      try {
        setUploadProgress((i / uploadedFiles.length) * 100);
        const response = await ApiService.uploadDocument(file);
        results.push({
          file,
          status: 'success' as const,
          message: `Uploaded successfully: ${response.message}`,
        });
        onUploadSuccess(response);
      } catch (error: any) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed';
        results.push({
          file,
          status: 'error' as const,
          message: errorMessage,
        });
        onUploadError(errorMessage);
      }
    }

    setUploadResults(results);
    setUploading(false);
    setUploadProgress(100);
  };

  const clearFiles = () => {
    setUploadedFiles([]);
    setUploadResults([]);
    setUploadProgress(0);
  };

  return (
    <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Upload Documents
      </Typography>

      <Box
        {...getRootProps()}
        sx={{
          border: '2px dashed',
          borderColor: isDragActive ? 'primary.main' : 'grey.300',
          borderRadius: 2,
          p: 4,
          textAlign: 'center',
          cursor: 'pointer',
          bgcolor: isDragActive ? 'action.hover' : 'background.paper',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'action.hover',
          },
        }}
      >
        <input {...getInputProps()} />
        <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
        <Typography variant="h6" gutterBottom>
          {isDragActive ? 'Drop files here' : 'Drag & drop documents here'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          or click to select files
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          Supported formats: PDF, TXT, JPG, PNG (max 50MB each)
        </Typography>
      </Box>

      {uploadedFiles.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Selected Files:
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {uploadedFiles.map((file, index) => (
              <Chip
                key={index}
                icon={<InsertDriveFile />}
                label={`${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`}
                variant="outlined"
                size="small"
              />
            ))}
          </Stack>

          <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={uploading}
              startIcon={<CloudUpload />}
            >
              {uploading ? 'Uploading...' : 'Upload Files'}
            </Button>
            <Button variant="outlined" onClick={clearFiles} disabled={uploading}>
              Clear
            </Button>
          </Box>

          {uploading && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress variant="determinate" value={uploadProgress} />
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                Uploading... {Math.round(uploadProgress)}%
              </Typography>
            </Box>
          )}
        </Box>
      )}

      {uploadResults.length > 0 && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Upload Results:
          </Typography>
          <Stack spacing={1}>
            {uploadResults.map((result, index) => (
              <Alert
                key={index}
                severity={result.status === 'success' ? 'success' : 'error'}
                icon={result.status === 'success' ? <CheckCircle /> : <ErrorIcon />}
              >
                <strong>{result.file.name}:</strong> {result.message}
              </Alert>
            ))}
          </Stack>
        </Box>
      )}
    </Paper>
  );
};