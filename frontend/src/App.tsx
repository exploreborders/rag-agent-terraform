import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Snackbar,
  Alert,
  Chip,
  Stack,
  Paper,
  Grid,
  IconButton,
} from '@mui/material';
import {
  SmartToy,
  Storage,
  Memory,
  Assessment,
  Brightness4,
  Brightness7,
} from '@mui/icons-material';
import { DocumentUpload } from './components/DocumentUpload';
import { DocumentList } from './components/DocumentList';
import { ChatInterface } from './components/ChatInterface';
import { ApiService } from './services/api';
import {
  Document,
  DocumentUploadResponse,
  HealthStatus,
  StatsResponse,
} from './types/api';

// Create dynamic theme function
const createAppTheme = (darkMode: boolean) => createTheme({
  palette: {
    mode: darkMode ? 'dark' : 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: darkMode ? '#121212' : '#fafafa',
      paper: darkMode ? '#1e1e1e' : '#ffffff',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    const saved = localStorage.getItem('app_dark_mode');
    return saved ? JSON.parse(saved) : false;
  });

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  // Reload documents when refresh trigger changes
  useEffect(() => {
    if (refreshTrigger > 0) {
      loadDocuments();
    }
  }, [refreshTrigger]);

  const loadDocuments = async () => {
    try {
      const docs = await ApiService.getDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  const loadInitialData = async () => {
    try {
      const [docs, health, statsData] = await Promise.all([
        ApiService.getDocuments(),
        ApiService.getHealth(),
        ApiService.getStats(),
      ]);
      setDocuments(docs);
      setHealthStatus(health);
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  };

  const handleUploadSuccess = (_response: DocumentUploadResponse) => {
    setRefreshTrigger(prev => prev + 1);
    setUploadError(null);
  };

  const handleUploadError = (error: string) => {
    setUploadError(error);
  };

  const handleChatError = (error: string) => {
    setUploadError(error);
  };

  const handleDocumentDeleted = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  const getServiceStatusColor = (status: string) => {
    return status === 'healthy' ? 'success' : 'error';
  };

  const toggleDarkMode = () => {
    setDarkMode((prev: boolean) => {
      const newMode = !prev;
      localStorage.setItem('app_dark_mode', JSON.stringify(newMode));
      return newMode;
    });
  };

  // Create theme based on dark mode state
  const theme = createAppTheme(darkMode);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{
        flexGrow: 1,
        minHeight: '100vh',
        bgcolor: darkMode ? 'background.default' : 'grey.50',
        transition: 'background-color 0.3s ease'
      }}>
        {/* Header */}
        <AppBar position="static" elevation={2}>
          <Toolbar>
            <SmartToy sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              RAG Agent - Document Intelligence System
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              {healthStatus && (
                <Stack direction="row" spacing={1}>
                  <Chip
                    icon={<SmartToy />}
                    label={`Ollama: ${healthStatus.services.ollama}`}
                    color={getServiceStatusColor(healthStatus.services.ollama)}
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={<Storage />}
                    label={`Vector Store: ${healthStatus.services.vector_store}`}
                    color={getServiceStatusColor(healthStatus.services.vector_store)}
                    size="small"
                    variant="outlined"
                  />
                  <Chip
                    icon={<Memory />}
                    label={`Redis: ${healthStatus.services.redis}`}
                    color={getServiceStatusColor(healthStatus.services.redis)}
                    size="small"
                    variant="outlined"
                  />
                </Stack>
              )}
              <IconButton onClick={toggleDarkMode} color="inherit">
                {darkMode ? <Brightness7 /> : <Brightness4 />}
              </IconButton>
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ py: 4 }}>
          {/* Stats Overview */}
          {stats && (
            <Grid container spacing={3} sx={{ mb: 4 }}>
              <Grid item xs={12} sm={6} md={3}>
                <Paper elevation={1} sx={{
                  p: 2,
                  textAlign: 'center',
                  bgcolor: darkMode ? 'grey.800' : 'inherit',
                  color: darkMode ? 'grey.100' : 'inherit'
                }}>
                  <Assessment color="primary" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="h4">{stats.total_documents}</Typography>
                  <Typography variant="body2" color={darkMode ? 'grey.400' : 'text.secondary'}>Documents</Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper elevation={1} sx={{
                  p: 2,
                  textAlign: 'center',
                  bgcolor: darkMode ? 'grey.800' : 'inherit',
                  color: darkMode ? 'grey.100' : 'inherit'
                }}>
                  <Storage color="secondary" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="h4">{stats.total_chunks}</Typography>
                  <Typography variant="body2" color={darkMode ? 'grey.400' : 'text.secondary'}>Text Chunks</Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper elevation={1} sx={{
                  p: 2,
                  textAlign: 'center',
                  bgcolor: darkMode ? 'grey.800' : 'inherit',
                  color: darkMode ? 'grey.100' : 'inherit'
                }}>
                  <SmartToy color="success" sx={{ fontSize: 40, mb: 1 }} />
                  <Typography variant="h4">{stats.total_queries}</Typography>
                  <Typography variant="body2" color={darkMode ? 'grey.400' : 'text.secondary'}>Queries</Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper elevation={1} sx={{
                  p: 2,
                  textAlign: 'center',
                  bgcolor: darkMode ? 'grey.800' : 'inherit',
                  color: darkMode ? 'grey.100' : 'inherit'
                }}>
                  <Memory color="info" sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="h4">{stats?.average_response_time?.toFixed(1) || 'N/A'}s</Typography>
                  <Typography variant="body2" color={darkMode ? 'grey.400' : 'text.secondary'}>Avg Response</Typography>
                </Paper>
              </Grid>
            </Grid>
          )}

          {/* Main Content */}
          <Grid container spacing={3}>
            <Grid item xs={12} lg={6}>
              {/* Document Upload */}
              <DocumentUpload
                onUploadSuccess={handleUploadSuccess}
                onUploadError={handleUploadError}
                darkMode={darkMode}
              />

              {/* Document List */}
              <DocumentList
                refreshTrigger={refreshTrigger}
                onDocumentDeleted={handleDocumentDeleted}
                darkMode={darkMode}
              />
            </Grid>

              <Grid item xs={12} lg={6}>
                {/* Chat Interface */}
                <ChatInterface
                  documents={documents}
                  onError={handleChatError}
                  darkMode={darkMode}
                />
              </Grid>
          </Grid>
        </Container>

        {/* Error Notifications */}
        <Snackbar
          open={!!uploadError}
          autoHideDuration={6000}
          onClose={() => setUploadError(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        >
          <Alert onClose={() => setUploadError(null)} severity="error">
            Upload Error: {uploadError}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

export default App;
