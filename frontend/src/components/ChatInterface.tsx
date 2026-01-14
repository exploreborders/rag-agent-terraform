import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  Stack,
  IconButton,
  Drawer,
  CircularProgress,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Send,
  SmartToy,
  Person,
  Menu,
  Add,
  Delete,
  Close,
  ExpandMore,
} from '@mui/icons-material';
import { ApiService } from '../services/api';
import {
  ChatMessage,
  ChatSession,
  Document,
} from '../types/api';

interface ChatInterfaceProps {
  documents: Document[];
  onError?: (_error: string) => void;
}

const DRAWER_WIDTH = 300;

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  documents,
  onError,
}) => {
  // State management
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [message, setMessage] = useState('');
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [newSessionTitle, setNewSessionTitle] = useState('');
  const [isCreatingSession, setIsCreatingSession] = useState(false);

  // Refs for scrolling
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    if (messagesEndRef.current?.scrollIntoView) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages]);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      // For now, we'll use local storage. Later this will be API calls
      const savedSessions = localStorage.getItem('chat_sessions');
      if (savedSessions) {
        const parsedSessions = JSON.parse(savedSessions).map((session: any) => ({
          ...session,
          messages: session.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })),
          created_at: new Date(session.created_at),
          updated_at: new Date(session.updated_at),
        }));
        setSessions(parsedSessions);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const saveSessions = (updatedSessions: ChatSession[]) => {
    try {
      localStorage.setItem('chat_sessions', JSON.stringify(updatedSessions));
      setSessions(updatedSessions);
    } catch (error) {
      console.error('Failed to save sessions:', error);
    }
  };

  const createNewSession = useCallback(async (title?: string) => {
    if (isCreatingSession) return;

    setIsCreatingSession(true);
    try {
      const sessionTitle = title || newSessionTitle || `Chat ${sessions.length + 1}`;
      const newSession: ChatSession = {
        id: `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        title: sessionTitle,
        messages: [],
        created_at: new Date(),
        updated_at: new Date(),
        document_ids: [],
      };

      const updatedSessions = [...sessions, newSession];
      saveSessions(updatedSessions);
      setCurrentSession(newSession);
      setNewSessionTitle('');
    } catch (error) {
      console.error('Failed to create session:', error);
      onError?.('Failed to create new chat session');
    } finally {
      setIsCreatingSession(false);
    }
  }, [isCreatingSession, newSessionTitle, sessions, onError]);

  // Auto-create first session if none exist
  useEffect(() => {
    if (sessions.length === 0 && !isCreatingSession) {
      createNewSession('New Chat');
    }
  }, [sessions.length, isCreatingSession, createNewSession]);

  const deleteSession = async (sessionId: string) => {
    try {
      const updatedSessions = sessions.filter(s => s.id !== sessionId);
      saveSessions(updatedSessions);

      if (currentSession?.id === sessionId) {
        setCurrentSession(updatedSessions.length > 0 ? updatedSessions[0] : null);
      }
    } catch (_error) {
      console.error('Failed to delete session:', _error);
      onError?.('Failed to delete chat session');
    }
  };

  const sendMessage = async () => {
    if (!message.trim() || !currentSession || isLoading) return;

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      role: 'user',
      content: message.trim(),
      timestamp: new Date(),
    };

    // Add user message immediately
    const updatedSession = {
      ...currentSession,
      messages: [...currentSession.messages, userMessage],
      updated_at: new Date(),
    };

    setCurrentSession(updatedSession);
    saveSessions(sessions.map(s => s.id === currentSession.id ? updatedSession : s));

    const messageToSend = message;
    setMessage('');
    setIsLoading(true);

    try {
      // Call the existing API with session context
      const response = await ApiService.queryDocuments({
        query: messageToSend,
        document_ids: selectedDocuments.length > 0 ? selectedDocuments : undefined,
        top_k: 5,
      });

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        sources: response.sources,
        processing_time: response.processing_time,
        confidence: response.confidence,
      };

      // Add assistant message
      const finalSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, assistantMessage],
        updated_at: new Date(),
      };

      setCurrentSession(finalSession);
      saveSessions(sessions.map(s => s.id === currentSession.id ? finalSession : s));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';

      const errorMsg: ChatMessage = {
        id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
        error: errorMessage,
      };

      const errorSession = {
        ...updatedSession,
        messages: [...updatedSession.messages, errorMsg],
        updated_at: new Date(),
      };

      setCurrentSession(errorSession);
      saveSessions(sessions.map(s => s.id === currentSession.id ? errorSession : s));

      onError?.(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box sx={{ display: 'flex', height: '600px' }}>
      {/* Sessions Drawer */}
      <Drawer
        variant="temporary"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: DRAWER_WIDTH,
            boxSizing: 'border-box',
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Chat Sessions
          </Typography>

          <Button
            fullWidth
            variant="outlined"
            startIcon={<Add />}
            onClick={() => createNewSession()}
            sx={{ mb: 2 }}
            disabled={isCreatingSession}
          >
            {isCreatingSession ? <CircularProgress size={20} /> : 'New Chat'}
          </Button>

          <List>
            {sessions.map((session) => (
              <ListItem
                key={session.id}
                button
                selected={currentSession?.id === session.id}
                onClick={() => {
                  setCurrentSession(session);
                  setDrawerOpen(false);
                }}
                sx={{ borderRadius: 1 }}
              >
                <ListItemText
                  primary={session.title}
                  secondary={`${session.messages.length} messages`}
                />
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteSession(session.id);
                  }}
                >
                  <Delete fontSize="small" />
                </IconButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>

      {/* Main Chat Area */}
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Chat Header */}
        <Paper elevation={1} sx={{ p: 2, borderRadius: 0 }}>
          <Box display="flex" alignItems="center" justifyContent="space-between">
            <Box display="flex" alignItems="center" gap={1}>
              <IconButton onClick={() => setDrawerOpen(true)}>
                <Menu />
              </IconButton>
              <Typography variant="h6">
                {currentSession?.title || 'Chat'}
              </Typography>
              {selectedDocuments.length > 0 && (
                <Chip
                  label={`${selectedDocuments.length} docs selected`}
                  size="small"
                  variant="outlined"
                />
              )}
            </Box>
            <Typography variant="body2" color="text.secondary">
              {currentSession?.messages.length || 0} messages
            </Typography>
          </Box>
        </Paper>

        {/* Messages Area */}
        <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
          {!currentSession && (
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              height="100%"
            >
              <Typography variant="h6" color="text.secondary" gutterBottom>
                Welcome to RAG Chat
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Start a conversation with your documents
              </Typography>
            </Box>
          )}

          {currentSession && (
            <List>
              {currentSession.messages.map((msg) => (
                <ListItem key={msg.id} sx={{ alignItems: 'flex-start', px: 0 }}>
                  <ListItemAvatar sx={{ minWidth: 40 }}>
                    <Avatar sx={{ width: 32, height: 32 }}>
                      {msg.role === 'user' ? <Person /> : <SmartToy />}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle2">
                          {msg.role === 'user' ? 'You' : 'Assistant'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {formatTime(msg.timestamp)}
                        </Typography>
                        {msg.processing_time && (
                          <Chip
                            label={`${msg.processing_time.toFixed(1)}s`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography
                          variant="body1"
                          sx={{
                            whiteSpace: 'pre-wrap',
                            lineHeight: 1.6,
                            mb: msg.sources && msg.sources.length > 0 ? 2 : 0,
                          }}
                        >
                          {msg.content}
                        </Typography>

                        {msg.error && (
                          <Alert severity="error" sx={{ mt: 1 }}>
                            {msg.error}
                          </Alert>
                        )}

                        {msg.sources && msg.sources.length > 0 && (
                          <Accordion>
                            <AccordionSummary expandIcon={<ExpandMore />}>
                              <Typography variant="body2">
                                View {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}
                              </Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              {msg.sources.map((source, idx) => (
                                <Box key={idx} sx={{ mb: 2 }}>
                                  <Typography variant="caption" color="text.secondary">
                                    {source.filename || `Source ${idx + 1}`} • Score: {typeof source.similarity_score === 'number' && !isNaN(source.similarity_score) ? `${(source.similarity_score * 100).toFixed(1)}%` : 'N/A'}
                                  </Typography>
                                  <Typography
                                    variant="body2"
                                    sx={{
                                      mt: 1,
                                      p: 1,
                                      bgcolor: 'grey.50',
                                      borderRadius: 1,
                                      fontFamily: 'monospace',
                                    }}
                                  >
                                    {source.chunk_text}
                                  </Typography>
                                </Box>
                              ))}
                            </AccordionDetails>
                          </Accordion>
                        )}
                      </Box>
                    }
                  />
                </ListItem>
              ))}

              {isLoading && (
                <ListItem sx={{ alignItems: 'flex-start', px: 0 }}>
                  <ListItemAvatar sx={{ minWidth: 40 }}>
                    <Avatar sx={{ width: 32, height: 32 }}>
                      <SmartToy />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Typography variant="subtitle2">Assistant</Typography>
                    }
                    secondary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <CircularProgress size={16} />
                        <Typography variant="body2" color="text.secondary">
                          Thinking...
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              )}
            </List>
          )}
          <div ref={messagesEndRef} />
        </Box>

        {/* Document Selection */}
        {documents.length > 0 && (
          <Paper elevation={1} sx={{ p: 2, borderRadius: 0 }}>
            <Typography variant="subtitle2" gutterBottom>
              Document Context ({documents.length} available)
            </Typography>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {documents
                .filter(doc => doc.status === 'completed')
                .map((doc) => (
                  <Chip
                    key={doc.id}
                    label={`${doc.filename} (${doc.chunks_count || 0} chunks)`}
                    onClick={() => {
                      setSelectedDocuments(prev =>
                        prev.includes(doc.id)
                          ? prev.filter(id => id !== doc.id)
                          : [...prev, doc.id]
                      );
                    }}
                    color={selectedDocuments.includes(doc.id) ? 'primary' : 'default'}
                    variant={selectedDocuments.includes(doc.id) ? 'filled' : 'outlined'}
                    size="small"
                    sx={{ cursor: 'pointer' }}
                  />
                ))}
            </Stack>
            {selectedDocuments.length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Button
                  size="small"
                  onClick={() => setSelectedDocuments([])}
                  startIcon={<Close />}
                >
                  Clear Selection ({selectedDocuments.length})
                </Button>
              </Box>
            )}
          </Paper>
        )}

        {/* Message Input */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 0 }}>
          <Box display="flex" gap={1}>
            <TextField
              fullWidth
              multiline
              rows={2}
              placeholder="Ask a question about your documents..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading || !currentSession}
              inputRef={inputRef}
            />
            <Button
              variant="contained"
              onClick={sendMessage}
              disabled={isLoading || !message.trim() || !currentSession}
              sx={{ minWidth: 100 }}
              startIcon={isLoading ? <CircularProgress size={20} /> : <Send />}
            >
              {isLoading ? 'Sending' : 'Send'}
            </Button>
          </Box>
        </Paper>
      </Box>
    </Box>
  );
};