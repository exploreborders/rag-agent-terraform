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
  Description,
  PictureAsPdf,
  Image,
  Article,
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
  darkMode?: boolean;
}

const DRAWER_WIDTH = 300;

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  documents,
  onError,
  darkMode = false,
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

  const getFileIcon = (filename: string) => {
    const extension = filename.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return <PictureAsPdf fontSize="small" />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'webp':
        return <Image fontSize="small" />;
      case 'txt':
        return <Article fontSize="small" />;
      default:
        return <Description fontSize="small" />;
    }
  };

  const getConfidenceColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Box
      sx={{
        display: 'flex',
        height: '80vh',
        bgcolor: darkMode ? 'grey.900' : 'background.default',
        color: darkMode ? 'grey.100' : 'text.primary',
        transition: 'background-color 0.3s ease, color 0.3s ease'
      }}
    >
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
        <Box sx={{
          p: 2,
          bgcolor: darkMode ? 'grey.900' : 'inherit',
          color: darkMode ? 'grey.100' : 'inherit',
          height: '100%'
        }}>
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
                sx={{
                  borderRadius: 1,
                  bgcolor: darkMode ? 'grey.800' : 'inherit',
                  color: darkMode ? 'grey.100' : 'inherit',
                  '&:hover': {
                    bgcolor: darkMode ? 'grey.700' : 'inherit',
                  },
                  '&.Mui-selected': {
                    bgcolor: darkMode ? 'grey.600' : 'primary.main',
                    color: darkMode ? 'grey.100' : 'inherit',
                    '&:hover': {
                      bgcolor: darkMode ? 'grey.600' : 'primary.main',
                    },
                  },
                }}
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
                   sx={{
                     bgcolor: darkMode ? 'grey.800' : 'inherit',
                     color: darkMode ? 'grey.100' : 'inherit'
                   }}
                 />
               )}
             </Box>
             <Typography variant="body2" color="text.secondary">
               {currentSession?.messages.length || 0} messages
             </Typography>
           </Box>
        </Paper>

        {/* Messages Area */}
        <Box sx={{
          flexGrow: 1,
          overflow: 'auto',
          p: 2,
          bgcolor: darkMode ? 'grey.800' : 'grey.50'
        }}>
          {!currentSession && (
            <Box
              display="flex"
              flexDirection="column"
              alignItems="center"
              justifyContent="center"
              height="100%"
            >
              <Typography variant="h6" color={darkMode ? 'grey.300' : 'text.secondary'} gutterBottom>
                Welcome to RAG Chat
              </Typography>
              <Typography variant="body2" color={darkMode ? 'grey.400' : 'text.secondary'}>
                Start a conversation with your documents
              </Typography>
            </Box>
          )}

          {currentSession && (
            <List>
              {currentSession.messages.map((msg) => (
                <ListItem key={msg.id} sx={{
                  alignItems: 'flex-start',
                  px: 0,
                  bgcolor: darkMode ? 'grey.800' : 'inherit',
                  '&:hover': {
                    bgcolor: darkMode ? 'grey.700' : 'grey.50'
                  }
                }}>
                  <ListItemAvatar sx={{ minWidth: 40 }}>
                    <Avatar sx={{
                      width: 32,
                      height: 32,
                      bgcolor: darkMode ? 'grey.700' : 'primary.main'
                    }}>
                      {msg.role === 'user' ? <Person /> : <SmartToy />}
                    </Avatar>
                  </ListItemAvatar>
                  <Box sx={{ width: '100%' }}>
                     <Box display="flex" alignItems="center" gap={1} mb={1}>
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

                     <Box
                       sx={{
                         whiteSpace: 'pre-wrap',
                         lineHeight: 1.6,
                         mb: msg.sources && msg.sources.length > 0 ? 2 : 0,
                       }}
                     >
                       {msg.content}
                     </Box>

                         {msg.error && (
                           <Alert severity="error" sx={{
                             mt: 1,
                             bgcolor: darkMode ? 'error.dark' : 'inherit',
                             color: darkMode ? 'error.contrastText' : 'inherit'
                           }}>
                             {msg.error}
                           </Alert>
                         )}

                     {msg.sources && msg.sources.length > 0 && (
                       <Accordion sx={{ mt: 1, '&:before': { display: 'none' } }}>
                         <AccordionSummary
                           expandIcon={<ExpandMore />}
                           sx={{
                             minHeight: 36,
                             '& .MuiAccordionSummary-content': { my: 1 }
                           }}
                         >
                           <Box display="flex" alignItems="center" gap={1}>
                             <Description fontSize="small" color="action" />
                             <Typography variant="body2" color="primary">
                               {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''} cited
                             </Typography>
                             {msg.confidence && (
                               <Chip
                                 label={`${(msg.confidence * 100).toFixed(0)}%`}
                                 size="small"
                                 color={getConfidenceColor(msg.confidence)}
                                 variant="outlined"
                               />
                             )}
                           </Box>
                         </AccordionSummary>
                         <AccordionDetails sx={{ pt: 0 }}>
                           <Stack spacing={2}>
                             {msg.sources.map((source, idx) => (
                               <Paper
                                 key={idx}
                                 elevation={0}
                                 sx={{
                                   p: 2,
                                   bgcolor: darkMode ? 'grey.700' : 'grey.50',
                                   borderRadius: 1,
                                   border: '1px solid',
                                   borderColor: darkMode ? 'grey.600' : 'divider'
                                 }}
                               >
                                 <Box display="flex" alignItems="center" gap={1} mb={1}>
                                   {getFileIcon(source.filename || 'unknown')}
                                   <Box sx={{ flexGrow: 1 }}>
                                     {source.filename || `Source ${idx + 1}`}
                                   </Box>
                                   <Box display="flex" alignItems="center" gap={1}>
                                     {typeof source.similarity_score === 'number' && !isNaN(source.similarity_score) && (
                                       <Chip
                                         label={`${(source.similarity_score * 100).toFixed(1)}%`}
                                         size="small"
                                         color={getConfidenceColor(source.similarity_score)}
                                         variant="filled"
                                       />
                                     )}
                                     <Box sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
                                       {source.content_type || 'text'}
                                     </Box>
                                   </Box>
                                 </Box>
                                 <Box
                                   sx={{
                                     lineHeight: 1.6,
                                     color: 'text.secondary',
                                     fontStyle: 'italic',
                                   }}
                                 >
                                   {source.chunk_text.length > 200
                                     ? `${source.chunk_text.substring(0, 200)}...`
                                     : source.chunk_text
                                   }
                                 </Box>
                               </Paper>
                             ))}
                           </Stack>
                         </AccordionDetails>
                       </Accordion>
                     )}
                   </Box>
                </ListItem>
              ))}

              {isLoading && (
                 <ListItem sx={{ alignItems: 'flex-start', px: 0 }}>
                   <ListItemAvatar sx={{ minWidth: 40 }}>
                     <Avatar sx={{ width: 32, height: 32 }}>
                       <SmartToy />
                     </Avatar>
                   </ListItemAvatar>
                   <Box sx={{ width: '100%' }}>
                     <Typography variant="subtitle2" sx={{ mb: 1 }}>
                       Assistant
                     </Typography>
                     <Box display="flex" alignItems="center" gap={1}>
                       <CircularProgress size={16} />
                       <Typography variant="body2" color="text.secondary">
                         Thinking...
                       </Typography>
                     </Box>
                   </Box>
                 </ListItem>
               )}
            </List>
          )}
          <div ref={messagesEndRef} />
        </Box>

        {/* Document Selection */}
        {documents.length > 0 && (
          <Paper elevation={1} sx={{
            p: 2,
            borderRadius: 0,
            bgcolor: darkMode ? 'grey.800' : 'inherit',
            color: darkMode ? 'grey.100' : 'inherit'
          }}>
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
        <Paper elevation={2} sx={{
          p: 2,
          borderRadius: 0,
          bgcolor: darkMode ? 'grey.800' : 'inherit',
          color: darkMode ? 'grey.100' : 'inherit'
        }}>
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
              sx={{
                '& .MuiOutlinedInput-root': {
                  bgcolor: darkMode ? 'grey.700' : 'inherit',
                  color: darkMode ? 'grey.100' : 'inherit',
                  '& fieldset': {
                    borderColor: darkMode ? 'grey.600' : 'inherit',
                  },
                  '&:hover fieldset': {
                    borderColor: darkMode ? 'grey.500' : 'inherit',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: darkMode ? 'primary.light' : 'inherit',
                  },
                },
                '& .MuiInputBase-input::placeholder': {
                  color: darkMode ? 'grey.400' : 'inherit',
                },
              }}
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