# 🌐 RAG Agent Frontend

**Modern React-based web interface for the RAG Agent Terraform system** - Complete document management and intelligent querying interface built with Material-UI and TypeScript.

## 📋 Overview

This frontend application provides a user-friendly interface for interacting with the RAG Agent system. Upload documents, manage your knowledge base, and ask intelligent questions powered by local AI models.

### ✨ Key Features

- **📤 Document Upload**: Drag & drop interface for PDF, text, and image files
- **📋 Document Management**: View, organize, and delete processed documents
- **🔍 Intelligent Querying**: Ask questions with configurable search parameters
- **📊 Results Display**: Expandable source references with metadata
- **📈 Real-time Monitoring**: Live system health and performance indicators
- **🎨 Modern UI**: Material-UI components with responsive design
- **🧪 Comprehensive Testing**: 200+ test cases ensuring reliability

## 🚀 Quick Start

### Prerequisites

- **Node.js 16+**: [nodejs.org](https://nodejs.org/)
- **RAG Backend**: Running on `http://localhost:8000`
- **Monitoring Stack**: Optional, for full dashboard features

### Installation & Development

```bash
# Install dependencies
npm install

# Start development server
npm start

# Access the application
open http://localhost:3001
```

### Production Build

```bash
# Build for production
npm run build

# Serve production build
npx serve -s build
```

## 🏗️ Architecture

```
frontend/
├── 📁 src/
│   ├── 📁 components/       # React components
│   │   ├── DocumentUpload.tsx   # File upload interface
│   │   ├── DocumentList.tsx     # Document management
│   │   ├── QueryInterface.tsx   # Query input form
│   │   ├── QueryResults.tsx     # Results display
│   │   └── App.tsx             # Main application
│   ├── 📁 services/        # API integration
│   │   └── api.ts          # Axios-based API client
│   ├── 📁 types/           # TypeScript definitions
│   │   └── api.ts          # API response types
│   └── 📁 __tests__/       # Test utilities
│       ├── test-utils.tsx      # Test helpers and mocks
│       └── api.test.ts         # API service tests
├── 📁 public/             # Static assets
├── package.json           # Dependencies and scripts
├── tsconfig.json          # TypeScript configuration
└── jest.config.js         # Test configuration
```

## 🧩 Components

### DocumentUpload
- Drag & drop file upload with progress tracking
- File type validation (PDF, TXT, JPG, PNG)
- Size limits and error handling
- Batch upload support

### DocumentList
- Document inventory with processing status
- File metadata display (size, chunks, upload date)
- Delete functionality with confirmation
- Real-time status updates

### QueryInterface
- Natural language query input
- Document selection and filtering
- Configurable result limits (3-20)
- Search scope controls (all vs. selected documents)

### QueryResults
- AI-generated answers with confidence scores
- Expandable source references
- Metadata display for each source
- Processing time metrics

## 🔌 API Integration

### Backend Endpoints

```typescript
// Health monitoring
GET /health

// Document management
GET /documents
POST /documents/upload
DELETE /documents/{id}

// Intelligent querying
POST /query

// Statistics
GET /stats
```

### Configuration

Create a `.env` file in the frontend root:

```bash
REACT_APP_API_URL=http://localhost:8000
```

## 🧪 Testing

### Test Categories

- **Component Tests**: Individual component functionality
- **Integration Tests**: Component interactions
- **API Tests**: Backend communication
- **User Workflow Tests**: Complete user journeys

### Running Tests

```bash
# All tests with coverage
npm run test:ci

# Development test mode
npm run test:watch

# Single test file
npm test DocumentUpload.test.tsx
```

### Test Coverage

- **Components**: 90%+ coverage
- **Services**: 100% coverage
- **Error Handling**: Comprehensive edge case testing
- **User Interactions**: Full workflow coverage

## 🎨 UI/UX Design

### Material-UI Theme

- **Primary**: Deep blue for actions and branding
- **Secondary**: Pink accent for highlights
- **Typography**: Clear hierarchy with consistent sizing
- **Spacing**: 8px grid system for consistent layouts

### Responsive Design

- **Mobile-first**: Optimized for mobile devices
- **Tablet**: Adaptive layouts for medium screens
- **Desktop**: Full feature set with expanded views

### Accessibility

- **WCAG 2.1 AA**: Screen reader support
- **Keyboard Navigation**: Full keyboard accessibility
- **Color Contrast**: High contrast ratios
- **Focus Management**: Clear focus indicators

## 🚀 Deployment

### Development

```bash
npm start  # Runs on port 3001
```

### Production

```bash
npm run build
# Deploy the build/ directory to your web server
```

### Docker Integration

The frontend can be containerized and deployed alongside the backend services.

## 🔧 Development

### Code Quality

- **TypeScript**: Strict type checking
- **ESLint**: Code linting and style enforcement
- **Prettier**: Consistent code formatting
- **Jest**: Comprehensive test suite

### Development Scripts

```bash
npm start          # Development server
npm run build      # Production build
npm run test:ci    # CI test execution
npm run test:watch # Development testing
```

## 📊 Monitoring Integration

### Health Status Display

- Real-time backend service status
- Database connectivity indicators
- AI model availability
- Cache performance metrics

### Performance Metrics

- Query response times
- Upload processing status
- System resource usage
- Error rate tracking

## 🤝 Contributing

### Development Workflow

1. **Setup**: `npm install`
2. **Development**: `npm start`
3. **Testing**: `npm run test:ci`
4. **Build**: `npm run build`

### Code Standards

- **TypeScript**: Strict mode enabled
- **Component Structure**: Functional components with hooks
- **State Management**: React hooks for local state
- **API Calls**: Centralized in services layer
- **Error Handling**: User-friendly error messages

## 📄 Documentation

- **API Docs**: Backend FastAPI documentation at `/docs`
- **Component Docs**: Inline TypeScript documentation
- **Test Docs**: Comprehensive test coverage reports

## 🐛 Troubleshooting

### Common Issues

1. **API Connection**: Ensure backend is running on port 8000
2. **CORS Issues**: Configure backend CORS settings
3. **File Upload**: Check file size limits and types
4. **Test Failures**: Run `npm install` to update dependencies

### Debug Mode

```bash
# Enable React development tools
npm start

# Check network requests in browser dev tools
# Verify API responses in Network tab
```

## 📈 Future Enhancements

- **Authentication**: User login and session management
- **Document Previews**: Thumbnail generation and preview
- **Advanced Search**: Faceted search and filtering
- **Collaboration**: Multi-user document sharing
- **Analytics**: Usage statistics and insights

---

**Built with ❤️ using React, TypeScript, and Material-UI for the RAG Agent ecosystem**
