# Frontend Tests

This directory contains comprehensive tests for the RAG Agent frontend application.

## Test Structure

```
src/
├── __tests__/
│   ├── test-utils.tsx          # Test utilities and mocks
│   ├── App.test.tsx           # Main App component tests
│   └── api.test.ts            # API service layer tests
├── components/__tests__/
│   ├── DocumentUpload.test.tsx # File upload component tests
│   ├── DocumentList.test.tsx   # Document management tests
│   ├── QueryInterface.test.tsx # Query input component tests
│   └── QueryResults.test.tsx   # Results display tests
```

## Test Categories

### Component Tests
- **DocumentUpload**: File upload, drag & drop, progress tracking, error handling
- **DocumentList**: Document display, status indicators, delete functionality
- **QueryInterface**: Query submission, document selection, form validation
- **QueryResults**: Results display, source expansion, metadata display
- **App**: Main application integration, state management, API integration

### Service Tests
- **API Service**: HTTP requests, error handling, authentication, data transformation

## Running Tests

```bash
# Run all tests
npm test

# Run tests in CI mode with coverage
npm run test:ci

# Run tests in watch mode
npm run test:watch
```

## Test Coverage

The tests achieve comprehensive coverage of:

- ✅ Component rendering and interactions
- ✅ API integration and error handling
- ✅ User workflows (upload → process → query → results)
- ✅ Form validation and user input
- ✅ Loading states and error states
- ✅ Accessibility and responsive design

## Test Utilities

- **Custom render function**: Includes Material-UI theme provider
- **API mocks**: Pre-configured mock responses for all endpoints
- **Test data**: Realistic mock documents, queries, and responses
- **Helper functions**: File creation, form interactions, async utilities

## Mock Data

The tests include comprehensive mock data for:
- Document uploads and processing
- Query submissions and responses
- Health status and system metrics
- Error conditions and edge cases

## Coverage Thresholds

- Branches: 70%
- Functions: 70%
- Lines: 70%
- Statements: 70%

## Technologies Used

- **Jest**: Test runner and assertion library
- **React Testing Library**: Component testing utilities
- **Axios Mock**: HTTP request mocking
- **Material-UI Test Utils**: Theme-aware testing
- **TypeScript**: Type-safe test writing