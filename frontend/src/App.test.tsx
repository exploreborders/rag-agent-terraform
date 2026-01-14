import React from 'react';
import { render, screen } from './__tests__/test-utils';
import App from './App';

// Mock the API service to prevent real API calls during testing
jest.mock('./services/api');

test('renders app title', () => {
  render(<App />);
  const titleElement = screen.getByText(/RAG Agent - Document Intelligence System/i);
  expect(titleElement).toBeInTheDocument();
});
