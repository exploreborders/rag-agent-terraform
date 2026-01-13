import React from 'react';
import { render, screen } from '../../__tests__/test-utils';
import { QueryResults } from '../QueryResults';

describe('QueryResults', () => {
  it('renders no results message when no result provided', () => {
    render(<QueryResults result={null} error={null} />);
    expect(screen.getByText('No Results Yet')).toBeInTheDocument();
  });

  it('renders error message when error provided', () => {
    const errorMessage = 'Test error';
    render(<QueryResults result={null} error={errorMessage} />);
    expect(screen.getByText('Query Error')).toBeInTheDocument();
    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('renders query results correctly', () => {
    const mockResult = {
      answer: 'This is a test answer',
      sources: [
        {
          id: 'source-1',
          content: 'Source content',
          score: 0.95,
          metadata: { filename: 'test.pdf' },
        },
      ],
      confidence: 0.9,
      processing_time: 1.2,
    };

    render(<QueryResults result={mockResult} error={null} />);
    expect(screen.getByText('Answer Found')).toBeInTheDocument();
    expect(screen.getByText('This is a test answer')).toBeInTheDocument();
    expect(screen.getByText('Source References (1)')).toBeInTheDocument();
  });
});