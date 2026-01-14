// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

// Mock react-dropzone for jsdom compatibility
jest.mock('react-dropzone', () => ({
  useDropzone: () => ({
    getRootProps: () => ({}),
    getInputProps: () => ({
      type: 'file',
      accept: '.pdf,.txt,.jpg,.png',
      multiple: true,
    }),
    isDragActive: false,
    acceptedFiles: [],
    fileRejections: [],
  }),
}));

// Mock axios globally with proper export
const mockAxiosInstance = {
  interceptors: {
    response: {
      use: jest.fn(),
    },
  },
  get: jest.fn(),
  post: jest.fn(),
  delete: jest.fn(),
};

jest.mock('axios', () => ({
  create: jest.fn(() => mockAxiosInstance),
  isAxiosError: jest.fn(() => false),
}));

// Export the mock instance for use in tests
export { mockAxiosInstance as mockedAxios };
