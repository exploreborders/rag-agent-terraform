module.exports = {
  extends: ['react-app', 'react-app/jest'],
  rules: {
    // Add custom rules for RAG app
    'react/react-in-jsx-scope': 'off', // Not needed in React 17+
    'no-unused-vars': ['error', { 'argsIgnorePattern': '^_' }],
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
};