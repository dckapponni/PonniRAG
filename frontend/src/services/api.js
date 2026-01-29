import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Ask a question (AI Search)
export const askQuestion = async (question, topK = 10, useLLM = true) => {
  const response = await api.post('/api/ask', {
    question,
    top_k: topK,
    use_llm: useLLM,
  });
  return response.data;
};

// Search (GET alternative)
export const search = async (query, topK = 10, useLLM = false) => {
  const response = await api.get('/api/search', {
    params: { q: query, top_k: topK, use_llm: useLLM },
  });
  return response.data;
};

// Get all authors
export const getAuthors = async () => {
  const response = await api.get('/api/authors');
  return response.data;
};

// Get articles by author
export const getAuthorArticles = async (authorName) => {
  const response = await api.get(`/api/authors/${encodeURIComponent(authorName)}/articles`);
  return response.data;
};

// Search by topic
export const searchByTopic = async (topic) => {
  const response = await api.get('/api/topics/search', {
    params: { topic },
  });
  return response.data;
};

// Get issue statistics
export const getIssueStats = async () => {
  const response = await api.get('/api/issues/stats');
  return response.data;
};

// Get all volumes
export const getVolumes = async () => {
  const response = await api.get('/api/library/volumes');
  return response.data;
};

// Get issues for a volume
export const getVolumeIssues = async (volumeId) => {
  const response = await api.get(`/api/library/volumes/${volumeId}/issues`);
  return response.data;
};

// Get PDF link for an issue
export const getPDFLink = async (volumeId, issueId) => {
  const response = await api.get(`/api/library/volumes/${volumeId}/issues/${issueId}/pdf`);
  return response.data;
};

export default api;
