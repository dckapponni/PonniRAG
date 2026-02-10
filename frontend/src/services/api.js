import axios from 'axios';

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes timeout for LLM queries
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.config.url} - ${response.status}`);
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    
    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      if (status === 404) {
        throw new Error(data.detail || 'Resource not found');
      } else if (status === 500) {
        throw new Error(data.detail || 'Server error occurred');
      } else if (status === 422) {
        throw new Error('Validation error: ' + JSON.stringify(data.detail));
      }
    } else if (error.request) {
      // Request made but no response
      throw new Error('No response from server. Please check if the API is running.');
    } else {
      // Error in request setup
      throw new Error('Error setting up request: ' + error.message);
    }
    
    return Promise.reject(error);
  }
);

/**
 * Check API and database health
 * @returns {Promise<{status: string, database: object, api: string}>}
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

/**
 * Get root API info
 * @returns {Promise<{name: string, version: string, description: string}>}
 */
export const getApiInfo = async () => {
  const response = await api.get('/');
  return response.data;
};

/**
 * Ask a question and get AI-generated answer with sources
 * @param {string} question - The question to ask
 * @param {boolean} useLLM - Whether to use LLM for answer generation
 * @returns {Promise<{answer: string, sources: Array, query_type: string, error: object}>}
 */
export const askQuestion = async (question, useLLM = true) => {
  const response = await api.post('/api/ask', {
    question,
    use_llm: useLLM,
  });
  return response.data;
};

/**
 * Search the archive (GET alternative to askQuestion)
 * @param {string} query - Search query
 * @param {boolean} useLLM - Whether to use LLM
 * @returns {Promise<{answer: string, sources: Array}>}
 */
export const search = async (query, useLLM = false) => {
  const response = await api.get('/api/search', {
    params: { q: query, use_llm: useLLM },
  });
  return response.data;
};

/**
 * Get list of all authors with article counts
 * @returns {Promise<{success: boolean, total_authors: number, total_articles: number, authors: Array}>}
 */
export const getAuthors = async () => {
  const response = await api.get('/api/authors');
  return response.data;
};

/**
 * Get all articles by a specific author
 * @param {string} authorName - Name of the author (supports Tamil)
 * @returns {Promise<{success: boolean, author: string, count: number, articles: Array}>}
 */
export const getAuthorArticles = async (authorName) => {
  const response = await api.get(`/api/authors/${encodeURIComponent(authorName)}/articles`);
  return response.data;
};


/**
 * Find articles about a specific topic
 * @param {string} topic - Topic keyword to search
 * @returns {Promise<{success: boolean, topic: string, count: number, articles: Array}>}
 */
export const searchByTopic = async (topic) => {
  const response = await api.get('/api/topics/search', {
    params: { topic },
  });
  return response.data;
};

/**
 * Get statistics about all issues
 * @returns {Promise<{success: boolean, count: number, total_articles: number, issues: Array}>}
 */
export const getIssueStats = async () => {
  const response = await api.get('/api/issues/stats');
  return response.data;
};

/**
 * Get list of all volumes
 * @returns {Promise<Array<{id: number, year: string, issue_count: number}>>}
 */
export const getVolumes = async () => {
  const response = await api.get('/api/library/volumes');
  return response.data;
};

/**
 * Get all issues for a specific volume
 * @param {number} volumeId - Volume number (1-8)
 * @returns {Promise<Array<{issue_number: number, has_pdf: boolean, pdf_url: string}>>}
 */
export const getVolumeIssues = async (volumeId) => {
  const response = await api.get(`/api/library/volumes/${volumeId}/issues`);
  return response.data;
};

/**
 * Get PDF link for a specific issue
 * @param {number} volumeId - Volume number (1-8)
 * @param {number} issueId - Issue number
 * @returns {Promise<{volume_id: number, issue_id: number, pdf_url: string, embed_url: string, found: boolean}>}
 */
export const getPDFLink = async (volumeId, issueId) => {
  const response = await api.get(`/api/library/volumes/${volumeId}/issues/${issueId}/pdf`);
  return response.data;
};

/**
 * Test API connection
 * @returns {Promise<boolean>}
 */
export const testConnection = async () => {
  try {
    const health = await checkHealth();
    return health.status === 'healthy';
  } catch (error) {
    console.error('Connection test failed:', error);
    return false;
  }
};

/**
 * Format author name for URL (handles Tamil text)
 * @param {string} authorName
 * @returns {string}
 */
export const formatAuthorName = (authorName) => {
  return encodeURIComponent(authorName);
};

/**
 * Check if API is reachable
 * @returns {Promise<{reachable: boolean, message: string}>}
 */
export const checkApiReachability = async () => {
  try {
    const response = await api.get('/');
    return {
      reachable: true,
      message: 'API is reachable',
      data: response.data
    };
  } catch (error) {
    return {
      reachable: false,
      message: error.message || 'API is not reachable'
    };
  }
};

export default api;