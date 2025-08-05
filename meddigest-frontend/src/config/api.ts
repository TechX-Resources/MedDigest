// API Configuration for MedDigest Frontend

const isDevelopment = process.env.NODE_ENV === 'development';

// API Base URL - use environment variable or stable domain
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 
  (isDevelopment ? 'http://localhost:8000' : 'https://meddigest.onrender.com');

// API Endpoints
export const API_ENDPOINTS = {
  newsletter: `${API_BASE_URL}/api/newsletter`,
  signup: `${API_BASE_URL}/api/signup`,
} as const;

// Helper function to get API URL
export const getApiUrl = (endpoint: keyof typeof API_ENDPOINTS): string => {
  return API_ENDPOINTS[endpoint];
};

// Log the API URL for debugging (only in development)
if (isDevelopment) {
  console.log('API Base URL:', API_BASE_URL);
} 