// API Configuration for Production and Development
export const API_BASE_URL = import.meta.env.VITE_API_URL || 
  (import.meta.env.PROD 
    ? 'https://emotion-aware-backend-xb6b.onrender.com'
    : 'http://localhost:8001');

export const WS_BASE_URL = import.meta.env.VITE_WS_URL || 
  (import.meta.env.PROD
    ? 'wss://emotion-aware-backend-xb6b.onrender.com'
    : 'ws://localhost:8001');

console.log('🌐 API Configuration:', {
  mode: import.meta.env.MODE,
  apiUrl: API_BASE_URL,
  wsUrl: WS_BASE_URL
});
