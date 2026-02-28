// src/api.js
import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

api.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token');
  if (token) cfg.headers.Authorization = `Bearer ${token}`;
  return cfg;
});

api.interceptors.response.use(
  r => r,
  err => {
    if (err.response?.status === 401) {
      localStorage.clear();
      window.location.href = '/login';
    }
    return Promise.reject(err);
  }
);

export const login  = (email, password) => api.post('/auth/login', { email, password });
export const getMe  = ()                 => api.get('/auth/me');
export const score  = (data)             => api.post('/score', data);
export const health = ()                 => api.get('/health');

export default api;
