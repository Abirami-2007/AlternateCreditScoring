// src/AuthContext.js
import { createContext, useContext, useState, useEffect } from 'react';
import { getMe } from './api';

const AuthCtx = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser]     = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      getMe()
        .then(r => setUser(r.data))
        .catch(() => localStorage.clear())
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const signin = (token, userData) => {
    localStorage.setItem('token', token);
    setUser(userData);
  };

  const signout = () => {
    localStorage.clear();
    setUser(null);
  };

  return (
    <AuthCtx.Provider value={{ user, loading, signin, signout }}>
      {children}
    </AuthCtx.Provider>
  );
}

export const useAuth = () => useContext(AuthCtx);
