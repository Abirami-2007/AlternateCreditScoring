// src/App.js
import './index.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './AuthContext';
import Login    from './pages/Login';
import Home     from './pages/Home';
import Calculate from './pages/Calculate';
import Result   from './pages/Result';
import Batch    from './pages/Batch';

function Guard({ children }) {
  const { user, loading } = useAuth();
  if (loading) return (
    <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',background:'#f7f7fc'}}>
      <div style={{width:32,height:32,border:'3px solid #e8e8f0',borderTopColor:'#c9a84c',borderRadius:'50%',animation:'spin .7s linear infinite'}}/>
    </div>
  );
  return user ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login"     element={<Login />} />
          <Route path="/"          element={<Guard><Home /></Guard>} />
          <Route path="/calculate" element={<Guard><Calculate /></Guard>} />
          <Route path="/result"    element={<Guard><Result /></Guard>} />
          <Route path="/batch"     element={<Guard><Batch /></Guard>} />
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
