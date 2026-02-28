// src/pages/Login.js
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../AuthContext';
import { login } from '../api';
import styles from './Login.module.css';

export default function Login() {
  const [email, setEmail]     = useState('demo@zenith.com');
  const [password, setPassword] = useState('demo123');
  const [error, setError]     = useState('');
  const [loading, setLoading] = useState(false);
  const { signin }  = useAuth();
  const navigate    = useNavigate();

  const handleSubmit = async e => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const { data } = await login(email, password);
      signin(data.token, { name: data.name, role: data.role, email: data.email });
      navigate('/');
    } catch {
      setError('Invalid email or password. Try demo@zenith.com / demo123');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.root}>
      {/* Decorative background */}
      <div className={styles.bg}>
        <div className={styles.orb1} />
        <div className={styles.orb2} />
        <div className={styles.grid} />
      </div>

      <div className={styles.split}>
        {/* Left panel */}
        <motion.div className={styles.left}
          initial={{ opacity:0, x:-40 }} animate={{ opacity:1, x:0 }}
          transition={{ duration:.7, ease:[.4,0,.2,1] }}>
          <div className={styles.brand}>
            <div className={styles.logo}>Z</div>
            <span>ZENITH</span>
          </div>
          <div className={styles.hero}>
            <h1>Credit Intelligence<br /><em>Reimagined</em></h1>
            <p>Score every applicant fairly — including those without a credit history — using 80 alternative financial signals trained on 1.8 million real loan records.</p>
          </div>
          <div className={styles.stats}>
            {[['1.8M','Training Records'],['80','AI Features'],['4','Global Datasets'],['AUC 0.80+','Model Accuracy']].map(([v,l])=>(
              <div key={l} className={styles.stat}>
                <span className={styles.statVal}>{v}</span>
                <span className={styles.statLbl}>{l}</span>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Right panel - form */}
        <motion.div className={styles.right}
          initial={{ opacity:0, x:40 }} animate={{ opacity:1, x:0 }}
          transition={{ duration:.7, ease:[.4,0,.2,1], delay:.1 }}>
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <h2>Welcome back</h2>
              <p>Sign in to your account</p>
            </div>

            <form onSubmit={handleSubmit} className={styles.form}>
              <div className={styles.field}>
                <label>Email address</label>
                <input type="email" value={email}
                  onChange={e=>setEmail(e.target.value)}
                  placeholder="you@zenith.com" required />
              </div>
              <div className={styles.field}>
                <label>Password</label>
                <input type="password" value={password}
                  onChange={e=>setPassword(e.target.value)}
                  placeholder="••••••••" required />
              </div>

              {error && <div className={styles.error}>{error}</div>}

              <button type="submit" className={styles.btn} disabled={loading}>
                {loading ? <span className={styles.spinner}/> : 'Sign in →'}
              </button>
            </form>

            <div className={styles.demo}>
              <p>Demo credentials</p>
              <div className={styles.demoRow}>
                <code>demo@zenith.com</code><span>/</span><code>demo123</code>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
