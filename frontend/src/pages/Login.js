// src/pages/Login.js
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../AuthContext';
import { login } from '../api';
import { ArrowRight, Lock, Mail, AlertCircle } from 'lucide-react';
import styles from './Login.module.css';

export default function Login() {
  const [email, setEmail] = useState('demo@zenith.com');
  const [password, setPassword] = useState('demo123');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [focusedField, setFocusedField] = useState(null);

  const { signin } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async e => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const { data } = await login(email, password);
      // Simulate slight network delay for premium feel of the spinner
      await new Promise(r => setTimeout(r, 600));
      signin(data.token, { name: data.name, role: data.role, email: data.email });
      navigate('/');
    } catch {
      setError('Invalid email or password. Try demo@zenith.com / demo123');
      setLoading(false);
    }
  };

  return (
    <div className={styles.root}>
      {/* Decorative premium background */}
      <div className={styles.bg}>
        <div className={styles.orb1} />
        <div className={styles.orb2} />
        <div className={styles.grid} />
      </div>

      <div className={styles.split}>
        {/* Left panel - Branding */}
        <motion.div className={styles.left}
          initial={{ opacity: 0, x: -40 }} animate={{ opacity: 1, x: 0 }}
          transition={{ duration: .8, ease: [.22, 1, .36, 1] }}>

          <div className={styles.brand}>
            <div className={styles.logo}>F</div>
            <span>FINOVA</span>
          </div>

          <div className={styles.hero}>
            <h1>Alternate<br /><span className="text-gradient-primary">Credit Scoring</span></h1>
            <p>Score every applicant fairly — including those without a credit history — using 80+ alternative financial signals trained on millions of real loan records.</p>
          </div>

          <div className={styles.stats}>
            {[
              ['1.8M+', 'Training Records'],
              ['80+', ' Features'],
              ['0.81', 'Model AUC']
            ].map(([v, l], i) => (
              <motion.div key={l} className={styles.stat}
                initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + (i * 0.1), duration: 0.5 }}>
                <span className={styles.statVal}>{v}</span>
                <span className={styles.statLbl}>{l}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Right panel - Auth Form */}
        <div className={styles.right}>
          <motion.div className={`glass-panel ${styles.card}`}
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: .6, delay: .2, ease: [.22, 1, .36, 1] }}>

            <div className={styles.cardHeader}>
              <h2>Welcome back</h2>
              <p>Sign in to your encrypted workspace</p>
            </div>

            <form onSubmit={handleSubmit} className={styles.form}>

              <div className={`${styles.field} ${focusedField === 'email' ? styles.fieldFocused : ''}`}>
                <label>Email Address</label>
                <div className={styles.inputWrapper}>
                  <Mail size={18} className={styles.inputIcon} />
                  <input type="email" value={email}
                    onChange={e => setEmail(e.target.value)}
                    onFocus={() => setFocusedField('email')}
                    onBlur={() => setFocusedField(null)}
                    placeholder="you@zenith.com" required />
                </div>
              </div>

              <div className={`${styles.field} ${focusedField === 'password' ? styles.fieldFocused : ''}`}>
                <label>Password</label>
                <div className={styles.inputWrapper}>
                  <Lock size={18} className={styles.inputIcon} />
                  <input type="password" value={password}
                    onChange={e => setPassword(e.target.value)}
                    onFocus={() => setFocusedField('password')}
                    onBlur={() => setFocusedField(null)}
                    placeholder="••••••••" required />
                </div>
              </div>

              <AnimatePresence>
                {error && (
                  <motion.div className={styles.error}
                    initial={{ opacity: 0, height: 0, marginTop: 0 }}
                    animate={{ opacity: 1, height: 'auto', marginTop: '0.5rem' }}
                    exit={{ opacity: 0, height: 0, marginTop: 0 }}>
                    <AlertCircle size={16} />
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <button type="submit" className={`btn-primary ${styles.btn}`} disabled={loading}>
                {loading ? <span className={styles.spinner} /> : (
                  <>Secure Sign In <ArrowRight size={18} /></>
                )}
              </button>
            </form>

            {/* <div className={styles.demo}>
              <p>Demo credentials</p>
              <div className={styles.demoRow}>
                <code>demo@zenith.com</code>
                <span className={styles.demoDiv}></span>
                <code>demo123</code>
              </div>
            </div>  */}

          </motion.div>
        </div>
      </div>
    </div>
  );
}
