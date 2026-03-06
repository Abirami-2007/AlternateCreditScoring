// src/pages/Home.js
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../AuthContext';
import Navbar from '../components/Navbar';
import { ArrowRight, Shield, Zap, Database, TrendingUp } from 'lucide-react';
import styles from './Home.module.css';

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 30 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-50px" },
  transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1], delay }
});

export default function Home() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const features = [
    {
      icon: <Shield size={24} />,
      title: 'Fair Evaluation',
      desc: 'Scores new users with zero loan history using savings behaviour, employment stability, and bank account health.',
      glowColor: 'var(--success)'
    },
    {
      icon: <Zap size={24} />,
      title: 'Sub-300ms Inference',
      desc: 'Blazing fast LightGBM inference. Full SHAP explanation generated in real time for every single applicant.',
      glowColor: 'var(--warning)'
    },
    {
      icon: <Database size={24} />,
      title: '1.8M+ Training Records',
      desc: 'Trained on major datasets across 80+ meticulously engineered features for high reliability.',
      glowColor: 'var(--primary)'
    },
    {
      icon: <TrendingUp size={24} />,
      title: 'Fully Explainable',
      desc: 'Every score includes the top factors that helped and hurt it — meeting modern regulatory requirements.',
      glowColor: 'var(--secondary)'
    },
  ];

  return (
    <div className={styles.root}>
      <Navbar />

      <main className={styles.main}>
        {/* Hero Section */}
        <section className={styles.hero}>
          <motion.div {...fadeUp(0.1)} className={styles.badge}>
            <span className={styles.badgeGlow}></span>
            <span className={styles.badgeText}>Powered by LightGBM & SHAP AI</span>
          </motion.div>

          <motion.h1 {...fadeUp(0.2)}>
            Score Every Applicant <br />
            <span className="text-gradient-primary"></span>
          </motion.h1>

          <motion.p {...fadeUp(0.3)} className={styles.sub}>
            Welcome back, <strong className={styles.highlight}>{user?.name || 'User'}</strong>. Your unified credit scoring platform is ready. Evaluate applicants in milliseconds, even those with thin files.
          </motion.p>

          <motion.div {...fadeUp(0.4)} className={styles.actions}>
            <button className="btn-primary" onClick={() => navigate('/calculate')}>
              Start Scoring <ArrowRight size={18} />
            </button>
            <button className="btn-secondary" onClick={() => navigate('/batch')}>
              Batch Processing
            </button>
          </motion.div>
        </section>

        {/* Floating Stats Bar */}
        <motion.div
          className={`glass-panel ${styles.statsBar}`}
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.5, ease: "easeOut" }}
        >
          {[
            ['1.8M+', 'Training Rows'],
            ['80+', 'Features'],
            ['16.1%', 'Avg Default Rate'],
            ['0.81', 'AUC Score']
          ].map(([val, label], idx) => (
            <div key={label} className={styles.statItem}>
              <motion.span
                className={styles.statVal}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 + (idx * 0.1) }}
              >
                {val}
              </motion.span>
              <span className={styles.statLbl}>{label}</span>
            </div>
          ))}
        </motion.div>

        {/* Feature Grid */}
        <section className={styles.features}>
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              {...fadeUp(0.2 + (i * 0.1))}
              className={`glass-panel ${styles.featureCard}`}
              whileHover={{ y: -5, scale: 1.02 }}
              style={{ '--hover-glow': f.glowColor }}
            >
              <div className={styles.featureIconWrapper} style={{ color: f.glowColor }}>
                <div className={styles.iconGlowBg} style={{ backgroundColor: f.glowColor }} />
                <div className={styles.featureIcon}>{f.icon}</div>
              </div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </motion.div>
          ))}
        </section>

        {/* Call to Action */}
        <motion.div {...fadeUp(0.3)} className={styles.ctaWrapper}>
          <div className={styles.ctaGlow}></div>
          <div className={`glass-panel ${styles.cta}`}>
            <div className={styles.ctaTop}>
              <h2>Ready to evaluate an applicant?</h2>
              <p>Enter financial boundaries, employment history, and personal details. Our model returns a precise 300–900 score instantly.</p>
            </div>
            <button className="btn-primary" onClick={() => navigate('/calculate')}>
              Initialize Calculator <ArrowRight size={18} />
            </button>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
