// src/pages/Home.js
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../AuthContext';
import Navbar from '../components/Navbar';
import { ArrowRight, Shield, Zap, Users, TrendingUp } from 'lucide-react';
import styles from './Home.module.css';

const fade = (delay=0) => ({
  initial:{opacity:0,y:24},
  animate:{opacity:1,y:0},
  transition:{duration:.55,ease:[.4,0,.2,1],delay}
});

export default function Home() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const features = [
    { icon:<Shield size={22}/>, color:'emerald',
      title:'Fair for Everyone',
      desc:'Scores new users with zero loan history using savings behaviour, employment stability, and bank account health.' },
    { icon:<Zap size={22}/>, color:'gold',
      title:'Under 300ms',
      desc:'LightGBM inference in under 10ms. Full SHAP explanation generated in real time for every applicant.' },
    { icon:<Users size={22}/>, color:'sapphire',
      title:'1.8M Training Records',
      desc:'Trained on Home Credit, Give Me Some Credit, Lending Club, and PKDD Czech across 80 engineered features.' },
    { icon:<TrendingUp size={22}/>, color:'ruby',
      title:'Explainable by Design',
      desc:'Every score includes the top factors that helped and hurt it — meeting regulatory explainability requirements.' },
  ];

  return (
    <div className={styles.root}>
      <Navbar />

      <main className={styles.main}>
        {/* Hero */}
        <section className={styles.hero}>
          <motion.div {...fade(0)} className={styles.badge}>
            Alternative Credit Scoring · Built with LightGBM + SHAP
          </motion.div>
          <motion.h1 {...fade(.08)}>
            Score Every Applicant<br /><em>Intelligently</em>
          </motion.h1>
          <motion.p {...fade(.16)} className={styles.sub}>
            Welcome back, <strong>{user?.name}</strong>. Your AI-powered credit scoring platform
            is ready. Score applicants in seconds — including those with no prior loan history.
          </motion.p>
          <motion.div {...fade(.22)} className={styles.actions}>
            <button className={styles.primary} onClick={()=>navigate('/calculate')}>
              Calculate Score <ArrowRight size={18}/>
            </button>
          </motion.div>
        </section>

        {/* Stats bar */}
        <motion.div {...fade(.28)} className={styles.statsBar}>
          {[['1,806,325','Training rows'],['80','AI features'],['16.05%','Base default rate'],['AUC 0.80+','Target accuracy'],['< 300ms','Scoring time']].map(([v,l])=>(
            <div key={l} className={styles.statItem}>
              <span className={styles.statVal}>{v}</span>
              <span className={styles.statLbl}>{l}</span>
            </div>
          ))}
        </motion.div>

        {/* Feature cards */}
        <section className={styles.features}>
          {features.map((f,i)=>(
            <motion.div key={f.title} {...fade(.1+i*.06)} className={`${styles.featureCard} ${styles[f.color]}`}>
              <div className={styles.featureIcon}>{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </motion.div>
          ))}
        </section>

        {/* CTA */}
        <motion.div {...fade(.4)} className={styles.cta}>
          <div className={styles.ctaInner}>
            <div>
              <h2>Ready to score an applicant?</h2>
              <p>Fill in financial and personal details — the model returns a 300–900 score with full SHAP explanation in seconds.</p>
            </div>
            <button className={styles.ctaBtn} onClick={()=>navigate('/calculate')}>
              Start Scoring <ArrowRight size={18}/>
            </button>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
