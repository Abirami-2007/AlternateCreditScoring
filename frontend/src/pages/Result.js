// src/pages/Result.js
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import { ArrowLeft, RefreshCw, TrendingUp, TrendingDown, Clock, Zap, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';
import styles from './Result.module.css';

const BAND_COLORS = {
  'Excellent': '#059669', // Muted Emerald
  'Very Good': '#047857',
  'Good': '#D97706', // Muted Amber
  'Fair': '#C2410C', // Muted Orange
  'Poor': '#B91C1C', // Muted Red
  'Very Poor': '#991B1B',
};

const RECOMMEND_STYLE = {
  'Approve': { bg: 'rgba(5, 150, 105, 0.08)', color: '#059669', label: 'Recommended for Approval', icon: CheckCircle2 },
  'Conditional': { bg: 'rgba(217, 119, 6, 0.08)', color: '#D97706', label: 'Conditional Approval', icon: AlertTriangle },
  'Reject': { bg: 'rgba(185, 28, 28, 0.08)', color: '#B91C1C', label: 'High Risk — Manual Review', icon: XCircle },
};

function GaugeCanvas({ score, color }) {
  const ref = useRef();
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H * 0.85, r = H * 0.7;

    ctx.clearRect(0, 0, W, H);

    // Background Track
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.lineWidth = 20;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineCap = 'round';
    ctx.stroke();

    // Active Track (Glow effect)
    const norm = Math.max(0, Math.min(1, (score - 300) / 600));
    const activeAngle = Math.PI + norm * Math.PI;

    if (norm > 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, activeAngle);
      ctx.lineWidth = 20;

      // Gradient for active track
      const grad = ctx.createLinearGradient(0, 0, W, 0);
      grad.addColorStop(0, '#EF4444');
      if (norm > 0.5) grad.addColorStop(0.5, '#F59E0B');
      if (norm > 0.8) grad.addColorStop(1, '#10B981');

      ctx.strokeStyle = grad;
      ctx.lineCap = 'round';
      ctx.shadowColor = color;
      ctx.shadowBlur = 15;
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
    }

    // Needle
    const nx = cx + (r - 2) * Math.cos(activeAngle);
    const ny = cy + (r - 2) * Math.sin(activeAngle);

    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(nx, ny);
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineCap = 'round';
    ctx.stroke();

    // Needle Center Dot
    ctx.beginPath();
    ctx.arc(cx, cy, 10, 0, 2 * Math.PI);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#0F172A';
    ctx.fill();

    // Labels
    ctx.font = '600 12px Inter, sans-serif';
    ctx.fillStyle = '#94A3B8';
    ctx.textAlign = 'center';
    [['300', Math.PI], ['600', 1.5 * Math.PI], ['900', 2 * Math.PI]].forEach(([t, a]) => {
      ctx.fillText(t, cx + (r + 25) * Math.cos(a), cy + (r + 25) * Math.sin(a) + 4);
    });
  }, [score, color]);

  return <canvas ref={ref} width={300} height={180} className={styles.gauge} />;
}

export default function Result() {
  const { state } = useLocation();
  const navigate = useNavigate();

  if (!state?.result) {
    navigate('/calculate');
    return null;
  }

  const { result, isNew } = state;
  const color = BAND_COLORS[result.band] || '#94A3B8';
  const rec = RECOMMEND_STYLE[result.recommendation] || RECOMMEND_STYLE['Reject'];
  const RecIcon = rec.icon;

  const fade = (delay = 0) => ({
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1], delay }
  });

  return (
    <div className={styles.root}>
      <Navbar />

      {/* Ambient background glows tailored to the result */}
      <div className={styles.bgOrbs}>
        <div className={styles.orb1} style={{ background: `radial-gradient(circle, ${color}20 0%, transparent 60%)` }} />
        <div className={styles.orb2} />
      </div>

      <main className={styles.main}>

        {/* Back */}
        <motion.button {...fade(0)} className={styles.back} onClick={() => navigate('/calculate')}>
          <ArrowLeft size={18} /> <span>Back to Assessment</span>
        </motion.button>

        {/* Score Hero */}
        <motion.div {...fade(0.1)} className={`glass-panel ${styles.hero}`}>
          <div className={styles.scoreLeft}>
            <div className={styles.scoreTag}>
              {isNew ? 'New User Assessment' : 'Full Credit Assessment'}
            </div>

            <motion.div
              className={styles.scoreNum}
              style={{ color }}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.3, type: "spring", stiffness: 200 }}
            >
              {result.score}
            </motion.div>

            <div className={styles.scoreBand} style={{ background: `${color}15`, color, border: `1px solid ${color}30` }}>
              {result.band}
            </div>

            <div className={styles.defaultProb}>
              Estimated Default Probability: <strong style={{ color: '#F8FAFC' }}>{result.prob_default}%</strong>
            </div>

            <div className={styles.rec} style={{ background: rec.bg, color: rec.color, border: `1px solid ${rec.color}30` }}>
              <RecIcon size={18} /> {rec.label}
            </div>
          </div>

          <div className={styles.scoreRight}>
            <GaugeCanvas score={result.score} color={color} />
            <div className={styles.scaleTable}>
              {[['750–900', 'Excellent', '#10B981'], ['700–749', 'Very Good', '#059669'],
              ['650–699', 'Good', '#F59E0B'], ['600–649', 'Fair', '#F97316'],
              ['550–599', 'Poor', '#EF4444'], ['300–549', 'Very Poor', '#DC2626']].map(([r, b, c]) => (
                <div key={b} className={styles.scaleRow}
                  style={result.band === b ? { background: `${c}20`, fontWeight: 600, borderLeft: `2px solid ${c}` } : { borderLeft: '2px solid transparent' }}>
                  <span className={styles.scaleDot} style={{ background: c, boxShadow: result.band === b ? `0 0 8px ${c}` : 'none' }} />
                  <span className={styles.scaleRange}>{r}</span>
                  <span style={{ color: result.band === b ? '#F8FAFC' : 'var(--text-secondary)' }}>{b}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Timing */}
        <motion.div {...fade(0.15)} className={styles.timing}>
          {[
            [<Clock size={16} />, 'Total Time', `${result.timing.total_ms} ms`],
            [<Zap size={16} />, 'Feature Build', `${result.timing.feature_ms} ms`],
            [<Zap size={16} />, 'Model Predict', `${result.timing.model_ms} ms`],
            [<Zap size={16} />, 'SHAP Explain', `${result.timing.shap_ms} ms`],
          ].map(([icon, lbl, val]) => (
            <div key={lbl} className={`glass-panel ${styles.timingItem}`}>
              <div className={styles.timingIconWrap}>
                <span className={styles.timingIcon}>{icon}</span>
                <span className={styles.timingLbl}>{lbl}</span>
              </div>
              <span className={styles.timingVal}>{val}</span>
            </div>
          ))}
        </motion.div>

        {/* Key Metrics */}
        <motion.div {...fade(0.2)} className={`glass-panel ${styles.metrics}`}>
          <h2>Key Financial Metrics</h2>
          <div className={styles.metricsGrid}>
            {[
              ['Credit-to-Income', `${result.key_metrics.credit_to_income}x`, result.key_metrics.credit_to_income < 3],
              ['Payment Burden', `${result.key_metrics.payment_burden}%`, result.key_metrics.payment_burden < 35],
              ['CC Utilization', `${result.key_metrics.cc_utilization}%`, result.key_metrics.cc_utilization < 30],
              ['Late Pay Rate', `${result.key_metrics.late_payment_rate}%`, result.key_metrics.late_payment_rate < 5],
            ].map(([l, v, good]) => (
              <div key={l} className={styles.metricCard}>
                <div className={styles.metricIcon}>
                  {good ? <CheckCircle2 size={20} color="#059669" /> : <AlertTriangle size={20} color="#B91C1C" />}
                </div>
                <div className={styles.metricVal}>{v}</div>
                <div className={styles.metricLbl}>{l}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* SHAP Reasons */}
        <motion.div {...fade(0.25)} className={styles.reasons}>
          <div className={`glass-panel ${styles.reasonCol}`}>
            <h3 className={styles.reasonTitle} style={{ color: '#34D399' }}>
              <TrendingUp size={20} /> Factors Helping Your Score
            </h3>
            <div className={styles.reasonList}>
              {result.positive_factors.length ? result.positive_factors.map((r, i) => (
                <div key={i} className={styles.reasonPos}>
                  <CheckCircle2 size={16} /> <span>{r} is working in your favour</span>
                </div>
              )) : <div className={styles.reasonEmpty}>No strong positive factors detected</div>}
            </div>
          </div>
          <div className={`glass-panel ${styles.reasonCol}`}>
            <h3 className={styles.reasonTitle} style={{ color: '#F87171' }}>
              <TrendingDown size={20} /> Factors Hurting Your Score
            </h3>
            <div className={styles.reasonList}>
              {result.negative_factors.length ? result.negative_factors.map((r, i) => (
                <div key={i} className={styles.reasonNeg}>
                  <AlertTriangle size={16} /> <span>{r} is pulling your score down</span>
                </div>
              )) : <div className={styles.reasonGood}>
                <CheckCircle2 size={16} /> <span>No significant negative factors!</span>
              </div>}
            </div>
          </div>
        </motion.div>

        {/* Tips */}
        {result.improvement_tips.length > 0 && (
          <motion.div {...fade(0.3)} className={`glass-panel ${styles.tips}`}>
            <h2>AI Recommendations for Improvement</h2>
            <ul className={styles.tipsList}>
              {result.improvement_tips.map((t, i) => (
                <li key={i} className={styles.tip}>
                  <div className={styles.tipDot} />
                  {t}
                </li>
              ))}
            </ul>
          </motion.div>
        )}

        {/* Actions */}
        <motion.div {...fade(0.35)} className={styles.actions}>
          <button className="btn-secondary" onClick={() => navigate('/calculate')}>
            <RefreshCw size={18} /> New Assessment
          </button>
          <button className="btn-primary" onClick={() => navigate('/')}>
            Back to Home
          </button>
        </motion.div>

      </main>
    </div>
  );
}
