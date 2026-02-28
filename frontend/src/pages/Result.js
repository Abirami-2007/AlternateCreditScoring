// src/pages/Result.js
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import { ArrowLeft, RefreshCw, TrendingUp, TrendingDown, Clock, Zap } from 'lucide-react';
import styles from './Result.module.css';

const BAND_COLORS = {
  'Excellent':  '#2E7D32',
  'Very Good':  '#388E3C',
  'Good':       '#F9A825',
  'Fair':       '#F57F17',
  'Poor':       '#E64A19',
  'Very Poor':  '#B71C1C',
};

const RECOMMEND_STYLE = {
  'Approve':     { bg:'#E8F5E9', color:'#1B5E20', label:'✅ Recommended for Approval' },
  'Conditional': { bg:'#FFF8E1', color:'#E65100', label:'⚠️ Conditional Approval' },
  'Reject':      { bg:'#FFEBEE', color:'#B71C1C', label:'❌ High Risk — Manual Review' },
};

function GaugeCanvas({ score }) {
  const ref = useRef();
  useEffect(()=>{
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W/2, cy = H*0.82, r = H*0.65;
    ctx.clearRect(0,0,W,H);
    const zones = [
      [Math.PI, 1.2*Math.PI, '#B71C1C'],
      [1.2*Math.PI, 1.4*Math.PI, '#E64A19'],
      [1.4*Math.PI, 1.6*Math.PI, '#F57F17'],
      [1.6*Math.PI, 1.8*Math.PI, '#F9A825'],
      [1.8*Math.PI, 1.9*Math.PI, '#388E3C'],
      [1.9*Math.PI, 2*Math.PI,   '#2E7D32'],
    ];
    zones.forEach(([s,e,c])=>{
      ctx.beginPath(); ctx.arc(cx,cy,r,s,e);
      ctx.lineWidth=28; ctx.strokeStyle=c; ctx.stroke();
    });
    // Needle
    const norm = (score-300)/600;
    const angle = Math.PI + norm*Math.PI;
    const nx = cx + (r-14)*Math.cos(angle);
    const ny = cy + (r-14)*Math.sin(angle);
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(nx,ny);
    ctx.lineWidth=3; ctx.strokeStyle='#0a0a0f';
    ctx.lineCap='round'; ctx.stroke();
    ctx.beginPath(); ctx.arc(cx,cy,8,0,2*Math.PI);
    ctx.fillStyle='#0a0a0f'; ctx.fill();
    // Labels
    ctx.font = '11px DM Sans, sans-serif';
    ctx.fillStyle='#6b6b80'; ctx.textAlign='center';
    [['300',Math.PI],['600',1.5*Math.PI],['900',2*Math.PI]].forEach(([t,a])=>{
      ctx.fillText(t, cx+(r+22)*Math.cos(a), cy+(r+22)*Math.sin(a));
    });
  },[score]);
  return <canvas ref={ref} width={280} height={160} className={styles.gauge}/>;
}

export default function Result() {
  const { state } = useLocation();
  const navigate  = useNavigate();

  if (!state?.result) {
    navigate('/calculate'); return null;
  }

  const { result, isNew } = state;
  const color  = BAND_COLORS[result.band] || '#888';
  const rec    = RECOMMEND_STYLE[result.recommendation] || RECOMMEND_STYLE['Reject'];

  const fade = (delay=0) => ({
    initial:{opacity:0,y:20}, animate:{opacity:1,y:0},
    transition:{duration:.5, ease:[.4,0,.2,1], delay}
  });

  return (
    <div className={styles.root}>
      <Navbar />
      <main className={styles.main}>

        {/* Back */}
        <motion.button {...fade(0)} className={styles.back} onClick={()=>navigate('/calculate')}>
          <ArrowLeft size={16}/> Back to Assessment
        </motion.button>

        {/* Score Hero */}
        <motion.div {...fade(.05)} className={styles.hero}>
          <div className={styles.scoreLeft}>
            <div className={styles.scoreTag}>
              {isNew ? '🆕 New User Score' : '👤 Full Credit Score'}
            </div>
            <div className={styles.scoreNum} style={{color}}>{result.score}</div>
            <div className={styles.scoreBand} style={{background:`${color}18`,color}}>
              {result.band}
            </div>
            <div className={styles.defaultProb}>
              Default Probability: <strong>{result.prob_default}%</strong>
            </div>
            <div className={styles.rec} style={{background:rec.bg, color:rec.color}}>
              {rec.label}
            </div>
          </div>
          <div className={styles.scoreRight}>
            <GaugeCanvas score={result.score} />
            <div className={styles.scaleTable}>
              {[['750–900','Excellent','#2E7D32'],['700–749','Very Good','#388E3C'],
                ['650–699','Good','#F9A825'],['600–649','Fair','#F57F17'],
                ['550–599','Poor','#E64A19'],['300–549','Very Poor','#B71C1C']].map(([r,b,c])=>(
                <div key={b} className={styles.scaleRow}
                  style={result.band===b?{background:`${c}12`,fontWeight:700}:{}}>
                  <span className={styles.scaleDot} style={{background:c}}/>
                  <span className={styles.scaleRange}>{r}</span>
                  <span>{b}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Timing */}
        <motion.div {...fade(.1)} className={styles.timing}>
          {[
            [<Clock size={14}/>, 'Total Time',    `${result.timing.total_ms} ms`],
            [<Zap size={14}/>,   'Feature Build', `${result.timing.feature_ms} ms`],
            [<Zap size={14}/>,   'Model Predict', `${result.timing.model_ms} ms`],
            [<Zap size={14}/>,   'SHAP Explain',  `${result.timing.shap_ms} ms`],
          ].map(([icon,lbl,val])=>(
            <div key={lbl} className={styles.timingItem}>
              <span className={styles.timingIcon}>{icon}</span>
              <span className={styles.timingVal}>{val}</span>
              <span className={styles.timingLbl}>{lbl}</span>
            </div>
          ))}
        </motion.div>

        {/* Key Metrics */}
        <motion.div {...fade(.15)} className={styles.metrics}>
          <h2>Key Financial Metrics</h2>
          <div className={styles.metricsGrid}>
            {[
              ['Credit-to-Income', `${result.key_metrics.credit_to_income}x`,   result.key_metrics.credit_to_income < 3],
              ['Payment Burden',   `${result.key_metrics.payment_burden}%`,      result.key_metrics.payment_burden < 35],
              ['CC Utilization',  `${result.key_metrics.cc_utilization}%`,       result.key_metrics.cc_utilization < 30],
              ['Late Pay Rate',   `${result.key_metrics.late_payment_rate}%`,    result.key_metrics.late_payment_rate < 5],
            ].map(([l,v,good])=>(
              <div key={l} className={styles.metricCard}>
                <div className={styles.metricIcon}>{good?'🟢':'🔴'}</div>
                <div className={styles.metricVal}>{v}</div>
                <div className={styles.metricLbl}>{l}</div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* SHAP Reasons */}
        <motion.div {...fade(.2)} className={styles.reasons}>
          <div className={styles.reasonCol}>
            <h3 className={styles.reasonTitle} style={{color:'#1B5E20'}}>
              <TrendingUp size={18}/> Factors Helping Your Score
            </h3>
            {result.positive_factors.length ? result.positive_factors.map((r,i)=>(
              <div key={i} className={styles.reasonPos}>✅ {r} is working in your favour</div>
            )) : <div className={styles.reasonEmpty}>No strong positive factors detected</div>}
          </div>
          <div className={styles.reasonCol}>
            <h3 className={styles.reasonTitle} style={{color:'#B71C1C'}}>
              <TrendingDown size={18}/> Factors Hurting Your Score
            </h3>
            {result.negative_factors.length ? result.negative_factors.map((r,i)=>(
              <div key={i} className={styles.reasonNeg}>⚠️ {r} is pulling your score down</div>
            )) : <div className={styles.reasonGood}>✅ No significant negative factors!</div>}
          </div>
        </motion.div>

        {/* Tips */}
        {result.improvement_tips.length > 0 && (
          <motion.div {...fade(.25)} className={styles.tips}>
            <h2>💡 How to Improve This Score</h2>
            <ul className={styles.tipsList}>
              {result.improvement_tips.map((t,i)=>(
                <li key={i} className={styles.tip}>{t}</li>
              ))}
            </ul>
          </motion.div>
        )}

        {/* Actions */}
        <motion.div {...fade(.3)} className={styles.actions}>
          <button className={styles.secondary} onClick={()=>navigate('/calculate')}>
            <RefreshCw size={16}/> Score Another Applicant
          </button>
          <button className={styles.primary} onClick={()=>navigate('/')}>
            Back to Home
          </button>
        </motion.div>

      </main>
    </div>
  );
}
