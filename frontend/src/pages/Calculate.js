// src/pages/Calculate.js
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from '../components/Navbar';
import { score as scoreAPI } from '../api';
import { AlertCircle, Zap } from 'lucide-react';
import styles from './Calculate.module.css';

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, scale: 0.95 },
  transition: { duration: 0.4 }
};

const Section = ({ title, children, delay = 0 }) => (
  <motion.div
    className={`glass-panel ${styles.section}`}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
  >
    <h3 className={styles.sectionTitle}>{title}</h3>
    <div className={styles.sectionBody}>{children}</div>
  </motion.div>
);

const Field = ({ label, hint, children }) => (
  <div className={styles.field}>
    <label className={styles.label}>{label}</label>
    {hint && <span className={styles.hint}>{hint}</span>}
    {children}
  </div>
);

export default function Calculate() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [isNew, setIsNew] = useState(false);
  const [focusedSection, setFocusedSection] = useState(null);

  const [form, setForm] = useState({
    annual_income: 480000, loan_amount: 250000,
    monthly_payment: 8500, loan_term: 36, dti: 25,
    has_checking: true, checking_tier: 3,
    has_savings: true, savings_amount: 25000, savings_tier: 3,
    age: 30, family_size: 3,
    employed: true, employment_years: 4, employment_stability: 3,
    owns_property: false, region_rating: 1,
    high_risk_purpose: false, has_guarantor: false,
    existing_credits: 2, cc_utilization: 30, enquiries: 1,
    late_payment_rate: 5, max_days_late: 0, has_past_delays: false,
    credit_history: 4, approval_rate: 80, prev_applications: 2,
    docs_rate: 85,
  });

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleSubmit = async e => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const payload = {
        ...form,
        is_new_user: isNew,
        cc_utilization: form.cc_utilization / 100,
        late_payment_rate: form.late_payment_rate / 100,
        approval_rate: form.approval_rate / 100,
        docs_rate: form.docs_rate / 100,
      };
      const { data } = await scoreAPI(payload);

      // Artificial delay for animation smoothness
      await new Promise(res => setTimeout(res, 500));

      navigate('/result', { state: { result: data, inputs: form, isNew } });
    } catch (err) {
      setError(err.response?.data?.detail || 'Scoring failed. Is the backend running?');
      setLoading(false);
    }
  };

  return (
    <div className={styles.root}>
      <Navbar />

      {/* Background Orbs */}
      <div className={styles.bgOrbs}>
        <div className={styles.orb1} />
        <div className={styles.orb2} />
      </div>

      <main className={styles.main}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .5 }}>
          {/* Header */}
          <div className={styles.header}>
            <div>
              <h1>Credit Assessment</h1>
              <p>Fill in the applicant details to generate a credit score with full AI explanation.</p>
            </div>

            <label className={styles.toggle}>
              <input type="checkbox" checked={isNew} onChange={e => setIsNew(e.target.checked)} />
              <motion.span className={styles.toggleTrack} animate={{ backgroundColor: isNew ? 'var(--primary)' : 'var(--bg-surface)' }}>
                <motion.span className={styles.toggleThumb} layout transition={{ type: "spring", stiffness: 500, damping: 30 }} />
              </motion.span>
              <span className={styles.toggleLabel}>
                {isNew ? '🆕 New User Mode' : '👤 General Mode'}
              </span>
            </label>
          </div>

          <AnimatePresence>
            {isNew && (
              <motion.div {...fadeUp} className={styles.newBanner} layout>
                <AlertCircle size={18} />
                <span><strong>New User Mode Active</strong> — Loan history fields are hidden. Score is driven by savings, employment, and bank account behaviour.</span>
              </motion.div>
            )}
          </AnimatePresence>

          <form onSubmit={handleSubmit} className={styles.form}>
            <div className={styles.grid}>

              {/* Col 1 */}
              <div onFocus={() => setFocusedSection(1)} onBlur={() => setFocusedSection(null)} className={focusedSection === 1 ? styles.colFocused : ''}>
                <Section title="💰 Financial Information" delay={0.1}>
                  <Field label="Annual Income (₹)">
                    <input type="number" value={form.annual_income} min={0}
                      onChange={e => set('annual_income', +e.target.value)} className={styles.input} />
                  </Field>
                  <Field label="Loan Amount (₹)">
                    <input type="number" value={form.loan_amount} min={0}
                      onChange={e => set('loan_amount', +e.target.value)} className={styles.input} />
                  </Field>
                  <Field label="Monthly Payment (₹)">
                    <input type="number" value={form.monthly_payment} min={0}
                      onChange={e => set('monthly_payment', +e.target.value)} className={styles.input} />
                  </Field>
                  <Field label="Loan Term">
                    <select value={form.loan_term} onChange={e => set('loan_term', +e.target.value)} className={styles.input}>
                      {[12, 24, 36, 48, 60].map(v => <option key={v} value={v}>{v} months</option>)}
                    </select>
                  </Field>
                  <Field label={`Debt-to-Income: ${form.dti}%`}>
                    <input type="range" min={0} max={80} value={form.dti}
                      onChange={e => set('dti', +e.target.value)} className={styles.range} />
                  </Field>
                </Section>

                <Section title="🏦 Banking & Savings" delay={0.2}>
                  <label className={styles.checkCard}>
                    <input type="checkbox" checked={form.has_checking} onChange={e => set('has_checking', e.target.checked)} />
                    <span className={styles.checkCardText}>Has active checking account</span>
                  </label>

                  <Field label="Account Balance Tier">
                    <select value={form.checking_tier} onChange={e => set('checking_tier', +e.target.value)} className={styles.input} disabled={!form.has_checking}>
                      {[[0, 'No Account'], [1, 'Below 0'], [2, '0–200'], [3, '200–1000'], [4, '1000+']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select>
                  </Field>

                  <div className={styles.divider} />

                  <label className={styles.checkCard}>
                    <input type="checkbox" checked={form.has_savings} onChange={e => set('has_savings', e.target.checked)} />
                    <span className={styles.checkCardText}>Has active savings account</span>
                  </label>

                  <Field label="Savings Balance (₹)">
                    <input type="number" value={form.savings_amount} min={0}
                      onChange={e => set('savings_amount', +e.target.value)} className={styles.input} disabled={!form.has_savings} />
                  </Field>
                  <Field label="Savings Tier">
                    <select value={form.savings_tier} onChange={e => set('savings_tier', +e.target.value)} className={styles.input} disabled={!form.has_savings}>
                      {[[0, 'None'], [1, '<100'], [2, '100–500'], [3, '500–1000'], [4, '1000+']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select>
                  </Field>
                </Section>
              </div>

              {/* Col 2 */}
              <div onFocus={() => setFocusedSection(2)} onBlur={() => setFocusedSection(null)} className={focusedSection === 2 ? styles.colFocused : ''}>
                <Section title="👤 Personal & Employment" delay={0.3}>
                  <Field label={`Age: ${form.age} years`}>
                    <input type="range" min={18} max={75} value={form.age}
                      onChange={e => set('age', +e.target.value)} className={styles.range} />
                  </Field>
                  <Field label="Family Members">
                    <input type="number" value={form.family_size} min={1} max={15}
                      onChange={e => set('family_size', +e.target.value)} className={styles.input} />
                  </Field>

                  <label className={`${styles.checkCard} ${styles.mt1}`}>
                    <input type="checkbox" checked={form.employed} onChange={e => set('employed', e.target.checked)} />
                    <span className={styles.checkCardText}>Currently employed</span>
                  </label>

                  <Field label={`Years at Job: ${form.employment_years}`}>
                    <input type="range" min={0} max={40} value={form.employment_years}
                      onChange={e => set('employment_years', +e.target.value)} className={styles.range} disabled={!form.employed} />
                  </Field>
                  <Field label="Employment Stability">
                    <select value={form.employment_stability}
                      onChange={e => set('employment_stability', +e.target.value)} className={styles.input} disabled={!form.employed}>
                      {[[0, 'Unemployed'], [1, '< 1 year'], [2, '1–3 years'], [3, '4–7 years'], [4, '7+ years']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select>
                  </Field>

                  <div className={styles.divider} />

                  <div className={styles.checkGroup}>
                    <label className={styles.checkCardSmall}>
                      <input type="checkbox" checked={form.owns_property} onChange={e => set('owns_property', e.target.checked)} />
                      <span>Owns property</span>
                    </label>
                    <label className={styles.checkCardSmall}>
                      <input type="checkbox" checked={form.has_guarantor} onChange={e => set('has_guarantor', e.target.checked)} />
                      <span>Has guarantor</span>
                    </label>
                    <label className={styles.checkCardSmall}>
                      <input type="checkbox" checked={form.high_risk_purpose} onChange={e => set('high_risk_purpose', e.target.checked)} />
                      <span className={form.high_risk_purpose ? styles.textWarn : ''}>High-risk purpose</span>
                    </label>
                  </div>

                  <Field label="Region Risk">
                    <select value={form.region_rating} onChange={e => set('region_rating', +e.target.value)} className={styles.input}>
                      {[[1, 'Low Risk'], [2, 'Medium Risk'], [3, 'High Risk']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                    </select>
                  </Field>
                </Section>
              </div>

              {/* Col 3 */}
              <div onFocus={() => setFocusedSection(3)} onBlur={() => setFocusedSection(null)} className={focusedSection === 3 ? styles.colFocused : ''}>
                <Section title="💳 Credit Utilisation" delay={0.4}>
                  <Field label={`Credit Card Utilization: ${form.cc_utilization}%`}>
                    <input type="range" min={0} max={100} value={form.cc_utilization}
                      onChange={e => set('cc_utilization', +e.target.value)} className={styles.range} />
                  </Field>
                  <Field label="Existing Open Credits">
                    <input type="number" value={form.existing_credits} min={0} max={20}
                      onChange={e => set('existing_credits', +e.target.value)} className={styles.input} />
                  </Field>
                  <Field label="Credit Enquiries (last year)">
                    <input type="number" value={form.enquiries} min={0} max={20}
                      onChange={e => set('enquiries', +e.target.value)} className={styles.input} />
                  </Field>
                  <Field label={`Documents Submitted: ${form.docs_rate}%`}>
                    <input type="range" min={0} max={100} value={form.docs_rate}
                      onChange={e => set('docs_rate', +e.target.value)} className={styles.range} />
                  </Field>
                </Section>

                <AnimatePresence>
                  {!isNew && (
                    <motion.div {...fadeUp} layout>
                      <Section title="📅 Payment History" delay={0}>
                        <Field label={`Late Payment Rate: ${form.late_payment_rate}%`}>
                          <input type="range" min={0} max={50} value={form.late_payment_rate}
                            onChange={e => set('late_payment_rate', +e.target.value)} className={styles.range} />
                        </Field>
                        <Field label="Max Days Late Ever">
                          <select value={form.max_days_late} onChange={e => set('max_days_late', +e.target.value)} className={styles.input}>
                            {[[0, 'Never'], [30, '30 days'], [60, '60 days'], [90, '90 days'], [120, '120+ days']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                          </select>
                        </Field>
                        <Field label="Credit History">
                          <select value={form.credit_history} onChange={e => set('credit_history', +e.target.value)} className={styles.input}>
                            {[[0, 'Critical Account'], [1, 'Major Delays'], [2, 'Some Delays'], [3, 'Mostly Good'], [4, 'Perfect']].map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                          </select>
                        </Field>
                        <Field label={`Past Approval Rate: ${form.approval_rate}%`}>
                          <input type="range" min={0} max={100} value={form.approval_rate}
                            onChange={e => set('approval_rate', +e.target.value)} className={styles.range} />
                        </Field>
                        <Field label="Previous Applications">
                          <input type="number" value={form.prev_applications} min={0} max={20}
                            onChange={e => set('prev_applications', +e.target.value)} className={styles.input} />
                        </Field>
                        <label className={`${styles.checkCard} ${styles.mt1}`}>
                          <input type="checkbox" checked={form.has_past_delays}
                            onChange={e => set('has_past_delays', e.target.checked)} />
                          <span className={form.has_past_delays ? styles.textError : ''}>Has past payment delays</span>
                        </label>
                      </Section>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            <AnimatePresence>
              {error && (
                <motion.div className={styles.error} {...fadeUp}>
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div
              className={`glass-panel ${styles.footer}`}
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              <div className={styles.summary}>
                <div className={styles.summaryItem}>
                  <span>Credit-to-Income</span>
                  <strong>{(form.loan_amount / Math.max(form.annual_income, 1)).toFixed(2)}x</strong>
                </div>
                <div className={styles.summaryItem}>
                  <span>Payment Burden</span>
                  <strong>{(form.monthly_payment * 12 / Math.max(form.annual_income, 1) * 100).toFixed(1)}%</strong>
                </div>
                <div className={styles.summaryItem}>
                  <span>Evaluation Mode</span>
                  <strong className={styles.textGradient}>{isNew ? 'New User' : 'General'}</strong>
                </div>
              </div>
              <button type="submit" className="btn-primary" disabled={loading} style={{ padding: '1rem 2.5rem', fontSize: '1.1rem' }}>
                {loading
                  ? <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}><span className={styles.spinner} /> Processing Data...</div>
                  : <><Zap size={18} /> Initialize Assessment</>}
              </button>
            </motion.div>
          </form>
        </motion.div>
      </main>
    </div>
  );
}
