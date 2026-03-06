// src/pages/Batch.js
import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from '../components/Navbar';
import api from '../api';
import { Upload, Download, FileText, CheckCircle2, XCircle, AlertTriangle, Clock, Zap } from 'lucide-react';
import styles from './Batch.module.css';

const BAND_COLOR = {
  'Excellent': '#059669', // Muted Emerald
  'Very Good': '#047857',
  'Good': '#D97706', // Muted Amber
  'Fair': '#C2410C', // Muted Orange
  'Poor': '#B91C1C', // Muted Red
  'Very Poor': '#991B1B',
};

const REC_STYLE = {
  'Approve': { bg: 'rgba(5, 150, 105, 0.08)', color: '#059669' },
  'Conditional': { bg: 'rgba(217, 119, 6, 0.08)', color: '#D97706' },
  'Reject': { bg: 'rgba(185, 28, 28, 0.08)', color: '#B91C1C' },
};

const fadeUp = {
  initial: { opacity: 0, y: 30 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

export default function Batch() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  const handleFile = f => {
    if (!f || !f.name.endsWith('.csv')) {
      setError('Please upload a .csv file'); return;
    }
    setFile(f); setError(''); setResults(null);
    // Parse preview
    const reader = new FileReader();
    reader.onload = e => {
      const lines = e.target.result.split('\n').filter(Boolean);
      const headers = lines[0].split(',').map(h => h.trim());
      const rows = lines.slice(1, 6).map(l =>
        Object.fromEntries(l.split(',').map((v, i) => [headers[i], v?.trim()]))
      );
      setPreview({ headers, rows, total: lines.length - 1 });
    };
    reader.readAsText(f);
  };

  const handleDrop = e => {
    e.preventDefault(); setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const downloadTemplate = async () => {
    try {
      const res = await api.get('/batch-template', { responseType: 'blob' });
      const url = URL.createObjectURL(res.data);
      const a = document.createElement('a');
      a.href = url; a.download = 'zenith_batch_template.csv'; a.click();
      URL.revokeObjectURL(url);
    } catch { setError('Failed to download template'); }
  };

  const handleSubmit = async () => {
    if (!file) { setError('Please select a CSV file first'); return; }
    setLoading(true); setError(''); setProgress(0);

    // Simulate progress while uploading
    const interval = setInterval(() => {
      setProgress(p => p < 85 ? p + Math.random() * 12 : p);
    }, 300);

    try {
      const form = new FormData();
      form.append('file', file);
      const res = await api.post('/batch-score', form, {
        responseType: 'blob',
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      clearInterval(interval); setProgress(100);

      // Parse the returned CSV for preview
      const text = await res.data.text();
      const lines = text.split('\n').filter(Boolean);
      const headers = lines[0].split(',').map(h => h.trim());
      const dataRows = lines.slice(1, -1); // exclude summary row

      const parsed = dataRows.slice(0, 10).map(l => {
        const vals = l.split(',');
        return Object.fromEntries(headers.map((h, i) => [h, (vals[i] || '').trim()]));
      });

      // Parse summary (last row)
      const summaryVals = lines[lines.length - 1].split(',');
      const summary = Object.fromEntries(headers.map((h, i) => [h, (summaryVals[i] || '').trim()]));

      setTimeout(() => {
        setResults({ headers, rows: parsed, summary, blob: res.data, total: dataRows.length });
        setLoading(false);
      }, 500); // Artificial delay for smooth animation

      // Trigger download
      const url = URL.createObjectURL(res.data);
      const a = document.createElement('a');
      a.href = url; a.download = `zenith_scores_${Date.now()}.csv`; a.click();
      URL.revokeObjectURL(url);

    } catch (err) {
      clearInterval(interval);
      setError(err.response?.data?.detail || 'Batch scoring failed. Is the backend running?');
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!results?.blob) return;
    const url = URL.createObjectURL(results.blob);
    const a = document.createElement('a');
    a.href = url; a.download = `zenith_scores_${Date.now()}.csv`; a.click();
    URL.revokeObjectURL(url);
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
              <h1>Batch Credit Scoring</h1>
              <p>Upload a CSV of up to 500 applicants — scores are calculated and returned as a downloadable CSV in seconds.</p>
            </div>
            <button className="btn-secondary" onClick={downloadTemplate}>
              <Download size={18} /> Download Template CSV
            </button>
          </div>

          <div className={styles.layout}>
            {/* Left — Upload */}
            <div className={styles.uploadCol}>
              {/* Drop zone */}
              <motion.div
                className={`glass-panel ${styles.dropzone} ${dragOver ? styles.dragOver : ''} ${file ? styles.hasFile : ''}`}
                onDragOver={e => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => inputRef.current.click()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <input ref={inputRef} type="file" accept=".csv"
                  onChange={e => handleFile(e.target.files[0])} style={{ display: 'none' }} />

                <AnimatePresence mode="wait">
                  {file ? (
                    <motion.div
                      key="file"
                      className={styles.fileInfo}
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                    >
                      <FileText size={40} className={styles.fileIconActive} />
                      <div className={styles.fileDetails}>
                        <div className={styles.fileName}>{file.name}</div>
                        <div className={styles.fileSize}>{(file.size / 1024).toFixed(1)} KB · <span className={styles.textHighlight}>{preview?.total} applicants</span></div>
                      </div>
                      <CheckCircle2 size={24} className={styles.fileStatusIcon} />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="prompt"
                      className={styles.dropPrompt}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <Upload size={40} className={styles.uploadIcon} />
                      <p><strong>Drop CSV here</strong> or click to browse</p>
                      <span>Max 500 rows · .csv format only</span>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

              {/* Instructions */}
              <motion.div className={`glass-panel ${styles.instructions}`} {...fadeUp} transition={{ delay: 0.1 }}>
                <h3>How it works</h3>
                <ol>
                  <li><strong>Download the template</strong> using the button above</li>
                  <li><strong>Fill in your applicants</strong> — one row per person</li>
                  <li><strong>Upload the CSV</strong> using the drop zone above</li>
                  <li><strong>Click Score</strong> — results download automatically</li>
                </ol>
                <div className={styles.outputCols}>
                  <h4>Output columns added:</h4>
                  <div className={styles.colPills}>
                    {['credit_score', 'score_band', 'default_prob', 'recommendation', 'scoring_time', 'status'].map(c => (
                      <span key={c} className={styles.pill}>{c}</span>
                    ))}
                  </div>
                </div>
              </motion.div>

              <AnimatePresence>
                {error && (
                  <motion.div className={styles.error} initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
                    <XCircle size={18} /> <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <button className={`btn-primary ${styles.scoreBtn}`} onClick={handleSubmit} disabled={loading || !file}>
                {loading ? (
                  <div className={styles.btnLoading}>
                    <span className={styles.spinner} /> <span>Processing {preview?.total} applicants...</span>
                  </div>
                ) : (
                  <><Zap size={18} /> Initialize Batch Scoring</>
                )}
              </button>

              {/* Progress bar */}
              <AnimatePresence>
                {loading && (
                  <motion.div className={styles.progressWrap} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <div className={styles.progressBar} style={{ width: `${progress}%` }}>
                      <div className={styles.progressGlow} />
                    </div>
                    <span className={styles.progressText}>{Math.round(progress)}%</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Right — Preview / Results */}
            <div className={styles.previewCol}>
              <AnimatePresence mode="wait">
                {results ? (
                  <motion.div key="results" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -20 }} className={styles.resultsPanel}>
                    {/* Summary cards */}
                    <div className={styles.summaryCards}>
                      <div className={`glass-panel ${styles.summaryCard}`}>
                        <CheckCircle2 size={24} style={{ color: '#10B981' }} />
                        <div className={styles.summaryVal}>{results.total}</div>
                        <div className={styles.summaryLbl}>Applicants Scored</div>
                      </div>
                      <div className={`glass-panel ${styles.summaryCard}`}>
                        <Zap size={24} style={{ color: '#F59E0B' }} />
                        <div className={styles.summaryVal}>{results.summary.score_band?.replace('Avg score: ', '') || '—'}</div>
                        <div className={styles.summaryLbl}>Average Score</div>
                      </div>
                      <div className={`glass-panel ${styles.summaryCard}`}>
                        <Clock size={24} style={{ color: '#7C3AED' }} />
                        <div className={styles.summaryVal}>{results.summary.scoring_time_ms?.split('|')[0]?.trim() || '—'}</div>
                        <div className={styles.summaryLbl}>Total Time</div>
                      </div>
                    </div>

                    {/* Recommendation breakdown */}
                    {results.summary.recommendation && (
                      <div className={`glass-panel ${styles.recBreakdown}`}>
                        {results.summary.recommendation.split(' | ').map(part => {
                          const [label, count] = part.split(':');
                          const style = REC_STYLE[label.trim()] || {};
                          return (
                            <div key={label} className={styles.recPart}
                              style={{ background: style.bg, color: style.color, border: `1px solid ${style.color}30` }}>
                              <span className={styles.recCount}>{count}</span>
                              <span className={styles.recLabel}>{label.trim()}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Results table */}
                    <div className={`glass-panel ${styles.tableWrap}`}>
                      <div className={styles.tableHeader}>
                        <span>Preview (first 10 rows)</span>
                        <button className="btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }} onClick={downloadResults}>
                          <Download size={16} /> Download Full CSV
                        </button>
                      </div>
                      <div className={styles.tableScroll}>
                        <table className={styles.table}>
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>Score</th>
                              <th>Band</th>
                              <th>Default %</th>
                              <th>Decision</th>
                              <th>Time</th>
                            </tr>
                          </thead>
                          <tbody>
                            {results.rows.map((row, i) => {
                              const score = parseInt(row.credit_score);
                              const band = row.score_band;
                              const rec = row.recommendation;
                              const recStyle = REC_STYLE[rec] || {};
                              return (
                                <tr key={i}>
                                  <td className={styles.rowNum}>{i + 1}</td>
                                  <td className={styles.scoreCell}>
                                    <span style={{ color: BAND_COLOR[band] || '#888', fontWeight: 600 }}>
                                      {row.credit_score || '—'}
                                    </span>
                                  </td>
                                  <td>
                                    <span className={styles.bandPill}
                                      style={{ background: `${BAND_COLOR[band]}15`, color: BAND_COLOR[band], border: `1px solid ${BAND_COLOR[band]}40` }}>
                                      {band || '—'}
                                    </span>
                                  </td>
                                  <td className={styles.monoCell}>{row.default_probability_pct ? `${row.default_probability_pct}%` : '—'}</td>
                                  <td>
                                    <span className={styles.recPill}
                                      style={{ background: recStyle.bg, color: recStyle.color, border: `1px solid ${recStyle.color}40` }}>
                                      {rec || '—'}
                                    </span>
                                  </td>
                                  <td className={styles.timeCell}>{row.scoring_time_ms ? `${row.scoring_time_ms}ms` : '—'}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </motion.div>
                ) : preview ? (
                  <motion.div key="preview" initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }} className={`glass-panel ${styles.previewPanel}`}>
                    <div className={styles.previewHeader}>
                      <FileText size={18} style={{ color: 'var(--text-muted)' }} />
                      <span>Dataset Preview — <strong className={styles.textHighlight}>{preview.total} applicants</strong> detected</span>
                    </div>
                    <div className={styles.tableScroll}>
                      <table className={styles.table}>
                        <thead>
                          <tr>{preview.headers.slice(0, 8).map(h => <th key={h}>{h}</th>)}</tr>
                        </thead>
                        <tbody>
                          {preview.rows.map((row, i) => (
                            <tr key={i}>
                              {preview.headers.slice(0, 8).map(h => (
                                <td key={h}>{row[h] ?? '—'}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className={styles.previewNote}>Showing first 5 rows · first 8 columns</p>
                  </motion.div>
                ) : (
                  <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className={`glass-panel ${styles.emptyState}`}>
                    <div className={styles.emptyIconWrap}>
                      <FileText size={48} className={styles.emptyIcon} />
                      <div className={styles.emptyGlow} />
                    </div>
                    <h3>No Data Loaded</h3>
                    <p>Upload a CSV file to preview your dataset here</p>
                    <span>Results will appear after scoring</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
