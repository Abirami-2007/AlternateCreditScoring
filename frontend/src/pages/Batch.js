// src/pages/Batch.js
import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from '../components/Navbar';
import api from '../api';
import { Upload, Download, FileText, CheckCircle, XCircle, AlertCircle, Clock } from 'lucide-react';
import styles from './Batch.module.css';

const BAND_COLOR = {
  'Excellent':'#2E7D32','Very Good':'#388E3C','Good':'#F9A825',
  'Fair':'#F57F17','Poor':'#E64A19','Very Poor':'#B71C1C',
};
const REC_STYLE = {
  'Approve':     { bg:'#E8F5E9', color:'#1B5E20' },
  'Conditional': { bg:'#FFF8E1', color:'#E65100' },
  'Reject':      { bg:'#FFEBEE', color:'#B71C1C' },
};

export default function Batch() {
  const [file, setFile]         = useState(null);
  const [preview, setPreview]   = useState(null);
  const [loading, setLoading]   = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults]   = useState(null);
  const [error, setError]       = useState('');
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
        Object.fromEntries(l.split(',').map((v,i) => [headers[i], v.trim()]))
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
      const a   = document.createElement('a');
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
        return Object.fromEntries(headers.map((h,i) => [h, (vals[i]||'').trim()]));
      });

      // Parse summary (last row)
      const summaryVals = lines[lines.length-1].split(',');
      const summary = Object.fromEntries(headers.map((h,i) => [h, (summaryVals[i]||'').trim()]));

      setResults({ headers, rows: parsed, summary, blob: res.data, total: dataRows.length });

      // Trigger download
      const url = URL.createObjectURL(res.data);
      const a   = document.createElement('a');
      a.href = url; a.download = `zenith_scores_${Date.now()}.csv`; a.click();
      URL.revokeObjectURL(url);

    } catch (err) {
      clearInterval(interval);
      setError(err.response?.data?.detail || 'Batch scoring failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  const downloadResults = () => {
    if (!results?.blob) return;
    const url = URL.createObjectURL(results.blob);
    const a   = document.createElement('a');
    a.href = url; a.download = `zenith_scores_${Date.now()}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className={styles.root}>
      <Navbar />
      <main className={styles.main}>
        <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} transition={{duration:.5}}>

          {/* Header */}
          <div className={styles.header}>
            <div>
              <h1>Batch Credit Scoring</h1>
              <p>Upload a CSV of up to 500 applicants — scores are calculated and returned as a downloadable CSV in seconds.</p>
            </div>
            <button className={styles.templateBtn} onClick={downloadTemplate}>
              <Download size={16}/> Download Template CSV
            </button>
          </div>

          <div className={styles.layout}>
            {/* Left — Upload */}
            <div className={styles.uploadCol}>
              {/* Drop zone */}
              <div className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''} ${file ? styles.hasFile : ''}`}
                onDragOver={e=>{e.preventDefault();setDragOver(true)}}
                onDragLeave={()=>setDragOver(false)}
                onDrop={handleDrop}
                onClick={()=>inputRef.current.click()}>
                <input ref={inputRef} type="file" accept=".csv"
                  onChange={e=>handleFile(e.target.files[0])} style={{display:'none'}}/>
                {file ? (
                  <div className={styles.fileInfo}>
                    <FileText size={32} color="#2E7D32"/>
                    <div>
                      <div className={styles.fileName}>{file.name}</div>
                      <div className={styles.fileSize}>{(file.size/1024).toFixed(1)} KB · {preview?.total} applicants</div>
                    </div>
                    <CheckCircle size={20} color="#2E7D32"/>
                  </div>
                ) : (
                  <div className={styles.dropPrompt}>
                    <Upload size={36} className={styles.uploadIcon}/>
                    <p><strong>Drop CSV here</strong> or click to browse</p>
                    <span>Max 500 rows · .csv format only</span>
                  </div>
                )}
              </div>

              {/* Instructions */}
              <div className={styles.instructions}>
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
                    {['credit_score','score_band','default_probability_pct','recommendation','scoring_time_ms','status'].map(c=>(
                      <span key={c} className={styles.pill}>{c}</span>
                    ))}
                  </div>
                </div>
              </div>

              {error && (
                <div className={styles.error}>
                  <XCircle size={16}/> {error}
                </div>
              )}

              <button className={styles.scoreBtn} onClick={handleSubmit} disabled={loading || !file}>
                {loading ? (
                  <><span className={styles.spinner}/> Scoring {preview?.total} applicants...</>
                ) : (
                  <>⚡ Score All Applicants →</>
                )}
              </button>

              {/* Progress bar */}
              {loading && (
                <div className={styles.progressWrap}>
                  <div className={styles.progressBar} style={{width:`${progress}%`}}/>
                  <span className={styles.progressText}>{Math.round(progress)}%</span>
                </div>
              )}
            </div>

            {/* Right — Preview / Results */}
            <div className={styles.previewCol}>
              {results ? (
                <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{duration:.4}}>
                  {/* Summary cards */}
                  <div className={styles.summaryCards}>
                    <div className={styles.summaryCard}>
                      <CheckCircle size={18} color="#2E7D32"/>
                      <div className={styles.summaryVal}>{results.total}</div>
                      <div className={styles.summaryLbl}>Applicants Scored</div>
                    </div>
                    <div className={styles.summaryCard}>
                      <AlertCircle size={18} color="#F9A825"/>
                      <div className={styles.summaryVal}>{results.summary.score_band?.replace('Avg score: ','') || '—'}</div>
                      <div className={styles.summaryLbl}>Average Score</div>
                    </div>
                    <div className={styles.summaryCard}>
                      <Clock size={18} color="#1a237e"/>
                      <div className={styles.summaryVal}>{results.summary.scoring_time_ms?.split('|')[0]?.trim() || '—'}</div>
                      <div className={styles.summaryLbl}>Total Time</div>
                    </div>
                  </div>

                  {/* Recommendation breakdown */}
                  {results.summary.recommendation && (
                    <div className={styles.recBreakdown}>
                      {results.summary.recommendation.split(' | ').map(part => {
                        const [label, count] = part.split(':');
                        const style = REC_STYLE[label.trim()] || {};
                        return (
                          <div key={label} className={styles.recPart}
                            style={{background:style.bg,color:style.color}}>
                            <span className={styles.recCount}>{count}</span>
                            <span>{label.trim()}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {/* Results table */}
                  <div className={styles.tableWrap}>
                    <div className={styles.tableHeader}>
                      <span>Preview (first 10 rows)</span>
                      <button className={styles.downloadBtn} onClick={downloadResults}>
                        <Download size={14}/> Download Full CSV
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
                            const band  = row.score_band;
                            const rec   = row.recommendation;
                            const recStyle = REC_STYLE[rec] || {};
                            return (
                              <tr key={i}>
                                <td className={styles.rowNum}>{i+1}</td>
                                <td className={styles.scoreCell}>
                                  <span style={{color: BAND_COLOR[band] || '#888', fontWeight:700}}>
                                    {row.credit_score || '—'}
                                  </span>
                                </td>
                                <td>
                                  <span className={styles.bandPill}
                                    style={{background:`${BAND_COLOR[band]}18`, color:BAND_COLOR[band]}}>
                                    {band || '—'}
                                  </span>
                                </td>
                                <td>{row.default_probability_pct ? `${row.default_probability_pct}%` : '—'}</td>
                                <td>
                                  <span className={styles.recPill}
                                    style={{background:recStyle.bg, color:recStyle.color}}>
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
                <motion.div initial={{opacity:0}} animate={{opacity:1}} transition={{duration:.4}}>
                  <div className={styles.previewHeader}>
                    <span>📄 Preview — {preview.total} applicants detected</span>
                  </div>
                  <div className={styles.tableScroll}>
                    <table className={styles.table}>
                      <thead>
                        <tr>{preview.headers.slice(0,8).map(h=><th key={h}>{h}</th>)}</tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((row,i)=>(
                          <tr key={i}>
                            {preview.headers.slice(0,8).map(h=>(
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
                <div className={styles.emptyState}>
                  <FileText size={48} color="#d0d0e0"/>
                  <p>Upload a CSV to preview your data here</p>
                  <span>Results will appear after scoring</span>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}
