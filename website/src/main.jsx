import { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDownToLine, ArrowRight, BellRing, BrainCircuit, CircleHelp, Cpu, Database, Download, FileSpreadsheet, Flame, Gauge, LoaderCircle, Menu, Moon, Network, Play, RotateCcw, ScanSearch, ShieldCheck, Sparkles, Sun, Upload, X, Zap } from 'lucide-react';
import { Area, AreaChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import './styles.css';
import ModelSlot from './ModelSlot';
import ModelWorkspace from './ModelPage';
import FaultReport from './FaultReport';

const navItems = [
  { id: 'test', label: 'Prediction console', icon: ScanSearch },
  { id: 'model', label: 'Model + dataset', icon: BrainCircuit },
];

const fallbackMetrics = [
  { metric: 'RMSE', value: '—', note: 'Upload labeled data' },
  { metric: 'MAE', value: '—', note: 'Upload labeled data' },
  { metric: 'R²', value: '—', note: 'Upload labeled data' },
  { metric: 'NASA score', value: '—', note: 'Upload labeled data' },
];

function formatNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatMetric(value, suffix = '') {
  return value === null || value === undefined ? '—' : `${formatNumber(value)}${suffix}`;
}

function downloadBlob(name, body, type = 'application/json') {
  const blob = new Blob([body], { type });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

function IconButton({ label, children, onClick, className = '' }) {
  return <button className={`icon-button ${className}`} aria-label={label} title={label} onClick={onClick}>{children}</button>;
}

function SectionLabel({ children, icon: Icon = Sparkles }) {
  return <div className="section-label"><Icon size={14} /> {children}</div>;
}

function Topbar({ active, setActive, status, onMenu, theme, onToggleTheme }) {
  const [localTheme, setLocalTheme] = useState(() => localStorage.getItem('grace-rul-theme') || 'dark');
  const activeTheme = theme || localTheme;
  const toggleTheme = onToggleTheme || (() => setLocalTheme(current => current === 'dark' ? 'light' : 'dark'));
  useEffect(() => {
    const nextTheme = theme || localTheme;
    document.documentElement.dataset.theme = nextTheme;
    localStorage.setItem('grace-rul-theme', nextTheme);
  }, [theme, localTheme]);
  return <header className="topbar">
    <div className="brand" onClick={() => setActive('test')} role="button" tabIndex={0}>
      <span className="brand-mark"><Cpu size={18} /></span>
      <span>GRACE<span className="brand-divider">:</span>RUL</span>
      <span className="brand-badge">C-MAPSS</span>
    </div>
    <nav className="topnav" aria-label="Main navigation">
      {navItems.map(({ id, label, icon: Icon }) => <button key={id} className={active === id ? 'active' : ''} onClick={() => setActive(id)}>
        <Icon size={15} /> {label}
      </button>)}
    </nav>
    <div className="top-actions">
      <div className="mission-clock">UTC 00:00:00</div>
      <div className={`system-status ${status?.ready ? 'ready' : ''}`}><span className="status-dot" /> {status?.ready ? 'INFERENCE ONLINE' : 'MODEL OFFLINE'}</div>
      <IconButton className="theme-toggle" label={activeTheme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} onClick={toggleTheme}>{activeTheme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}</IconButton>
      <IconButton label="Help"><CircleHelp size={18} /></IconButton>
      <button className="menu-button" onClick={onMenu} aria-label="Open navigation"><Menu size={20} /></button>
    </div>
  </header>;
}

function Hero({ setActive, status }) {
  return <section className="hero section-shell">
    <div className="telemetry-grid" aria-hidden="true" />
    <div className="hero-copy">
      <div className="eyebrow"><span className="eyebrow-pulse" /> NASA C-MAPSS / PROPULSION HEALTH MONITOR</div>
      <div className="hero-title-row"><div><h1>Remaining useful life<br /><em>prediction console</em></h1><p className="hero-subtitle">Inference workspace for turbofan degradation trajectories. Load a C-MAPSS engine history to reconstruct the sensor stream, estimate time-to-failure, and inspect model confidence.</p></div><div className="mission-stamp"><span>GRACE-RUL</span><b>OPS // 01</b><small>FLIGHT DATA SYSTEM</small></div></div>
      <div className="hero-actions">
        <button className="button button-primary" onClick={() => setActive('test')}><Upload size={16} /> Load engine history <ArrowRight size={16} /></button>
        <button className="button button-quiet" onClick={() => setActive('model')}><Network size={16} /> View model protocol</button>
      </div>
      <div className="hero-trust"><ShieldCheck size={15} /> SEPARATE CALIBRATION <span /> <Database size={15} /> 4 DATA REGIMES <span /> <Activity size={15} /> 21 SENSORS + 3 SETTINGS</div>
    </div>
    <div className="hero-readout">
      <div className="readout-heading"><span>MISSION STATUS</span><b className={status?.ready ? 'ready' : ''}><i /> {status?.ready ? 'READY FOR INFERENCE' : 'AWAITING MODEL BUNDLE'}</b></div>
      <div className="readout-display"><span className="display-label">RUL PREDICTION</span><strong>—</strong><small>CYCLES REMAINING</small></div>
      <div className="readout-rule" />
      <div className="readout-grid"><div><span>DATA SOURCE</span><b>TEST HISTORY</b></div><div><span>WINDOW</span><b>70 / 30 CYCLES</b></div><div><span>UNCERTAINTY</span><b>MC DROPOUT</b></div><div><span>OUTPUT</span><b>EMPIRICAL INTERVAL</b></div></div>
    </div>
    <div className="hero-scroll">SYSTEM OVERVIEW <ArrowDownToLine size={14} /></div>
  </section>;
}

function UploadPanel({ onResult, modelStatus }) {
  const [file, setFile] = useState(null);
  const [subset, setSubset] = useState('FD001');
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  async function inspect(nextFile) {
    if (!nextFile) return;
    setError(''); setFile(nextFile); setPreview(null);
    const inferredSubset = nextFile.name.match(/FD00[1-4]/i)?.[0].toUpperCase() || subset;
    setSubset(inferredSubset);
    const form = new FormData(); form.append('file', nextFile); form.append('subset', inferredSubset);
    try {
      const response = await fetch('/api/preview', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Preview failed.');
      setPreview(data);
    } catch (err) { setError(err.message); }
  }

  async function runPrediction() {
    if (!file) return setError('Choose a data file first.');
    setBusy(true); setError('');
    const form = new FormData(); form.append('file', file); form.append('subset', subset); form.append('mc_dropout', 'true');
    try {
      const response = await fetch('/api/predict', { method: 'POST', body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Inference failed.');
      onResult(data, file);
    } catch (err) { setError(err.message); }
    finally { setBusy(false); }
  }

  return <section className="upload-card card-glow">
    <div className="card-heading">
      <div><SectionLabel icon={Upload}>TELEMETRY INGEST</SectionLabel><h2>Register test data</h2></div>
      <span className="step-index">01 <span>/ 03</span></span>
    </div>
    <p className="card-description">Register a C-MAPSS test history. The inference service accepts the canonical 26-column format, preserves missing sensor observations, and returns one remaining-life forecast per engine.</p>
    <div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); inspect(event.dataTransfer.files[0]); }} onClick={() => inputRef.current?.click()}>
      <input ref={inputRef} type="file" accept=".csv,.txt,text/csv,text/plain" hidden onChange={event => inspect(event.target.files[0])} />
      {file ? <><span className="file-icon"><FileSpreadsheet size={20} /></span><strong>{file.name}</strong><small>{preview ? `${formatNumber(preview.rows, 0)} observations · ${formatNumber(preview.engines, 0)} engines` : 'Reading file…'}</small><button className="remove-file" onClick={event => { event.stopPropagation(); setFile(null); setPreview(null); }}><X size={14} /> replace</button></> : <><span className="upload-icon"><Upload size={19} /></span><strong>Drop your CSV or text file here</strong><small>or click to browse · max 25 MB</small></>}
    </div>
     <div className="form-row input-meta"><label className="field-label">DATASET <select aria-label="Dataset subset" value={subset} onChange={e => { setSubset(e.target.value); setPreview(null); }}>{["FD001", "FD002", "FD003", "FD004"].map(s => <option key={s}>{s}</option>)}</select></label><a className="button button-quiet" href={`/api/sample?subset=${subset}`}>Download sample</a></div>
    {preview && <div className="upload-preview"><span><b>{formatNumber(preview.availability)}%</b> available</span><span><b>{formatNumber(preview.engines, 0)}</b> engines</span><span><b>{preview.has_targets ? 'Yes' : 'No'}</b> targets</span></div>}
    {error && <div className="error-message"><BellRing size={15} /> {error}{!modelStatus?.ready && <small> The model bundle will be detected automatically after notebook training finishes.</small>}</div>}
     <button className="button button-primary run-button" disabled={!file || busy || !modelStatus?.ready} onClick={runPrediction}>{busy ? <><LoaderCircle className="spin" size={17} /> EXECUTING INFERENCE…</> : <><Play size={16} fill="currentColor" /> {modelStatus?.ready ? 'Execute prediction' : 'Awaiting trained model'}</>}<ArrowRight size={16} /></button>
    <div className="upload-footnote"><ShieldCheck size={13} /> Your file stays in this local workspace.</div>
  </section>;
}

function MetricStrip({ result }) {
  const metrics = result?.metrics ? [
    { metric: 'RMSE', value: formatMetric(result.metrics.RMSE, ' cycles'), note: `${result.metrics.N} labeled engines` },
    { metric: 'MAE', value: formatMetric(result.metrics.MAE, ' cycles'), note: 'absolute error' },
    { metric: 'R²', value: formatMetric(result.metrics.R2), note: 'variance explained' },
    { metric: 'NASA score', value: formatNumber(result.metrics.NASA_Score, 0), note: 'lower is better' },
  ] : fallbackMetrics;
  return <div className="metric-strip">{metrics.map(item => <div className="metric-cell" key={item.metric}><span>{item.metric}</span><strong>{item.value}</strong><small>{item.note}</small></div>)}</div>;
}

function OverviewChart({ engines }) {
  const data = engines.map(engine => ({ name: displayEngineId(engine), predicted: engine.predicted_rul, lower: engine.interval_lower, truth: engine.true_rul }));
  return <div className="chart-wrap tall-chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 8 }}>
    <defs><linearGradient id="rul-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff734d" stopOpacity=".35" /><stop offset="100%" stopColor="#ff734d" stopOpacity="0" /></linearGradient></defs>
    <CartesianGrid strokeDasharray="3 3" stroke="#242427" vertical={false} /><XAxis dataKey="name" hide /><YAxis stroke="#69696f" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} /><Tooltip content={<ChartTooltip />} />
    <Area type="monotone" dataKey="predicted" stroke="#ff734d" strokeWidth={2.5} fill="url(#rul-fill)" name="Predicted RUL" />
    <Line type="monotone" dataKey="truth" stroke="#e9ddca" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="True RUL" />
    <Line type="monotone" dataKey="lower" stroke="#ffad64" strokeOpacity={.6} dot={false} name="Interval floor" />
  </ComposedChart></ResponsiveContainer></div>;
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><small>{label}</small>{payload.map(item => <div key={item.name}><i style={{ background: item.color }} /> {item.name}<b>{formatNumber(item.value)} cycles</b></div>)}</div>;
}

function displayEngineId(engine) {
  return `ENGINE ${String(engine?.unit ?? engine?.id ?? '').replace(/^.*:/, '').padStart(3, '0')}`;
}

function ResultsWorkspace({ result, setResult, sourceFile }) {
  const [selectedId, setSelectedId] = useState(null);
  const [tab, setTab] = useState('overview');
  const selected = result?.engines?.find(engine => engine.id === selectedId) || result?.engines?.[0];
  const trajectory = selected?.trajectory || [];
  const lowest = useMemo(() => [...(result?.engines || [])].sort((a, b) => a.predicted_rul - b.predicted_rul).slice(0, 5), [result]);
  if (!result) return null;
  function downloadReport() { downloadBlob('grace-rul-report.json', JSON.stringify(result, null, 2)); }
  return <section className="results-section section-shell" id="results">
    <div className="section-topline"><div><SectionLabel icon={Activity}>POST-FLIGHT DIAGNOSTICS / OUTPUT</SectionLabel><h2>Fleet prediction register</h2><p>Engine-level remaining life, empirical interval bounds, and sensor quality for the uploaded mission data.</p></div><div className="result-actions"><button className="button button-quiet" onClick={() => setResult(null)}><RotateCcw size={15} /> New upload</button><button className="button button-primary" onClick={downloadReport}><Download size={15} /> Export JSON</button></div></div>
    <div className="result-banner"><div className="result-banner-orb"><Flame size={20} /></div><div><strong>{result.filename}</strong><span>Processed with {result.model.mode === 'quick' ? 'quick-validation' : 'trained'} GRACE-RUL · {result.model.bundle}</span></div><div className="banner-right"><span className="live-chip"><span /> ANALYSIS COMPLETE</span><small>{formatNumber(result.summary.elapsed_seconds, 2)}s runtime</small></div></div>
    <MetricStrip result={result} />
     <div className="result-tabs"><button className={tab === 'overview' ? 'active' : ''} onClick={() => setTab('overview')}>MISSION OVERVIEW</button><button className={tab === 'engines' ? 'active' : ''} onClick={() => setTab('engines')}>ENGINE REGISTER <span>{result.engines.length}</span></button><button className={tab === 'sensors' ? 'active' : ''} onClick={() => setTab('sensors')}>SENSOR FAULTS + EXPLANATIONS</button></div>
    {tab === 'overview' && <div className="result-grid">
      <div className="panel chart-panel wide-panel"><div className="panel-head"><div><span className="panel-kicker">FLEET TRAJECTORY</span><h3>Predicted life across the uploaded fleet</h3></div><div className="legend"><span><i className="legend-predicted" /> prediction</span><span><i className="legend-true" /> ground truth</span></div></div><OverviewChart engines={result.engines} /><div className="chart-axis-note"><span>Each point is an engine · interval floor is shown alongside RUL</span><span>cycles remaining →</span></div></div>
      <div className="panel risk-panel"><div className="panel-head"><div><span className="panel-kicker">PRIORITY QUEUE</span><h3>Earliest intervention</h3></div><Zap size={17} className="orange-icon" /></div><div className="risk-list">{lowest.map((engine, index) => <button className="risk-row" key={engine.id} onClick={() => { setSelectedId(engine.id); setTab('engines'); }}><span className="risk-rank">0{index + 1}</span><span className="risk-engine"><b>{displayEngineId(engine)}</b><small>cycle {engine.last_cycle} · {formatNumber(engine.availability)}% signal</small></span><strong>{formatNumber(engine.predicted_rul, 0)}<small>cycles</small></strong><ArrowRight size={14} /></button>)}</div></div>
      <div className="panel summary-panel"><div className="panel-head"><div><span className="panel-kicker">READOUT</span><h3>System summary</h3></div><Gauge size={17} className="orange-icon" /></div><div className="summary-rows"><div><span>Engines analyzed</span><b>{result.summary.engines}</b></div><div><span>Signal availability</span><b>{formatNumber(result.summary.availability)}%</b></div><div><span>Near-failure ≤ 25 cycles</span><b className={result.summary.near_failure ? 'text-alert' : ''}>{result.summary.near_failure}</b></div><div><span>Nominal interval level</span><b>{formatNumber(result.summary.interval_confidence, 0)}%</b></div></div><div className="confidence-bar"><span style={{ width: `${Math.min(100, result.summary.interval_confidence)}%` }} /></div></div>
    </div>}
    {tab === 'engines' && <div className="engine-layout"><div className="panel engine-list-panel"><div className="panel-head"><div><span className="panel-kicker">ENGINE EXPLORER</span><h3>Select a trajectory</h3></div><span className="engine-count">{result.engines.length} units</span></div><div className="engine-list">{result.engines.map(engine => <button key={engine.id} className={selected?.id === engine.id ? 'selected' : ''} onClick={() => setSelectedId(engine.id)}><span className={`risk-dot ${engine.predicted_rul <= 25 ? 'danger' : engine.predicted_rul <= 60 ? 'warn' : ''}`} /><span><b>{displayEngineId(engine)}</b><small>cycle {engine.last_cycle} · {engine.missing_readings} missing</small></span><strong>{formatNumber(engine.predicted_rul, 0)} <small>RUL</small></strong></button>)}</div></div><EngineDetail engine={selected} trajectory={trajectory} /></div>}
    {tab === 'sensors' && <><div className="sensor-engine-selector"><label htmlFor="sensor-engine">Engine</label><select id="sensor-engine" value={selected?.id} onChange={e => setSelectedId(e.target.value)}>{result.engines.map(e => <option key={e.id} value={e.id}>{displayEngineId(e)}</option>)}</select></div><FaultReport key={selected.id} engine={selected} sourceFile={sourceFile} onExplanation={(id, data) => setResult(current => current ? ({ ...current, explanations: { ...current.explanations, [id]: data } }) : current)} /></>}
  </section>;
}

function EngineDetail({ engine, trajectory }) {
  if (!engine) return null;
  return <div className="engine-detail"><div className="panel detail-hero"><div className="detail-head"><div><span className="panel-kicker">SELECTED ENGINE</span><h3>{displayEngineId(engine)}</h3><p>Last observed cycle {engine.last_cycle} · operating regime {engine.operating_regime}</p></div><div className="rul-readout"><span>REMAINING LIFE</span><strong>{formatNumber(engine.predicted_rul, 1)} <small>cycles</small></strong><em>{formatNumber(engine.interval_lower, 0)}—{formatNumber(engine.interval_upper, 0)} interval</em></div></div><div className="detail-stats"><div><span>Failure estimate</span><b>cycle {formatNumber(engine.estimated_failure_cycle, 0)}</b></div><div><span>Availability</span><b>{formatNumber(engine.availability)}%</b></div><div><span>Local / global gate</span><b>{formatNumber(engine.local_gate * 100, 0)}% / {formatNumber(engine.global_gate * 100, 0)}%</b></div><div><span>MC-dropout spread</span><b>±{formatNumber(engine.mc_std_cycles, 1)} cycles</b></div></div></div><div className="panel chart-panel"><div className="panel-head"><div><span className="panel-kicker">ENGINE SIGNAL</span><h3>RUL projection over history</h3></div></div><div className="chart-wrap detail-chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={trajectory} margin={{ top: 12, right: 20, left: -10, bottom: 5 }}><defs><linearGradient id="engine-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff734d" stopOpacity=".28" /><stop offset="100%" stopColor="#ff734d" stopOpacity="0" /></linearGradient></defs><CartesianGrid stroke="#252528" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="cycle" stroke="#69696f" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} /><YAxis stroke="#69696f" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} /><Tooltip content={<ChartTooltip />} /><Area type="monotone" dataKey="predicted_rul" stroke="#ff734d" fill="url(#engine-fill)" strokeWidth={2} name="Predicted RUL" /><Line type="monotone" dataKey="interval_upper" stroke="#ffb36d" strokeOpacity={.5} dot={false} name="Upper interval" /><Line type="monotone" dataKey="interval_lower" stroke="#ffb36d" strokeOpacity={.5} dot={false} name="Lower interval" /></AreaChart></ResponsiveContainer></div></div></div>;
}

function App() {
  const [active, setActive] = useState('test');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [status, setStatus] = useState(null);
  const [modelData, setModelData] = useState(null);
  const [result, setResult] = useState(null);
  const [sourceFile, setSourceFile] = useState(null);
  async function refreshModel() {
    try {
      const [statusResponse, modelResponse] = await Promise.all([fetch('/api/status'), fetch('/api/model')]);
      if (!statusResponse.ok || !modelResponse.ok) throw new Error('Model service is unavailable.');
      setStatus(await statusResponse.json()); setModelData(await modelResponse.json());
    } catch (error) { const errorState = { ready: false, state: 'error', message: error.message }; setStatus(errorState); setModelData(errorState); }
  }
  useEffect(() => { refreshModel(); const timer = setInterval(refreshModel, 15000); return () => clearInterval(timer); }, []);
  function navigate(id) { setActive(id); setMobileOpen(false); window.scrollTo({ top: 0, behavior: 'smooth' }); }
  return <div className="app"><Topbar active={active} setActive={navigate} status={status} onMenu={() => setMobileOpen(!mobileOpen)} />{mobileOpen && <div className="mobile-nav">{navItems.map(item => <button key={item.id} onClick={() => navigate(item.id)}>{item.label}</button>)}</div>}{active === 'test' ? <><Hero setActive={navigate} status={status} /><main className="test-page"><section className="workspace section-shell" id="test"><div className="workspace-intro"><SectionLabel icon={ScanSearch}>TEST WORKSPACE</SectionLabel><h2>From file to forecast<br /><em>in one pass.</em></h2><p>Start with the raw history. GRACE-RUL handles windows, missing observations, repairs, uncertainty, and report-ready visualizations in a single analysis.</p><ModelSlot status={status} onRefresh={refreshModel} onDetails={() => navigate('model')} /><div className="workflow-list"><div><span>01</span><b>Upload</b><small>your test history</small></div><div><span>02</span><b>Analyze</b><small>the failure signal</small></div><div><span>03</span><b>Act</b><small>before the window closes</small></div></div></div><UploadPanel onResult={(data, file) => { setSourceFile(file); setResult(data); setTimeout(() => document.getElementById('results')?.scrollIntoView({ behavior: 'smooth' }), 80); }} modelStatus={status} /></section>{result && <ResultsWorkspace result={result} setResult={setResult} sourceFile={sourceFile} />}{!result && <section className="capability-section section-shell"><div className="section-topline"><div><SectionLabel icon={Sparkles}>BUILT FOR THE SIGNAL</SectionLabel><h2>Everything you need<br /><em>after the upload.</em></h2></div></div><div className="capability-grid"><Capability icon={Gauge} title="Remaining-life estimates" copy="Non-negative remaining-life estimates with a forecasted failure cycle for each engine." /><Capability icon={ShieldCheck} title="Measured uncertainty" copy="Empirical intervals and Monte Carlo dropout describe uncertainty within the model’s validation scope." /><Capability icon={Activity} title="Sensor forensics" copy="Inspect suspected faults, recovered readings, and SHAP/LIME evidence for individual sensors." /></div></section>}</main></> : <ModelWorkspace modelData={modelData || status} setActive={navigate} onRefresh={refreshModel} />}<footer className="footer section-shell"><div className="brand"><span className="brand-mark"><Flame size={17} fill="currentColor" /></span><span>GRACE<span className="brand-divider">/</span>RUL</span></div><span>Experimental predictive intelligence · local-first analysis</span><span>© 2026</span></footer></div>;
}

function Capability({ icon: Icon, title, copy }) { return <div className="capability-card"><span className="capability-icon"><Icon size={20} /></span><h3>{title}</h3><p>{copy}</p><ArrowRight size={16} /></div>; }

export default App;

createRoot(document.getElementById('root')).render(<App />);
