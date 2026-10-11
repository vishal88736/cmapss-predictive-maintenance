import { Cpu, RefreshCw, CheckCircle2, AlertCircle, ArrowRight } from 'lucide-react';

export default function ModelSlot({ status, onRefresh, onDetails }) {
  const ready = status?.ready;
  const quick = status?.mode === 'quick';
  const error = status?.state === 'error';
  return <section className={`model-slot ${ready ? 'connected' : ''}`} aria-live="polite">
    <div className="model-slot-title"><span className="model-slot-icon"><Cpu size={23} /></span><div><span className="panel-kicker">YOUR MODEL</span><h3>{ready ? (quick ? 'Quick validation model' : 'Trained model connected') : error ? 'Model connection needs attention' : 'Ready for your trained model'}</h3></div>{ready ? <CheckCircle2 size={20} /> : error ? <AlertCircle size={20} /> : <span className="placeholder-badge">PLACEHOLDER</span>}</div>
    <p>{status?.message || 'Checking the local model service…'}</p>
    <div className="slot-stages"><span>Detect faults</span><ArrowRight size={12} /><span>Recover sensors</span><ArrowRight size={12} /><span>Predict RUL</span><ArrowRight size={12} /><span>SHAP + LIME</span></div>
    {!ready && <p className="slot-note">Upload and inspect your data now. Prediction and explanations become available after full notebook training finishes.</p>}
    {ready && <div className="slot-meta"><span>{status.bundle_name}</span><span>{status.supported_subsets?.join(' · ')}</span></div>}
    <div className="slot-actions"><button className="button button-quiet" onClick={onRefresh}><RefreshCw size={14} /> Refresh model</button><button className="button button-quiet" onClick={onDetails}>Model details <ArrowRight size={14} /></button></div>
  </section>;
}
