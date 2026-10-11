import { useEffect, useRef, useState } from 'react';
import { ScanSearch, LoaderCircle, AlertCircle } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ReferenceLine } from 'recharts';

const fmt = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(3);

export default function FaultReport({ engine, sourceFile, onExplanation }) {
  const [sensor, setSensor] = useState(engine?.sensors?.find(s => s.status.includes('suspected'))?.sensor || engine?.sensors?.find(s => s.availability > 0)?.sensor || 'sensor_3');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [explanation, setExplanation] = useState(null);
  const request = useRef(null);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => { setExplanation(null); setError(''); }, [sensor]);
  if (!engine) return null;
  const index = Number(sensor.replace('sensor_', '')) - 1;
  const selected = engine.sensors[index];
  const chart = engine.sensor_history.map(point => ({ cycle: point.cycle, observed: point.observed[index], recovered: point.repaired[index] }));
  async function explain() {
    if (!sourceFile) return;
    setBusy(true); setError('');
    request.current = new AbortController();
    const form = new FormData(); form.append('file', sourceFile); form.append('subset', engine.subset); form.append('unit', engine.unit); form.append('sensor', index + 1);
    try {
      const response = await fetch('/api/explain', { method: 'POST', body: form, signal: request.current.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Explanation failed.');
      setExplanation(data); onExplanation?.(engine.id, data);
    } catch (err) { if (err.name !== 'AbortError') setError(err.message); }
    finally { setBusy(false); }
  }
  return <div className="fault-workspace"><section className="panel"><div className="panel-head"><div><span className="panel-kicker">SENSOR FAULT REGISTER</span><h3>{engine.suspected_sensors} sensors with suspected finite faults</h3></div><span className="spec-chip">{engine.missing_readings} missing readings</span></div><p className="scope-note">Flags combine observed missingness and validation-thresholded detector scores. They describe suspected faults.</p><div className="fault-table-wrap"><table className="fault-table"><thead><tr><th>Sensor</th><th>Condition</th><th>Peak score / threshold</th><th>Affected cycles</th><th>Repair magnitude*</th></tr></thead><tbody>{engine.sensors.map(item => <tr key={item.sensor} className={item.sensor === sensor ? 'selected' : ''} onClick={() => { if (!busy) setSensor(item.sensor); }}><td><button disabled={busy} onClick={() => setSensor(item.sensor)}>{item.sensor.replace('sensor_', 'S')} · {item.name}</button></td><td><span className={`fault-status ${item.status === 'no flag' ? '' : 'flagged'}`}>{item.status}</span></td><td>{fmt(item.fault_score)} / {fmt(item.threshold)}</td><td>{[...item.fault_spans, ...item.missing_spans].map(span => `${span.start}–${span.end}`).join(', ') || '—'}</td><td>{fmt(item.repair_magnitude)}</td></tr>)}</tbody></table></div><p className="scope-note">*Mean change in normalized sensor space. Select a sensor to inspect its readings.</p></section>
    <section className="panel"><div className="panel-head"><div><span className="panel-kicker">OBSERVED / RECOVERED</span><h3>{selected.name} · {sensor.replace('sensor_', 'S')}</h3></div><select aria-label="Sensor to inspect" value={sensor} disabled={busy} onChange={e => setSensor(e.target.value)}>{engine.sensors.map(s => <option key={s.sensor} value={s.sensor}>{s.sensor.replace('sensor_', 'S')} · {s.name}</option>)}</select></div><div className="chart-wrap detail-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><CartesianGrid stroke="var(--line)" /><XAxis dataKey="cycle" tick={{ fontSize: 10 }} /><YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} /><Tooltip /><Line dataKey="observed" name="Observed" stroke="var(--amber)" dot={false} connectNulls={false} /><Line dataKey="recovered" name="Recovered" stroke="var(--cyan)" dot={false} /></LineChart></ResponsiveContainer></div><p className="scope-note">Readings use the original dataset's sensor units. Missing observations appear as gaps.</p></section>
    <section className="panel explanation-panel"><div className="panel-head"><div><span className="panel-kicker">SHAP + LIME / SENSOR EVIDENCE</span><h3>Explain this sensor and its RUL influence</h3></div><button className="button button-primary" disabled={busy || !sourceFile || selected.availability === 0} onClick={explain}>{busy ? <LoaderCircle className="spin" size={15} /> : <ScanSearch size={15} />}{busy ? 'Computing explanations…' : 'Run SHAP + LIME'}</button></div><p>Explanations rerun the model with coherent sensor/time perturbations. They explain the selected sensor's estimated fault score and the final RUL separately.</p>{selected.availability === 0 && <p className="scope-note">This sensor is entirely missing. Its outage is directly observable; select an available channel to explain finite-fault evidence.</p>}{error && <div className="error-message"><AlertCircle size={16} />{error}</div>}
      {explanation ? <div className="explanation-grid">{explanation.targets.map(target => <div key={target.target} className="explanation-target"><h4>{target.target === 'final_rul' ? 'Final RUL' : `${target.sensor.replace('sensor_', 'S')} finite-fault score`}</h4><p className="scope-note">{target.target === 'final_rul' ? 'Predicted RUL' : 'Mean available-cycle fault score'}: {fmt(target.prediction)} {target.target === 'final_rul' ? 'cycles' : ''}<br />LIME held-out fidelity: {fmt(target.lime_holdout_fidelity)} · SHAP units: {target.units}</p><div className="chart-wrap model-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={target.sensors.slice(0, 8).map(s => ({ ...s, label: s.sensor.replace('sensor_', 'S') }))} layout="vertical" margin={{ left: 0, right: 12 }}><XAxis type="number" tick={{ fontSize: 10 }} /><YAxis type="category" dataKey="label" width={40} tick={{ fontSize: 11 }} /><Tooltip /><ReferenceLine x={0} stroke="var(--muted)" /><Bar dataKey="shap" name="SHAP contribution" fill="var(--blue)" /></BarChart></ResponsiveContainer></div><table className="fault-table"><thead><tr><th>Evidence sensor</th><th>SHAP</th><th>LIME</th></tr></thead><tbody>{target.sensors.slice(0, 8).map(item => <tr key={item.sensor}><td>{item.sensor.replace('sensor_', 'S')} · {item.name}</td><td>{fmt(item.shap)}</td><td>{fmt(item.lime)}</td></tr>)}</tbody></table></div>)}</div> : <div className="chart-placeholder"><ScanSearch size={30} /><p>SHAP and LIME evidence appears here after you run an explanation.</p></div>}
    </section>
  </div>;
}
