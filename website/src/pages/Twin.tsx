import { useState } from 'react';
import { Maximize2, GitCompareArrows, FlaskConical, Presentation } from 'lucide-react';
import { api } from '../services/api';
import { statusColor } from '../data/generator';
import { useApp } from '../store/AppContext';
import { DegradationTimeline } from '../components/twin/DegradationTimeline';
import { TimelineScrubber } from '../components/twin/TimelineScrubber';
import { TelemetryCanvas } from '../components/telemetry/TelemetryCanvas';
import { AIInsight } from '../components/ai/AIInsight';
import { Simulator } from '../components/simulation/Simulator';
import { ModelExplanation } from '../components/model/ModelPanels';
import { Comparison } from '../components/compare/Comparison';

export function Twin() {
  const { twinEngineId, setTwinEngineId, scrubCycle, setFocusMode, setPresentationMode, focusMode } = useApp();
  const [tab, setTab] = useState<'analyze' | 'simulate' | 'compare'>('analyze');
  const engine = api.getEngine(twinEngineId)!;
  const series = api.getSeries(twinEngineId)!;
  const effCycle = scrubCycle ?? engine.currentCycle;
  const pt = series.points[Math.min(series.points.length - 1, effCycle - 1)];
  // interpolated RUL at scrub position
  const rulAt = effCycle <= engine.currentCycle ? engine.rul + (engine.currentCycle - effCycle) : Math.max(0, engine.rul - (effCycle - engine.currentCycle));
  const c = statusColor(engine.status);

  return (
    <div className={`grid gap-3 pb-24 ${focusMode ? '' : 'xl:grid-cols-12'}`}>
      <div className={focusMode ? '' : 'xl:col-span-12'}>
        {/* Twin header */}
        <header className="surface flex flex-wrap items-center gap-4 rounded-2xl p-4" aria-label="Twin header">
          <div>
            <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">FOCUSED ENGINE · DIGITAL TWIN</div>
            <h2 className="flex items-center gap-2 text-2xl font-extrabold tracking-tight">
              ENGINE #{String(engine.id).padStart(3, '0')}
              <span className="rounded-full px-2.5 py-0.5 text-[11px] font-bold" style={{ background: c + '1e', color: c, border: `1px solid ${c}55` }}>{engine.status}</span>
            </h2>
          </div>
          <div className="flex gap-5 font-mono text-[12px]">
            <span>RUL<br /><b className="text-2xl text-white">{rulAt}</b> <span className="text-muted2">CYC</span></span>
            <span>RISK<br /><b className="text-2xl" style={{ color: c }}>{engine.risk}%</b></span>
            <span className="hidden sm:block">CYCLE<br /><b className="text-2xl text-white">{effCycle}</b></span>
            <span className="hidden sm:block">HEALTH<br /><b className="text-2xl text-white">{pt.health.toFixed(0)}%</b></span>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <select value={twinEngineId} onChange={(e) => { setTwinEngineId(Number(e.target.value)); }} className="rounded-lg border border-line bg-raised px-2.5 py-1.5 font-mono text-[12px] font-bold" aria-label="Switch twin engine">
              {api.getEngines().map((e) => <option key={e.id} value={e.id}>#{String(e.id).padStart(3, '0')} · RUL {e.rul}</option>)}
            </select>
            <button onClick={() => setFocusMode(true)} className="flex items-center gap-1 rounded-lg border border-line px-2.5 py-1.5 text-[12px] text-muted2 hover:text-white" title="Focus mode (F)"><Maximize2 size={13} />Focus</button>
            <button onClick={() => setPresentationMode(true)} className="flex items-center gap-1 rounded-lg border border-line px-2.5 py-1.5 text-[12px] text-muted2 hover:text-white"><Presentation size={13} />Present</button>
          </div>
        </header>
        {/* Engine schematic strip */}
        <div className="surface mt-3 flex items-center gap-4 overflow-x-auto rounded-2xl p-4" aria-label="Engine schematic">
          <EngineSchematic temp={pt.values['T24']} />
          <div className="grid min-w-[220px] flex-1 grid-cols-2 gap-2 sm:grid-cols-4">
            {[['T24', 'Temperature', '°K'], ['P30', 'Pressure', 'psia'], ['Nc', 'Core', 'rpm'], ['VIB', 'Vibration', 'in/s']].map(([k, n, u]) => (
              <div key={k} className="rounded-xl bg-black/30 p-2.5">
                <div className="font-mono text-[10px] text-muted2">{n}</div>
                <div className="font-mono text-[15px] font-bold tabular-nums">{pt.values[k].toFixed(k === 'VIB' ? 3 : 1)} <span className="text-[10px] font-medium text-muted2">{u}</span></div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className={focusMode ? '' : 'xl:col-span-8'}>
        <DegradationTimeline engineId={twinEngineId} scrub={scrubCycle} />
        <div className="mt-3"><TimelineScrubber /></div>
        <div className="mt-3"><TelemetryCanvas engineId={twinEngineId} /></div>
      </div>

      <div className={focusMode ? '' : 'xl:col-span-4'}>
        <div className="flex gap-1.5 rounded-xl border border-line bg-panel p-1 text-[12px]" role="tablist" aria-label="Twin analysis tabs">
          {([['analyze', 'Analyze'], ['simulate', 'Simulate'], ['compare', 'Compare']] as const).map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id} onClick={() => setTab(id)} className={`flex flex-1 items-center justify-center gap-1 rounded-lg px-2 py-1.5 font-semibold ${tab === id ? 'bg-white/10 text-white' : 'text-muted2'}`}>
              {id === 'analyze' ? null : id === 'simulate' ? <FlaskConical size={13} /> : <GitCompareArrows size={13} />}{label}
            </button>
          ))}
        </div>
        <div className="mt-3 space-y-3">
          {tab === 'analyze' && (
            <>
              <AIInsight engineId={twinEngineId} />
              <ModelExplanation engineId={twinEngineId} />
            </>
          )}
          {tab === 'simulate' && <Simulator engineId={twinEngineId} />}
          {tab === 'compare' && <Comparison />}
          {tab !== 'compare' && <div className="opacity-90"><Comparison /></div>}
        </div>
      </div>
    </div>
  );
}

function EngineSchematic({ temp }: { temp: number }) {
  const heat = Math.min(1, Math.max(0, (temp - 628) / 30));
  return (
    <svg viewBox="0 0 300 90" className="h-[86px] w-[260px] shrink-0" role="img" aria-label="Stylized turbofan cross-section">
      <defs>
        <linearGradient id="engHeat" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#22D3EE" />
          <stop offset={`${Math.round(heat * 100)}%`} stopColor={heat > 0.6 ? '#FBBF24' : '#34D399'} />
          <stop offset="100%" stopColor={heat > 0.6 ? '#F87171' : '#34D399'} />
        </linearGradient>
      </defs>
      <ellipse cx={60} cy={45} rx={26} ry={30} fill="none" stroke="#5b6673" strokeWidth={3} />
      <ellipse cx={60} cy={45} rx={10} ry={12} fill="none" stroke="#8b949e" strokeWidth={2} />
      <rect x={86} y={22} width={150} height={46} rx={10} fill="none" stroke="url(#engHeat)" strokeWidth={3} />
      {[110, 135, 160, 185].map((x) => <line key={x} x1={x} y1={26} x2={x} y2={64} stroke="rgba(255,255,255,0.25)" />)}
      <path d="M236,30 L282,45 L236,60" fill="none" stroke="#8b949e" strokeWidth={2.5} />
      <text x={150} y={86} textAnchor="middle" fill="#6b7684" fontSize={9} fontFamily="JetBrains Mono">TURBOFAN · CORE TEMP {temp.toFixed(1)}°K</text>
    </svg>
  );
}
