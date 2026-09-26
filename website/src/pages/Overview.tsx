import { api } from '../services/api';
import { useApp } from '../store/AppContext';
import { FleetRiskMap } from '../components/engine/FleetRiskMap';
import { RadialGauge, Sparkline } from '../components/charts/primitives';
import { AlertsTimeline } from '../components/alerts/AlertsTimeline';
import { statusColor } from '../data/generator';

export function Overview() {
  const { setSelectedEngineId, openTwin, aiResponse } = useApp();
  const f = api.fleetHealth();
  const engines = api.getEngines();
  const worst = [...engines].sort((a, b) => a.rul - b.rul).slice(0, 4);
  const healthTrend = engines.slice(0, 40).map((e) => e.health);

  return (
    <div className="grid gap-3 pb-24 lg:grid-cols-12">
      {/* HERO — fleet health (spans 7) */}
      <section className="surface relative overflow-hidden rounded-2xl p-6 lg:col-span-7" aria-label="Fleet health hero">
        <div className="grid-bg absolute inset-0 opacity-60" aria-hidden />
        <div className="relative flex flex-wrap items-center gap-6">
          <RadialGauge value={f.health} stroke={f.health > 75 ? '#34D399' : '#FBBF24'} label="FLEET HEALTH" />
          <div>
            <div className="font-mono text-[10px] tracking-[0.24em] text-muted2">AEROGUARD AI · AIRCRAFT ENGINE INTELLIGENCE</div>
            <h2 className="text-3xl font-extrabold tracking-tight">FLEET HEALTH <span className="text-emerald-300">{f.health}%</span></h2>
            <div className="mt-1 text-[15px] font-bold text-emerald-300">● NOMINAL</div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-4xl font-extrabold tabular-nums">{f.avgRul}</span>
              <span className="text-sm text-muted2">average remaining cycles</span>
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-muted2">
              <span className="text-emerald-300">+4.2% vs previous window</span>
              <span>{f.nominal} / 100 engines normal</span>
            </div>
            <div className="mt-2"><Sparkline values={healthTrend} width={260} height={36} stroke="#34D399" /></div>
          </div>
        </div>
      </section>

      {/* SYSTEM STATUS (spans 5) */}
      <section className="surface rounded-2xl p-5 lg:col-span-5" aria-label="System status detail">
        <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">SYSTEM STATUS</div>
        <div className="mt-2 space-y-2">
          {[['Telemetry stream', 'CONNECTED · 4.8k msgs/s', true], ['ML inference · LSTM v2.4', 'READY · 42ms', true], ['Prediction service', 'ONLINE · drift 0.8%', true]].map(([k, v, ok]) => (
            <div key={k as string} className="flex items-center justify-between rounded-xl bg-black/30 px-3 py-2.5">
              <span className="flex items-center gap-2 text-[13px]"><span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse-dot" aria-hidden />{k}</span>
              <span className="font-mono text-[10px] text-emerald-300">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <div className="rounded-xl bg-raised p-3"><div className="font-mono text-[10px] text-muted2">AVG RUL</div><div className="text-2xl font-extrabold tabular-nums">{f.avgRul} <span className="text-xs font-medium text-muted2">cycles</span></div></div>
          <div className="rounded-xl bg-raised p-3"><div className="font-mono text-[10px] text-muted2">HIGH RISK</div><div className="text-2xl font-extrabold tabular-nums text-amber-300">{f.highRisk} <span className="text-xs font-medium text-muted2">engines</span></div></div>
        </div>
      </section>

      {/* RISK MAP (spans 8) */}
      <section className="surface rounded-2xl p-5 lg:col-span-8" aria-label="Fleet risk map">
        <div className="mb-1 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">FLEET RISK MAP · CLICK NODE TO INSPECT</span>
        </div>
        <FleetRiskMap compact />
      </section>

      {/* ATTENTION QUEUE (spans 4) */}
      <section className="surface rounded-2xl p-5 lg:col-span-4" aria-label="Engines needing attention">
        <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">REQUIRES ATTENTION</div>
        <div className="mt-2 space-y-2">
          {worst.map((e) => (
            <button key={e.id} onClick={() => setSelectedEngineId(e.id)} className="flex w-full items-center gap-3 rounded-xl border border-white/[0.06] bg-raised p-2.5 text-left transition hover:border-white/20" aria-label={`Inspect ${e.label}`}>
              <span className="h-8 w-1 rounded-full" style={{ background: statusColor(e.status) }} aria-hidden />
              <span>
                <span className="block font-mono text-[13px] font-bold">{e.label}</span>
                <span className="block text-[11px] text-muted2">RUL {e.rul} · Risk {e.risk}% · {e.status}</span>
              </span>
              <span className="ml-auto font-mono text-[11px] text-muted2">→</span>
            </button>
          ))}
        </div>
        <button onClick={() => openTwin(24)} className="mt-3 w-full rounded-xl bg-white py-2 text-[13px] font-bold text-black hover:bg-cyan-200">Open Engine #024 Twin →</button>
      </section>

      {/* DEGRADATION TIMELINE strip (spans 12) */}
      <section className="surface rounded-2xl p-5 lg:col-span-12" aria-label="Degradation timeline preview">
        <div className="mb-2 flex items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">DEGRADATION TIMELINE · FLEET TRAJECTORIES</span>
          <span className="ml-auto font-mono text-[10px] text-muted2">solid = history · dashed = predicted</span>
        </div>
        <svg viewBox="0 0 900 140" className="w-full" role="img" aria-label="Fleet degradation trajectories">
          {worst.map((e, i) => {
            const s = api.getSeries(e.id)!;
            const hist = s.points.filter((_, k) => k % 4 === 0);
            const vals = hist.map((p) => p.health);
            const min = 20, max = 100;
            const d = vals.map((v, k) => `${k === 0 ? 'M' : 'L'}${(30 + (k / Math.max(1, vals.length - 1)) * 700).toFixed(1)},${(14 + (1 - (v - min) / (max - min)) * 100).toFixed(1)}`).join(' ');
            const y = 14 + (1 - (e.health - min) / (max - min)) * 100;
            return (
              <g key={e.id} onClick={() => setSelectedEngineId(e.id)} className="cursor-pointer">
                <path d={d} fill="none" stroke={statusColor(e.status)} strokeWidth={2} opacity={0.9} />
                <circle cx={730} cy={y} r={4} fill={statusColor(e.status)} stroke="#08090B" />
                <text x={742} y={y + 4} fill="#E6E9EE" fontSize={11} fontFamily="JetBrains Mono">#{String(e.id).padStart(3, '0')} RUL {e.rul}</text>
              </g>
            );
          })}
          <line x1={30} y1={124} x2={730} y2={124} stroke="rgba(255,255,255,0.12)" />
        </svg>
      </section>

      {/* ALERTS strip */}
      <section className="lg:col-span-7"><AlertsTimeline /></section>
      <section className="surface h-fit rounded-2xl p-5 lg:col-span-5" aria-label="Journey hint">
        <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">OBSERVE → UNDERSTAND → INVESTIGATE → PREDICT → SIMULATE</div>
        <p className="mt-2 text-[13px] leading-relaxed text-white/80">Select any engine node to open the <b>Engine Explorer</b> without leaving this view. Then escalate to the <b>Digital Twin</b> for scrubbing, telemetry forensics, prediction rationale and what-if simulation.</p>
        {aiResponse && <div className="mt-3 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.06] p-3 text-[13px] text-cyan-100" role="status">✦ {aiResponse}</div>}
      </section>
    </div>
  );
}
