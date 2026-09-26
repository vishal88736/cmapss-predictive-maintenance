import { useMemo, useState } from 'react';
import { Pin } from 'lucide-react';
import { SENSORS } from '../../data/generator';
import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';
import { SensorInsight } from './SensorInsight';

const COLORS: Record<string, string> = { T24: '#FBBF24', P30: '#22D3EE', S11: '#A78BFA', VIB: '#F87171', Nc: '#34D399' };

export function TelemetryCanvas({ engineId }: { engineId: number }) {
  const { primarySensor, setPrimarySensor, pinned, togglePin, scrubCycle } = useApp();
  const [inspectCycle, setInspectCycle] = useState<number | null>(null);
  const series = api.getSeries(engineId);
  const engine = api.getEngine(engineId);
  const def = SENSORS.find((s) => s.key === primarySensor) ?? SENSORS[0];

  const { pathMain, overlayPaths, anomalyPts, curVal, stats } = useMemo(() => {
    if (!series) return { pathMain: '', overlayPaths: [] as string[], anomalyPts: [] as number[], curVal: 0, stats: { dev: 0 } };
    const pts = series.points;
    const W = 760, H = 220, PL = 46, PR = 12, PT = 12, PB = 24;
    const build = (key: string) => {
      const vals = pts.map((p) => p.values[key]);
      const min = Math.min(...vals) * 0.998, max = Math.max(...vals) * 1.002;
      const X = (i: number) => PL + (i / Math.max(1, pts.length - 1)) * (W - PL - PR);
      const Y = (v: number) => PT + (1 - (v - min) / Math.max(1e-6, max - min)) * (H - PT - PB);
      return { d: pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${X(i).toFixed(1)},${Y(p.values[key]).toFixed(1)}`).join(' '), X, Y, min, max, W, H, PL, PB, PT, PR };
    };
    const main = build(primarySensor);
    const overlays = pinned.filter((p) => p !== primarySensor).slice(0, 2).map((p) => {
      const b = build(p);
      // normalize overlay into main's frame for comparison readability
      return b.d;
    });
    const effCycle = scrubCycle ?? engine?.currentCycle ?? pts.length;
    const cur = pts[Math.min(pts.length - 1, effCycle - 1)];
    const mid = (def.expected[0] + def.expected[1]) / 2;
    const dev = ((cur.values[primarySensor] - mid) / mid) * 100;
    return { pathMain: main.d, overlayPaths: overlays, anomalyPts: pts.filter((p) => p.anomaly).map((p) => main.X(p.cycle - 1)), curVal: cur.values[primarySensor], stats: { dev }, geom: main };
  }, [series, primarySensor, pinned, scrubCycle, engine, def]);

  if (!series || !engine) return null;
  const effCycle = scrubCycle ?? engine.currentCycle;

  return (
    <div className="surface rounded-2xl p-4">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">PRIMARY TELEMETRY CANVAS</span>
        <div className="relative ml-2">
          <select value={primarySensor} onChange={(e) => setPrimarySensor(e.target.value)} aria-label="Select primary sensor" className="appearance-none rounded-lg border border-line bg-raised py-1.5 pl-3 pr-8 text-[13px] font-bold outline-none">
            {SENSORS.map((s) => <option key={s.key} value={s.key}>{s.name} ({s.key})</option>)}
          </select>
          <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-muted2">▼</span>
        </div>
        <span className="ml-auto font-mono text-[11px] text-muted2">CYCLE {effCycle} · PINS {pinned.length}/3</span>
      </div>
      <div className="flex items-baseline gap-3">
        <span className="text-4xl font-extrabold tabular-nums">{curVal.toFixed(def.key === 'VIB' ? 3 : 1)} <span className="text-base font-semibold text-muted2">{def.unit}</span></span>
        <span className={`rounded-full px-2 py-0.5 font-mono text-[11px] font-bold ${stats.dev > 1 ? 'bg-amber-400/15 text-amber-300' : 'bg-emerald-400/10 text-emerald-300'}`}>
          {stats.dev >= 0 ? '+' : ''}{stats.dev.toFixed(1)}% deviation
        </span>
      </div>
      <svg viewBox="0 0 760 220" className="mt-2 w-full cursor-crosshair" role="img" aria-label={`${def.name} telemetry chart`}
        onClick={(e) => {
          const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const frac = (e.clientX - rect.left) / rect.width;
          setInspectCycle(Math.max(1, Math.min(series.points.length, Math.round(frac * series.points.length))));
        }}>
        {[0.2, 0.5, 0.8].map((f) => <line key={f} x1={46} y1={220 * f} x2={748} y2={220 * f} stroke="rgba(255,255,255,0.06)" />)}
        {/* expected band */}
        <rect x={46} y={60} width={702} height={60} fill="rgba(52,211,153,0.07)" />
        <text x={52} y={72} className="tick-label">EXPECTED {def.expected[0]}–{def.expected[1]}</text>
        {overlayPaths.map((d, i) => <path key={i} d={d} fill="none" stroke={COLORS[pinned.filter((p) => p !== primarySensor)[i]] ?? '#A78BFA'} strokeWidth={1.4} opacity={0.75} strokeDasharray={i === 0 ? undefined : '4 3'} />)}
        <path d={pathMain} fill="none" stroke={COLORS[primarySensor] ?? '#E6E9EE'} strokeWidth={2.4} strokeLinecap="round" />
        {anomalyPts.map((x, i) => <circle key={i} cx={x} cy={118} r={4} fill="#F87171" stroke="#08090B" strokeWidth={1.5} />)}
        {/* scrub marker */}
        <line x1={46 + ((effCycle - 1) / Math.max(1, series.points.length - 1)) * 702} y1={8} x2={46 + ((effCycle - 1) / Math.max(1, series.points.length - 1)) * 702} y2={196} stroke="#fff" strokeWidth={1.4} />
      </svg>
      <div className="mt-2 flex flex-wrap gap-1.5" role="group" aria-label="Sensor selection">
        {SENSORS.slice(0, 10).map((s) => {
          const active = s.key === primarySensor;
          const pin = pinned.includes(s.key);
          return (
            <span key={s.key} className={`flex items-center gap-1 rounded-full border py-1 pl-3 pr-1.5 text-[12px] ${active ? 'border-white/40 bg-white/10 text-white' : 'border-line text-muted2'}`}>
              <button onClick={() => setPrimarySensor(s.key)} className="font-semibold" aria-pressed={active}>{s.short}</button>
              <button onClick={() => togglePin(s.key)} aria-label={`${pin ? 'Unpin' : 'Pin'} ${s.name} overlay`} aria-pressed={pin} className={pin ? 'text-cyan-300' : 'text-muted2/60 hover:text-white'}>
                <Pin size={12} fill={pin ? 'currentColor' : 'none'} />
              </button>
            </span>
          );
        })}
      </div>
      <div className="mt-2 font-mono text-[10px] text-muted2">Click chart to inspect a cycle · pin up to 3 overlays · model overlays are normalized for comparison</div>
      {inspectCycle && <SensorInsight engineId={engineId} cycle={inspectCycle} sensor={primarySensor} onClose={() => setInspectCycle(null)} />}
    </div>
  );
}
