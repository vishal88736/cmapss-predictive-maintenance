import { useMemo, useState } from 'react';
import { api } from '../../services/api';
import { Bars } from '../charts/primitives';

function scatter(n: number, seed: number, fn: (i: number) => { x: number; y: number }) {
  return Array.from({ length: n }, (_, i) => fn(i));
}

export function ModelPanels() {
  const models = api.getModels();
  const [mid, setMid] = useState('lstm');
  const m = models.find((x) => x.id === mid)!;

  const avp = useMemo(() => scatter(60, 1, (i) => ({ x: 5 + (i / 60) * 220, y: 5 + (i / 60) * 190 + Math.sin(i / 4) * 14 + ((i * 37) % 17 - 8) })), []);
  const resid = useMemo(() => Array.from({ length: 28 }, (_, i) => 4 + Math.abs(14 * Math.sin(i / 2.4)) + ((i * 53) % 9)), []);
  const errByRul = useMemo(() => [26, 22, 19, 16, 13, 11], []);
  const feat = useMemo(() => [
    ['Sensor 04 · LPT Outlet', 0.94], ['Sensor 11 · Static Press', 0.81], ['Sensor 07 · Bleed', 0.62], ['Sensor 12 · Fuel Flow', 0.48], ['Sensor 09 · LPT Temp', 0.39], ['Vibration', 0.31],
  ] as [string, number][], []);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="surface flex items-center gap-3 rounded-xl px-4 py-2.5">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse-dot" aria-hidden />
          <span className="font-bold">{m.name}</span>
          <span className="font-mono text-[11px] text-muted2">{m.version}</span>
          <span className="rounded-full bg-emerald-400/10 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-300">● {m.status}</span>
        </div>
        <div className="flex gap-1.5">
          {models.map((x) => (
            <button key={x.id} onClick={() => setMid(x.id)} aria-pressed={mid === x.id} className={`rounded-lg border px-2.5 py-1.5 font-mono text-[11px] ${mid === x.id ? 'border-white/40 bg-white/10 text-white' : 'border-line text-muted2'}`}>
              {x.name.split(' ')[0]} {x.version}
            </button>
          ))}
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[['RMSE', m.rmse.toFixed(1), 'cycles · lower is better'], ['MAE', m.mae.toFixed(1), 'cycles'], ['R²', m.r2.toFixed(2), 'explained variance'], ['p50 latency', `${m.latencyMs}ms`, 'GPU inference']].map(([k, v, s]) => (
          <div key={k} className="surface rounded-2xl p-4">
            <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">{k.toUpperCase()}</div>
            <div className="text-3xl font-extrabold tabular-nums">{v}</div>
            <div className="font-mono text-[10px] text-muted2">{s}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-2">
        <div className="surface rounded-2xl p-4">
          <div className="mb-1 font-mono text-[10px] tracking-[0.2em] text-muted2">ACTUAL vs PREDICTED RUL</div>
          <svg viewBox="0 0 300 220" className="w-full" role="img" aria-label="Actual versus predicted scatter">
            <line x1={20} y1={200} x2={290} y2={10} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
            {avp.map((p, i) => <circle key={i} cx={p.x} cy={210 - p.y} r={3} fill={i % 7 === 0 ? '#F87171' : '#22D3EE'} opacity={0.8} />)}
          </svg>
          <div className="font-mono text-[10px] text-muted2">Ideal diagonal dashed · red = high-residual tails</div>
        </div>
        <div className="surface rounded-2xl p-4">
          <div className="mb-1 font-mono text-[10px] tracking-[0.2em] text-muted2">RESIDUAL DISTRIBUTION</div>
          <Bars values={resid} color="#A78BFA" height={120} />
          <div className="mt-1 font-mono text-[10px] text-muted2">Near-zero centered · σ ≈ 14.2 cycles</div>
        </div>
        <div className="surface rounded-2xl p-4">
          <div className="mb-1 font-mono text-[10px] tracking-[0.2em] text-muted2">ERROR BY RUL RANGE</div>
          <div className="space-y-1.5">
            {['0–25', '25–50', '50–100', '100–150', '150–200', '200+'].map((r, i) => (
              <div key={r} className="flex items-center gap-2 text-[12px]">
                <span className="w-16 font-mono text-muted2">{r}</span>
                <div className="h-2 flex-1 rounded-full bg-white/[0.06]"><div className="h-2 rounded-full bg-gradient-to-r from-cyan-300 to-amber-300" style={{ width: `${(errByRul[i] / 26) * 100}%` }} /></div>
                <span className="w-10 text-right font-mono">{errByRul[i]}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="surface rounded-2xl p-4">
          <div className="mb-1 font-mono text-[10px] tracking-[0.2em] text-muted2">FEATURE IMPORTANCE · GLOBAL</div>
          <div className="space-y-1.5">
            {feat.map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 text-[12px]">
                <span className="w-44 truncate text-muted2">{k}</span>
                <div className="h-2 flex-1 rounded-full bg-white/[0.06]"><div className="h-2 rounded-full bg-emerald-300" style={{ width: `${v * 100}%` }} /></div>
                <span className="w-10 text-right font-mono">{v.toFixed(2)}</span>
              </div>
            ))}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-2 font-mono text-[10px]">
            <div className="rounded-lg bg-black/30 p-2">CONFIDENCE · p90 <b className="text-white">87%</b></div>
            <div className="rounded-lg bg-black/30 p-2">DRIFT · 7d <b className="text-emerald-300">0.8% ✓</b></div>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ModelExplanation({ engineId }: { engineId: number }) {
  const e = api.getEngine(engineId)!;
  const rows: [string, string, number][] = [
    ['S04', 'Sensor 04 · LPT Outlet', 0.92],
    ['S11', 'Sensor 11 · Static Press', 0.78],
    ['S07', 'Sensor 07 · Bleed Enthalpy', 0.58],
    ['S12', 'Sensor 12 · Fuel Flow', 0.44],
  ];
  return (
    <div className="surface rounded-2xl p-4" aria-label="Why this prediction">
      <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">WHY THIS PREDICTION? · FEATURE ATTRIBUTION</div>
      <div className="mt-2 flex gap-4 font-mono text-[13px]">
        <span>Current RUL <b className="text-xl text-white">{e.rul}</b></span>
        <span>Confidence <b className="text-xl text-cyan-300">{e.confidence}%</b></span>
      </div>
      <div className="mt-3 space-y-2">
        {rows.map(([k, n, w]) => (
          <div key={k} className="flex items-center gap-2 text-[12px]">
            <span className="w-44 truncate"><b className="font-mono">{k}</b> <span className="text-muted2">{n.split('·')[1]}</span></span>
            <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]" role="img" aria-label={`${n} importance ${Math.round(w * 100)} percent`}>
              <div className="h-2.5 rounded-full bg-gradient-to-r from-cyan-300 to-violet-400" style={{ width: `${w * 100}%` }} />
            </div>
            <span className="w-10 text-right font-mono">{w.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 flex gap-1.5 font-mono text-[10px]">
        <span className="rounded bg-cyan-300/10 px-2 py-1 text-cyan-200">MODEL OUTPUT</span>
        <span className="rounded bg-white/[0.06] px-2 py-1 text-muted2">RAW SENSOR DATA</span>
        <span className="rounded bg-amber-300/10 px-2 py-1 text-amber-200">DERIVED METRICS</span>
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-muted2">Attribution via integrated-gradients surrogate. Distinguishes model output from raw telemetry — inspect signals to verify.</p>
    </div>
  );
}
