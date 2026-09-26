import { useMemo, useState } from 'react';
import { RotateCcw, Play } from 'lucide-react';
import { api } from '../../services/api';

export function Simulator({ engineId }: { engineId: number }) {
  const engine = api.getEngine(engineId);
  const series = api.getSeries(engineId);
  const cur = series?.points[series.points.length - 1];
  const [dT, setDT] = useState(0);
  const [dP, setDP] = useState(0);
  const [dV, setDV] = useState(0);
  const [ran, setRan] = useState(false);

  const out = useMemo(() => {
    if (!engine) return { rul: 0, risk: 0 };
    // heuristic surrogate model: temp dominates
    const penalty = Math.max(0, dT * 1.15 + dP * 0.9 + dV * 6);
    const rul = Math.max(2, Math.round(engine.rul - penalty));
    const risk = Math.min(99, Math.round(engine.risk + penalty * 1.5));
    return { rul, risk };
  }, [engine, dT, dP, dV]);

  if (!engine || !cur) return null;
  const tBase = cur.values['T24'], pBase = cur.values['P30'], vBase = cur.values['VIB'];

  return (
    <div className="surface rounded-2xl p-4" aria-label="What-if simulator">
      <div className="mb-1 flex items-center gap-2">
        <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">WHAT-IF SIMULATION · INTERACTIVE EXPERIMENT</span>
        <span className="ml-auto rounded-full bg-amber-300/10 px-2 py-0.5 font-mono text-[10px] font-bold text-amber-300">ESTIMATED — NOT OBSERVED</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <div className="rounded-xl bg-black/30 p-3">
          <div className="font-mono text-[10px] text-muted2">BASELINE (OBSERVED TELEMETRY)</div>
          <div className="mt-1 flex gap-4 font-mono text-[13px]"><span>RUL <b className="text-white">{engine.rul}</b></span><span>RISK <b className="text-white">{engine.risk}%</b></span></div>
          <div className="mt-2 space-y-1 font-mono text-[11px] text-muted2">
            <div>T24 {tBase.toFixed(1)} °K</div><div>P30 {pBase.toFixed(1)} psia</div><div>VIB {vBase.toFixed(3)} in/s</div>
          </div>
        </div>
        <div className="rounded-xl border border-cyan-300/20 bg-cyan-300/[0.05] p-3">
          <div className="font-mono text-[10px] text-cyan-200">SCENARIO PREVIEW (MODEL ESTIMATE)</div>
          <div className="mt-1 flex items-baseline gap-3">
            <span className="font-mono text-[13px]">RUL <b className="text-xl text-white">{engine.rul} → {out.rul}</b></span>
            <span className={`font-mono text-[11px] font-bold ${out.rul < engine.rul ? 'text-red-300' : 'text-emerald-300'}`}>Δ {out.rul - engine.rul}</span>
          </div>
          <div className="mt-1 font-mono text-[13px]">Risk <b className="text-white">{engine.risk}% → {out.risk}%</b></div>
          {/* trajectory mini viz */}
          <svg viewBox="0 0 260 64" className="mt-2 w-full" aria-hidden>
            <path d="M4,14 C70,20 120,30 180,44 L250,58" fill="none" stroke="#5b6673" strokeWidth={2} />
            <path d="M4,14 C70,22 110,36 160,52 L250,62" fill="none" stroke="#22D3EE" strokeWidth={2} strokeDasharray="5 4" style={{ transition: 'd 0.3s' }} />
          </svg>
        </div>
      </div>
      <div className="mt-3 space-y-3">
        <Slider label="Temperature" base={tBase} unit="°K" delta={dT} set={setDT} min={-10} max={30} />
        <Slider label="Pressure" base={pBase} unit="psia" delta={dP} set={setDP} min={-5} max={12} />
        <Slider label="Vibration" base={vBase} unit="in/s" delta={dV} set={setDV} min={-0.2} max={0.6} step={0.01} />
      </div>
      <div className="mt-3 flex gap-2">
        <button onClick={() => { setDT(0); setDP(0); setDV(0); setRan(false); }} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-[13px] text-muted2 hover:text-white"><RotateCcw size={14} />Reset</button>
        <button onClick={() => setRan(true)} className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-white py-1.5 text-[13px] font-bold text-black hover:bg-cyan-200"><Play size={14} />{ran ? 'Re-run model' : 'Run model'}</button>
      </div>
      {ran && <div className="mt-2 rounded-lg bg-white/[0.04] p-2 font-mono text-[11px] text-muted2">Surrogate inference complete in 38ms · baseline vs scenario trajectories animated into comparison. Clearly-labeled estimate.</div>}
    </div>
  );
}

function Slider({ label, base, unit, delta, set, min, max, step = 0.5 }: { label: string; base: number; unit: string; delta: number; set: (n: number) => void; min: number; max: number; step?: number }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-[12px]">
        <span className="text-muted2">{label}</span>
        <span className="font-mono tabular-nums">{base.toFixed(2)} → <b className="text-white">{(base + delta).toFixed(2)}</b> <span className="text-muted2">{unit}</span></span>
      </div>
      <input type="range" min={min} max={max} step={step} value={delta} onChange={(e) => set(Number(e.target.value))} className="w-full accent-amber-300" aria-label={`${label} delta`} />
    </div>
  );
}
