import { useMemo } from 'react';
import { api } from '../services/api';
import { useApp } from '../store/AppContext';
import { Comparison } from '../components/compare/Comparison';
import { Bars } from '../components/charts/primitives';

export function Predictions() {
  const { setSelectedEngineId } = useApp();
  const engines = useMemo(() => api.getEngines(), []);
  const buckets = useMemo(() => {
    const b = [0, 0, 0, 0, 0, 0];
    engines.forEach((e) => {
      if (e.rul < 25) b[0]++; else if (e.rul < 50) b[1]++; else if (e.rul < 100) b[2]++;
      else if (e.rul < 150) b[3]++; else if (e.rul < 200) b[4]++; else b[5]++;
    });
    return b;
  }, [engines]);
  return (
    <div className="grid gap-3 pb-24 lg:grid-cols-12">
      <section className="surface rounded-2xl p-5 lg:col-span-5" aria-label="RUL distribution">
        <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">RUL DISTRIBUTION · FLEET HORIZON</div>
        <Bars values={buckets} color="#22D3EE" height={130} />
        <div className="mt-1 flex justify-between font-mono text-[10px] text-muted2"><span>&lt;25</span><span>50</span><span>100</span><span>150</span><span>200+</span></div>
        <div className="mt-3 grid grid-cols-3 gap-2 text-center font-mono">
          <div className="rounded-lg bg-red-400/10 p-2"><div className="text-[10px] text-muted2">&lt;25 CYC</div><div className="text-xl font-bold text-red-300">{buckets[0]}</div></div>
          <div className="rounded-lg bg-amber-300/10 p-2"><div className="text-[10px] text-muted2">25–50</div><div className="text-xl font-bold text-amber-300">{buckets[1]}</div></div>
          <div className="rounded-lg bg-emerald-400/10 p-2"><div className="text-[10px] text-muted2">100+</div><div className="text-xl font-bold text-emerald-300">{buckets[2] + buckets[3] + buckets[4] + buckets[5]}</div></div>
        </div>
      </section>
      <section className="surface rounded-2xl p-5 lg:col-span-7" aria-label="Lowest RUL queue">
        <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">LOWEST RUL · PRIORITY QUEUE</div>
        <div className="mt-2 max-h-[260px] space-y-1.5 overflow-y-auto">
          {[...engines].sort((a, b) => a.rul - b.rul).slice(0, 10).map((e, i) => (
            <button key={e.id} onClick={() => setSelectedEngineId(e.id)} className="flex w-full items-center gap-3 rounded-lg border border-white/[0.06] bg-raised px-3 py-2 text-left hover:border-white/20">
              <span className="font-mono text-[11px] text-muted2">#{i + 1}</span>
              <span className="font-mono text-[13px] font-bold">#{String(e.id).padStart(3, '0')}</span>
              <span className="h-1.5 flex-1 rounded-full bg-white/[0.07]"><span className="block h-1.5 rounded-full bg-gradient-to-r from-red-400 via-amber-300 to-emerald-300" style={{ width: `${Math.min(100, (e.rul / 200) * 100)}%` }} /></span>
              <span className="font-mono text-[12px] tabular-nums">{e.rul} cyc · {e.confidence}%</span>
            </button>
          ))}
        </div>
      </section>
      <section className="lg:col-span-12"><Comparison /></section>
    </div>
  );
}
