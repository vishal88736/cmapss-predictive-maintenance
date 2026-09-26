import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';
import { statusColor } from '../../data/generator';

export function Comparison() {
  const { compareA, compareB, setCompareA, setCompareB, openTwin } = useApp();
  const a = api.getEngine(compareA), b = api.getEngine(compareB);
  const sa = api.getSeries(compareA), sb = api.getSeries(compareB);
  if (!a || !b || !sa || !sb) return null;

  const chart = (key: string, color: string) => {
    const W = 520, H = 120;
    const mk = (pts: { values: Record<string, number> }[], kk: string) => {
      const vals = pts.map((p) => p.values[kk]);
      const min = Math.min(...vals), max = Math.max(...vals);
      return vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${(8 + (i / Math.max(1, vals.length - 1)) * (W - 16)).toFixed(1)},${(8 + (1 - (v - min) / Math.max(1e-6, max - min)) * (H - 16)).toFixed(1)}`).join(' ');
    };
    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-hidden>
        <path d={mk(sa.points, key)} fill="none" stroke={color} strokeWidth={2} opacity={0.95} />
        <path d={mk(sb.points, key)} fill="none" stroke="#5b6673" strokeWidth={1.6} strokeDasharray="5 4" opacity={0.9} />
      </svg>
    );
  };

  const Card = ({ id, other }: { id: number; other: boolean }) => {
    const e = other ? b : a;
    return (
      <div className={`rounded-2xl border p-4 ${other ? 'border-dashed border-white/15 bg-raised' : 'surface'}`}>
        <div className="flex items-center gap-2">
          <select value={e.id} onChange={(ev) => (other ? setCompareB(Number(ev.target.value)) : setCompareA(Number(ev.target.value)))} className="rounded-lg border border-line bg-black/40 px-2 py-1 font-mono text-[13px] font-bold" aria-label="Comparison engine">
            {api.getEngines().slice(0, 100).map((en) => <option key={en.id} value={en.id}>ENGINE #{String(en.id).padStart(3, '0')}</option>)}
          </select>
          <span className="font-mono text-[10px] font-bold" style={{ color: statusColor(e.status) }}>{e.status}</span>
          <button onClick={() => openTwin(e.id)} className="ml-auto text-[11px] text-muted2 underline hover:text-white">Twin →</button>
        </div>
        <div className="mt-2 grid grid-cols-3 gap-2 font-mono text-center">
          <div className="rounded-lg bg-black/30 p-2"><div className="text-[10px] text-muted2">HEALTH</div><div className="text-lg font-bold">{e.health}%</div></div>
          <div className="rounded-lg bg-black/30 p-2"><div className="text-[10px] text-muted2">RUL</div><div className="text-lg font-bold">{e.rul}</div></div>
          <div className="rounded-lg bg-black/30 p-2"><div className="text-[10px] text-muted2">RISK</div><div className="text-lg font-bold">{e.risk}%</div></div>
        </div>
      </div>
    );
  };

  return (
    <div className="surface rounded-2xl p-4" aria-label="Split-screen comparison">
      <div className="mb-3 flex items-center gap-2">
        <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">SPLIT-SCREEN COMPARISON · SYNCHRONIZED TIMELINES</span>
        <span className="ml-auto flex items-center gap-2 font-mono text-[10px]"><i className="h-[2px] w-5 inline-block bg-cyan-300" />A · <i className="h-[2px] w-5 inline-block bg-[#5b6673]" />B</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        <Card id={compareA} other={false} />
        <Card id={compareB} other={true} />
      </div>
      <div className="mt-3 space-y-3">
        {[['T24', 'Temperature', '#FBBF24'], ['P30', 'Pressure', '#22D3EE'], ['VIB', 'Vibration', '#F87171']].map(([k, n, c]) => (
          <div key={k} className="rounded-xl bg-black/25 p-3">
            <div className="mb-1 font-mono text-[11px] text-muted2">{n} · normalized per-signal</div>
            {chart(k, c as string)}
          </div>
        ))}
      </div>
    </div>
  );
}
