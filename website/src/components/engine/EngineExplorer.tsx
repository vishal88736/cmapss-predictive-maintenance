import { X } from 'lucide-react';
import { SENSORS } from '../../data/generator';
import { statusColor } from '../../data/generator';
import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';

export function EngineExplorer() {
  const { selectedEngineId, setSelectedEngineId, openTwin } = useApp();
  if (selectedEngineId == null) return null;
  const e = api.getEngine(selectedEngineId);
  if (!e) return null;
  const series = api.getSeries(e.id);
  const cur = series?.points[series.points.length - 1];
  const c = statusColor(e.status);

  return (
    <aside
      className="fixed right-0 top-0 z-50 flex h-screen w-full max-w-[360px] animate-slide-in flex-col border-l border-line bg-panel/98 shadow-glow backdrop-blur"
      role="complementary" aria-label={`Engine ${e.label} explorer`}
    >
      <div className="flex items-center justify-between border-b border-line px-5 py-4">
        <div>
          <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">ENGINE EXPLORER · PROGRESSIVE DISCLOSURE</div>
          <h2 className="text-lg font-extrabold tracking-tight">{e.label}</h2>
        </div>
        <button onClick={() => setSelectedEngineId(null)} aria-label="Close explorer" className="rounded-lg border border-line p-1.5 text-muted2 hover:text-white"><X size={16} /></button>
      </div>
      <div className="flex-1 overflow-y-auto p-5">
        <div className="mb-4 flex items-center gap-2 rounded-lg border px-3 py-2 text-[13px] font-bold" style={{ borderColor: c + '55', background: c + '14', color: c }} role="status">
          <span className="h-2 w-2 rounded-full animate-pulse-dot" style={{ background: c }} aria-hidden />
          {e.status === 'NOMINAL' ? '● NOMINAL — operating normally' : `▲ ${e.status} — attention advised`}
          <span className="sr-only">Status {e.status}</span>
        </div>
        <div className="surface-2 rounded-xl p-4">
          <div className="font-mono text-[10px] tracking-[0.2em] text-muted2">REMAINING USEFUL LIFE</div>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-extrabold tabular-nums">{e.rul}</span>
            <span className="text-sm text-muted2">cycles</span>
            <span className="ml-auto font-mono text-[11px] text-muted2">conf {e.confidence}%</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <div className="rounded-lg bg-black/30 p-2.5"><div className="font-mono text-[10px] text-muted2">HEALTH</div><div className="text-xl font-bold tabular-nums">{e.health}%</div></div>
            <div className="rounded-lg bg-black/30 p-2.5"><div className="font-mono text-[10px] text-muted2">RISK</div><div className="text-xl font-bold tabular-nums" style={{ color: c }}>{e.risk}%</div></div>
          </div>
        </div>
        <div className="mt-4">
          <div className="mb-2 font-mono text-[10px] tracking-[0.2em] text-muted2">TELEMETRY · CYCLE {e.currentCycle}</div>
          <div className="space-y-1.5">
            {['T24', 'P30', 'Nc', 'VIB'].map((k) => {
              const def = SENSORS.find((s) => s.key === k)!;
              const v = cur?.values[k];
              return (
                <div key={k} className="flex items-center justify-between rounded-lg border border-white/[0.06] bg-raised px-3 py-2">
                  <span className="text-[13px] text-muted2">{def.name}</span>
                  <span className="font-mono text-[13px] font-bold tabular-nums">{v} <span className="text-[10px] font-medium text-muted2">{def.unit}</span></span>
                </div>
              );
            })}
          </div>
        </div>
        <div className="mt-4 rounded-xl border border-cyan-300/20 bg-cyan-300/[0.06] p-3 text-[12px] leading-relaxed text-cyan-100/90">
          <span className="font-bold text-cyan-300">✦ Model insight —</span> sustained degradation pattern across recent cycles. Most relevant: Temperature · Pressure · Sensor 11. Estimated RUL {e.rul} cycles.
        </div>
      </div>
      <div className="border-t border-line p-4">
        <button onClick={() => openTwin(e.id)} className="w-full rounded-xl bg-white py-2.5 text-[14px] font-bold text-black transition hover:bg-cyan-200" aria-label={`Open digital twin for ${e.label}`}>
          Open Digital Twin →
        </button>
        <div className="mt-2 text-center font-mono text-[10px] text-muted2">ESC to close · E to toggle</div>
      </div>
    </aside>
  );
}
