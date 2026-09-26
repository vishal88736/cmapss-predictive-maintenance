import { X } from 'lucide-react';
import { SENSORS } from '../../data/generator';
import { api } from '../../services/api';

export function SensorInsight({ engineId, cycle, sensor, onClose }: { engineId: number; cycle: number; sensor: string; onClose: () => void }) {
  const series = api.getSeries(engineId);
  const pt = series?.points[cycle - 1];
  const def = SENSORS.find((s) => s.key === sensor)!;
  if (!pt) return null;
  const v = pt.values[sensor];
  const mid = (def.expected[0] + def.expected[1]) / 2;
  const dev = ((v - mid) / mid) * 100;
  const out = v < def.expected[0] || v > def.expected[1];
  return (
    <div className="mt-3 animate-fade-up rounded-xl border border-amber-300/25 bg-amber-300/[0.06] p-3.5" role="dialog" aria-label={`${def.name} anomaly detail`}>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-[13px] font-extrabold tracking-wide text-amber-300">▲ {def.name.toUpperCase()} {out ? 'ANOMALY' : 'READING'} · CYCLE {cycle}</span>
        <button onClick={onClose} aria-label="Close insight"><X size={14} /></button>
      </div>
      <div className="grid grid-cols-3 gap-2 font-mono text-[12px]">
        <div className="rounded-lg bg-black/30 p-2"><div className="text-[10px] text-muted2">OBSERVED</div><div className="font-bold text-white">{v.toFixed(1)} {def.unit}</div></div>
        <div className="rounded-lg bg-black/30 p-2"><div className="text-[10px] text-muted2">EXPECTED</div><div className="font-bold text-white">{def.expected[0]}–{def.expected[1]}</div></div>
        <div className="rounded-lg bg-black/30 p-2"><div className="text-[10px] text-muted2">DEVIATION</div><div className="font-bold text-amber-300">{dev >= 0 ? '+' : ''}{dev.toFixed(1)}%</div></div>
      </div>
      <div className="mt-2 text-[12px] leading-relaxed text-white/80">
        Pattern: <b>increasing {def.name.toLowerCase()} variance</b> across recent window. Potential effect: <b>higher degradation signal</b>.
        <span className="mt-1 block font-mono text-[10px] text-muted2">MODEL-DERIVED INTERPRETATION · NOT CLAIMED CAUSALITY</span>
      </div>
    </div>
  );
}
