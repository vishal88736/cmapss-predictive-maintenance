import { useState } from 'react';
import { Check } from 'lucide-react';
import { severityColor } from '../../data/generator';
import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';

export function AlertsTimeline() {
  const { openTwin, setScrubCycle } = useApp();
  const [filter, setFilter] = useState('ALL');
  const [acked, setAcked] = useState<Set<string>>(new Set());
  const alerts = api.getAlerts().filter((a) => (filter === 'ALL' ? true : a.severity === filter));

  const focus = (engineId: number, cycle: number) => {
    openTwin(engineId);
    setScrubCycle(cycle);
  };

  return (
    <div>
      <div className="mb-3 flex gap-1.5">
        {['ALL', 'CRITICAL', 'WARNING', 'INFO'].map((s) => (
          <button key={s} onClick={() => setFilter(s)} aria-pressed={filter === s} className={`rounded-full border px-3 py-1 font-mono text-[11px] ${filter === s ? 'border-white/40 bg-white/10 text-white' : 'border-line text-muted2'}`}>{s}</button>
        ))}
      </div>
      <div className="relative ml-2 border-l border-white/10 pl-6" role="list" aria-label="Alert events">
        {alerts.map((a) => {
          const c = severityColor(a.severity);
          const done = acked.has(a.id);
          return (
            <div key={a.id} role="listitem" className={`relative mb-3 rounded-xl border border-white/[0.07] bg-panel p-3.5 transition ${done ? 'opacity-55' : ''}`}>
              <span className="absolute -left-[29px] top-4 h-2.5 w-2.5 rounded-full border-2 border-void animate-pulse-dot" style={{ background: c }} aria-hidden />
              <div className="flex items-center gap-2">
                <span className="font-mono text-[11px] text-muted2">{a.time}</span>
                <span className="rounded px-1.5 py-0.5 font-mono text-[10px] font-bold" style={{ background: c + '1e', color: c }}>{a.severity}</span>
                <span className="font-mono text-[12px] font-bold">ENGINE #{String(a.engineId).padStart(3, '0')}</span>
                {done && <span className="ml-auto flex items-center gap-1 font-mono text-[10px] text-emerald-300"><Check size={12} />ACKED</span>}
              </div>
              <div className="mt-1 text-[13px] font-semibold">{a.title}</div>
              <div className="text-[12px] text-muted2">{a.detail} · cycle {a.cycle}</div>
              <div className="mt-2 flex gap-2">
                <button onClick={() => focus(a.engineId, a.cycle)} className="rounded-lg bg-white px-2.5 py-1 text-[11px] font-bold text-black hover:bg-cyan-200">Focus engine @ cycle {a.cycle}</button>
                {!done && <button onClick={() => setAcked(new Set(acked).add(a.id))} className="rounded-lg border border-line px-2.5 py-1 text-[11px] text-muted2 hover:text-white">Acknowledge</button>}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
