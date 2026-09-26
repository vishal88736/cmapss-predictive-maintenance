import { X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useApp } from '../../store/AppContext';

export function SystemStatus() {
  const { sysOpen, setSysOpen, presentationMode } = useApp();
  const [now, setNow] = useState('14:32:08');
  useEffect(() => {
    const t = setInterval(() => setNow(new Date().toISOString().slice(11, 19)), 1000);
    return () => clearInterval(t);
  }, []);
  if (presentationMode) return null;
  return (
    <>
      <button
        onClick={() => setSysOpen(!sysOpen)}
        aria-expanded={sysOpen}
        aria-label="System status, activate to expand diagnostics"
        className="fixed bottom-20 right-4 z-40 w-[218px] rounded-xl border border-line bg-panel/95 p-3 text-left shadow-glow backdrop-blur transition hover:border-white/20"
      >
        <div className="mb-2 flex items-center justify-between">
          <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-muted2">SYSTEM</span>
          <span className="flex items-center gap-1 font-mono text-[10px] text-emerald-300"><span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse-dot" />LIVE</span>
        </div>
        <StatusRow label="Telemetry stream" value="CONNECTED" ok />
        <StatusRow label="ML inference" value="READY" ok />
        <StatusRow label="Prediction svc" value="ONLINE" ok />
        <div className="mt-2 border-t border-line pt-1.5 font-mono text-[10px] text-muted2">Last sync <span className="text-white">{now}</span></div>
      </button>
      {sysOpen && (
        <div className="fixed bottom-20 right-[240px] z-40 w-[300px] animate-slide-in rounded-xl border border-line bg-panel p-4 shadow-glow" role="dialog" aria-label="System diagnostics">
          <div className="mb-3 flex items-center justify-between">
            <span className="text-sm font-bold">Diagnostics</span>
            <button onClick={() => setSysOpen(false)} aria-label="Close diagnostics"><X size={16} /></button>
          </div>
          {[
            ['Ingest throughput', '4,812 msgs/s · p99 38ms'],
            ['Feature pipeline', 'C-MAPSS window=30 · normalized'],
            ['LSTM v2.4 latency', '42ms median · GPU:1'],
            ['Prediction drift', '0.8% · within tolerance'],
            ['Data freshness', 'FD001–FD004 mirrors synced'],
          ].map(([k, v]) => (
            <div key={k} className="mb-2 flex items-start justify-between gap-3 border-b border-white/[0.05] pb-2 last:border-0">
              <span className="text-[12px] text-muted2">{k}</span>
              <span className="text-right font-mono text-[11px] text-white">{v}</span>
            </div>
          ))}
          <div className="mt-2 rounded-lg bg-emerald-400/10 p-2 text-[12px] text-emerald-300">● All subsystems nominal. No action required.</div>
        </div>
      )}
    </>
  );
}

function StatusRow({ label, value, ok }: { label: string; value: string; ok?: boolean }) {
  return (
    <div className="flex items-center justify-between py-0.5">
      <span className="flex items-center gap-1.5 text-[12px] text-muted2">
        <span className={`h-1.5 w-1.5 rounded-full ${ok ? 'bg-emerald-400' : 'bg-amber-400'} animate-pulse-dot`} aria-hidden />{label}
      </span>
      <span className="font-mono text-[10px] font-bold text-white">{value}</span>
    </div>
  );
}
