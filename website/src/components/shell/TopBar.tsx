import { useEffect, useState } from 'react';
import { useApp } from '../../store/AppContext';

const TITLES: Record<string, { title: string; sub: string }> = {
  overview: { title: 'Fleet Command', sub: 'OBSERVE → UNDERSTAND · live operational picture' },
  fleet: { title: 'Fleet Map', sub: 'SELECT → INSPECT · 100 engines · risk / health space' },
  twin: { title: 'Engine Digital Twin', sub: 'INVESTIGATE → PREDICT · focused engine analysis' },
  telemetry: { title: 'Telemetry Canvas', sub: 'ANALYZE · layered sensor intelligence' },
  predictions: { title: 'Predictions & Risk', sub: 'PREDICT · RUL distribution & horizon' },
  alerts: { title: 'Event Timeline', sub: 'RESPOND · alerts as operational narrative' },
  models: { title: 'Model Observatory', sub: 'TRUST · performance, drift & explanations' },
};

export function TopBar() {
  const { view, focusMode, presentationMode, setCommandOpen, setFocusMode, twinEngineId } = useApp();
  const [clock, setClock] = useState('');
  useEffect(() => {
    const f = () => setClock(new Date().toISOString().slice(11, 19) + ' UTC');
    f();
    const t = setInterval(f, 1000);
    return () => clearInterval(t);
  }, []);
  if (presentationMode) return null;
  const meta = TITLES[view] ?? TITLES.overview;
  return (
    <header className={`sticky top-0 z-30 border-b border-line bg-void/85 backdrop-blur-md ${focusMode ? 'ml-0' : 'ml-[64px]'}`}>
      <div className="flex h-[60px] items-center gap-4 px-5">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse-dot" aria-hidden />
            <span className="font-mono text-[10px] tracking-[0.22em] text-muted2">LIVE · NASA C-MAPSS FEED</span>
          </div>
          <h1 className="truncate text-[17px] font-bold tracking-tight">
            {view === 'twin' ? `Engine Digital Twin — #${String(twinEngineId).padStart(3, '0')}` : meta.title}
            <span className="ml-3 hidden font-mono text-[11px] font-medium text-muted2 md:inline">{meta.sub}</span>
          </h1>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden font-mono text-[11px] text-muted2 lg:block" aria-live="off">{clock}</span>
          <button onClick={() => setCommandOpen(true)} className="hidden items-center gap-2 rounded-lg border border-line bg-raised px-3 py-1.5 text-[13px] text-muted2 transition hover:border-white/20 hover:text-white sm:flex" aria-label="Open command palette">
            <span className="text-muted2">Search engines, sensors, actions…</span>
            <span className="cmd-kbd">⌘K</span>
          </button>
          <button
            onClick={() => setFocusMode(!focusMode)}
            className="rounded-lg border border-line bg-raised px-2.5 py-1.5 font-mono text-[11px] text-muted2 hover:text-white"
            title="Toggle focus mode (F)"
            aria-pressed={focusMode}
          >{focusMode ? 'EXIT FOCUS' : 'FOCUS [F]'}</button>
        </div>
      </div>
    </header>
  );
}
