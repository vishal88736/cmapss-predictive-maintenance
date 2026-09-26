import { AlertTriangle, Boxes, Gauge, Home, Orbit, Radar, Settings, TerminalSquare, Waves } from 'lucide-react';
import { clsx } from 'clsx';
import { useApp, type View } from '../../store/AppContext';

const ITEMS: { id: View; icon: any; label: string; kbd?: string }[] = [
  { id: 'overview', icon: Home, label: 'Overview' },
  { id: 'fleet', icon: Orbit, label: 'Fleet' },
  { id: 'twin', icon: Boxes, label: 'Engines' },
  { id: 'telemetry', icon: Waves, label: 'Telemetry' },
  { id: 'predictions', icon: Radar, label: 'Predictions' },
  { id: 'alerts', icon: AlertTriangle, label: 'Alerts' },
  { id: 'models', icon: Gauge, label: 'Models' },
];

export function Rail() {
  const { view, setView, setCommandOpen, focusMode, presentationMode } = useApp();
  if (focusMode || presentationMode) return null;
  return (
    <nav aria-label="Primary" className="group fixed left-0 top-0 z-40 flex h-screen w-[64px] flex-col items-center border-r border-line bg-panel/95 py-3 backdrop-blur transition-all duration-200 hover:w-[196px]">
      <button onClick={() => setView('overview')} className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-400 to-emerald-400 font-black text-black shadow-glow" aria-label="AeroGuard home">
        <span className="text-lg">▲</span>
      </button>
      <div className="mb-1 hidden text-[9px] font-bold tracking-[0.22em] text-muted2 group-hover:block">AEROGUARD</div>
      <div className="flex flex-1 flex-col gap-1.5">
        {ITEMS.map((it) => {
          const active = view === it.id;
          const Icon = it.icon;
          return (
            <button
              key={it.id}
              onClick={() => setView(it.id)}
              aria-current={active ? 'page' : undefined}
              className={clsx(
                'flex h-10 w-10 items-center gap-3 overflow-hidden rounded-lg px-0 text-sm transition-all group-hover:w-[164px] group-hover:px-3',
                active ? 'bg-white/[0.08] text-white shadow-card' : 'text-muted2 hover:bg-white/[0.05] hover:text-white',
              )}
              title={it.label}
            >
              <span className={clsx('flex h-10 w-10 shrink-0 items-center justify-center', active && 'text-cyan-300')}>
                <Icon size={18} strokeWidth={active ? 2.4 : 1.8} />
                {active && <span className="absolute ml-[-30px] h-5 w-[3px] rounded-full bg-cyan-300" aria-hidden />}
              </span>
              <span className="hidden whitespace-nowrap font-medium group-hover:block">{it.label}</span>
              {it.id === 'alerts' && (
                <span className="ml-auto hidden rounded-full bg-red-400/15 px-1.5 py-0.5 font-mono text-[10px] font-bold text-critical group-hover:block">7</span>
              )}
            </button>
          );
        })}
      </div>
      <div className="flex flex-col gap-1.5">
        <button onClick={() => setCommandOpen(true)} className="flex h-10 w-10 items-center gap-3 overflow-hidden rounded-lg px-0 text-muted2 transition-all hover:bg-white/[0.05] hover:text-white group-hover:w-[164px] group-hover:px-3" title="Command palette (Ctrl+K)">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center"><TerminalSquare size={18} /></span>
          <span className="hidden whitespace-nowrap text-sm group-hover:block">Command</span>
          <span className="cmd-kbd ml-auto hidden group-hover:block">⌘K</span>
        </button>
        <button className="flex h-10 w-10 items-center gap-3 overflow-hidden rounded-lg px-0 text-muted2 transition-all hover:bg-white/[0.05] hover:text-white group-hover:w-[164px] group-hover:px-3" title="Settings">
          <span className="flex h-10 w-10 shrink-0 items-center justify-center"><Settings size={18} /></span>
          <span className="hidden whitespace-nowrap text-sm group-hover:block">Settings</span>
        </button>
      </div>
    </nav>
  );
}
