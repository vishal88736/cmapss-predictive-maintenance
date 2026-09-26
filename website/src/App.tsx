import { useEffect } from 'react';
import { X } from 'lucide-react';
import { useApp } from './store/AppContext';
import { Rail } from './components/shell/Rail';
import { TopBar } from './components/shell/TopBar';
import { SystemStatus } from './components/shell/SystemStatus';
import { CommandPalette } from './components/command/CommandPalette';
import { AICommandBar } from './components/command/AICommandBar';
import { KeyboardHelp } from './components/command/KeyboardHelp';
import { EngineExplorer } from './components/engine/EngineExplorer';
import { Overview } from './pages/Overview';
import { Fleet } from './pages/Fleet';
import { Twin } from './pages/Twin';
import { TelemetryPage } from './pages/TelemetryPage';
import { Predictions } from './pages/Predictions';
import { AlertsPage } from './pages/AlertsPage';
import { ModelsPage } from './pages/ModelsPage';
import { api } from './services/api';

export default function App() {
  const { view, setView, setCommandOpen, setHelpOpen, focusMode, setFocusMode, presentationMode, setPresentationMode, selectedEngineId, setSelectedEngineId, twinEngineId, setTwinEngineId, scrubCycle, setScrubCycle } = useApp();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const typing = tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setCommandOpen(true); return; }
      if (typing) { if (e.key === 'Escape') (e.target as HTMLElement).blur(); return; }
      if (e.key === '?') setHelpOpen(true);
      else if (e.key.toLowerCase() === 'f') setFocusMode(!focusMode);
      else if (e.key.toLowerCase() === 'p') setPresentationMode(!presentationMode);
      else if (e.key === 'Escape') {
        if (presentationMode) setPresentationMode(false);
        else if (focusMode) setFocusMode(false);
        else if (selectedEngineId != null) setSelectedEngineId(null);
      } else if (e.key.toLowerCase() === 'e' && selectedEngineId == null) {
        setSelectedEngineId(twinEngineId);
      } else if (e.key === '/' ) { e.preventDefault(); setView('fleet'); }
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        if (view === 'twin') {
          const eng = api.getEngine(twinEngineId);
          const s = api.getSeries(twinEngineId);
          if (eng && s) {
            const cur = scrubCycle ?? eng.currentCycle;
            const nx = e.key === 'ArrowRight' ? Math.min(s.failureCycle, cur + 2) : Math.max(1, cur - 2);
            setScrubCycle(nx);
          }
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focusMode, presentationMode, selectedEngineId, twinEngineId, scrubCycle, view, setCommandOpen, setFocusMode, setHelpOpen, setPresentationMode, setScrubCycle, setSelectedEngineId, setView]);

  // Presentation mode: fullscreen story Engine → Telemetry → Degradation → Prediction → Risk
  if (presentationMode) {
    return (
      <PresentationStory
        onExit={() => setPresentationMode(false)}
        twinEngineId={twinEngineId}
        setTwinEngineId={setTwinEngineId}
      />
    );
  }

  return (
    <div className="min-h-screen bg-void text-white">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-[100] focus:rounded focus:bg-white focus:px-3 focus:py-1 focus:text-black">Skip to content</a>
      <Rail />
      <TopBar />
      <main id="main" className={`${focusMode ? 'ml-0' : 'ml-[64px]'} px-3 pt-4 sm:px-5`} aria-label="Main content">
        <div className="mx-auto max-w-[1280px]">
          {view === 'overview' && <Overview />}
          {view === 'fleet' && <Fleet />}
          {view === 'twin' && <Twin />}
          {view === 'telemetry' && <TelemetryPage />}
          {view === 'predictions' && <Predictions />}
          {view === 'alerts' && <AlertsPage />}
          {view === 'models' && <ModelsPage />}
          <footer className="mt-2 border-t border-line py-4 pb-24 font-mono text-[10px] text-muted2">
            AEROGUARD AI · NASA C-MAPSS FD001–FD004 · LSTM RUL v2.4 · Shortcuts: ⌘K palette · F focus · E explorer · P present · ? help · status via shape+text, never color alone
          </footer>
        </div>
      </main>
      {/* contextual side panel (progressive disclosure) */}
      <EngineExplorer />
      <SystemStatus />
      <AICommandBar context={view} />
      <CommandPalette />
      <KeyboardHelp />
      {/* mobile: floating command button */}
      <button onClick={() => setCommandOpen(true)} aria-label="Open command search" className="fixed bottom-4 right-4 z-40 rounded-full bg-white p-3.5 text-black shadow-glow sm:hidden">⌘K</button>
    </div>
  );
}

function PresentationStory({ onExit, twinEngineId, setTwinEngineId }: { onExit: () => void; twinEngineId: number; setTwinEngineId: (n: number) => void }) {
  const steps = ['Engine', 'Telemetry', 'Degradation', 'Prediction', 'Risk'] as const;
  const engine = api.getEngine(twinEngineId)!;
  const series = api.getSeries(twinEngineId)!;
  return (
    <div className="fixed inset-0 z-[90] flex flex-col bg-void text-white" role="dialog" aria-label="Presentation mode">
      <div className="flex items-center gap-3 border-b border-line px-6 py-3">
        <span className="font-mono text-[11px] tracking-[0.22em] text-muted2">PRESENTATION MODE · AEROGUARD AI</span>
        <span className="font-bold">Engine → Telemetry → Degradation → Prediction → Risk</span>
        <select value={twinEngineId} onChange={(e) => setTwinEngineId(Number(e.target.value))} className="ml-2 rounded-lg border border-line bg-raised px-2 py-1 font-mono text-[12px]" aria-label="Presentation engine">
          {[24, 71, 38, 52].map((id) => <option key={id} value={id}>ENGINE #{String(id).padStart(3, '0')}</option>)}
        </select>
        <button onClick={onExit} className="ml-auto flex items-center gap-1 rounded-lg border border-line px-3 py-1.5 text-[13px] hover:border-white/30"><X size={14} />Exit (ESC)</button>
      </div>
      <div className="grid-bg flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
        <div className="font-mono text-[12px] tracking-[0.3em] text-cyan-300">ENGINE #{String(engine.id).padStart(3, '0')} · {engine.status}</div>
        <div className="text-6xl font-extrabold tracking-tight tabular-nums">{engine.rul} <span className="text-2xl text-muted2">cycles RUL</span></div>
        <svg viewBox="0 0 700 180" className="w-full max-w-[760px]" aria-label="Story degradation curve">
          <path d={series.points.filter((_, i) => i % 3 === 0).map((p, i, a) => `${i === 0 ? 'M' : 'L'}${(20 + (i / Math.max(1, a.length - 1)) * 560).toFixed(1)},${(20 + (1 - p.health / 100) * 120).toFixed(1)}`).join(' ')} fill="none" stroke="#fff" strokeWidth={3} />
          <path d={`M580,${(20 + (1 - series.points[series.points.length - 1].health / 100) * 120).toFixed(1)} L680,150`} fill="none" stroke="#22D3EE" strokeWidth={3} strokeDasharray="8 6" />
        </svg>
        <div className="flex items-center gap-2 font-mono text-[12px] text-muted2">
          {steps.map((s, i) => <span key={s} className="flex items-center gap-2"><span className={i === 0 ? 'text-white' : ''}>{s}</span>{i < steps.length - 1 && <span>↓</span>}</span>)}
        </div>
        <div className="max-w-[560px] text-[15px] text-white/75">T24 {series.points[series.points.length - 1].values['T24'].toFixed(1)}°K · Health {engine.health}% · Risk {engine.risk}% · Confidence {engine.confidence}% — telemetry, degradation and prediction in one glance.</div>
      </div>
    </div>
  );
}
