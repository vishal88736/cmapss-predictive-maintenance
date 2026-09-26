import { X } from 'lucide-react';
import { useApp } from '../../store/AppContext';

const ROWS: [string, string][] = [
  ['⌘ K / Ctrl K', 'Command palette'],
  ['/', 'Focus fleet search'],
  ['F', 'Focus mode'],
  ['E', 'Engine explorer (selected engine)'],
  ['P', 'Presentation mode'],
  ['Esc', 'Close panel / exit mode'],
  ['?', 'This help'],
  ['← →', 'Scrub timeline cycle'],
];

export function KeyboardHelp() {
  const { helpOpen, setHelpOpen } = useApp();
  if (!helpOpen) return null;
  return (
    <div className="fixed inset-0 z-[85] flex items-center justify-center bg-black/60 p-4" onClick={() => setHelpOpen(false)}>
      <div className="w-full max-w-[380px] rounded-2xl border border-white/10 bg-panel p-5 shadow-glow" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Keyboard shortcuts">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-bold">Keyboard-first navigation</h2>
          <button onClick={() => setHelpOpen(false)} aria-label="Close help"><X size={16} /></button>
        </div>
        {ROWS.map(([k, v]) => (
          <div key={k} className="flex items-center justify-between border-b border-white/[0.05] py-2 text-[13px] last:border-0">
            <span className="text-muted2">{v}</span>
            <span className="cmd-kbd">{k}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
