import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, Search } from 'lucide-react';
import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';

function fuzzy(hay: string, needle: string): boolean {
  hay = hay.toLowerCase(); needle = needle.toLowerCase().trim();
  if (!needle) return true;
  let hi = 0;
  for (let ni = 0; ni < needle.length; ni++) {
    const c = needle[ni];
    if (c === ' ') continue;
    hi = hay.indexOf(c, hi);
    if (hi === -1) return false;
    hi++;
  }
  return true;
}

export function CommandPalette() {
  const { commandOpen, setCommandOpen, setView, setSelectedEngineId, openTwin, setPrimarySensor, setFleetQuery, setFleetStatus, setCompareA, setCompareB } = useApp();
  const [q, setQ] = useState('');
  const [idx, setIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const engines = useMemo(() => api.getEngines(), []);

  useEffect(() => {
    if (commandOpen) { setQ(''); setIdx(0); setTimeout(() => inputRef.current?.focus(), 30); }
  }, [commandOpen]);

  const results = useMemo(() => {
    const out: { group: string; label: string; hint: string; run: () => void }[] = [];
    const ql = q.toLowerCase();
    // smart interpretations
    if (/lowest rul|critical|high risk/.test(ql)) {
      const worst = [...engines].sort((a, b) => a.rul - b.rul).slice(0, 5);
      worst.forEach((e) => out.push({ group: 'SMART · LOWEST RUL', label: `${e.label} — RUL ${e.rul} · ${e.status}`, hint: 'Open twin', run: () => openTwin(e.id) }));
      out.push({ group: 'SMART', label: 'Show all critical engines in Fleet', hint: 'Filter', run: () => { setFleetStatus('CRITICAL'); setFleetQuery(''); setView('fleet'); } });
    }
    if (/compare/.test(ql)) {
      const nums = (ql.match(/\d+/g) ?? []).map(Number).filter((n) => n >= 1 && n <= 100);
      const a = nums[0] ?? 24, b = nums[1] ?? 71;
      out.push({ group: 'ACTION', label: `Compare Engine ${a} and ${b}`, hint: 'Split view', run: () => { setCompareA(a); setCompareB(b); setView('predictions'); } });
    }
    if (/temp/.test(ql)) {
      out.push({ group: 'SENSOR', label: 'Temperature (T24) — analyze signal', hint: 'Telemetry', run: () => { setPrimarySensor('T24'); setView('telemetry'); } });
    }
    if (/alert/.test(ql)) out.push({ group: 'NAVIGATION', label: 'Show alerts timeline', hint: 'Go', run: () => setView('alerts') });

    engines.filter((e) => fuzzy(`${e.label} engine ${e.id} ${e.status} rul ${e.rul}`, q)).slice(0, 6).forEach((e) =>
      out.push({ group: 'ENGINES', label: `${e.label} — RUL ${e.rul} · ${e.health}% health`, hint: e.status, run: () => setSelectedEngineId(e.id) }),
    );
    [
      { label: 'Open Fleet map', hint: 'Navigation', run: () => setView('fleet') },
      { label: 'Run RUL prediction (Engine twin)', hint: 'Action', run: () => setView('twin') },
      { label: 'Compare Engines 24 and 71', hint: 'Action', run: () => { setCompareA(24); setCompareB(71); setView('predictions'); } },
      { label: 'Analyze Temperature sensor', hint: 'Telemetry', run: () => { setPrimarySensor('T24'); setView('telemetry'); } },
      { label: 'Open Model insights', hint: 'Navigation', run: () => setView('models') },
      { label: 'Overview command center', hint: 'Navigation', run: () => setView('overview') },
    ].filter((a) => fuzzy(a.label, q)).forEach((a) => out.push({ group: 'ACTIONS', label: a.label, hint: a.hint, run: a.run }));
    ['Overview', 'Fleet', 'Telemetry', 'Predictions', 'Alerts', 'Models'].filter((n) => fuzzy(n, q)).forEach((n) =>
      out.push({ group: 'NAVIGATION', label: `Go to ${n}`, hint: 'Go', run: () => setView(n.toLowerCase() as any) }),
    );
    return out.slice(0, 14);
  }, [q, engines, openTwin, setView, setSelectedEngineId, setPrimarySensor, setFleetStatus, setFleetQuery, setCompareA, setCompareB]);

  useEffect(() => setIdx(0), [q]);

  if (!commandOpen) return null;
  return (
    <div className="fixed inset-0 z-[80] flex items-start justify-center bg-black/60 p-4 pt-[12vh] backdrop-blur-sm" onClick={() => setCommandOpen(false)} role="presentation">
      <div
        className="w-full max-w-[560px] animate-fade-up overflow-hidden rounded-2xl border border-white/10 bg-[#0E1013] shadow-glow"
        onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Command palette"
      >
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <Search size={16} className="text-muted2" />
          <input
            ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') { e.preventDefault(); setIdx((i) => Math.min(results.length - 1, i + 1)); }
              if (e.key === 'ArrowUp') { e.preventDefault(); setIdx((i) => Math.max(0, i - 1)); }
              if (e.key === 'Enter' && results[idx]) { results[idx].run(); setCommandOpen(false); }
              if (e.key === 'Escape') setCommandOpen(false);
            }}
            placeholder="Search engines, sensors, actions…  try “lowest RUL” or “compare 24 and 71”"
            className="w-full bg-transparent text-[14px] outline-none placeholder:text-muted2"
            aria-label="Command search"
          />
          <span className="cmd-kbd">ESC</span>
        </div>
        <div className="max-h-[380px] overflow-y-auto p-2" role="listbox" aria-label="Results">
          {!results.length && <div className="p-6 text-center text-sm text-muted2">No matches. Try “engine 24”, “critical engines”, “temperature sensor”.</div>}
          {results.map((r, i) => (
            <button
              key={i} role="option" aria-selected={i === idx}
              onMouseEnter={() => setIdx(i)}
              onClick={() => { r.run(); setCommandOpen(false); }}
              className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px] ${i === idx ? 'bg-cyan-400/10 text-white' : 'text-muted2'}`}
            >
              <span className="font-mono text-[9px] tracking-widest text-muted2/80">{r.group}</span>
              <span className="flex-1 truncate font-medium">{r.label}</span>
              <span className="flex items-center gap-1 font-mono text-[10px] text-muted2">{r.hint}<ArrowRight size={12} /></span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-4 border-t border-line px-4 py-2 font-mono text-[10px] text-muted2">
          <span><span className="cmd-kbd">↑↓</span> navigate</span>
          <span><span className="cmd-kbd">↵</span> select</span>
          <span className="ml-auto">RECENT → Engine #024 · Engine #071</span>
        </div>
      </div>
    </div>
  );
}
