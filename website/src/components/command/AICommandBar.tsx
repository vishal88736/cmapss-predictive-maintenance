import { useState } from 'react';
import { Sparkles } from 'lucide-react';
import { useApp } from '../../store/AppContext';
import { api } from '../../services/api';

export function AICommandBar({ context }: { context: string }) {
  const { setView, openTwin, setFleetStatus, setPrimarySensor, setAiResponse, setCompareA, setCompareB } = useApp();
  const [val, setVal] = useState('');
  const [thinking, setThinking] = useState(false);

  const run = (text: string) => {
    const q = text.toLowerCase();
    setThinking(true);
    setTimeout(() => {
      setThinking(false);
      const engines = api.getEngines();
      if (/lowest|critical|risk/.test(q)) {
        const worst = [...engines].sort((a, b) => a.rul - b.rul).slice(0, 3);
        setAiResponse(`Lowest-RUL engines: ${worst.map((e) => `#${String(e.id).padStart(3, '0')} (RUL ${e.rul}, ${e.status})`).join(' · ')}. Focused #${String(worst[0].id).padStart(3, '0')} for inspection.`);
        openTwin(worst[0].id);
      } else if (/why.*24|engine 24.*risk/.test(q)) {
        setAiResponse('Engine #024 risk (68%) is driven by sustained T24 temperature drift (+8.4% above expected band) and P30 pressure variance across cycles 240–261. Model confidence 87%.');
        openTwin(24);
      } else if (/compare/.test(q)) {
        const nums = (q.match(/\d+/g) ?? []).map(Number);
        setCompareA(nums[0] || 24); setCompareB(nums[1] || 71);
        setAiResponse(`Split-screen comparison armed: #${String(nums[0] || 24).padStart(3, '0')} vs #${String(nums[1] || 71).padStart(3, '0')}. Timelines synchronized.`);
        setView('predictions');
      } else if (/temp|abnormal/.test(q)) {
        setPrimarySensor('T24');
        setAiResponse('Isolated Temperature (T24): +2.8% fleet deviation, anomaly cluster at cycles 259–261 on #024. Pinned P30 overlay for cross-signal check.');
        setView('telemetry');
      } else if (/simulat/.test(q)) {
        setAiResponse('What-if staged: +5% temperature scenario projects RUL 37 → ~29 cycles. Open the Digital Twin simulator to run the model.');
        openTwin(24);
      } else {
        setFleetStatus('ALL');
        setAiResponse(`Understood — “${text}”. Applied as fleet context filter and surfaced the most relevant engines. Use ⌘K for precise jumps.`);
        setView('fleet');
      }
      setVal('');
    }, 650);
  };

  return (
    <div className="pointer-events-none fixed bottom-4 left-[64px] right-0 z-40 flex justify-center px-4">
      <form
        onSubmit={(e) => { e.preventDefault(); if (val.trim()) run(val); }}
        className="pointer-events-auto flex w-full max-w-[640px] items-center gap-2 rounded-full border border-white/10 bg-panel/90 py-2 pl-4 pr-2 shadow-glow backdrop-blur"
        role="search" aria-label="AI command bar"
      >
        <Sparkles size={15} className="shrink-0 text-cyan-300" />
        <input
          value={val} onChange={(e) => setVal(e.target.value)}
          placeholder={thinking ? 'Reasoning over fleet state…' : `Ask about this ${context}…  e.g. “Which engines have the lowest RUL?”`}
          className="w-full bg-transparent text-[13px] outline-none placeholder:text-muted2"
          aria-label={`Ask about this ${context}`}
        />
        <span className="cmd-kbd hidden sm:block">⌘K</span>
        <button type="submit" className="rounded-full bg-white px-3.5 py-1.5 text-[12px] font-bold text-black hover:bg-cyan-200" disabled={thinking}>
          {thinking ? '…' : 'Ask'}
        </button>
      </form>
    </div>
  );
}
