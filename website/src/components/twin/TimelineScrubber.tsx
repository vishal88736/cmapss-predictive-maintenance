import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';

export function TimelineScrubber() {
  const { twinEngineId, scrubCycle, setScrubCycle } = useApp();
  const engine = api.getEngine(twinEngineId);
  const series = api.getSeries(twinEngineId);
  if (!engine || !series) return null;
  const max = series.failureCycle;
  const cur = scrubCycle ?? engine.currentCycle;
  const anomalies = series.points.filter((p) => p.anomaly).map((p) => p.cycle);

  return (
    <div className="surface rounded-2xl p-4" aria-label="Timeline scrubber">
      <div className="mb-2 flex items-center gap-3">
        <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">CYCLE SCRUBBER · DRAG TO REPLAY DEGRADATION STORY</span>
        <span className="ml-auto rounded-lg bg-white/10 px-2.5 py-1 font-mono text-[12px] font-bold tabular-nums">● CYCLE {cur} / {max}</span>
      </div>
      <div className="relative px-1 pb-5 pt-1">
        <input
          type="range" min={1} max={max} value={cur}
          onChange={(e) => setScrubCycle(Number(e.target.value))}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') setScrubCycle(Math.max(1, cur - 1));
            if (e.key === 'ArrowRight') setScrubCycle(Math.min(max, cur + 1));
          }}
          className="w-full accent-cyan-300" aria-label="Scrub operating cycle" aria-valuetext={`Cycle ${cur}`}
        />
        <div className="pointer-events-none absolute left-1 right-1 top-[26px] flex justify-between font-mono text-[10px] text-muted2">
          <span>0</span><span>100</span><span>200</span><span className="text-white">● {engine.currentCycle} now</span><span>{max}</span>
        </div>
        <div className="pointer-events-none absolute left-1 right-1 top-[13px] h-[3px]">
          {anomalies.map((c) => (
            <span key={c} className="absolute h-[7px] w-[7px] -translate-x-1/2 rounded-full bg-red-400" style={{ left: `${(c / max) * 100}%` }} title={`Anomaly cycle ${c}`} />
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2 text-[11px] text-muted2">
        <span>Scrubbing updates <b className="text-white">sensors · health · RUL · twin render</b>.</span>
        <button onClick={() => setScrubCycle(null)} className="ml-auto rounded-md border border-line px-2 py-0.5 font-mono text-[10px] hover:text-white">RESET TO LIVE ({engine.currentCycle})</button>
      </div>
    </div>
  );
}
