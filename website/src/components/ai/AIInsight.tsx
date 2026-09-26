import { api } from '../../services/api';

export function AIInsight({ engineId }: { engineId: number }) {
  const e = api.getEngine(engineId);
  if (!e) return null;
  return (
    <section className="rounded-2xl border border-cyan-300/20 bg-gradient-to-b from-cyan-300/[0.07] to-transparent p-4" aria-label="Model insight">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-cyan-300">✦</span>
        <span className="font-mono text-[10px] font-bold tracking-[0.2em] text-cyan-200">MODEL INSIGHT · CONTEXTUAL</span>
        <span className="ml-auto rounded-full bg-cyan-300/10 px-2 py-0.5 font-mono text-[10px] text-cyan-200">conf {e.confidence}%</span>
      </div>
      <p className="text-[13px] leading-relaxed text-white/85">
        Engine <b>#{String(engineId).padStart(3, '0')}</b> is showing a sustained degradation pattern across recent operating cycles.
      </p>
      <div className="mt-2 text-[12px] text-muted2">Most relevant signals: <span className="text-white">Temperature · Pressure · Sensor 11</span></div>
      <div className="mt-2 flex items-center gap-3 font-mono text-[12px]">
        <span>EST RUL <b className="text-white">{e.rul}</b></span>
        <span>HEALTH <b className="text-white">{e.health}%</b></span>
      </div>
    </section>
  );
}
