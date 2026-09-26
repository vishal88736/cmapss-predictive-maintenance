import { useMemo } from 'react';
import { api } from '../../services/api';

export function DegradationTimeline({ engineId, scrub }: { engineId: number; scrub: number | null }) {
  const series = api.getSeries(engineId);
  const engine = api.getEngine(engineId);
  const W = 860, H = 300, PL = 44, PR = 16, PT = 18, PB = 30;
  const hist = series?.points ?? [];
  const pred = series?.predicted ?? [];

  const { x, y, histPath, predPath, bandPath, curX } = useMemo(() => {
    const allH = [...hist.map((p) => p.health), ...pred.map((p) => p.hi)];
    const allL = [...hist.map((p) => p.health), ...pred.map((p) => p.lo)];
    const max = Math.max(...allH, 100), min = Math.min(...allL, 0);
    const maxC = Math.max(...pred.map((p) => p.cycle), ...hist.map((p) => p.cycle));
    const x = (c: number) => PL + (c / maxC) * (W - PL - PR);
    const y = (h: number) => PT + (1 - (h - min) / Math.max(1, max - min)) * (H - PT - PB);
    const hp = hist.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.cycle).toFixed(1)},${y(p.health).toFixed(1)}`).join(' ');
    const pp = pred.length ? `M${x(hist[hist.length - 1].cycle).toFixed(1)},${y(hist[hist.length - 1].health).toFixed(1)} ` + pred.map((p) => `L${x(p.cycle).toFixed(1)},${y(p.health).toFixed(1)}`).join(' ') : '';
    const band = pred.length
      ? `M${pred.map((p) => `${x(p.cycle).toFixed(1)},${y(p.hi).toFixed(1)}`).join(' L')} L${[...pred].reverse().map((p) => `${x(p.cycle).toFixed(1)},${y(p.lo).toFixed(1)}`).join(' L')} Z`
      : '';
    const cur = scrub ?? engine?.currentCycle ?? hist.length;
    return { x, y, histPath: hp, predPath: pp, bandPath: band, curX: x(cur) };
  }, [hist, pred, engine, scrub, W, H]);

  if (!series || !engine) return null;
  const onsetX = x(engine.degradationOnset);
  const failX = x(series.failureCycle);
  const curCycle = scrub ?? engine.currentCycle;

  return (
    <div className="surface rounded-2xl p-4">
      <div className="mb-2 flex flex-wrap items-center gap-3 text-[12px]">
        <span className="font-mono text-[10px] tracking-[0.2em] text-muted2">DEGRADATION TIMELINE · HEALTH SCORE</span>
        <span className="ml-auto flex items-center gap-3 font-mono text-[10px] text-muted2">
          <span className="flex items-center gap-1"><i className="inline-block h-[2px] w-5 bg-white" />HISTORICAL</span>
          <span className="flex items-center gap-1"><i className="inline-block h-[2px] w-5 border-t-2 border-dashed border-cyan-300" />PREDICTED</span>
          <span className="flex items-center gap-1"><i className="inline-block h-2 w-4 rounded-sm bg-cyan-300/20" />UNCERTAINTY</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Degradation timeline for engine ${engineId}. Current cycle ${curCycle}, predicted failure ${series.failureCycle}.`}>
        {[90, 70, 50, 30].map((g) => (
          <g key={g}>
            <line x1={PL} y1={y(g)} x2={W - PR} y2={y(g)} stroke="rgba(255,255,255,0.07)" />
            <text x={PL - 8} y={y(g) + 3} textAnchor="end" className="tick-label">{g}</text>
          </g>
        ))}
        {/* phase bands */}
        <rect x={PL} y={PT} width={Math.max(0, onsetX - PL)} height={H - PT - PB} fill="rgba(52,211,153,0.05)" />
        <rect x={onsetX} y={PT} width={Math.max(0, curX - onsetX)} height={H - PT - PB} fill="rgba(251,191,36,0.06)" />
        <rect x={curX} y={PT} width={Math.max(0, failX - curX)} height={H - PT - PB} fill="rgba(34,211,238,0.05)" />
        <text x={(PL + onsetX) / 2} y={PT + 12} textAnchor="middle" className="tick-label">HEALTHY</text>
        <text x={(onsetX + curX) / 2} y={PT + 12} textAnchor="middle" className="tick-label">DEGRADATION</text>
        <text x={(curX + failX) / 2} y={PT + 12} textAnchor="middle" className="tick-label">PREDICTED</text>
        {bandPath && <path d={bandPath} fill="rgba(34,211,238,0.16)" stroke="none" />}
        <path d={histPath} fill="none" stroke="#E6E9EE" strokeWidth={2.2} strokeLinecap="round" style={{ animation: 'draw-line 1.2s ease-out' }} />
        {predPath && <path d={predPath} fill="none" stroke="#22D3EE" strokeWidth={2.2} strokeDasharray="7 5" strokeLinecap="round" />}
        {/* anomaly markers */}
        {hist.filter((p) => p.anomaly).map((p) => (
          <circle key={p.cycle} cx={x(p.cycle)} cy={y(p.health)} r={4} fill="#F87171" stroke="#08090B" strokeWidth={1.5}>
            <title>Anomaly · cycle {p.cycle}</title>
          </circle>
        ))}
        {/* onset marker */}
        <line x1={onsetX} y1={PT} x2={onsetX} y2={H - PB} stroke="#FBBF24" strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
        {/* current marker */}
        <line x1={curX} y1={PT - 4} x2={curX} y2={H - PB} stroke="#fff" strokeWidth={1.5} />
        <circle cx={curX} cy={y(hist[Math.min(hist.length - 1, curCycle - 1)]?.health ?? 70)} r={6} fill="#fff" stroke="#22D3EE" strokeWidth={2.5} />
        {/* failure marker */}
        <g>
          <line x1={failX} y1={PT} x2={failX} y2={H - PB} stroke="#F87171" strokeWidth={1.5} strokeDasharray="4 3" />
          <text x={failX} y={PT - 4} textAnchor="middle" fill="#F87171" fontSize={10} fontFamily="JetBrains Mono" fontWeight={700}>✕ FAILURE ~{series.failureCycle}</text>
        </g>
        {[0, 100, 200, curCycle, 300].filter((v, i, a) => a.indexOf(v) === i).map((c) => (
          <text key={c} x={x(c)} y={H - 10} textAnchor="middle" className="tick-label">{c}</text>
        ))}
      </svg>
      <div className="mt-1 flex flex-wrap gap-x-5 gap-y-1 font-mono text-[10px] text-muted2">
        <span>ONSET CYCLE {engine.degradationOnset}</span>
        <span className="text-white">CURRENT {curCycle}</span>
        <span className="text-critical">FAILURE BOUNDARY ~{series.failureCycle}</span>
        <span className="ml-auto">SOLID = observed · DASHED = predicted · SHADE = uncertainty</span>
      </div>
    </div>
  );
}
