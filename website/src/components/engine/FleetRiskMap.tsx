import { useMemo } from 'react';
import { statusColor } from '../../data/generator';
import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';

export function FleetRiskMap({ compact = false }: { compact?: boolean }) {
  const { setSelectedEngineId } = useApp();
  const engines = useMemo(() => api.getEngines(), []);
  const W = 560, H = compact ? 220 : 300;
  const pad = 26;
  const x = (risk: number) => pad + (risk / 100) * (W - pad * 2);
  const y = (health: number) => H - pad - (health / 100) * (H - pad * 2);
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Fleet risk map: engine risk versus health">
        <defs>
          <radialGradient id="riskGlow" cx="80%" cy="20%" r="80%">
            <stop offset="0%" stopColor="#F87171" stopOpacity="0.14" />
            <stop offset="100%" stopColor="transparent" />
          </radialGradient>
        </defs>
        <rect x={0} y={0} width={W} height={H} fill="url(#riskGlow)" rx={12} />
        {[25, 50, 75].map((g) => (
          <g key={g}>
            <line x1={pad + (g / 100) * (W - pad * 2)} y1={pad} x2={pad + (g / 100) * (W - pad * 2)} y2={H - pad} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 4" />
            <line x1={pad} y1={H - pad - (g / 100) * (H - pad * 2)} x2={W - pad} y2={H - pad - (g / 100) * (H - pad * 2)} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 4" />
          </g>
        ))}
        <text x={W - pad} y={H - 8} textAnchor="end" className="tick-label">RISK →</text>
        <text x={10} y={pad} className="tick-label">HEALTH ↑</text>
        {engines.map((e) => (
          <g key={e.id} onClick={() => setSelectedEngineId(e.id)} className="cursor-pointer" role="button" aria-label={`${e.label} risk ${e.risk} health ${e.health}`} tabIndex={0}
            onKeyDown={(ev) => { if (ev.key === 'Enter') setSelectedEngineId(e.id); }}>
            <circle
              cx={x(e.risk)} cy={y(e.health)}
              r={3 + (e.currentCycle / 300) * 5}
              fill={statusColor(e.status)} fillOpacity={e.status === 'NOMINAL' ? 0.55 : 0.95}
              stroke="rgba(0,0,0,0.5)" strokeWidth={1}
            >
              <title>{e.label} · RUL {e.rul} · Risk {e.risk}% · Health {e.health}%</title>
            </circle>
            {(e.id === 24 || e.id === 71) && (
              <text x={x(e.risk) + 9} y={y(e.health) + 3} fill="#E6E9EE" fontSize={10} fontFamily="JetBrains Mono">#{String(e.id).padStart(3, '0')}</text>
            )}
          </g>
        ))}
      </svg>
      <div className="mt-1 flex items-center gap-3 font-mono text-[10px] text-muted2">
        <span>SIZE = operating cycles</span>
        <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-full bg-emerald-400 inline-block" />Nominal</span>
        <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-full bg-amber-400 inline-block" />Warning</span>
        <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-full bg-red-400 inline-block" />Critical</span>
      </div>
    </div>
  );
}
