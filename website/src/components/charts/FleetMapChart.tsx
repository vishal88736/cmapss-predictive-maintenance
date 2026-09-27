import { useMemo } from 'react'
import type { Engine } from '@/types/engine'
import { STATUS_META } from '@/types/engine'
import { useMeasure } from './chartUtils'
import { cn } from '@/lib/utils'

interface Props {
  engines: Engine[]
  onEngineClick?: (engine: Engine) => void
  onEngineHover?: (engine: Engine | null, pos: { x: number; y: number } | null) => void
  selectedId?: string | null
  className?: string
  compact?: boolean
}

export default function FleetMapChart({ engines, onEngineClick, onEngineHover, selectedId, className }: Props) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const pad = { l: 34, r: 14, t: 14, b: 26 }

  const maxCycles = useMemo(() => Math.max(...engines.map((e) => e.currentCycle), 1), [engines])

  const px = (risk: number) => pad.l + (risk / 100) * (w - pad.l - pad.r)
  const py = (health: number) => pad.t + (1 - health / 100) * (h - pad.t - pad.b)
  const pr = (cycles: number) => 3.2 + (cycles / maxCycles) * 4.4

  return (
    <div ref={ref} className={cn('relative h-full w-full', className)}>
      <svg width={w} height={h} className="block" role="img" aria-label="Fleet risk map. Each node is an engine positioned by risk and health.">
        <rect x={px(65)} y={pad.t} width={w - pad.r - px(65)} height={py(65) - pad.t} fill="var(--crit)" fillOpacity={0.035} rx={8} />
        <text x={w - pad.r - 6} y={pad.t + 12} textAnchor="end" fontSize={8.5} fill="var(--crit)" opacity={0.7} letterSpacing={1} className="font-semibold">
          WATCHLIST
        </text>
        {[25, 50, 75].map((v) => (
          <g key={v}>
            <line x1={px(v)} x2={px(v)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.045)" />
            <line x1={pad.l} x2={w - pad.r} y1={py(v)} y2={py(v)} stroke="rgba(255,255,255,0.045)" />
          </g>
        ))}
        <line x1={px(50)} x2={px(50)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.08)" strokeDasharray="2 4" />
        <line x1={pad.l} x2={w - pad.r} y1={py(50)} y2={h - pad.b} stroke="rgba(255,255,255,0.08)" strokeDasharray="2 4" />

        <text x={pad.l} y={h - 10} fontSize={9} fill="var(--ink3)" className="font-mono">
          0
        </text>
        <text x={w - pad.r} y={h - 10} textAnchor="end" fontSize={8.5} fill="var(--ink3)" letterSpacing={1}>
          RISK →
        </text>
        <text x={pad.l - 6} y={pad.t + 4} fontSize={8.5} fill="var(--ink3)" letterSpacing={1} transform={`rotate(-90 ${pad.l - 6} ${pad.t + 4})`} textAnchor="end">
          HEALTH ↑
        </text>

        {engines.map((e, i) => {
          const meta = STATUS_META[e.status]
          const r = pr(e.currentCycle)
          const isSel = selectedId === e.id
          return (
            <g
              key={e.id}
              transform={`translate(${px(e.risk)},${py(e.health)})`}
              className="cursor-pointer"
              style={{ animation: `rise-in 0.5s cubic-bezier(0.22,1,0.36,1) ${Math.min(i * 6, 500)}ms both` }}
              onClick={() => onEngineClick?.(e)}
              onPointerEnter={(ev) => {
                const rect = (ev.currentTarget.ownerSVGElement?.parentElement)?.getBoundingClientRect()
                if (!rect) return
                const g = ev.currentTarget.ownerSVGElement!.getBoundingClientRect()
                onEngineHover?.(e, { x: ev.clientX - g.left, y: ev.clientY - g.top })
              }}
              onPointerLeave={() => onEngineHover?.(null, null)}
            >
              <circle r={r + 5} fill="transparent" />
              <circle r={r} fill={meta.color} fillOpacity={0.16} stroke={meta.color} strokeWidth={1.2} />
              {e.status === 'critical' && <circle r={r} fill="none" stroke={meta.color} strokeWidth={1} opacity={0.4} className="animate-pulse-dot" />}
              <circle r={Math.max(1.2, r * 0.28)} fill={meta.color} />
              {isSel && <circle r={r + 4} fill="none" stroke="var(--accent)" strokeWidth={1.4} strokeDasharray="2 3" />}
              <text y={-r - 4} textAnchor="middle" fontSize={7.5} fill="var(--ink3)" className="font-mono pointer-events-none" opacity={isSel ? 1 : 0}>
                {e.unit}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
