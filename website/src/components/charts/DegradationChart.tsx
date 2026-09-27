import { useMemo } from 'react'
import type { Engine } from '@/types/engine'
import type { TelemetryBundle, Anomaly } from '@/types/telemetry'
import type { Prediction } from '@/types/prediction'
import { useMeasure, scaleLinear, niceTicks, linePath, bandPath, smoothPath, useTooltip, nearestIndex } from './chartUtils'
import { cn } from '@/lib/utils'

interface Props {
  engine: Engine
  bundle: TelemetryBundle
  prediction: Prediction
  scrub: number
  onScrub?: (cycle: number) => void
  onAnomalySelect?: (a: Anomaly) => void
  compact?: boolean
  animate?: boolean
  className?: string
}

const PHASES = [
  { id: 'healthy', label: 'HEALTHY', color: 'var(--ok)', tint: 'rgba(52,211,153,0.045)', threshold: 85 },
  { id: 'early', label: 'EARLY DEGRADATION', color: 'var(--info)', tint: 'rgba(76,194,255,0.045)', threshold: 68 },
  { id: 'degradation', label: 'DEGRADATION', color: 'var(--warn)', tint: 'rgba(245,184,74,0.05)', threshold: 45 },
  { id: 'critical', label: 'CRITICAL', color: 'var(--crit)', tint: 'rgba(240,98,93,0.05)', threshold: -1 }
]

function healthAtPrediction(prediction: Prediction, cycle: number): number {
  const cs = prediction.horizonCycles
  if (cycle <= cs[0]) return prediction.horizonHealth[0]
  if (cycle >= cs[cs.length - 1]) return prediction.horizonHealth[cs.length - 1]
  const i = nearestIndex(cs, cycle)
  const c0 = cs[i]
  const c1 = cs[Math.min(i + 1, cs.length - 1)]
  const h0 = prediction.horizonHealth[i]
  const h1 = prediction.horizonHealth[Math.min(i + 1, cs.length - 1)]
  const t = c1 === c0 ? 0 : (cycle - c0) / (c1 - c0)
  return h0 + (h1 - h0) * t
}

export function healthAtCycle(bundle: TelemetryBundle, prediction: Prediction, cycle: number): number {
  if (cycle <= bundle.cycles.length - 1) return bundle.health[cycle]
  return healthAtPrediction(prediction, cycle)
}

export default function DegradationChart({ engine, bundle, prediction, scrub, onScrub, onAnomalySelect, compact = false, animate = true, className }: Props) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const { ref: ttRef, pos, onMove, onLeave } = useTooltip<HTMLDivElement>()
  const pad = compact ? { l: 30, r: 14, t: 12, b: 18 } : { l: 46, r: 16, t: 22, b: 30 }
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const innerW = w - pad.l - pad.r
  const innerH = h - pad.t - pad.b
  const domainX = useMemo((): [number, number] => [0, Math.round(prediction.failureCycle * 1.08)], [prediction.failureCycle])
  const sx = scaleLinear(domainX, [pad.l, w - pad.r])
  const sy = scaleLinear([0, 104], [h - pad.b, pad.t])

  const historyPts = useMemo(() => bundle.cycles.map((c, i) => [sx(c), sy(bundle.health[i])] as [number, number]), [bundle, sx, sy])
  const predPts = useMemo(
    () => prediction.horizonCycles.map((c, i) => [sx(c), sy(prediction.horizonHealth[i])] as [number, number]),
    [prediction, sx, sy]
  )
  const band = useMemo(
    () =>
      bandPath(
        prediction.horizonCycles.map((c, i) => [sx(c), sy(prediction.horizonHigh[i])] as [number, number]),
        prediction.horizonCycles.map((c, i) => [sx(c), sy(prediction.horizonLow[i])] as [number, number])
      ),
    [prediction, sx, sy]
  )

  const xTicks = useMemo(() => niceTicks(domainX[0], domainX[1], compact ? 4 : 7), [domainX, compact])
  const yTicks = useMemo(() => niceTicks(0, 100, compact ? 2 : 5), [compact])

  const phaseBands = useMemo(() => {
    const all = [...bundle.cycles.map((c, i) => ({ c, h: bundle.health[i] })), ...prediction.horizonCycles.map((c, i) => ({ c, h: prediction.horizonHealth[i] }))]
    const boundaries: { x0: number; x1: number; phase: (typeof PHASES)[number] }[] = []
    for (let p = 0; p < PHASES.length; p++) {
      const startCycle = p === 0 ? 0 : cycleWhereHealthBelow(all, PHASES[p - 1].threshold)
      const endCycle = p === PHASES.length - 1 ? domainX[1] : cycleWhereHealthBelow(all, PHASES[p].threshold)
      boundaries.push({ x0: sx(startCycle), x1: sx(endCycle), phase: PHASES[p] })
    }
    return boundaries
  }, [bundle, prediction, domainX, sx])

  const scrubHealth = healthAtCycle(bundle, prediction, scrub)
  const hoverCycle = pos ? Math.round(sx.invert(pos.x)) : null
  const hoverHealth = hoverCycle != null ? healthAtCycle(bundle, prediction, hoverCycle) : null

  return (
    <div
      ref={(el) => {
        ;
        (ref as React.MutableRefObject<HTMLDivElement | null>).current = el
        ;
        (ttRef as React.MutableRefObject<HTMLDivElement | null>).current = el
      }}
      className={cn('relative touch-none select-none', className)}
      onPointerMove={(e) => {
        onMove(e)
        if (!onScrub || pos == null) return
      }}
      onPointerLeave={onLeave}
      onPointerDown={(e) => {
        if (!onScrub) return
        const r = e.currentTarget.getBoundingClientRect()
        const cycle = Math.round(sx.invert(e.clientX - r.left))
        onScrub(Math.max(0, Math.min(cycle, domainX[1])))
      }}
      role="img"
      aria-label={`Degradation timeline for engine ${engine.unit}. Current cycle ${engine.currentCycle}, health ${engine.health}%, predicted failure at cycle ${prediction.failureCycle}.`}
    >
      <svg width={w} height={h} className="block">
        {phaseBands.map((b, i) => (
          <g key={i}>
            <rect x={b.x0} y={pad.t} width={Math.max(0, b.x1 - b.x0)} height={innerH} fill={b.phase.tint} />
            {!compact && b.x1 - b.x0 > 60 && (
              <text x={(b.x0 + b.x1) / 2} y={pad.t - 8} textAnchor="middle" fontSize={8.5} letterSpacing={1.1} fill={b.phase.color} opacity={0.75} className="font-semibold">
                {b.phase.label}
              </text>
            )}
          </g>
        ))}

        {yTicks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.05)" />
            {!compact && (
              <text x={pad.l - 8} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--ink3)" className="font-mono tabular">
                {t}
              </text>
            )}
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={h - pad.b} y2={h - pad.b + 4} stroke="rgba(255,255,255,0.14)" />
            <text x={sx(t)} y={h - pad.b + 14} textAnchor="middle" fontSize={9} fill="var(--ink3)" className="font-mono tabular">
              {t}
            </text>
          </g>
        ))}
        <line x1={pad.l} x2={w - pad.r} y1={sy(15)} y2={sy(15)} stroke="var(--crit)" strokeOpacity={0.3} strokeDasharray="2 5" />
        {!compact && (
          <text x={w - pad.r} y={sy(15) - 4} textAnchor="end" fontSize={8} fill="var(--crit)" opacity={0.7} letterSpacing={1}>
            FAILURE THRESHOLD
          </text>
        )}

        <path d={band} fill="var(--crit)" fillOpacity={0.07} />
        <path d={smoothPath(historyPts)} fill="none" stroke="var(--accent)" strokeWidth={compact ? 1.6 : 2.2} strokeLinecap="round" pathLength={1} className={animate ? 'animate-draw' : undefined} style={{ ['--draw-dur' as string]: '1.1s' }} />
        <path d={smoothPath(predPts)} fill="none" stroke="var(--crit)" strokeWidth={compact ? 1.4 : 2} strokeDasharray="5 5" strokeLinecap="round" pathLength={1} className={animate ? 'animate-draw' : undefined} style={{ ['--draw-dur' as string]: '1s', animationDelay: '0.55s' }} />

        <line x1={sx(engine.currentCycle)} x2={sx(engine.currentCycle)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.25)" />
        <circle cx={sx(engine.currentCycle)} cy={sy(engine.health)} r={compact ? 3 : 4.5} fill="var(--accent)" stroke="var(--bg)" strokeWidth={2} />
        {!compact && (
          <g>
            <rect x={sx(engine.currentCycle) - 30} y={h - pad.b + 4} width={60} height={16} rx={4} fill="var(--accent)" />
            <text x={sx(engine.currentCycle)} y={h - pad.b + 15} textAnchor="middle" fontSize={9} fill="#0a0b0e" className="font-mono font-semibold">
              NOW {engine.currentCycle}
            </text>
          </g>
        )}

        <line x1={sx(prediction.failureCycle)} x2={sx(prediction.failureCycle)} y1={pad.t} y2={h - pad.b} stroke="var(--crit)" strokeOpacity={0.55} strokeDasharray="3 4" />
        {!compact && (
          <text x={sx(prediction.failureCycle) + 6} y={pad.t + 10} fontSize={8.5} fill="var(--crit)" letterSpacing={0.8} className="font-semibold">
            EST. FAILURE · {prediction.failureCycle}
          </text>
        )}

        {!compact &&
          bundle.anomalies.map((a) => (
            <g
              key={a.id}
              transform={`translate(${sx(a.cycle)},${sy(bundle.health[a.cycle])})`}
              className="cursor-pointer"
              onClick={(e) => {
                e.stopPropagation()
                onAnomalySelect?.(a)
              }}
            >
              <rect x={-3.6} y={-3.6} width={7.2} height={7.2} transform="rotate(45)" fill={a.severity === 'critical' ? 'var(--crit)' : a.severity === 'warning' ? 'var(--warn)' : 'var(--info)'} stroke="var(--bg)" strokeWidth={1.4} />
              <rect x={-7} y={-7} width={14} height={14} transform="rotate(45)" fill="transparent" />
            </g>
          ))}

        {scrub !== engine.currentCycle && (
          <g>
            <line x1={sx(scrub)} x2={sx(scrub)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.4)" strokeDasharray="1 3" />
            <circle cx={sx(scrub)} cy={sy(scrubHealth)} r={4} fill="var(--ink1)" stroke="var(--bg)" strokeWidth={2} />
          </g>
        )}

        {hoverCycle != null && hoverHealth != null && pos && hoverCycle >= 0 && hoverCycle <= domainX[1] && (
          <g>
            <line x1={sx(hoverCycle)} x2={sx(hoverCycle)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.18)" />
            <circle cx={sx(hoverCycle)} cy={sy(hoverHealth)} r={3} fill="rgba(255,255,255,0.7)" />
          </g>
        )}
      </svg>

      {hoverCycle != null && hoverHealth != null && pos && hoverCycle >= 0 && hoverCycle <= domainX[1] && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-line-2 bg-surface-3/95 px-2.5 py-2 shadow-pop backdrop-blur-sm"
          style={{ left: Math.min(Math.max(pos.x + 12, 8), w - 150), top: 12 }}
        >
          <div className="font-mono text-[10px] text-ink-3">CYCLE {hoverCycle}{hoverCycle > engine.currentCycle ? ' · PREDICTED' : ''}</div>
          <div className="font-mono text-[13px] font-semibold text-ink-1">
            {Math.round(hoverHealth)}<span className="text-[10px] text-ink-3"> health</span>
          </div>
          <div className="font-mono text-[10px] text-ink-2">
            RUL ≈ {Math.max(0, engine.rul - (hoverCycle - engine.currentCycle))} cycles
          </div>
        </div>
      )}
    </div>
  )
}

function cycleWhereHealthBelow(all: { c: number; h: number }[], threshold: number): number {
  for (const p of all) {
    if (p.h <= threshold) return p.c
  }
  return all[all.length - 1]?.c ?? 0
}
