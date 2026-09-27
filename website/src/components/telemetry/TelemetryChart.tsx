import { useMemo } from 'react'
import type { TelemetryBundle, Anomaly } from '@/types/telemetry'
import { useMeasure, scaleLinear, niceTicks, linePath, smoothPath, bandPath, useTooltip, nearestIndex, SERIES_COLORS } from '@/components/charts/chartUtils'
import { fmtSensor } from '@/lib/format'
import { cn } from '@/lib/utils'

interface Props {
  bundle: TelemetryBundle
  primarySensor: string
  pinnedSensors: string[]
  scrub: number
  anomalies: Anomaly[]
  tempUnit: 'R' | 'K'
  onAnomalySelect?: (a: Anomaly) => void
  onScrub?: (cycle: number) => void
  animate?: boolean
  className?: string
}

export default function TelemetryChart({ bundle, primarySensor, pinnedSensors, scrub, anomalies, tempUnit, onAnomalySelect, onScrub, animate = true, className }: Props) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const { ref: ttRef, pos, onMove, onLeave } = useTooltip<HTMLDivElement>()
  const pad = { l: 52, r: 14, t: 14, b: 22 }
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const innerW = w - pad.l - pad.r
  const innerH = h - pad.t - pad.b
  const cycles = bundle.cycles

  const primarySeries = bundle.sensors[primarySensor] ?? []
  const primaryDomain = useMemo<[number, number]>(() => {
    const vals = primarySeries
    if (!vals.length) return [0, 1]
    let min = Infinity
    let max = -Infinity
    for (const v of vals) {
      if (v < min) min = v
      if (v > max) max = v
    }
    const padv = (max - min) * 0.12 || Math.abs(max) * 0.05 || 1
    return [min - padv, max + padv]
  }, [primarySeries])

  const sx = scaleLinear([0, cycles[cycles.length - 1] ?? 1], [pad.l, w - pad.r])
  const sy = scaleLinear(primaryDomain, [h - pad.b, pad.t])

  const primaryPts = useMemo(() => cycles.map((c, i) => [sx(c), sy(primarySeries[i])] as [number, number]), [cycles, primarySeries, sx, sy])
  const expected = bundle.expected[primarySensor]

  const overlays = useMemo(
    () =>
      pinnedSensors
        .filter((s) => s !== primarySensor && bundle.sensors[s])
        .map((sensorId, idx) => {
          const vals = bundle.sensors[sensorId]
          let min = Infinity
          let max = -Infinity
          for (const v of vals) {
            if (v < min) min = v
            if (v > max) max = v
          }
          const span = max - min || 1
          const norm = scaleLinear([min - span * 0.08, max + span * 0.08], [h - pad.b, pad.t])
          return {
            sensorId,
            color: SERIES_COLORS[(idx + 1) % SERIES_COLORS.length],
            pts: cycles.map((c, i) => [sx(c), norm(vals[i])] as [number, number])
          }
        }),
    [pinnedSensors, bundle, cycles, sx, h, pad.b, pad.t]
  )

  const chartAnomalies = useMemo(() => anomalies.filter((a) => a.sensorId === primarySensor || pinnedSensors.includes(a.sensorId)), [anomalies, primarySensor, pinnedSensors])
  const xTicks = useMemo(() => niceTicks(0, cycles[cycles.length - 1] ?? 1, 6), [cycles])
  const yTicks = useMemo(() => niceTicks(primaryDomain[0], primaryDomain[1], 5), [primaryDomain])

  const hoverIdx = pos ? Math.max(0, Math.min(cycles.length - 1, Math.round(sx.invert(pos.x)))) : null

  return (
    <div
      ref={(el) => {
        ;
        (ref as React.MutableRefObject<HTMLDivElement | null>).current = el
        ;
        (ttRef as React.MutableRefObject<HTMLDivElement | null>).current = el
      }}
      className={cn('relative touch-none select-none', className)}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
      onPointerDown={(e) => {
        if (!onScrub) return
        const r = e.currentTarget.getBoundingClientRect()
        onScrub(Math.max(0, Math.min(Math.round(sx.invert(e.clientX - r.left)), cycles[cycles.length - 1])))
      }}
      role="img"
      aria-label={`Telemetry chart for ${primarySensor} with ${overlays.length} overlaid sensors`}
    >
      <svg width={w} height={h} className="block">
        {expected && (
          <g>
            <rect x={pad.l} y={Math.min(sy(expected[1]), sy(expected[0]))} width={innerW} height={Math.abs(sy(expected[0]) - sy(expected[1]))} fill="var(--info)" fillOpacity={0.05} />
            <line x1={pad.l} x2={w - pad.r} y1={sy(expected[0])} y2={sy(expected[0])} stroke="var(--info)" strokeOpacity={0.25} strokeDasharray="2 5" />
            <line x1={pad.l} x2={w - pad.r} y1={sy(expected[1])} y2={sy(expected[1])} stroke="var(--info)" strokeOpacity={0.25} strokeDasharray="2 5" />
          </g>
        )}
        {yTicks.map((t: number) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.045)" />
            <text x={pad.l - 7} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--ink3)" className="font-mono tabular">
              {primarySensor === 'VI' ? t.toFixed(1) : Math.abs(t) >= 1000 ? `${(t / 1000).toFixed(1)}k` : t.toFixed(Math.abs(primaryDomain[1]) < 10 ? 1 : 0)}
            </text>
          </g>
        ))}
        {xTicks.map((t: number) => (
          <text key={t} x={sx(t)} y={h - 6} textAnchor="middle" fontSize={9} fill="var(--ink3)" className="font-mono tabular">
            {t}
          </text>
        ))}

        {overlays.map((o) => (
          <path key={o.sensorId} d={smoothPath(o.pts)} fill="none" stroke={o.color} strokeWidth={1.3} strokeOpacity={0.85} pathLength={1} className={animate ? 'animate-draw' : undefined} style={{ ['--draw-dur' as string]: '1.2s' }} />
        ))}
        <path d={smoothPath(primaryPts)} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" pathLength={1} className={animate ? 'animate-draw' : undefined} style={{ ['--draw-dur' as string]: '1.3s' }} />

        {chartAnomalies.map((a) => {
          const isPrimary = a.sensorId === primarySensor
          const y = isPrimary ? sy(primarySeries[a.cycle]) : pad.t + 6
          return (
            <g key={a.id} className="cursor-pointer" onClick={(e) => { e.stopPropagation(); onAnomalySelect?.(a) }}>
              <rect x={sx(a.cycle) - 4} y={y - 4} width={8} height={8} transform={`rotate(45 ${sx(a.cycle)} ${y})`} fill={isPrimary ? (a.severity === 'critical' ? 'var(--crit)' : a.severity === 'warning' ? 'var(--warn)' : 'var(--info)') : 'none'} stroke={isPrimary ? 'var(--bg)' : a.severity === 'warning' ? 'var(--warn)' : 'var(--info)'} strokeWidth={1.2} />
              <rect x={sx(a.cycle) - 8} y={y - 8} width={16} height={16} fill="transparent" />
            </g>
          )
        })}

        <line x1={sx(scrub)} x2={sx(scrub)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.35)" strokeDasharray="1 3" />
        <circle cx={sx(scrub)} cy={sy(primarySeries[scrub] ?? 0)} r={4} fill="var(--ink1)" stroke="var(--bg)" strokeWidth={2} />

        {hoverIdx != null && pos && (
          <g>
            <line x1={sx(cycles[hoverIdx])} x2={sx(cycles[hoverIdx])} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.16)" />
            <circle cx={sx(cycles[hoverIdx])} cy={sy(primarySeries[hoverIdx])} r={3} fill="rgba(255,255,255,0.8)" />
          </g>
        )}
      </svg>

      {hoverIdx != null && pos && (
        <div className="pointer-events-none absolute z-10 min-w-[140px] rounded-lg border border-line-2 bg-surface-3/95 px-2.5 py-2 shadow-pop backdrop-blur-sm" style={{ left: Math.min(Math.max(pos.x + 12, 8), Math.max(w - 160, 8)), top: 10 }}>
          <div className="font-mono text-[10px] text-ink-3">CYCLE {cycles[hoverIdx]}</div>
          <div className="mt-0.5 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: 'var(--accent)' }} />
            <span className="font-mono text-[12px] font-semibold">{fmtSensor(primarySensor, primarySeries[hoverIdx], tempUnit)}</span>
          </div>
          {overlays.map((o) => (
            <div key={o.sensorId} className="mt-0.5 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: o.color }} />
              <span className="font-mono text-[11px] text-ink-2">{fmtSensor(o.sensorId, bundle.sensors[o.sensorId][hoverIdx], tempUnit)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
