import { useMemo } from 'react'
import { useMeasure, scaleLinear, niceTicks, linePath, smoothPath, bandPath } from './chartUtils'
import { cn } from '@/lib/utils'

interface ScatterProps {
  points: { x: number; y: number }[]
  identity?: boolean
  xLabel: string
  yLabel: string
  className?: string
  color?: string
}

export function ScatterChart({ points, identity = true, xLabel, yLabel, className, color = 'var(--accent)' }: ScatterProps) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const pad = { l: 40, r: 12, t: 12, b: 30 }
  const max = useMemo(() => Math.max(...points.flatMap((p) => [p.x, p.y]), 10) * 1.05, [points])
  const s = scaleLinear([0, max], [pad.l, w - pad.r])
  const sy = scaleLinear([0, max], [h - pad.b, pad.t])
  const ticks = niceTicks(0, max, 5)
  return (
    <div ref={ref} className={cn('h-full w-full', className)}>
      <svg width={w} height={h} className="block" role="img" aria-label={`${yLabel} versus ${xLabel}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={s(t)} x2={s(t)} y1={pad.t} y2={h - pad.b} stroke="rgba(255,255,255,0.04)" />
            <text x={s(t)} y={h - pad.b + 13} textAnchor="middle" fontSize={9} fill="var(--ink3)" className="font-mono">
              {t}
            </text>
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--ink3)" className="font-mono">
              {t}
            </text>
          </g>
        ))}
        {identity && <line x1={s(0)} y1={sy(0)} x2={s(max)} y2={sy(max)} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />}
        {points.map((p, i) => (
          <circle key={i} cx={s(p.x)} cy={sy(p.y)} r={2.4} fill={color} fillOpacity={0.5} />
        ))}
        <text x={w - pad.r} y={h - 4} textAnchor="end" fontSize={8.5} fill="var(--ink3)" letterSpacing={1}>
          {xLabel.toUpperCase()}
        </text>
        <text x={pad.l} y={pad.t - 2} fontSize={8.5} fill="var(--ink3)" letterSpacing={1}>
          {yLabel.toUpperCase()}
        </text>
      </svg>
    </div>
  )
}

interface HistProps {
  bins: { x: number; count: number }[]
  color?: string
  xLabel?: string
  className?: string
}

export function Histogram({ bins, color = 'var(--accent)', xLabel, className }: HistProps) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const pad = { l: 34, r: 12, t: 12, b: 26 }
  const maxCount = Math.max(...bins.map((b) => b.count), 1)
  const x0 = bins[0]?.x ?? 0
  const x1 = bins[bins.length - 1]?.x ?? 1
  const s = scaleLinear([x0, x1], [pad.l, w - pad.r])
  const sy = scaleLinear([0, maxCount * 1.1], [h - pad.b, pad.t])
  const bw = Math.max(1, (w - pad.l - pad.r) / bins.length - 2)
  return (
    <div ref={ref} className={cn('h-full w-full', className)}>
      <svg width={w} height={h} className="block" role="img" aria-label={xLabel ?? 'histogram'}>
        {niceTicks(0, maxCount * 1.1, 4).map((t) => (
          <text key={t} x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--ink3)" className="font-mono">
            {Math.round(t)}
          </text>
        ))}
        {bins.map((b, i) => (
          <rect key={i} x={s(b.x) - bw / 2} y={sy(b.count)} width={bw} height={Math.max(1, h - pad.b - sy(b.count))} fill={color} fillOpacity={0.45} rx={2} className="transition-all duration-500" style={{ animation: 'rise-in 0.4s ease-out both', animationDelay: `${i * 12}ms` }} />
        ))}
        <text x={w - pad.r} y={h - 4} textAnchor="end" fontSize={8.5} fill="var(--ink3)" letterSpacing={1}>
          {(xLabel ?? '').toUpperCase()}
        </text>
      </svg>
    </div>
  )
}

interface RangeBarsProps {
  bars: { range: string; mae: number; count: number }[]
  className?: string
}

export function RangeBars({ bars, className }: RangeBarsProps) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const pad = { l: 34, r: 12, t: 14, b: 24 }
  const max = Math.max(...bars.map((b) => b.mae), 1)
  const sy = scaleLinear([0, max * 1.15], [h - pad.b, pad.t])
  const bw = (w - pad.l - pad.r) / bars.length
  return (
    <div ref={ref} className={cn('h-full w-full', className)}>
      <svg width={w} height={h} className="block" role="img" aria-label="Error by RUL range">
        {niceTicks(0, max * 1.15, 4).map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.04)" />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--ink3)" className="font-mono">
              {t}
            </text>
          </g>
        ))}
        {bars.map((b, i) => (
          <g key={b.range}>
            <rect x={pad.l + i * bw + 4} y={sy(b.mae)} width={bw - 8} height={h - pad.b - sy(b.mae)} fill="var(--info)" fillOpacity={0.4} rx={3} style={{ animation: 'rise-in 0.4s ease-out both', animationDelay: `${i * 40}ms` }} />
            <text x={pad.l + i * bw + bw / 2} y={sy(b.mae) - 5} textAnchor="middle" fontSize={9} fill="var(--ink2)" className="font-mono">
              {b.mae}
            </text>
            <text x={pad.l + i * bw + bw / 2} y={h - pad.b + 13} textAnchor="middle" fontSize={8.5} fill="var(--ink3)">
              {b.range}
            </text>
          </g>
        ))}
      </svg>
    </div>
  )
}

interface DualLineProps {
  series: { pts: { x: number; y: number }[]; color: string; dashed?: boolean }[]
  yLabel?: string
  className?: string
}

export function DualLine({ series, className }: DualLineProps) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const pad = { l: 34, r: 12, t: 12, b: 22 }
  const allY = series.flatMap((s) => s.pts.map((p) => p.y))
  const minY = Math.min(...allY)
  const maxY = Math.max(...allY)
  const sx = scaleLinear([series[0].pts[0]?.x ?? 0, series[0].pts[series[0].pts.length - 1]?.x ?? 1], [pad.l, w - pad.r])
  const sy = scaleLinear([minY, maxY], [h - pad.b, pad.t])
  const xTickIdx = [0, Math.floor(series[0].pts.length / 2), series[0].pts.length - 1]
  return (
    <div ref={ref} className={cn('h-full w-full', className)}>
      <svg width={w} height={h} className="block" role="img" aria-label="trend chart">
        {niceTicks(minY, maxY, 4).map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={sy(t)} y2={sy(t)} stroke="rgba(255,255,255,0.04)" />
            <text x={pad.l - 6} y={sy(t) + 3} textAnchor="end" fontSize={9} fill="var(--ink3)" className="font-mono">
              {Math.abs(maxY) < 10 ? t.toFixed(1) : Math.round(t)}
            </text>
          </g>
        ))}
        {xTickIdx.map((i) =>
          series[0].pts[i] ? (
            <text key={i} x={sx(series[0].pts[i].x)} y={h - 6} textAnchor="middle" fontSize={9} fill="var(--ink3)" className="font-mono">
              {series[0].pts[i].x}
            </text>
          ) : null
        )}
        {series.map((s, si) => (
          <path key={si} d={smoothPath(s.pts.map((p) => [sx(p.x), sy(p.y)] as [number, number]))} fill="none" stroke={s.color} strokeWidth={1.6} strokeDasharray={s.dashed ? '4 4' : undefined} pathLength={1} className="animate-draw" style={{ ['--draw-dur' as string]: '1s' }} />
        ))}
      </svg>
    </div>
  )
}

interface MultiLineProps {
  lines: { label: string; points: { x: number; y: number }[]; color: string }[]
  className?: string
}

export function MultiLine({ lines, className }: MultiLineProps) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 100)
  const h = Math.max(size.h, 100)
  const pad = { l: 30, r: 10, t: 10, b: 18 }
  const allY = lines.flatMap((l) => l.points.map((p) => p.y))
  const sy = scaleLinear([Math.min(...allY) * 0.95, Math.max(...allY) * 1.05], [h - pad.b, pad.t])
  const sx = scaleLinear([0, Math.max(...lines[0].points.map((p) => p.x), 1)], [pad.l, w - pad.r])
  return (
    <div ref={ref} className={cn('h-full w-full', className)}>
      <svg width={w} height={h} className="block" role="img" aria-label="cohort degradation curves">
        {lines.map((l) => (
          <path key={l.label} d={linePath(l.points.map((p) => [sx(p.x), sy(p.y)] as [number, number]))} fill="none" stroke={l.color} strokeWidth={1.6} strokeLinecap="round" />
        ))}
        {[0, 50, 100].map((t) => (
          <text key={t} x={sx(t)} y={h - 6} textAnchor={t === 0 ? 'start' : t === 100 ? 'end' : 'middle'} fontSize={8.5} fill="var(--ink3)" className="font-mono">
            {t}%
          </text>
        ))}
      </svg>
    </div>
  )
}

interface AttributionBarsProps {
  items: { sensorId: string; weight: number }[]
  nameOf: (id: string) => string
  className?: string
  compact?: boolean
}

export function AttributionBars({ items, nameOf, className, compact }: AttributionBarsProps) {
  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {items.map((a, i) => (
        <div key={a.sensorId} className="group flex items-center gap-3">
          <span className="w-[130px] shrink-0 truncate text-[11px] text-ink-2 transition-colors group-hover:text-ink-1">{nameOf(a.sensorId)}</span>
          <div className="h-[7px] flex-1 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full transition-[width] duration-700"
              style={{
                width: `${a.weight * 100}%`,
                background: `linear-gradient(90deg, rgba(124,140,255,0.35), var(--accent))`,
                animation: 'rise-in 0.4s ease-out both',
                animationDelay: `${i * 45}ms`
              }}
            />
          </div>
          <span className="stat-num w-10 shrink-0 text-right text-[11px] text-ink-3">{a.weight.toFixed(2)}</span>
        </div>
      ))}
      {!compact && <p className="mt-1 text-[10px] leading-relaxed text-ink-3">MODEL OUTPUT — integrated gradient attribution over the last 30 cycles. Derived from raw sensor trajectories, not direct measurements.</p>}
    </div>
  )
}
