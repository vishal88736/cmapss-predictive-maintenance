import { useMeasure, scaleLinear, smoothPath, linePath } from './chartUtils'
import { cn } from '@/lib/utils'

interface SparklineProps {
  values: number[]
  color?: string
  className?: string
  strokeWidth?: number
  fill?: boolean
}

export function Sparkline({ values, color = 'var(--accent)', className, strokeWidth = 1.5, fill = true }: SparklineProps) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 2)
  const h = Math.max(size.h, 2)
  let min = Infinity
  let max = -Infinity
  for (const v of values) {
    if (v < min) min = v
    if (v > max) max = v
  }
  const sx = scaleLinear([0, values.length - 1 || 1], [1, w - 1])
  const sy = scaleLinear([min, max === min ? min + 1 : max], [h - 1.5, 1.5])
  const pts = values.map((v, i) => [sx(i), sy(v)] as [number, number])
  const d = smoothPath(pts)
  return (
    <div ref={ref} className={cn('h-full w-full', className)}>
      <svg width={w} height={h} className="block">
        {fill && <path d={`${d} L${pts[pts.length - 1][0]},${h} L${pts[0][0]},${h} Z`} fill={color} fillOpacity={0.08} />}
        <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinecap="round" />
      </svg>
    </div>
  )
}

interface RadialScoreProps {
  value: number
  size?: number
  stroke?: number
  color?: string
  trackColor?: string
  children?: React.ReactNode
  className?: string
}

export function RadialScore({ value, size = 200, stroke = 10, color = 'var(--accent)', trackColor = 'rgba(255,255,255,0.06)', children, className }: RadialScoreProps) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const sweep = 0.78
  const arcLen = c * sweep
  const filled = arcLen * Math.max(0, Math.min(100, value)) / 100
  const startAngle = 222
  return (
    <div className={cn('relative', className)} style={{ width: size, height: size }}>
      <svg width={size} height={size} className="block -rotate-[-0deg]" viewBox={`0 0 ${size} ${size}`}>
        <g transform={`rotate(${startAngle} ${size / 2} ${size / 2})`}>
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={trackColor} strokeWidth={stroke} strokeDasharray={`${arcLen} ${c}`} strokeLinecap="round" />
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke} strokeDasharray={`${filled} ${c}`} strokeLinecap="round" pathLength={undefined} style={{ transition: 'stroke-dasharray 1s cubic-bezier(0.22,1,0.36,1)' }} />
        </g>
        {Array.from({ length: 28 }, (_, i) => {
          const a = ((startAngle + (sweep * 360 * i) / 27.999) * Math.PI) / 180
          const rr = r - stroke / 2 - 5
          const cx = size / 2 + Math.cos(a) * (r + stroke / 2 + 5)
          const cy = size / 2 + Math.sin(a) * (r + stroke / 2 + 5)
          const cx2 = size / 2 + Math.cos(a) * (r + stroke / 2 + 9)
          const cy2 = size / 2 + Math.sin(a) * (r + stroke / 2 + 9)
          const active = (i / 28) * 100 <= value
          return (
            <g key={i}>
              <line x1={cx} y1={cy} x2={cx2} y2={cy2} stroke={active ? color : 'rgba(255,255,255,0.1)'} strokeWidth={1.4} opacity={active ? 0.9 : 0.6} />
              <line x1={cx} y1={cy} x2={cx} y2={cy} stroke="none" />
              <circle cx={size / 2 + Math.cos(a) * rr} cy={size / 2 + Math.sin(a) * rr} r={0} />
            </g>
          )
        })}
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">{children}</div>
    </div>
  )
}
