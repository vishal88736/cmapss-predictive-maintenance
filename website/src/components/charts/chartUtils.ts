import { useCallback, useEffect, useRef, useState } from 'react'
import { clamp } from '@/lib/prng'

export interface Size {
  w: number
  h: number
}

export function useMeasure<T extends HTMLElement>(): [React.RefObject<T>, Size] {
  const ref = useRef<T>(null)
  const [size, setSize] = useState<Size>({ w: 0, h: 0 })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect
      setSize((s) => (Math.abs(s.w - r.width) > 0.5 || Math.abs(s.h - r.height) > 0.5 ? { w: r.width, h: r.height } : s))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref as React.RefObject<T>, size]
}

export function scaleLinear(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain
  const [r0, r1] = range
  const m = (r1 - r0) / (d1 - d0 || 1e-9)
  const s = (v: number) => r0 + (v - d0) * m
  s.invert = (v: number) => d0 + (v - r0) / m
  return s
}

export function niceTicks(min: number, max: number, count = 5): number[] {
  const span = max - min
  if (span <= 0) return [min]
  const step0 = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(step0)))
  const norm = step0 / mag
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag
  const start = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let v = start; v <= max + step * 0.01; v += step) ticks.push(+v.toFixed(10))
  if (ticks.length < 2) ticks.push(min, max)
  return ticks
}

export function linePath(pts: [number, number][]): string {
  if (!pts.length) return ''
  return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('')
}

export function smoothPath(pts: [number, number][]): string {
  if (pts.length < 3) return linePath(pts)
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`
  for (let i = 1; i < pts.length - 1; i++) {
    const xc = (pts[i][0] + pts[i + 1][0]) / 2
    const yc = (pts[i][1] + pts[i + 1][1]) / 2
    d += ` Q${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)} ${xc.toFixed(1)},${yc.toFixed(1)}`
  }
  const last = pts[pts.length - 1]
  d += ` L${last[0].toFixed(1)},${last[1].toFixed(1)}`
  return d
}

export function areaPath(pts: [number, number][], baseline: number): string {
  if (!pts.length) return ''
  return `${linePath(pts)} L${pts[pts.length - 1][0].toFixed(1)},${baseline.toFixed(1)} L${pts[0][0].toFixed(1)},${baseline.toFixed(1)} Z`
}

export function bandPath(top: [number, number][], bottom: [number, number][]): string {
  if (!top.length || !bottom.length) return ''
  const rev = [...bottom].reverse()
  return `${linePath(top)} L${rev[0][0].toFixed(1)},${rev[0][1].toFixed(1)} ${linePath(rev).slice(1)} Z`
}

export const SERIES_COLORS = ['#7C8CFF', '#4CC2FF', '#34D399', '#F5B84A', '#F0625D', '#B58CFF', '#FF8A65', '#80CBC4', '#E9EBEE']

export function useTooltip<T extends HTMLElement>() {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const ref = useRef<T>(null)
  const onMove = useCallback((e: React.PointerEvent) => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    setPos({ x: e.clientX - r.left, y: e.clientY - r.top })
  }, [])
  const onLeave = useCallback(() => setPos(null), [])
  return { ref, pos, onMove, onLeave }
}

export function nearestIndex(xs: number[], x: number): number {
  let best = 0
  let bestD = Infinity
  for (let i = 0; i < xs.length; i++) {
    const d = Math.abs(xs[i] - x)
    if (d < bestD) {
      bestD = d
      best = i
    }
  }
  return best
}

export function pointerXToValue(e: React.PointerEvent, el: HTMLElement | null, domain: [number, number]): number | null {
  if (!el) return null
  const r = el.getBoundingClientRect()
  const t = clamp((e.clientX - r.left) / r.width, 0, 1)
  return domain[0] + t * (domain[1] - domain[0])
}
