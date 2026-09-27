import { useEffect, useMemo, useRef, useState } from 'react'
import { X, ChevronLeft, ChevronRight, Sparkles, Maximize2, Minimize2, Gauge, Activity, TrendingUp, AlertTriangle, Zap } from 'lucide-react'
import { getEngine, getTelemetry, getPrediction, getEngines } from '@/data/db'
import DegradationChart from '@/components/charts/DegradationChart'
import TelemetryChart from '@/components/telemetry/TelemetryChart'
import CycleScrubber from '@/components/timeline/CycleScrubber'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'
import { STATUS_META } from '@/types/engine'

const SLIDES = [
  { id: 'title', label: 'Title', icon: Sparkles },
  { id: 'engine', label: 'Engine', icon: Gauge },
  { id: 'telemetry', label: 'Telemetry', icon: Activity },
  { id: 'degradation', label: 'Degradation', icon: TrendingUp },
  { id: 'prediction', label: 'Prediction', icon: Zap },
  { id: 'risk', label: 'Risk', icon: AlertTriangle }
] as const

export default function Presentation() {
  const { active, engineId } = useAppStore((s) => s.presentation)
  const stop = useAppStore((s) => s.stopPresentation)
  const [idx, setIdx] = useState(0)
  const fsRef = useRef<HTMLDivElement>(null)

  const engine = useMemo(() => getEngine(engineId ?? '024'), [engineId])
  const bundle = useMemo(() => engine ? getTelemetry(engine.id) : undefined, [engine])
  const prediction = useMemo(() => engine ? getPrediction(engine.id) : undefined, [engine])

  useEffect(() => {
    if (active) {
      setIdx(0)
      requestAnimationFrame(() => fsRef.current?.requestFullscreen?.().catch(() => {}))
    }
  }, [active])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!active) return
      if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault()
        setIdx((i) => Math.min(i + 1, SLIDES.length - 1))
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        setIdx((i) => Math.max(i - 1, 0))
      } else if (e.key === 'Escape') {
        stop()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [active, stop])

  if (!active || !engine || !bundle || !prediction) return null

  const slide = SLIDES[idx]
  const scrub = idx === 3 ? engine.currentCycle : idx === 4 ? engine.currentCycle + Math.round(prediction.rul * 0.6) : engine.currentCycle

  return (
    <div
      ref={fsRef}
      className="fixed inset-0 z-[100] flex flex-col bg-bg"
      role="dialog"
      aria-modal="true"
      aria-label="Presentation mode"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-[var(--bg)] via-[var(--s1)] to-[var(--s2)]" />
      <div className="absolute top-6 right-6 z-10 flex items-center gap-2">
        <span className="rounded-lg border border-line bg-surface-2/80 px-3 py-1.5 text-[10px] font-medium text-ink-3">
          {idx + 1} / {SLIDES.length}
        </span>
        <button className="icon-btn h-8 w-8" onClick={stop} aria-label="Exit presentation">
          <Minimize2 size={15} />
        </button>
      </div>

      <div className="flex-1 overflow-hidden flex flex-col">
        {slide.id === 'title' && (
          <div className="flex h-full items-center justify-center px-8">
            <div className="text-center max-w-2xl animate-rise-in" style={{ animationDelay: '100ms' }}>
              <div className="mb-6 flex items-center justify-center">
                <div className="relative flex h-24 w-24 shrink-0 items-center justify-center">
                  <svg viewBox="0 0 32 32" className="h-24 w-24">
                    <path d="M16 4 L26 25 L16 21 L6 25 Z" fill="none" stroke="var(--accent)" strokeWidth={3} strokeLinejoin="round" />
                    <circle cx="16" cy="15" r="3" fill="var(--info)" />
                  </svg>
                </div>
              </div>
              <h1 className="font-display text-[clamp(36px,6vw,64px)] font-bold tracking-tight text-ink-1 leading-tight">
                AeroGuard AI
              </h1>
              <p className="mt-4 text-[clamp(18px,2.5vw,24px)] leading-relaxed text-ink-2">
                Aircraft Engine Intelligence
              </p>
              <p className="mt-6 text-[clamp(14px,2vw,18px)] text-ink-3 max-w-md mx-auto">
                Predictive maintenance operations for the NASA C-MAPSS turbofan fleet — real-time telemetry, RUL inference, and what-if simulation.
              </p>
              <div className="mt-10 flex items-center justify-center gap-4">
                <span className="chip">LSTM v2.4 · RMSE 19.8</span>
                <span className="chip">100 engines · FD001</span>
                <span className="chip">Live telemetry stream</span>
              </div>
            </div>
          </div>
        )}

        {slide.id === 'engine' && (
          <div className="flex h-full flex-col items-center justify-center px-8 gap-4">
            <div className="flex items-center justify-center gap-3 animate-rise-in">
              <span className="eyebrow">Engine identity</span>
              <span className="chip chip-active">Engine #{engine.id}</span>
              <span className="chip" style={{ borderColor: STATUS_META[engine.status].color, background: STATUS_META[engine.status].color + '10', color: STATUS_META[engine.status].color }}>
                {STATUS_META[engine.status].label}
              </span>
            </div>
            <div className="grid gap-4 md:grid-cols-3 max-w-3xl animate-rise-in" style={{ animationDelay: '120ms' }}>
              <div className="card p-6 text-center">
                <div className="eyebrow">Remaining Useful Life</div>
                <div className="mt-2 stat-num font-display text-[48px] font-bold" style={{ color: engine.rul < 20 ? 'var(--crit)' : engine.rul < 60 ? 'var(--warn)' : 'var(--ok)' }}>
                  {engine.rul}
                  <span className="text-[16px] font-normal text-ink-3"> cycles</span>
                </div>
              </div>
              <div className="card p-6 text-center">
                <div className="eyebrow">Health Index</div>
                <div className="mt-2 stat-num font-display text-[48px] font-bold" style={{ color: engine.health >= 70 ? 'var(--ok)' : engine.health >= 50 ? 'var(--warn)' : 'var(--crit)' }}>
                  {engine.health}%
                </div>
              </div>
              <div className="card p-6 text-center">
                <div className="eyebrow">Risk Score</div>
                <div className="mt-2 stat-num font-display text-[48px] font-bold" style={{ color: engine.risk >= 70 ? 'var(--crit)' : engine.risk >= 40 ? 'var(--warn)' : 'var(--ok)' }}>
                  {engine.risk}%
                </div>
              </div>
            </div>
            <div className="text-[11px] text-ink-3 animate-rise-in" style={{ animationDelay: '200ms' }}>
              Operating cycle {engine.currentCycle} · Fault: {engine.faultMode} · Dataset {engine.dataset}
            </div>
          </div>
        )}

        {slide.id === 'telemetry' && (
          <div className="flex h-full flex-col items-center justify-center px-4 gap-3">
            <div className="flex items-center justify-center gap-2 animate-rise-in">
              <span className="eyebrow">Live telemetry snapshot</span>
              <span className="chip">T24 · LPC Outlet Temp</span>
              <span className="chip">Ps30 · Static Pressure</span>
              <span className="chip">Nc · Core Speed</span>
            </div>
            <div className="flex-1 flex items-center justify-center animate-rise-in" style={{ animationDelay: '120ms' }}>
              <TelemetryChart
                bundle={bundle}
                primarySensor="T24"
                pinnedSensors={['Ps30', 'Nc']}
                scrub={engine.currentCycle}
                anomalies={bundle.anomalies}
                tempUnit="R"
                className="w-full max-w-4xl"
              />
            </div>
          </div>
        )}

        {slide.id === 'degradation' && (
          <div className="flex h-full flex-col items-center justify-center px-4 gap-3">
            <div className="flex items-center justify-center gap-2 animate-rise-in">
              <span className="eyebrow">Degradation timeline</span>
              <span className="chip">Solid = observed</span>
              <span className="chip" style={{ borderColor: 'var(--crit)', background: 'var(--crit)/10', color: 'var(--crit)' }}>Dashed = predicted</span>
              <span className="chip">Shaded = uncertainty</span>
            </div>
            <div className="flex-1 flex items-center justify-center animate-rise-in" style={{ animationDelay: '120ms' }}>
              <DegradationChart
                engine={engine}
                bundle={bundle}
                prediction={prediction}
                scrub={engine.currentCycle}
                compact={false}
                className="w-full max-w-4xl"
                animate={false}
              />
            </div>
            <div className="w-full max-w-4xl animate-rise-in" style={{ animationDelay: '200ms' }}>
              <CycleScrubber engine={engine} bundle={bundle} prediction={prediction} scrub={engine.currentCycle} onScrub={() => {}} className="mx-auto" />
            </div>
          </div>
        )}

        {slide.id === 'prediction' && (
          <div className="flex h-full flex-col items-center justify-center px-8 gap-4">
            <div className="flex items-center justify-center gap-2 animate-rise-in">
              <span className="eyebrow">RUL prediction</span>
              <span className="chip">LSTM v2.4 · {Math.round(prediction.confidence * 100)}% confidence</span>
            </div>
            <div className="grid gap-6 md:grid-cols-3 max-w-3xl animate-rise-in" style={{ animationDelay: '120ms' }}>
              <div className="card p-6 text-center">
                <div className="eyebrow">Estimated RUL</div>
                <div className="mt-2 stat-num font-display text-[52px] font-bold text-ink-1">
                  {prediction.rul}
                  <span className="text-[16px] font-normal text-ink-3"> cycles</span>
                </div>
                <div className="mt-1 text-[11px] text-ink-3">MAE {prediction.mae} cycles</div>
              </div>
              <div className="card p-6 text-center">
                <div className="eyebrow">Failure cycle</div>
                <div className="mt-2 stat-num font-display text-[52px] font-bold text-crit">
                  {prediction.failureCycle}
                </div>
                <div className="mt-1 text-[11px] text-ink-3">Estimated boundary</div>
              </div>
              <div className="card p-6 text-center">
                <div className="eyebrow">Model confidence</div>
                <div className="mt-2 stat-num font-display text-[52px] font-bold text-info">
                  {Math.round(prediction.confidence * 100)}%
                </div>
              </div>
            </div>
            <div className="max-w-2xl animate-rise-in" style={{ animationDelay: '200ms' }}>
              <p className="text-[13px] leading-relaxed text-ink-2 text-center">{prediction.insight}</p>
            </div>
          </div>
        )}

        {slide.id === 'risk' && (
          <div className="flex h-full flex-col items-center justify-center px-8 gap-4">
            <div className="flex items-center justify-center gap-2 animate-rise-in">
              <span className="eyebrow">Risk assessment</span>
              <span className="chip" style={{ borderColor: engine.risk >= 70 ? 'var(--crit)' : engine.risk >= 40 ? 'var(--warn)' : 'var(--ok)', background: engine.risk >= 70 ? 'var(--crit)/10' : engine.risk >= 40 ? 'var(--warn)/10' : 'var(--ok)/10', color: engine.risk >= 70 ? 'var(--crit)' : engine.risk >= 40 ? 'var(--warn)' : 'var(--ok)' }}>
                {engine.risk >= 70 ? 'CRITICAL' : engine.risk >= 40 ? 'WARNING' : 'NOMINAL'}
              </span>
            </div>
            <div className="card p-6 max-w-md text-center animate-rise-in" style={{ animationDelay: '120ms' }}>
              <div className="eyebrow">Fleet context</div>
              <div className="mt-4 stat-num font-display text-[56px] font-bold" style={{ color: engine.risk >= 70 ? 'var(--crit)' : engine.risk >= 40 ? 'var(--warn)' : 'var(--ok)' }}>
                {engine.risk}%
              </div>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-2">
                {engine.risk >= 70
                  ? 'Immediate inspection recommended. This engine is in the critical cohort.'
                  : engine.risk >= 40
                    ? 'Elevated degradation signal. Schedule maintenance within the planning window.'
                    : 'Operating within normal envelope. Continue standard monitoring.'}
              </p>
              <div className="mt-4 flex justify-center gap-2">
                <span className="chip">{getEngines().filter((e) => e.status === 'critical').length} critical</span>
                <span className="chip">{getEngines().filter((e) => e.status === 'warning').length} warning</span>
                <span className="chip">{getEngines().filter((e) => e.status === 'nominal').length} nominal</span>
              </div>
            </div>
          </div>
        )}
      </div>

<div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-4 py-2 rounded-full border border-line bg-surface-2/80 text-[10px] text-ink-3 backdrop-blur">
        <span>← →</span> navigate · <span className="kbd">Esc</span> exit · <span className="kbd">F</span> fullscreen
      </div>
    </div>
  )
}