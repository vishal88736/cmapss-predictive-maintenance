import { useEffect, useMemo, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ChevronLeft, ChevronRight, X, ListCollapse, ArrowRight, Maximize2, Layers } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import DegradationChart from '@/components/charts/DegradationChart'
import CycleScrubber from '@/components/timeline/CycleScrubber'
import TelemetryChart from '@/components/telemetry/TelemetryChart'
import { getEngines, getTelemetry, getPrediction, getEngine } from '@/data/db'
import { SENSOR_CHANNELS } from '@/data/sensors'
import { SENSOR_MAP } from '@/data/sensors'
import type { Engine } from '@/types/engine'
import { useAppStore } from '@/store/useAppStore'
import { pushRecent } from '@/commands/registry'
import { cn } from '@/lib/utils'

type Metric = 'health' | 'rul' | 'risk' | 'temp' | 'pressure' | 'speed' | 'vibration'

const METRIC_CHIPS: { key: Metric; label: string; sensor?: string }[] = [
  { key: 'health', label: 'Health' },
  { key: 'rul', label: 'RUL' },
  { key: 'risk', label: 'Risk' },
  { key: 'temp', label: 'Temperature', sensor: 'T24' },
  { key: 'pressure', label: 'Pressure', sensor: 'Ps30' },
  { key: 'speed', label: 'Speed', sensor: 'Nc' },
  { key: 'vibration', label: 'Vibration', sensor: 'VI' }
]

export default function ComparePage() {
  const { a, b } = useParams()
  const navigate = useNavigate()
  const openExplorer = useAppStore((s) => s.openExplorer)
  const scrubState = useAppStore((s) => s.scrubCycle)
  const setScrub = useAppStore((s) => s.setScrub)
  const primarySensor = useAppStore((s) => s.primarySensor)
  const setPrimarySensor = useAppStore((s) => s.setPrimarySensor)
  const pinnedSensors = useAppStore((s) => s.pinnedSensors)
  const tempUnit = useAppStore((s) => s.tempUnit)
  const [metric, setMetric] = useState<Metric>('health')
  const [scrub, setScrubLocal] = useState(0)

  const eA = useMemo(() => getEngine(a ?? '024'), [a])
  const eB = useMemo(() => getEngine(b ?? '071'), [b])
  const bundleA = useMemo(() => eA ? getTelemetry(eA.id) : undefined, [eA])
  const bundleB = useMemo(() => eB ? getTelemetry(eB.id) : undefined, [eB])
  const predA = useMemo(() => eA ? getPrediction(eA.id) : undefined, [eA])
  const predB = useMemo(() => eB ? getPrediction(eB.id) : undefined, [eB])

  useEffect(() => {
    if (eA) pushRecent(eA.id)
    if (eB) pushRecent(eB.id)
    const initial = Math.min(eA?.currentCycle ?? 0, eB?.currentCycle ?? 0)
    setScrubLocal(initial)
    setScrub(eA!.id, initial)
    setScrub(eB!.id, initial)
  }, [eA, eB, setScrub])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        const step = e.shiftKey ? 10 : 1
        const dir = e.key === 'ArrowRight' ? 1 : -1
        if (eA && eB && predA && predB) {
          const max = Math.min(eA.currentCycle + predA.rul, eB.currentCycle + predB.rul)
          setScrubLocal((s) => Math.max(0, Math.min(s + step * dir, max)))
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [eA, eB, predA, predB])

  useEffect(() => {
    setScrub(eA!.id, scrub)
    setScrub(eB!.id, scrub)
  }, [scrub, eA, eB, setScrub])

  if (!eA || !eB || !bundleA || !bundleB || !predA || !predB) {
    return <div className="flex h-full items-center justify-center text-ink-3">Loading…</div>
  }

  const scrubA = Math.min(scrub, eA.currentCycle + predA.rul)
  const scrubB = Math.min(scrub, eB.currentCycle + predB.rul)

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Split-Screen Analysis"
        title="Engine Comparison"
        sub="Synchronized timelines — scrub both engines together."
        actions={
          <div className="flex items-center gap-1.5">
            <div className="flex gap-0.5" role="tablist" aria-label="Compare metric">
              {METRIC_CHIPS.map((m) => (
                <button
                  key={m.key}
                  role="tab"
                  aria-selected={metric === m.key}
                  onClick={() => setMetric(m.key)}
                  className={cn('chip', metric === m.key && 'chip-active')}
                >
                  {m.label}
                </button>
              ))}
            </div>
            <button className="btn h-8 text-[11px]" onClick={() => navigate('/predictions')}>
              <X size={13} /> Exit
            </button>
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-2 text-[10px] text-ink-3">
          <span>Engine #{eA.id} — RUL {eA.rul} · Risk {eA.risk}% · Health {eA.health}%</span>
          <span className="mx-2">vs</span>
          <span>Engine #{eB.id} — RUL {eB.rul} · Risk {eB.risk}% · Health {eB.health}%</span>
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto">
        <div className="space-y-3 px-4 py-3.5 md:px-6">
          <div className="grid gap-3 lg:grid-cols-2">
            <section className="card overflow-hidden" aria-label={`Engine #{eA.id} degradation`}>
              <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
                <span className="flex items-center gap-2 font-semibold">
                  <span className="stat-num">#{eA.id}</span>
                  <span className="chip !py-0.5 !text-[9px]" style={{ borderColor: 'var(--accent)', background: 'var(--accent)/10', color: 'var(--accent)' }}>A</span>
                </span>
                <span className="text-[10px] text-ink-3">RUL {eA.rul} · cycle {eA.currentCycle}</span>
              </div>
              <div className="h-[220px] p-2">
                <DegradationChart
                  engine={eA}
                  bundle={bundleA}
                  prediction={predA}
                  scrub={scrubA}
                  onScrub={() => {}}
                  compact
                  animate={false}
                  className="h-full"
                />
              </div>
            </section>

            <section className="card overflow-hidden" aria-label={`Engine #{eB.id} degradation`}>
              <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
                <span className="flex items-center gap-2 font-semibold">
                  <span className="stat-num">#{eB.id}</span>
                  <span className="chip !py-0.5 !text-[9px]" style={{ borderColor: 'var(--crit)', background: 'var(--crit)/10', color: 'var(--crit)' }}>B</span>
                </span>
                <span className="text-[10px] text-ink-3">RUL {eB.rul} · cycle {eB.currentCycle}</span>
              </div>
              <div className="h-[220px] p-2">
                <DegradationChart
                  engine={eB}
                  bundle={bundleB}
                  prediction={predB}
                  scrub={scrubB}
                  onScrub={() => {}}
                  compact
                  animate={false}
                  className="h-full"
                />
              </div>
            </section>
          </div>

          <div className="card overflow-hidden">
            <div className="border-b border-line px-4 py-2.5">
              <span className="panel-title">Synchronized cycle scrubber</span>
            </div>
            <div className="p-3">
              <CycleScrubber
                engine={eA}
                bundle={bundleA}
                prediction={predA}
                scrub={scrub}
                onScrub={setScrubLocal}
              />
            </div>
          </div>

          {metric === 'health' || metric === 'rul' || metric === 'risk' ? (
            <section className="card overflow-hidden" aria-label="Delta comparison">
              <div className="border-b border-line px-4 py-3">
                <span className="panel-title">Instant delta at cycle {scrub}</span>
              </div>
              <div className="p-4">
                <div className="grid gap-2 md:grid-cols-3">
                  {([
                    ['Health', 'health', 'health'],
                    ['RUL', 'rul', 'rul'],
                    ['Risk', 'risk', 'risk']
                  ] as const).map(([label, _kA, _kB]) => {
                    const getVal = (e: Engine, b: typeof bundleA, p: typeof predA, s: number) => {
                      if (label === 'Health') {
                        const h = s <= e.currentCycle ? b.health[s] : s <= e.currentCycle + p.rul ? Math.max(10, e.health - (e.health - 14) * Math.pow((s - e.currentCycle) / p.rul, 2.3)) : 10
                        return Math.round(h)
                      }
                      if (label === 'RUL') return s <= e.currentCycle ? e.rul + (e.currentCycle - s) : Math.max(0, e.rul - (s - e.currentCycle))
                      const h = s <= e.currentCycle ? b.health[s] : s <= e.currentCycle + p.rul ? Math.max(10, e.health - (e.health - 14) * Math.pow((s - e.currentCycle) / p.rul, 2.3)) : 10
                      return Math.round(riskFromRul(s <= e.currentCycle ? e.rul + (e.currentCycle - s) : Math.max(0, e.rul - (s - e.currentCycle)), h))
                    }
                    const vA = getVal(eA, bundleA, predA, scrub)
                    const vB = getVal(eB, bundleB, predB, scrub)
                    const delta = vA - vB
                    const unit = label === 'Risk' ? '%' : label === 'RUL' ? ' cyc' : '%'
                    return (
                      <div key={label} className="rounded-xl border border-line bg-surface-2 px-3 py-2.5">
                        <div className="eyebrow">{label}</div>
                        <div className="mt-1 flex items-baseline gap-2">
                          <span className="stat-num text-[20px] font-semibold">{vA}{unit}</span>
                          <ArrowRight size={12} className="text-ink-3" />
                          <span className="stat-num text-[20px] font-semibold text-ink-2">{vB}{unit}</span>
                        </div>
                        <div className="mt-1 flex items-center gap-1.5">
                          <span className="stat-num text-[12px] font-semibold" style={{ color: delta > 0 ? (label === 'Risk' ? 'var(--crit)' : 'var(--ok)') : delta < 0 ? (label === 'Risk' ? 'var(--ok)' : 'var(--crit)') : 'var(--ink3)' }}>
                            {delta > 0 ? '+' : ''}{delta}{unit}
                          </span>
                          <span className="text-[9px] text-ink-3">A vs B</span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </section>
          ) : (
            <section className="card overflow-hidden" aria-label="Telemetry comparison">
              <div className="border-b border-line px-4 py-3">
                <span className="panel-title">Telemetry · {METRIC_CHIPS.find((m) => m.key === metric)?.label}</span>
              </div>
              <div className="grid gap-3 p-3 lg:grid-cols-2">
                <TelemetryChart
                  bundle={bundleA}
                  primarySensor={metric === 'vibration' ? 'VI' : METRIC_CHIPS.find((m) => m.key === metric)?.sensor ?? 'T24'}
                  pinnedSensors={pinnedSensors}
                  scrub={scrubA}
                  anomalies={bundleA.anomalies}
                  tempUnit={tempUnit}
                  className="h-[260px]"
                />
                <TelemetryChart
                  bundle={bundleB}
                  primarySensor={metric === 'vibration' ? 'VI' : METRIC_CHIPS.find((m) => m.key === metric)?.sensor ?? 'T24'}
                  pinnedSensors={pinnedSensors}
                  scrub={scrubB}
                  anomalies={bundleB.anomalies}
                  tempUnit={tempUnit}
                  className="h-[260px]"
                />
              </div>
            </section>
          )}

          <div className="grid gap-2 md:grid-cols-2">
            <button className="btn" onClick={() => openExplorer(eA.id)}>Open Engine #{eA.id} twin</button>
            <button className="btn" onClick={() => openExplorer(eB.id)}>Open Engine #{eB.id} twin</button>
          </div>
        </div>
      </div>

      <div className="px-4 pb-4 md:px-6 md:pb-5">
        <AICommandBar contextLabel="the comparison" />
      </div>
    </div>
  )
}

function riskFromRul(rul: number, health: number): number {
  const RISK_ANCHORS: [number, number][] = [
    [0, 100], [3, 97], [5, 94], [8, 92], [10, 90], [15, 86], [20, 80],
    [25, 77], [30, 73], [40, 64], [50, 57], [60, 50], [80, 36], [100, 26],
    [130, 16], [160, 10], [200, 6], [260, 3.5], [320, 2]
  ]
  let base = 50
  for (let i = 1; i < RISK_ANCHORS.length; i++) {
    const [x0, y0] = RISK_ANCHORS[i - 1]
    const [x1, y1] = RISK_ANCHORS[i]
    if (rul <= x1) {
      base = y0 + (y1 - y0) * ((rul - x0) / Math.max(x1 - x0, 1e-6))
      break
    }
  }
  if (rul > 320) base = 2
  return Math.round(Math.min(99, Math.max(1, base + (60 - health) * 0.12)))
}