import { useEffect, useMemo, useState } from 'react'
import { useParams, useSearchParams, useNavigate } from 'react-router-dom'
import { Sparkles, Brain, ChevronLeft, ChevronRight, Focus, Presentation, Columns2, ListCollapse, X, ArrowRight, Zap } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import DegradationChart, { healthAtCycle } from '@/components/charts/DegradationChart'
import CycleScrubber from '@/components/timeline/CycleScrubber'
import TelemetryCanvas from '@/components/telemetry/TelemetryCanvas'
import AnomalyInsight from '@/components/telemetry/AnomalyInsight'
import WhatIfSimulator from '@/components/simulation/WhatIfSimulator'
import { EngineBadge } from '@/components/engine/engineBits'
import { getEngine, getTelemetry, getPrediction, getSnapshot, getEngines } from '@/data/db'
import { SENSOR_MAP } from '@/data/sensors'
import { riskFromRul } from '@/data/generate'
import { useAppStore } from '@/store/useAppStore'
import { pushRecent } from '@/commands/registry'
import { cn } from '@/lib/utils'

export default function EngineTwin() {
  const { id } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const scrubCycle = useAppStore((s) => s.scrubCycle)
  const setScrub = useAppStore((s) => s.setScrub)
  const setPrimarySensor = useAppStore((s) => s.setPrimarySensor)
  const openExplorer = useAppStore((s) => s.openExplorer)
  const focusMode = useAppStore((s) => s.focusMode)
  const setFocusMode = useAppStore((s) => s.setFocusMode)
  const startPresentation = useAppStore((s) => s.startPresentation)
  const setCompareIds = useAppStore((s) => s.setCompareIds)
  const toggleExplorer = useAppStore((s) => s.closeExplorer)
  const [anomaly, setAnomaly] = useState<ReturnType<typeof getTelemetry> extends undefined ? never : NonNullable<ReturnType<typeof getTelemetry>>['anomalies'][number] | null>(null)

  const engine = useMemo(() => getEngine(id ?? '024'), [id])
  const bundle = useMemo(() => (engine ? getTelemetry(engine.id) : undefined), [engine])
  const prediction = useMemo(() => (engine ? getPrediction(engine.id) : undefined), [engine])

  const scrub = scrubCycle[engine?.id ?? ''] ?? engine?.currentCycle ?? 0

  useEffect(() => {
    if (engine) {
      pushRecent(engine.id)
      const c = params.get('cycle')
      if (c) setScrub(engine.id, Math.min(Math.max(0, Number(c)), engine.currentCycle + prediction!.rul))
      else if (scrubCycle[engine.id] == null) setScrub(engine.id, engine.currentCycle)
      if (params.get('explain') || params.get('simulate')) {
        setTimeout(() => {
          document.getElementById(params.get('simulate') ? 'simulator' : 'explanation')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }, 300)
      }
    }
  }, [engine?.id])

  useEffect(() => {
    if (!engine) return
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault()
        const step = e.shiftKey ? 10 : 1
        const dir = e.key === 'ArrowRight' ? 1 : -1
        const next = Math.max(0, Math.min(scrub + step * dir, engine.currentCycle + prediction!.rul))
        setScrub(engine.id, next)
      } else if (e.key === '[' || e.key === ']') {
        const engines = getEngines()
        const idx = engines.findIndex((x) => x.id === engine.id)
        const next = e.key === '[' ? engines[idx - 1] : engines[idx + 1]
        if (next) navigate(`/engine/${next.id}`)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [engine, scrub, navigate, setScrub])

  if (!engine || !bundle || !prediction) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="text-[13px] text-ink-3">Engine not found. <button className="text-accent hover:underline" onClick={() => navigate('/engines')}>Back to registry</button></p>
      </div>
    )
  }

  const snapshot = getSnapshot(engine.id, Math.min(scrub, engine.currentCycle))
  const observedOnly = scrub > engine.currentCycle
  const rulNow = scrub <= engine.currentCycle ? engine.rul + (engine.currentCycle - scrub) : Math.max(0, engine.rul - (scrub - engine.currentCycle))
  const healthNow = healthAtCycle(bundle, prediction, scrub)
  const riskNow = scrub === engine.currentCycle ? engine.risk : riskFromRul(rulNow, healthNow)
  const nearAnomalies = bundle.anomalies.filter((a) => Math.abs(a.cycle - scrub) <= 8 && a.cycle <= engine.currentCycle)

  const prevEngine = getEngines()[getEngines().findIndex((x) => x.id === engine.id) - 1]
  const nextEngine = getEngines()[getEngines().findIndex((x) => x.id === engine.id) + 1]

  const headerStats = (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5">
      {[
        ['RUL', `${rulNow}`, 'cycles', rulNow < 20 ? 'text-crit' : rulNow < 60 ? 'text-warn' : 'text-ok'],
        ['Risk', `${Math.round(riskNow)}`, '%', riskNow >= 70 ? 'text-crit' : riskNow >= 40 ? 'text-warn' : 'text-ok'],
        ['Health', `${Math.round(healthNow)}`, '%', healthNow >= 70 ? 'text-ok' : healthNow >= 45 ? 'text-warn' : 'text-crit'],
        ['Cycle', `${scrub}`, observedOnly ? 'predicted' : scrub === engine.currentCycle ? 'now' : 'past', 'text-ink-1']
      ].map(([k, v, u, tone]) => (
        <div key={k as string} className="flex items-baseline gap-1.5">
          <span className="eyebrow !tracking-[0.1em]">{k}</span>
          <span className={cn('stat-num text-[13px] font-semibold', tone as string)}>{v}</span>
          <span className="text-[9px] text-ink-3">{u}</span>
        </div>
      ))}
      {observedOnly && <span className="chip !py-0.5 !text-[9px] border-crit/30 bg-crit/10 text-crit">projected position</span>}
    </div>
  )

  const headerActions = (
    <>
      <button className="btn h-8 text-[11.5px]" onClick={() => navigate(`/compare/${engine.id}/${engine.id === '071' ? '024' : '071'}`)} aria-label="Compare engines">
        <Columns2 size={13} /> <span className="hidden sm:inline">Compare</span>
      </button>
      <button className="btn h-8 text-[11.5px]" onClick={() => startPresentation(engine.id)} aria-label="Start presentation mode">
        <Presentation size={13} /> <span className="hidden sm:inline">Present</span>
      </button>
      <button
        className={cn('btn h-8 text-[11.5px]', focusMode && 'btn-primary')}
        onClick={() => setFocusMode(!focusMode)}
        aria-pressed={focusMode}
        aria-label="Toggle focus mode"
      >
        <Focus size={13} /> <span className="hidden sm:inline">{focusMode ? 'Exit focus' : 'Focus'}</span>
        <kbd className="kbd hidden lg:inline-flex">F</kbd>
      </button>
    </>
  )

  if (focusMode) {
    return (
      <div className="flex h-full flex-col">
        <header className="sticky top-0 z-30 border-b border-line bg-bg/85 px-4 py-2.5 backdrop-blur-lg md:px-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <h1 className="font-display text-[16px] font-semibold tracking-tight">Engine #{engine.id}</h1>
              <EngineBadge status={engine.status} />
              {headerStats}
            </div>
            {headerActions}
          </div>
        </header>
        <div className="flex flex-1 flex-col gap-2.5 overflow-y-auto px-4 py-4 md:px-6">
          <div className="card min-h-[42vh] flex-1 p-3">
            <DegradationChart engine={engine} bundle={bundle} prediction={prediction} scrub={scrub} onScrub={(c) => setScrub(engine.id, c)} onAnomalySelect={setAnomaly} className="h-full" animate={false} />
          </div>
          <CycleScrubber engine={engine} bundle={bundle} prediction={prediction} scrub={scrub} onScrub={(c) => setScrub(engine.id, c)} className="mx-1" />
          <div className="card min-h-[38vh] flex-1 p-1.5">
            <TelemetryCanvas engine={engine} bundle={bundle} scrub={scrub} onAnomalySelect={setAnomaly} />
          </div>
          <div className="pb-2 text-center">
            <span className="chip">Focus mode — press <kbd className="kbd">Esc</kbd> to exit</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader eyebrow="Digital Twin" title={`Engine #${engine.id}`} sub={`${engine.faultMode} · dataset ${engine.dataset} · ${bundle.cycles.length - 1} observed cycles · ${bundle.anomalies.length} recorded anomalies`} meta={<EngineBadge status={engine.status} />} actions={headerActions}>
        <div className="flex items-center justify-between">
          {headerStats}
          <div className="hidden items-center gap-1 md:flex">
            <button className="icon-btn" disabled={!prevEngine} onClick={() => prevEngine && navigate(`/engine/${prevEngine.id}`)} aria-label="Previous engine">
              <ChevronLeft size={15} />
            </button>
            <button className="icon-btn" disabled={!nextEngine} onClick={() => nextEngine && navigate(`/engine/${nextEngine.id}`)} aria-label="Next engine">
              <ChevronRight size={15} />
            </button>
          </div>
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto">
        <div className="space-y-3 px-4 py-3.5 md:px-6">
          <section className="card overflow-hidden" aria-label="Degradation timeline">
            <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
              <span className="panel-title">Degradation timeline</span>
              <span className="flex items-center gap-2.5 text-[9.5px] text-ink-3">
                <span className="flex items-center gap-1"><span className="h-[3px] w-3.5 rounded-full bg-accent" /> observed</span>
                <span className="flex items-center gap-1"><span className="h-[3px] w-3.5 rounded-full border-t-2 border-dashed border-crit" /> predicted</span>
                <span className="flex items-center gap-1"><span className="h-2 w-2 rotate-45 rounded-[1px] bg-info" /> anomaly</span>
              </span>
            </div>
            <div className="h-[280px] px-2 py-2 md:h-[320px]">
              <DegradationChart engine={engine} bundle={bundle} prediction={prediction} scrub={scrub} onScrub={(c) => setScrub(engine.id, c)} onAnomalySelect={setAnomaly} className="h-full" animate />
            </div>
            <div className="border-t border-line px-4 py-2.5">
              <CycleScrubber engine={engine} bundle={bundle} prediction={prediction} scrub={scrub} onScrub={(c) => setScrub(engine.id, c)} />
            </div>
          </section>

          <div className="grid gap-3 xl:grid-cols-[1.55fr_1fr]">
            <div className="space-y-3">
              <TelemetryCanvas engine={engine} bundle={bundle} scrub={scrub} onAnomalySelect={setAnomaly} />
              {anomaly && (
                <AnomalyInsight
                  anomaly={anomaly}
                  onClose={() => {
                    setAnomaly(null)
                    toggleExplorer()
                  }}
                />
              )}
              {nearAnomalies.length > 0 && !anomaly && (
                <div className="flex items-center gap-2 rounded-xl border border-warn/25 bg-warn/5 px-3.5 py-2.5 text-[11px] text-ink-2">
                  <ListCollapse size={13} className="text-warn" />
                  {nearAnomalies.length} anomaly marker{nearAnomalies.length > 1 ? 's' : ''} near cycle {scrub} —
                  <button className="font-semibold text-accent hover:underline" onClick={() => { setScrub(engine.id, nearAnomalies[0].cycle); setPrimarySensor(nearAnomalies[0].sensorId); setAnomaly(nearAnomalies[0]) }}>
                    inspect {nearAnomalies[0].sensorId}
                  </button>
                </div>
              )}
              <div id="simulator" className="scroll-mt-4">
                <WhatIfSimulator engine={engine} autoOpen={params.has('simulate')} />
              </div>
            </div>

            <div className="space-y-3">
              <section className="card border-accent/25 p-4" aria-label="Model insight">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-accent">
                    <Sparkles size={12} /> Model Insight
                  </span>
                  <span className="chip !py-0.5 !text-[9px]">LSTM v2.4</span>
                </div>
                <p className="mt-2.5 text-[12px] leading-relaxed text-ink-2">{prediction.insight}</p>
                <div className="mt-3 flex items-center justify-between rounded-lg border border-line bg-surface-2 px-3 py-2">
                  <span className="text-[11px] text-ink-3">Estimated RUL</span>
                  <span className="stat-num text-[13px] font-semibold text-ink-1">{prediction.rul} cycles</span>
                </div>
                <div className="mt-1.5 flex items-center justify-between rounded-lg border border-line bg-surface-2 px-3 py-2">
                  <span className="text-[11px] text-ink-3">Confidence</span>
                  <span className="stat-num text-[13px] font-semibold text-info">{Math.round(prediction.confidence * 100)}%</span>
                </div>
                <button
                  className="btn btn-primary mt-3 h-8 w-full text-[12px]"
                  onClick={() => {
                    setPrimarySensor(prediction.attributions[0].sensorId)
                    document.getElementById('telemetry-canvas')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                  }}
                >
                  Inspect signals <ArrowRight size={12} />
                </button>
              </section>

              <section id="explanation" className="card scroll-mt-4 p-4" aria-label="Model explanation">
                <div className="flex items-center gap-1.5">
                  <Brain size={13} className="text-info" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-2">Why this prediction?</h3>
                </div>
                <div className="mt-2.5 grid grid-cols-2 gap-2">
                  <div className="rounded-lg border border-line bg-surface-2 px-3 py-2">
                    <div className="eyebrow">Current RUL</div>
                    <div className="stat-num mt-0.5 text-[16px] font-semibold">{prediction.rul} <span className="text-[10px] text-ink-3">cycles</span></div>
                  </div>
                  <div className="rounded-lg border border-line bg-surface-2 px-3 py-2">
                    <div className="eyebrow">Confidence</div>
                    <div className="stat-num mt-0.5 text-[16px] font-semibold text-info">{Math.round(prediction.confidence * 100)}%</div>
                  </div>
                </div>
                <div className="eyebrow mt-3.5">Important signals</div>
                <div className="mt-2 space-y-2">
                  {prediction.attributions.map((a) => {
                    const ch = SENSOR_MAP[a.sensorId]
                    return (
                      <div key={a.sensorId} className="group flex items-center gap-2.5">
                        <button className="w-[118px] shrink-0 truncate text-left text-[11px] text-ink-2 transition-colors group-hover:text-accent" onClick={() => setPrimarySensor(a.sensorId)}>
                          <span className="stat-num mr-1.5 text-[9.5px] text-ink-3">{a.sensorId}</span>
                          {ch?.short}
                        </button>
                        <div className="h-[7px] flex-1 overflow-hidden rounded-full bg-white/5">
                          <div className="h-full rounded-full bg-gradient-to-r from-accent/40 to-accent transition-[width] duration-700" style={{ width: `${a.weight * 100}%` }} />
                        </div>
                        <span className="stat-num w-9 shrink-0 text-right text-[10.5px] text-ink-3">{a.weight.toFixed(2)}</span>
                      </div>
                    )
                  })}
                </div>
                <p className="mt-3 border-t border-line pt-2.5 text-[9.5px] leading-relaxed text-ink-3">
                  MODEL OUTPUT (integrated gradients) · RAW SENSOR DATA (T-30 window) · DERIVED METRICS (variance envelopes). Attribution explains the model, not the physics.
                </p>
              </section>

              <section className="card p-4" aria-label="Anomaly log">
                <div className="flex items-center justify-between">
                  <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-2">Anomaly log</h3>
                  <span className="chip !py-0.5 !text-[9px]">{bundle.anomalies.length} events</span>
                </div>
                <div className="mt-2.5 space-y-1.5">
                  {bundle.anomalies.map((a) => (
                    <button
                      key={a.id}
                      onClick={() => {
                        setScrub(engine.id, a.cycle)
                        setPrimarySensor(a.sensorId)
                        setAnomaly(a)
                      }}
                      className={cn('flex w-full items-center gap-2.5 rounded-lg border border-line bg-surface-2 px-3 py-2 text-left transition-colors hover:border-line-2', anomaly?.id === a.id && 'border-accent/40 bg-accent/10')}
                    >
                      <span className={cn('h-2 w-2 shrink-0 rotate-45 rounded-[1px]', a.severity === 'critical' ? 'bg-crit' : a.severity === 'warning' ? 'bg-warn' : 'bg-info')} />
                      <span className="min-w-0 flex-1 text-[11px] text-ink-2">
                        <span className="stat-num font-semibold text-ink-1">{a.sensorId}</span> · cycle {a.cycle}
                      </span>
                      <span className={cn('stat-num shrink-0 text-[11px] font-semibold', a.deviation > 0 ? 'text-warn' : 'text-info')}>
                        {a.deviation > 0 ? '+' : ''}{a.deviation}%
                      </span>
                    </button>
                  ))}
                </div>
              </section>

              <div className="grid grid-cols-2 gap-2">
                <button className="btn" onClick={() => openExplorer(engine.id)}>
                  Engine explorer <kbd className="kbd">E</kbd>
                </button>
                <button className="btn" onClick={() => document.getElementById('simulator')?.scrollIntoView({ behavior: 'smooth', block: 'center' })}>
                  <Zap size={12} /> Simulate
                </button>
              </div>
            </div>
          </div>

          <div className="pt-1">
            <AICommandBar contextLabel={`engine #${engine.id}`} />
          </div>
        </div>
      </div>
    </div>
  )
}
