import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { X, ArrowRight, Columns2, Zap, History } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { getEngine, getTelemetry, getPrediction } from '@/data/db'
import { EngineBadge } from './engineBits'
import { Sparkline } from '@/components/charts/Sparkline'
import { fmtSensor } from '@/lib/format'
import { pushRecent } from '@/commands/registry'
import { cn } from '@/lib/utils'

export default function EngineExplorer() {
  const engineId = useAppStore((s) => s.explorerEngineId)
  const close = useAppStore((s) => s.closeExplorer)
  const tempUnit = useAppStore((s) => s.tempUnit)
  const setCompareIds = useAppStore((s) => s.setCompareIds)
  const navigate = useNavigate()

  useEffect(() => {
    if (engineId) pushRecent(engineId)
  }, [engineId])

  const engine = engineId ? getEngine(engineId) : undefined
  const bundle = engine ? getTelemetry(engine.id) : undefined
  const prediction = engine ? getPrediction(engine.id) : undefined

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && engineId) close()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [engineId, close])

  if (!engineId || !engine || !bundle || !prediction) return null

  const snapshotSensors = ['T24', 'Ps30', 'Nc']
  const rulColor = engine.status === 'critical' ? 'var(--crit)' : engine.status === 'warning' ? 'var(--warn)' : 'var(--ok)'

  return (
    <div className="fixed inset-0 z-[80]" role="dialog" aria-modal="true" aria-label={`Engine ${engine.id} explorer panel`}>
      <div className="absolute inset-0 bg-black/35 animate-fade-in lg:bg-transparent" onClick={close} />
      <aside
        className="absolute bottom-0 left-0 right-0 max-h-[82vh] rounded-t-2xl border-t border-line-2 bg-surface-1/95 shadow-pop backdrop-blur-xl animate-rise-in lg:bottom-auto lg:left-auto lg:top-0 lg:h-full lg:max-h-none lg:w-[350px] lg:rounded-none lg:rounded-l-2xl lg:border-l lg:border-t-0 lg:animate-slide-left"
        aria-label="Engine details"
      >
        <div className="flex items-center justify-between border-b border-line px-4 py-3.5">
          <div className="flex items-center gap-2.5">
            <h2 className="font-display text-[15px] font-semibold tracking-tight">Engine #{engine.id}</h2>
            <EngineBadge status={engine.status} />
          </div>
          <button className="icon-btn" onClick={close} aria-label="Close engine panel">
            <X size={15} />
          </button>
        </div>

        <div className="max-h-[calc(82vh-56px)] space-y-4 overflow-y-auto p-4 lg:max-h-[calc(100vh-56px)]">
          <div className="flex items-baseline gap-2.5">
            <span className="stat-num text-[34px] font-semibold leading-none" style={{ color: rulColor }}>
              {engine.rul}
            </span>
            <span className="text-[11px] text-ink-3">cycles of remaining useful life</span>
          </div>

          <div className="grid grid-cols-2 gap-2.5">
            {(
              [
                ['Health', `${engine.health}%`, engine.health >= 70 ? 'var(--ok)' : engine.health >= 50 ? 'var(--warn)' : 'var(--crit)'],
                ['Risk', `${engine.risk}%`, engine.risk >= 70 ? 'var(--crit)' : engine.risk >= 40 ? 'var(--warn)' : 'var(--ok)'],
                ['Current cycle', `${engine.currentCycle}`, 'var(--ink1)'],
                ['Confidence', `${Math.round(engine.confidence * 100)}%`, 'var(--info)']
              ] as const
            ).map(([label, value, color]) => (
              <div key={label} className="rounded-xl border border-line bg-surface-2 p-2.5">
                <div className="eyebrow">{label}</div>
                <div className="stat-num mt-1 text-[17px] font-semibold" style={{ color }}>
                  {value}
                </div>
              </div>
            ))}
          </div>

          <div className="rounded-xl border border-line bg-surface-2 p-2.5">
            <div className="eyebrow mb-1.5">Health trajectory</div>
            <div className="h-14">
              <Sparkline values={bundle.health.slice(-90)} color={rulColor} />
            </div>
            <div className="mt-1 flex justify-between text-[9.5px] text-ink-3">
              <span>cycle {Math.max(0, engine.currentCycle - 89)}</span>
              <span>now</span>
            </div>
          </div>

          <div>
            <div className="eyebrow mb-2">Telemetry · last cycle</div>
            <div className="space-y-1.5">
              {snapshotSensors.map((sid) => (
                <div key={sid} className="flex items-center justify-between rounded-lg border border-line bg-surface-2 px-3 py-2 text-[12px]">
                  <span className="text-ink-2">
                    <span className="stat-num mr-2 text-[10px] text-ink-3">{sid}</span>
                    {sid === 'T24' ? 'LPC Temp' : sid === 'Ps30' ? 'Static Pres' : 'Core Speed'}
                  </span>
                  <span className="stat-num font-semibold text-ink-1">{fmtSensor(sid, engine.sensorSnapshot[sid], tempUnit)}</span>
                </div>
              ))}
            </div>
          </div>

          {bundle.anomalies.length > 0 && (
            <div>
              <div className="eyebrow mb-2">Recent anomalies</div>
              <div className="space-y-1.5">
                {bundle.anomalies.slice(0, 2).map((a) => (
                  <button
                    key={a.id}
                    onClick={() => {
                      close()
                      navigate(`/engine/${engine.id}?cycle=${a.cycle}`)
                    }}
                    className="flex w-full items-center gap-2.5 rounded-lg border border-line bg-surface-2 px-3 py-2 text-left transition-colors hover:border-line-2"
                  >
                    <History size={12} className={a.severity === 'critical' ? 'text-crit' : a.severity === 'warning' ? 'text-warn' : 'text-info'} />
                    <span className="min-w-0 flex-1 text-[11.5px] text-ink-2">
                      <span className="font-semibold text-ink-1">{a.sensorId}</span> at cycle {a.cycle}
                    </span>
                    <span className={cn('stat-num text-[11px] font-semibold', a.deviation > 0 ? 'text-warn' : 'text-info')}>
                      {a.deviation > 0 ? '+' : ''}{a.deviation}%
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-2 pt-1">
            <button
              className="btn btn-primary w-full"
              onClick={() => {
                close()
                navigate(`/engine/${engine.id}`)
              }}
            >
              Open Digital Twin <ArrowRight size={13} />
            </button>
            <div className="grid grid-cols-2 gap-2">
              <button
                className="btn"
                onClick={() => {
                  setCompareIds([engine.id, engine.id === '071' ? '024' : '071'])
                  close()
                  navigate(`/compare/${engine.id}/${engine.id === '071' ? '024' : '071'}`)
                }}
              >
                <Columns2 size={13} /> Compare
              </button>
              <button
                className="btn"
                onClick={() => {
                  close()
                  navigate(`/engine/${engine.id}?simulate=1`)
                }}
              >
                <Zap size={13} /> What-if
              </button>
            </div>
          </div>
        </div>
      </aside>
    </div>
  )
}
