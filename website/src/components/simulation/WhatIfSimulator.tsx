import { useMemo, useRef, useState, useEffect } from 'react'
import { FlaskConical, RotateCcw, Play, ArrowRight } from 'lucide-react'
import type { Engine } from '@/types/engine'
import { simulate } from '@/data/db'
import { SENSOR_MAP } from '@/data/sensors'
import type { SimulationResult } from '@/types/prediction'
import { useMeasure, scaleLinear, smoothPath } from '@/components/charts/chartUtils'
import { fmtSensor } from '@/lib/format'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

export default function WhatIfSimulator({ engine, autoOpen = false }: { engine: Engine; autoOpen?: boolean }) {
  const tempUnit = useAppStore((s) => s.tempUnit)
  const scenarioSensors = useMemo(() => {
    const attr = ['T30', 'T24', 'Ps30', 'phi']
    return attr.filter((a) => engine.sensorSnapshot[a] != null)
  }, [engine])

  const [overrides, setOverrides] = useState<Record<string, number>>({})
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [running, setRunning] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const dirty = Object.keys(overrides).length > 0

  const run = (ov: Record<string, number>) => {
    setRunning(true)
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      if (Object.keys(ov).length) {
        setResult(simulate(engine.id, ov))
      } else {
        setResult(null)
      }
      setRunning(false)
    }, 480)
  }

  const apply = (id: string, v: number) => {
    const next = { ...overrides, [id]: v }
    setOverrides(next)
    run(next)
  }

  const reset = () => {
    setOverrides({})
    setResult(null)
    setRunning(false)
    if (timer.current) clearTimeout(timer.current)
  }

  return (
    <section className="card overflow-hidden" aria-label="What-if simulation">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-accent/30 bg-accent/10 text-accent">
            <FlaskConical size={14} />
          </span>
          <div>
            <span className="block text-[12.5px] font-semibold">What-if Simulator</span>
            <span className="block text-[10px] text-ink-3">Experiment on Engine #{engine.id} — outputs are model estimates, not observations</span>
          </div>
        </div>
        <div className="flex gap-2">
          <button className="btn h-7 text-[11px]" onClick={reset} disabled={!dirty}>
            <RotateCcw size={11} /> Reset
          </button>
          <button className="btn btn-primary h-7 text-[11px]" onClick={() => run(overrides)} disabled={!dirty}>
            <Play size={11} /> {running ? 'Estimating…' : 'Run model'}
          </button>
        </div>
      </div>

      <div className="grid gap-0 lg:grid-cols-[1fr_1fr]">
        <div className="border-b border-line p-4 lg:border-b-0 lg:border-r">
          <div className="eyebrow mb-3">Baseline</div>
          <div className="grid grid-cols-3 gap-2">
            {[
              ['RUL', `${engine.rul}`, 'cycles'],
              ['Risk', `${engine.risk}`, '%'],
              ['Health', `${engine.health}`, '%']
            ].map(([k, v, u]) => (
              <div key={k} className="rounded-xl border border-line bg-surface-2 px-3 py-2.5">
                <div className="eyebrow !tracking-[0.1em]">{k}</div>
                <div className="stat-num mt-1 text-[20px] font-semibold leading-none">
                  {v}
                  <span className="text-[11px] text-ink-3"> {u}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="eyebrow mb-2 mt-5">Create scenario</div>
          <div className="space-y-3.5">
            {scenarioSensors.map((sid) => {
              const ch = SENSOR_MAP[sid]
              const current = engine.sensorSnapshot[sid]
              const lo = current * 0.92
              const hi = current * 1.08
              const target = overrides[sid] ?? current
              const pct = ((target - current) / current) * 100
              return (
                <div key={sid}>
                  <div className="mb-1.5 flex items-center justify-between text-[11px]">
                    <span className="text-ink-2">
                      <span className="stat-num mr-1.5 text-[9.5px] text-ink-3">{sid}</span>
                      {ch.short}
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="stat-num text-[10px] text-ink-3">{fmtSensor(sid, current, tempUnit, 2)}</span>
                      <ArrowRight size={9} className="text-ink-3" />
                      <input
                        type="number"
                        value={+target.toFixed(2)}
                        step={ch.step}
                        min={lo}
                        max={hi}
                        onChange={(e) => apply(sid, Math.min(hi, Math.max(lo, Number(e.target.value) || current)))}
                        className="stat-num h-6 w-[86px] rounded-md border border-line bg-surface-2 px-1.5 text-right text-[10.5px] text-ink-1 outline-none focus:border-accent/50"
                        aria-label={`Scenario value for ${sid}`}
                      />
                      <span className={cn('stat-num w-11 text-right text-[10px] font-semibold', pct > 0.05 ? 'text-warn' : pct < -0.05 ? 'text-ok' : 'text-ink-3')}>
                        {pct > 0.05 ? '+' : ''}{pct.toFixed(1)}%
                      </span>
                    </span>
                  </div>
                  <input
                    type="range"
                    min={lo}
                    max={hi}
                    step={ch.step}
                    value={target}
                    onChange={(e) => apply(sid, Number(e.target.value))}
                    className="h-1.5 w-full rounded-full bg-white/[0.07]"
                    aria-label={`${sid} scenario slider`}
                  />
                </div>
              )
            })}
          </div>
        </div>

        <div className="flex flex-col p-4">
          <div className="eyebrow mb-3">Scenario preview</div>
          {!result ? (
            <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-line bg-surface-2/50 py-10 text-center">
              <p className="max-w-[240px] text-[11.5px] leading-relaxed text-ink-3">
                Adjust sensors on the left to see how the model expects RUL and risk to respond. The engine does not leave the hangar — this is a simulation.
              </p>
            </div>
          ) : (
            <div className="animate-rise-in">
              <div className="grid grid-cols-3 gap-2">
                {(
                  [
                    ['RUL', engine.rul, result.scenarioRul, result.scenarioRul - engine.rul, 'cycles'],
                    ['Risk', engine.risk, result.scenarioRisk, result.scenarioRisk - engine.risk, '%'],
                    ['Health', engine.health, result.scenarioHealth, result.scenarioHealth - engine.health, '%']
                  ] as const
                ).map(([k, base, scen, delta, unit]) => (
                  <div key={k} className="rounded-xl border border-line bg-surface-2 px-3 py-2.5">
                    <div className="eyebrow !tracking-[0.1em]">{k}</div>
                    <div className="stat-num mt-1 flex items-baseline gap-1.5 text-[19px] font-semibold leading-none">
                      <span className="text-ink-3">{base}</span>
                      <ArrowRight size={11} className="text-ink-3" />
                      <span className={delta < 0 ? (k === 'Health' ? 'text-crit' : 'text-crit') : delta > 0 ? (k === 'Risk' ? 'text-crit' : 'text-ok') : 'text-ink-1'}>{scen}</span>
                      <span className={cn('text-[11px] font-semibold', delta === 0 ? 'text-ink-3' : (k === 'Risk' ? delta > 0 : delta < 0) ? 'text-crit' : 'text-ok')}>
                        {delta > 0 ? '+' : ''}{Math.round(delta)}
                      </span>
                    </div>
                    <div className="mt-1 h-1 overflow-hidden rounded-full bg-white/5">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.min(100, (Number(scen) / (k === 'RUL' ? Math.max(engine.rul, scen) * 1.2 : 100)) * 100)}%`,
                          background: k === 'Risk' ? (delta > 0 ? 'var(--crit)' : 'var(--ok)') : delta < 0 ? 'var(--crit)' : 'var(--ok)'
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-3 h-[120px] rounded-xl border border-line bg-surface-2 p-1">
                <ScenarioTrajectory result={result} />
              </div>

              {result.contributions.length > 0 && (
                <div className="mt-3 text-[10.5px] leading-relaxed text-ink-3">
                  Dominant drivers:{' '}
                  {result.contributions.slice(0, 3).map((c, i) => (
                    <span key={c.sensorId}>
                      {i > 0 && ' · '}
                      <span className="stat-num text-ink-2">{c.sensorId}</span>{' '}
                      <span className={c.deltaRul < 0 ? 'text-crit' : 'text-ok'}>({c.deltaRul > 0 ? '+' : ''}{c.deltaRul} cyc)</span>
                    </span>
                  ))}
                </div>
              )}
              <p className="mt-2 text-[9.5px] leading-relaxed text-ink-3">
                ESTIMATED OUTPUT — scenario trajectories extrapolate the current model under shifted inputs. Not observed telemetry.
              </p>
            </div>
          )}
        </div>
      </div>
      {autoOpen && null}
    </section>
  )
}

function ScenarioTrajectory({ result }: { result: SimulationResult }) {
  const [ref, size] = useMeasure<HTMLDivElement>()
  const w = Math.max(size.w, 60)
  const h = Math.max(size.h, 60)
  const all = [...result.baselineTrajectory, ...result.scenarioTrajectory]
  const sy = scaleLinear([Math.min(...all) * 0.95, Math.max(...all) * 1.05], [h - 6, 6])
  const sx = scaleLinear([0, result.trajectoryCycles.length - 1], [4, w - 4])
  return (
    <div ref={ref} className="h-full w-full" aria-label="Baseline versus scenario health trajectory">
      <svg width={w} height={h}>
        <path d={smoothPath(result.baselineTrajectory.map((v, i) => [sx(i), sy(v)] as [number, number]))} fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth={1.4} strokeDasharray="3 3" />
        <path d={smoothPath(result.scenarioTrajectory.map((v, i) => [sx(i), sy(v)] as [number, number]))} fill="none" stroke="var(--accent)" strokeWidth={1.8} className="animate-draw" style={{ ['--draw-dur' as string]: '0.8s' }} />
      </svg>
      <div className="absolute right-2 top-1.5 flex gap-2.5 text-[8.5px] text-ink-3">
        <span className="flex items-center gap-1"><span className="h-[3px] w-3 rounded bg-ink-3" /> baseline</span>
        <span className="flex items-center gap-1"><span className="h-[3px] w-3 rounded bg-accent" /> scenario</span>
      </div>
    </div>
  )
}
