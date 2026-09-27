import { useMemo } from 'react'
import type { TelemetryBundle } from '@/types/telemetry'
import type { Prediction } from '@/types/prediction'
import type { Engine } from '@/types/engine'
import { cn } from '@/lib/utils'

interface Props {
  engine: Engine
  bundle: TelemetryBundle
  prediction: Prediction
  scrub: number
  onScrub: (cycle: number) => void
  className?: string
}

export default function CycleScrubber({ engine, bundle, prediction, scrub, onScrub, className }: Props) {
  const max = prediction.failureCycle
  const nowPct = (engine.currentCycle / max) * 100
  const scrubPct = (scrub / max) * 100

  const ticks = useMemo(() => {
    const step = max > 260 ? 50 : 25
    const out: number[] = []
    for (let t = 0; t <= max; t += step) out.push(t)
    return out
  }, [max])

  return (
    <div className={cn('relative select-none', className)}>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="eyebrow">Cycle scrubber</span>
        <div className="flex items-center gap-2 text-[10px] text-ink-3">
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-3 rounded-full bg-accent" /> observed
          </span>
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-3 rounded-full bg-crit/70" /> predicted
          </span>
          {scrub !== engine.currentCycle && (
            <button className="chip !py-0.5 hover:border-accent/40" onClick={() => onScrub(engine.currentCycle)}>
              back to now
            </button>
          )}
        </div>
      </div>

      <div className="relative h-10">
        <input
          type="range"
          min={0}
          max={max}
          value={scrub}
          onChange={(e) => onScrub(Number(e.target.value))}
          aria-label={`Cycle scrubber, currently at cycle ${scrub} of ${max}`}
          aria-valuetext={`Cycle ${scrub}${scrub === engine.currentCycle ? ' (now)' : scrub > engine.currentCycle ? ' (predicted region)' : ''}`}
          className="absolute inset-0 z-20 h-full w-full cursor-grab opacity-0 active:cursor-grabbing"
        />
        <div className="pointer-events-none absolute inset-0">
          <div className="absolute left-0 right-0 top-1/2 h-[3px] -translate-y-1/2 rounded-full bg-white/[0.07]">
            <div className="absolute left-0 h-full rounded-full bg-accent" style={{ width: `${nowPct}%` }} />
            <div className="absolute h-full rounded-full bg-crit/50" style={{ left: `${nowPct}%`, right: 0 }} />
            <div className="absolute h-full border-r-2 border-dashed border-crit/70" style={{ left: '100%', width: 0 }} />
          </div>

          {ticks.map((t) => (
            <div key={t} className="absolute top-1/2 -translate-x-1/2" style={{ left: `${(t / max) * 100}%` }}>
              <div className="h-1.5 w-px -translate-y-4 bg-white/15" />
              <div className="stat-num absolute top-[22px] -translate-x-1/2 text-[8.5px] text-ink-3">{t}</div>
            </div>
          ))}

          <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2" style={{ left: `${nowPct}%` }}>
            <div className="h-4 w-[2px] rounded bg-accent" />
            <div className="absolute left-1/2 top-full mt-1.5 -translate-x-1/2 whitespace-nowrap rounded border border-accent/30 bg-accent/10 px-1.5 py-px text-[8.5px] font-semibold text-[#aab4ff]">
              NOW {engine.currentCycle}
            </div>
          </div>

          <div className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 transition-[left] duration-75" style={{ left: `${scrubPct}%` }}>
            <div className={cn('h-5 w-5 rounded-full border-[3px] shadow-card transition-colors', scrub === engine.currentCycle ? 'border-accent bg-accent/30' : 'border-ink-1 bg-surface-1')} />
            <div className="absolute left-1/2 -top-6 -translate-x-1/2 whitespace-nowrap rounded-md border border-line-2 bg-surface-3 px-1.5 py-0.5 text-[9px] font-semibold text-ink-1 shadow-card">
              {scrub}
            </div>
          </div>

          {bundle.anomalies.map((a) => (
            <div key={a.id} className="absolute top-[calc(50%+9px)] -translate-x-1/2" style={{ left: `${(a.cycle / max) * 100}%` }} title={`${a.sensorId} anomaly · cycle ${a.cycle}`}>
              <div
                className={cn('h-2 w-2 rotate-45 rounded-[1px]', a.severity === 'critical' ? 'bg-crit' : a.severity === 'warning' ? 'bg-warn' : 'bg-info', Math.abs(a.cycle - scrub) <= 6 && 'ring-2 ring-white/30')}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
