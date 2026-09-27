import { useMemo } from 'react'
import { Pin, PinOff, Info } from 'lucide-react'
import TelemetryChart from './TelemetryChart'
import { SENSOR_CHANNELS } from '@/data/sensors'
import { getSnapshot } from '@/data/db'
import type { TelemetryBundle, Anomaly } from '@/types/telemetry'
import type { Engine } from '@/types/engine'
import { useAppStore } from '@/store/useAppStore'
import { fmtSensor } from '@/lib/format'
import { cn } from '@/lib/utils'

const VI = { id: 'VI', short: 'Vibration', name: 'Vibration Index', derived: true }

export default function TelemetryCanvas({ engine, bundle, scrub, onAnomalySelect }: { engine: Engine; bundle: TelemetryBundle; scrub: number; onAnomalySelect: (a: Anomaly) => void }) {
  const primarySensor = useAppStore((s) => s.primarySensor)
  const setPrimarySensor = useAppStore((s) => s.setPrimarySensor)
  const pinnedSensors = useAppStore((s) => s.pinnedSensors)
  const togglePin = useAppStore((s) => s.togglePin)
  const tempUnit = useAppStore((s) => s.tempUnit)

  const snapshot = useMemo(() => getSnapshot(engine.id, Math.min(scrub, engine.currentCycle)), [engine.id, scrub, engine.currentCycle])
  const primaryChannel = primarySensor === 'VI' ? VI : SENSOR_CHANNELS.find((c) => c.id === primarySensor)
  const expected = bundle.expected[primarySensor]
  const value = snapshot?.values[primarySensor] ?? 0
  const dev = snapshot?.deviations[primarySensor] ?? 0
  const devAbs = Math.abs(dev)
  const devTone = devAbs > 4 ? 'text-crit' : devAbs > 2 ? 'text-warn' : 'text-ok'
  const isPinned = pinnedSensors.includes(primarySensor)

  const chips = [...SENSOR_CHANNELS.map((c) => ({ id: c.id, short: c.short, derived: false })), VI]

  return (
    <div className="card flex flex-col overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="eyebrow">Sensor</span>
              <select
                value={primarySensor}
                onChange={(e) => setPrimarySensor(e.target.value)}
                className="h-7 rounded-md border border-line bg-surface-2 px-2 text-[12px] font-semibold text-ink-1 outline-none focus:border-accent/50"
                aria-label="Primary sensor"
              >
                {SENSOR_CHANNELS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.id} — {c.name}
                  </option>
                ))}
                <option value="VI">VI — Vibration Index (derived)</option>
              </select>
            </div>
            <div className="mt-1.5 flex items-baseline gap-2">
              <span className="stat-num font-display text-[26px] font-bold leading-none">{fmtSensor(primarySensor, value, tempUnit)}</span>
              <span className={cn('stat-num text-[11.5px] font-semibold', devTone)}>
                {dev >= 0 ? '+' : ''}{dev.toFixed(1)}% dev
              </span>
              {primarySensor === 'VI' && (
                <span className="chip !py-0 !text-[9px]">derived metric</span>
              )}
            </div>
          </div>
        </div>
        {expected && (
          <div className="text-right">
            <div className="eyebrow">Expected band</div>
            <div className="stat-num mt-0.5 text-[11.5px] text-ink-2">
              {fmtSensor(primarySensor, expected[0], tempUnit, 1)} — {fmtSensor(primarySensor, expected[1], tempUnit, 1)}
            </div>
          </div>
        )}
      </div>

      <div className="min-h-[220px] flex-1 px-2 py-2">
        <TelemetryChart
          key={engine.id + primarySensor + pinnedSensors.join(',')}
          bundle={bundle}
          primarySensor={primarySensor}
          pinnedSensors={pinnedSensors}
          scrub={Math.min(scrub, engine.currentCycle)}
          anomalies={bundle.anomalies}
          tempUnit={tempUnit}
          onAnomalySelect={onAnomalySelect}
          className="h-full min-h-[210px] w-full"
        />
      </div>

      <div className="flex items-center gap-1.5 overflow-x-auto border-t border-line px-3 py-2 no-scrollbar">
        {chips.map((c) => {
          const isPrimary = primarySensor === c.id
          const pinned = pinnedSensors.includes(c.id)
          return (
            <div key={c.id} className={cn('chip shrink-0 !gap-1 !py-0.5 !pl-2.5 !pr-1', isPrimary && 'chip-active')}>
              <button className="py-1 text-[10.5px]" onClick={() => setPrimarySensor(c.id)} aria-pressed={isPrimary}>
                {c.short}
              </button>
              <button
                className={cn('rounded p-1 transition-colors', pinned ? 'text-accent' : 'text-ink-3 hover:text-ink-1')}
                onClick={() => togglePin(c.id)}
                aria-label={pinned ? `Unpin ${c.short} overlay` : `Pin ${c.short} as overlay`}
                aria-pressed={pinned}
              >
                {pinned ? <Pin size={10} /> : <PinOff size={10} />}
              </button>
            </div>
          )
        })}
        <span className="ml-2 hidden shrink-0 items-center gap-1 text-[9.5px] text-ink-3 md:flex">
          <Info size={10} /> pin to overlay
        </span>
      </div>
    </div>
  )
}
