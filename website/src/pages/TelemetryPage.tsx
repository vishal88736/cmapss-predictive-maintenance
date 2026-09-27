import { useMemo, useState } from 'react'
import { Search, ChevronDown, Pin, PinOff, Layers, LayoutDashboard } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import TelemetryChart from '@/components/telemetry/TelemetryChart'
import { getEngines, getTelemetry } from '@/data/db'
import { SENSOR_CHANNELS, SENSOR_MAP } from '@/data/sensors'
import { VI_CHANNEL_ID } from '@/data/db'
import type { Engine } from '@/types/engine'
import type { TelemetryBundle } from '@/types/telemetry'
import { useAppStore } from '@/store/useAppStore'
import { fmtSensor, fmtNumber } from '@/lib/format'
import { Sparkline } from '@/components/charts/Sparkline'
import { cn } from '@/lib/utils'

const ALL_CHANNELS = [...SENSOR_CHANNELS, { id: VI_CHANNEL_ID, name: 'Vibration Index', short: 'Vibration', derived: true, baseline: 0.72 } as const]

export default function TelemetryPage() {
  const [engineId, setEngineId] = useState('024')
  const [query, setQuery] = useState('')
  const tempUnit = useAppStore((s) => s.tempUnit)
  const primarySensor = useAppStore((s) => s.primarySensor)
  const setPrimarySensor = useAppStore((s) => s.setPrimarySensor)
  const pinnedSensors = useAppStore((s) => s.pinnedSensors)
  const togglePin = useAppStore((s) => s.togglePin)
  const engines = useMemo(() => getEngines(), [])

  const engine = engines.find((e) => e.id === engineId)!
  const bundle = getTelemetry(engineId)!

  const filtered = engines.filter((e) => e.id.toLowerCase().includes(query.toLowerCase()) || String(e.unit).includes(query) || e.status.includes(query.toLowerCase()))

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Telemetry Lab"
        title="Sensor Analysis"
        sub="Dense multi-channel view — pick an engine, pin sensors, scrub cycles."
        actions={
          <div className="flex items-center gap-2">
            <div className="relative min-w-[180px] flex-1 md:max-w-[260px]">
              <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Filter units…"
                className="input h-8 pl-8 text-[12px]"
                aria-label="Filter engines"
              />
            </div>
            <select
              value={engineId}
              onChange={(e) => setEngineId(e.target.value)}
              className="input h-8 w-[140px] md:w-auto text-[12px]"
              aria-label="Select engine"
            >
              {engines.map((e) => (
                <option key={e.id} value={e.id}>
                  #{e.id} · {e.status} · RUL {e.rul}
                </option>
              ))}
            </select>
            <button className="btn h-8 text-[11px]" onClick={() => setPrimarySensor('VI')}>
              <Layers size={13} /> Vibration
            </button>
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="eyebrow">Pin sensors as overlays on the canvas</span>
          <div className="flex flex-wrap gap-1.5">
            {ALL_CHANNELS.map((c) => (
              <div key={c.id} className={cn('chip shrink-0 !gap-1 !py-0.5 !pl-2.5 !pr-1', primarySensor === c.id && 'chip-active')}>
                <button className="py-1 text-[10.5px]" onClick={() => setPrimarySensor(c.id)} aria-pressed={primarySensor === c.id}>
                  {c.short}
                </button>
                <button
                  className={cn('rounded p-1 transition-colors', pinnedSensors.includes(c.id) ? 'text-accent' : 'text-ink-3 hover:text-ink-1')}
                  onClick={() => togglePin(c.id)}
                  aria-label={pinnedSensors.includes(c.id) ? `Unpin ${c.short}` : `Pin ${c.short}`}
                  aria-pressed={pinnedSensors.includes(c.id)}
                >
                  {pinnedSensors.includes(c.id) ? <Pin size={10} /> : <PinOff size={10} />}
                </button>
              </div>
            ))}
          </div>
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto">
        <div className="grid gap-3 px-4 py-3.5 md:px-6 lg:grid-cols-[1fr_300px]">
          <section className="card overflow-hidden lg:col-span-1" aria-label="Primary sensor canvas">
            <div className="border-b border-line px-4 py-3">
              <div className="eyebrow mb-1">{SENSOR_MAP[primarySensor]?.name ?? primarySensor === VI_CHANNEL_ID ? 'Vibration Index (derived)' : primarySensor}</div>
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-ink-2">Engine #{engine.id} · cycle {engine.currentCycle} · {engine.status}</span>
                <span className={cn('rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide', engine.status === 'nominal' ? 'border-ok/30 bg-ok/10 text-ok' : engine.status === 'warning' ? 'border-warn/30 bg-warn/10 text-warn' : 'border-crit/30 bg-crit/10 text-crit')}>
                  {engine.status}
                </span>
              </div>
            </div>
            <div className="h-[380px] p-2 md:h-[420px]">
              <TelemetryChart
                key={engineId + primarySensor + pinnedSensors.join(',')}
                bundle={bundle}
                primarySensor={primarySensor}
                pinnedSensors={pinnedSensors}
                scrub={engine.currentCycle}
                anomalies={bundle.anomalies}
                tempUnit={tempUnit}
                className="h-full w-full"
              />
            </div>
          </section>

          <section className="card p-3 overflow-y-auto" aria-label="Sensor wall">
            <div className="mb-2 flex items-center justify-between">
              <span className="panel-title">All channels · snapshot</span>
              <span className="text-[9.5px] text-ink-3">{ALL_CHANNELS.length} sensors</span>
            </div>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
              {ALL_CHANNELS.map((ch) => {
                const isVI = ch.id === VI_CHANNEL_ID
                const arr = bundle.sensors[ch.id] ?? []
                const current = arr[arr.length - 1] ?? (isVI ? 0.72 : ch.baseline)
                const dev = bundle.expected[ch.id] ? ((current - (bundle.expected[ch.id][0] + bundle.expected[ch.id][1]) / 2) / ((bundle.expected[ch.id][0] + bundle.expected[ch.id][1]) / 2)) * 100 : 0
                const isPinned = pinnedSensors.includes(ch.id)
                return (
                  <button
                    key={ch.id}
                    onClick={() => setPrimarySensor(ch.id)}
                    className={cn('card-hover group flex w-full items-start gap-2.5 rounded-xl border border-line bg-surface-2 p-2.5 text-left transition-all hover:border-accent/30', isPinned && 'border-accent/30 bg-accent/5')}
                  >
                    <div className="h-[34px] w-[44px] shrink-0" aria-hidden>
                      <Sparkline values={arr.slice(-60).length ? arr.slice(-60) : [current, current]} color={isPinned ? 'var(--accent)' : 'var(--ink3)'} strokeWidth={1.2} fill />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="text-[10.5px] font-medium text-ink-1">{ch.id}</span>
                        <span className="text-[9px] text-ink-3">{ch.short}</span>
                        {isPinned && <span className="rounded-full border border-accent/30 bg-accent/10 px-1.5 py-0.5 text-[7.5px] font-semibold text-accent">PINNED</span>}
                        {isVI && <span className="rounded-full border border-info/30 bg-info/10 px-1.5 py-0.5 text-[7.5px] font-semibold text-info">DERIVED</span>}
                      </div>
                      <div className="mt-1 flex items-baseline gap-2">
                        <span className="stat-num font-display text-[15px] font-semibold">{fmtSensor(ch.id, current, tempUnit, 2)}</span>
                        <span className={cn('stat-num text-[10.5px] font-semibold', Math.abs(dev) > 4 ? 'text-crit' : Math.abs(dev) > 2 ? 'text-warn' : 'text-ok')}>
                          {dev > 0 ? '+' : ''}{dev.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <button
                      className={cn('shrink-0 rounded p-1 transition-colors', isPinned ? 'text-accent' : 'text-ink-3 hover:text-ink-1')}
                      onClick={(e) => { e.stopPropagation(); togglePin(ch.id) }}
                      aria-label={isPinned ? `Unpin ${ch.short}` : `Pin ${ch.short}`}
                    >
                      {isPinned ? <Pin size={11} /> : <PinOff size={11} />}
                    </button>
                  </button>
                )
              })}
            </div>
          </section>
        </div>
      </div>

      <div className="px-4 pb-4 md:px-6 md:pb-5">
        <AICommandBar contextLabel="telemetry data" />
      </div>
    </div>
  )
}