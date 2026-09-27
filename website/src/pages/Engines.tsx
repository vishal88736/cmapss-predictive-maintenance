import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, ChevronDown, ArrowRight, Columns2, Zap } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import DegradationChart from '@/components/charts/DegradationChart'
import { EngineBadge } from '@/components/engine/engineBits'
import { getEngines, getTelemetry, getPrediction, getFleetStats } from '@/data/db'
import { SENSOR_MAP } from '@/data/sensors'
import type { Engine } from '@/types/engine'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

type SortKey = 'unit' | 'rul' | 'health' | 'risk' | 'cycles'

export default function Engines() {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortKey>('unit')
  const [expanded, setExpanded] = useState<string | null>(null)
  const navigate = useNavigate()
  const openExplorer = useAppStore((s) => s.openExplorer)
  const engines = useMemo(() => getEngines(), [])
  const stats = useMemo(() => getFleetStats(), [])
  const tempUnit = useAppStore((s) => s.tempUnit)

  const rows = useMemo(() => {
    let list = [...engines]
    if (query.trim()) {
      const q = query.toLowerCase()
      list = list.filter((e) => e.id.includes(q) || String(e.unit) === q.replace('#', '') || e.status.includes(q))
    }
    list.sort((a, b) => (sort === 'unit' ? a.unit - b.unit : sort === 'rul' ? a.rul - b.rul : sort === 'cycles' ? a.currentCycle - b.currentCycle : (b[sort as 'health' | 'risk'] as number) - (a[sort as 'health' | 'risk'] as number)))
    return list
  }, [engines, query, sort])

  const headers: { key: SortKey; label: string; className?: string }[] = [
    { key: 'unit', label: 'Unit' },
    { key: 'cycles', label: 'Cycles' },
    { key: 'rul', label: 'RUL' },
    { key: 'health', label: 'Health', className: 'hidden sm:table-cell' },
    { key: 'risk', label: 'Risk', className: 'hidden sm:table-cell' }
  ]

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Asset Registry"
        title="Engines"
        sub="All 100 monitored units. Expand a row for the degradation profile."
        meta={<span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[10px] font-medium text-ink-2">{stats.monitored} units · FD001</span>}
      >
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[180px] flex-1 md:max-w-[240px]">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Jump to unit…" className="input h-8 pl-8 text-[12px]" aria-label="Search engines" />
          </div>
          <div className="flex gap-1.5" role="group" aria-label="Sort engines">
            {(['rul', 'risk', 'health', 'cycles'] as SortKey[]).map((k) => (
              <button key={k} onClick={() => setSort(k)} className={cn('chip', sort === k && 'chip-active')} aria-pressed={sort === k}>
                Sort · {k === 'cycles' ? 'cycles run' : k.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto px-4 py-3 md:px-6">
        <div className="mx-auto max-w-4xl">
          <table className="w-full border-separate border-spacing-y-1 text-left" role="table" aria-label="Engine registry">
            <thead>
              <tr className="text-[9.5px] uppercase tracking-[0.14em] text-ink-3">
                {headers.map((h) => (
                  <th key={h.key} className={cn('px-3 pb-1 font-semibold', h.className)}>
                    <button onClick={() => setSort(h.key)} className="transition-colors hover:text-ink-1">
                      {h.label} {sort === h.key && '↓'}
                    </button>
                  </th>
                ))}
                <th className="px-3 pb-1 font-semibold">Status</th>
                <th className="w-8" />
              </tr>
            </thead>
            <tbody>
              {rows.map((e, i) => {
                const isExp = expanded === e.id
                return (
                  <EngineTableRow key={e.id} engine={e} i={i} expanded={isExp} onToggle={() => setExpanded(isExp ? null : e.id)} tempUnit={tempUnit} onExplorer={() => openExplorer(e.id)} />
                )
              })}
            </tbody>
          </table>
          {!rows.length && <div className="py-16 text-center text-[12px] text-ink-3">No engines match “{query}”.</div>}
        </div>
      </div>

      <div className="px-4 pb-4 md:px-6 md:pb-5">
        <AICommandBar contextLabel="the engine registry" />
      </div>
    </div>
  )
}

function EngineTableRow({ engine, i, expanded, onToggle, tempUnit, onExplorer }: { engine: Engine; i: number; expanded: boolean; onToggle: () => void; tempUnit: 'R' | 'K'; onExplorer: () => void }) {
  const bundle = getTelemetry(engine.id)
  const prediction = getPrediction(engine.id)
  const navigate = useNavigate()
  const setCompareIds = useAppStore((s) => s.setCompareIds)
  if (!bundle || !prediction) return null
  return (
    <>
      <tr
        className={cn('cursor-pointer bg-surface-1 transition-colors hover:bg-surface-2 [&>td]:first:rounded-l-xl [&>td]:last:rounded-r-xl', expanded && '[&>td]:border-accent/30')}
        onClick={onToggle}
        style={{ animation: `rise-in 0.3s cubic-bezier(0.22,1,0.36,1) ${Math.min(i * 12, 260)}ms both` }}
      >
        <td className={cn('border-y border-l border-line px-3 py-2.5', expanded && 'border-accent/40')}>
          <span className="stat-num text-[12.5px] font-semibold">#{engine.id}</span>
          <button className="ml-2 text-[10px] text-accent hover:underline" onClick={(ev) => { ev.stopPropagation(); onExplorer() }}>
            inspect
          </button>
        </td>
        <td className={cn('stat-num border-y border-line px-3 py-2.5 text-[11.5px] text-ink-2', expanded && 'border-accent/40')}>{engine.currentCycle}</td>
        <td className={cn('stat-num border-y border-line px-3 py-2.5 text-[12px] font-semibold', engine.rul < 20 ? 'text-crit' : engine.rul < 60 ? 'text-warn' : 'text-ink-1', expanded && 'border-accent/40')}>
          {engine.rul}
        </td>
        <td className={cn('stat-num hidden border-y border-line px-3 py-2.5 text-[11.5px] text-ink-2 sm:table-cell', expanded && 'border-accent/40')}>{engine.health}%</td>
        <td className={cn('stat-num hidden border-y border-line px-3 py-2.5 text-[11.5px] sm:table-cell', engine.risk >= 70 ? 'text-crit' : engine.risk >= 40 ? 'text-warn' : 'text-ok', expanded && 'border-accent/40')}>
          {engine.risk}%
        </td>
        <td className={cn('border-y border-line px-3 py-2.5', expanded && 'border-accent/40')}>
          <EngineBadge status={engine.status} />
        </td>
        <td className={cn('border-y border-r border-line px-2 py-2.5', expanded && 'border-accent/40')}>
          <ChevronDown size={14} className={cn('mx-auto text-ink-3 transition-transform duration-200', expanded && 'rotate-180')} />
        </td>
      </tr>
      {expanded && (
        <tr>
          <td colSpan={7} className="pb-1">
            <div className="card overflow-hidden p-3 animate-rise-in md:p-4">
              <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr]">
                <div>
                  <div className="mb-1 flex items-center justify-between">
                    <span className="eyebrow">Degradation profile</span>
                    <span className="font-mono text-[9.5px] text-ink-3">conf {Math.round(prediction.confidence * 100)}% · mae {prediction.mae}</span>
                  </div>
                  <DegradationChart engine={engine} bundle={bundle} prediction={prediction} scrub={engine.currentCycle} compact className="h-[150px]" animate />
                </div>
                <div className="flex flex-col gap-3">
                  <div>
                    <div className="eyebrow mb-1.5">Top model signals</div>
                    <div className="space-y-1">
                      {prediction.attributions.slice(0, 3).map((a) => (
                        <div key={a.sensorId} className="flex items-center justify-between text-[11px]">
                          <span className="text-ink-2">
                            <span className="stat-num mr-1.5 text-ink-3">{a.sensorId}</span>
                            {SENSOR_MAP[a.sensorId]?.short}
                          </span>
                          <span className="stat-num text-ink-3">{a.weight.toFixed(2)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <button className="btn btn-primary h-7 flex-1 text-[11px]" onClick={() => navigate(`/engine/${engine.id}`)}>
                      Digital Twin <ArrowRight size={11} />
                    </button>
                    <button
                      className="btn h-7 text-[11px]"
                      onClick={() => {
                        setCompareIds([engine.id, engine.id === '071' ? '024' : '071'])
                        navigate(`/compare/${engine.id}/${engine.id === '071' ? '024' : '071'}`)
                      }}
                    >
                      <Columns2 size={11} />
                    </button>
                    <button className="btn h-7 text-[11px]" onClick={() => navigate(`/engine/${engine.id}?simulate=1`)}>
                      <Zap size={11} />
                    </button>
                  </div>
                  <div className="text-[10px] leading-relaxed text-ink-3">
                    Fault mode: {engine.faultMode} · baseline dataset {engine.dataset}. Telemetry values shown in {tempUnit === 'R' ? 'Rankine' : 'Kelvin'}.
                  </div>
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}
