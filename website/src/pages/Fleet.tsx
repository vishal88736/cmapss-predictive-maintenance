import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { LayoutGrid, Rows3, Search, Maximize2, LocateFixed } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import FleetMapChart from '@/components/charts/FleetMapChart'
import { EngineRow, EngineBadge } from '@/components/engine/engineBits'
import { getEngines } from '@/data/db'
import type { Engine } from '@/types/engine'
import { STATUS_META } from '@/types/engine'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

type FilterMode = 'all' | 'nominal' | 'warning' | 'critical' | 'attention'

export default function Fleet() {
  const [params, setParams] = useSearchParams()
  const [view, setView] = useState<'map' | 'list'>('map')
  const [query, setQuery] = useState('')
  const [hover, setHover] = useState<{ engine: Engine; x: number; y: number } | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const openExplorer = useAppStore((s) => s.openExplorer)
  const engines = useMemo(() => getEngines(), [])
  const mapWrapRef = useRef<HTMLDivElement>(null)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState<{ x: number; y: number } | null>(null)
  const dragRef = useRef<{ x: number; y: number; px: number; py: number } | null>(null)

  const filter = (params.get('status') ?? 'all') as FilterMode

  const filtered = useMemo(() => {
    let list = engines
    if (filter === 'attention') list = list.filter((e) => e.status === 'critical' || e.risk >= 78)
    else if (filter !== 'all') list = list.filter((e) => e.status === filter)
    if (query.trim()) {
      const q = query.toLowerCase()
      list = list.filter((e) => e.id.includes(q) || String(e.unit).includes(q) || e.status.includes(q))
    }
    return list
  }, [engines, filter, query])

  const setFilter = (f: FilterMode) => {
    const next = new URLSearchParams(params)
    if (f === 'all') next.delete('status')
    else next.set('status', f)
    setParams(next, { replace: true })
  }

  useEffect(() => {
    if (params.get('view')) setView(params.get('view') as 'map' | 'list')
  }, [params])

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    setZoom((z) => Math.min(2.6, Math.max(0.8, z * (e.deltaY > 0 ? 0.92 : 1.08))))
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Spatial View"
        title="Fleet Risk Map"
        sub="Each node is an engine — position by risk and health, size by operating cycles."
        meta={
          <span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[10px] font-medium text-ink-2">
            {filtered.length} / {engines.length} units
          </span>
        }
        actions={
          <div className="flex rounded-lg border border-line bg-surface-2 p-0.5" role="tablist" aria-label="View mode">
            {(['map', 'list'] as const).map((v) => (
              <button
                key={v}
                role="tab"
                aria-selected={view === v}
                onClick={() => setView(v)}
                className={cn('flex h-7 items-center gap-1.5 rounded-md px-2.5 text-[11px] font-medium transition-colors', view === v ? 'bg-accent/15 text-[#aab4ff]' : 'text-ink-3 hover:text-ink-1')}
              >
                {v === 'map' ? <LayoutGrid size={12} /> : <Rows3 size={12} />}
                {v === 'map' ? 'Canvas' : 'List'}
              </button>
            ))}
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-[180px] flex-1 md:max-w-[260px]">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter engines inline — try “24” or “critical”"
              className="input h-8 pl-8 text-[12px]"
              aria-label="Filter engines"
            />
          </div>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Status filter">
            {(
              [
                ['all', `All (${engines.length})`],
                ['attention', `Watchlist (${engines.filter((e) => e.status === 'critical' || e.risk >= 78).length})`],
                ['critical', `Critical (${engines.filter((e) => e.status === 'critical').length})`],
                ['warning', `Warning (${engines.filter((e) => e.status === 'warning').length})`],
                ['nominal', `Nominal (${engines.filter((e) => e.status === 'nominal').length})`]
              ] as [FilterMode, string][]
            ).map(([f, label]) => (
              <button key={f} onClick={() => setFilter(f)} className={cn('chip', filter === f && 'chip-active')} aria-pressed={filter === f}>
                {f === 'attention' && <LocateFixed size={10} />}
                {label}
              </button>
            ))}
          </div>
        </div>
      </PageHeader>

      <div className="relative flex-1 overflow-hidden">
        {view === 'map' ? (
          <div
            ref={mapWrapRef}
            className="grid-texture absolute inset-0 touch-none"
            onWheel={onWheel}
            onPointerDown={(e) => {
              dragRef.current = { x: e.clientX, y: e.clientY, px: pan?.x ?? 0, py: pan?.y ?? 0 }
            }}
            onPointerMove={(e) => {
              const d = dragRef.current
              if (!d || e.buttons === 0) {
                dragRef.current = null
                return
              }
              setPan({ x: d.px + (e.clientX - d.x), y: d.py + (e.clientY - d.y) })
            }}
            onPointerUp={() => (dragRef.current = null)}
          >
            <div
              className="absolute inset-0 origin-center transition-transform duration-100 ease-out"
              style={{ transform: `translate(${pan?.x ?? 0}px, ${pan?.y ?? 0}px) scale(${zoom})` }}
            >
              <FleetMapChart
                engines={filtered}
                selectedId={selected}
                onEngineHover={(engine, pos) => setHover(engine && pos ? { engine, x: pos.x, y: pos.y } : null)}
                onEngineClick={(e) => {
                  setSelected(e.id)
                  openExplorer(e.id)
                }}
              />
            </div>

            {hover && (
              <div
                className="pointer-events-none absolute z-20 w-[190px] rounded-xl border border-line-2 bg-surface-2/95 p-3 shadow-pop backdrop-blur-md animate-fade-in"
                style={{ left: Math.min(hover.x + 14, (mapWrapRef.current?.clientWidth ?? 400) - 200), top: Math.max(8, hover.y - 70) }}
                role="tooltip"
              >
                <div className="flex items-center justify-between">
                  <span className="stat-num text-[13px] font-semibold">Engine #{hover.engine.id}</span>
                  <EngineBadge status={hover.engine.status} />
                </div>
                <div className="mt-2 grid grid-cols-3 gap-1.5 text-center">
                  {[
                    ['RUL', `${hover.engine.rul}`],
                    ['Risk', `${hover.engine.risk}%`],
                    ['Health', `${hover.engine.health}%`]
                  ].map(([k, v]) => (
                    <div key={k} className="rounded-md border border-line bg-surface-3 px-1 py-1">
                      <div className="text-[8.5px] uppercase tracking-wide text-ink-3">{k}</div>
                      <div className="stat-num text-[12px] font-semibold" style={{ color: k === 'Risk' ? STATUS_META[hover.engine.status].color : 'var(--ink1)' }}>
                        {v}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 text-[9.5px] text-ink-3">Click to open the engine explorer</div>
              </div>
            )}

            <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-1.5">
              <button className="btn h-8 w-8 !px-0" onClick={() => setZoom((z) => Math.min(2.6, z * 1.15))} aria-label="Zoom in">
                <Maximize2 size={13} />
              </button>
              <button
                className="btn h-8 !px-2 text-[10px]"
                onClick={() => {
                  setZoom(1)
                  setPan({ x: 0, y: 0 })
                }}
              >
                Reset
              </button>
            </div>
            <div className="absolute bottom-4 left-4 z-10 hidden rounded-lg border border-line bg-surface-2/80 px-2.5 py-1.5 text-[9.5px] text-ink-3 backdrop-blur md:block">
              scroll to zoom · drag to pan · hover for details
            </div>
          </div>
        ) : (
          <div className="h-full overflow-y-auto px-4 py-3 md:px-6">
            <div className="mx-auto max-w-3xl space-y-1.5 pb-24 md:pb-6">
              {filtered.map((e, i) => (
                <div key={e.id} style={{ animation: `rise-in 0.35s cubic-bezier(0.22,1,0.36,1) ${Math.min(i * 18, 400)}ms both` }}>
                  <EngineRow engine={e} active={selected === e.id} onClick={() => { setSelected(e.id); openExplorer(e.id) }} />
                </div>
              ))}
              {!filtered.length && <div className="py-16 text-center text-[12px] text-ink-3">No engines match this filter.</div>}
            </div>
          </div>
        )}
      </div>

      <div className="px-4 pb-4 md:px-6 md:pb-5">
        <AICommandBar contextLabel="this fleet map" />
      </div>
    </div>
  )
}
