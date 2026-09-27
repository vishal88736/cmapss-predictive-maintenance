import { useMemo, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { Sparkles, Brain, ArrowRight, Columns2, ArrowUpDown } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import { RangeBars } from '@/components/charts/modelCharts'
import { EngineBadge } from '@/components/engine/engineBits'
import { getEngines, getPrediction, getModelMetrics, getEngine, getFleetStats, rulDistribution } from '@/data/db'
import { SENSOR_MAP } from '@/data/sensors'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

type SortKey = 'rul' | 'risk' | 'confidence'

export default function Predictions() {
  const [params] = useSearchParams()
  const [sort, setSort] = useState<SortKey>(params.get('sort') === 'rul' ? 'rul' : 'risk')
  const engines = useMemo(() => getEngines(), [])
  const stats = useMemo(() => getFleetStats(), [])
  const navigate = useNavigate()
  const openExplorer = useAppStore((s) => s.openExplorer)
  const setCompareIds = useAppStore((s) => s.setCompareIds)
  const dist = useMemo(() => rulDistribution(), [])

  const ranked = useMemo(() => {
    const list = [...engines]
    list.sort((a, b) => (sort === 'rul' ? a.rul - b.rul : sort === 'risk' ? b.risk - a.risk : getPrediction(b.id)!.confidence - getPrediction(a.id)!.confidence))
    return list.slice(0, 12)
  }, [engines, sort])

  const focusEngine = useMemo(() => [...engines].sort((a, b) => b.risk - a.risk)[0], [engines])
  const focusPrediction = getPrediction(focusEngine.id)!
  const metrics = getModelMetrics('lstm-rul-v2.4')!
  const compare = ['024', '071'].map((id) => getEngine(id)!)

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Prognostics"
        title="RUL Predictions"
        sub="Remaining useful life across the fleet, ranked by model output."
        meta={
          <span className="chip !py-0.5 !text-[9px] border-accent/30 bg-accent/10 text-[#aab4ff]">
            LSTM RUL PREDICTOR · v2.4
          </span>
        }
        actions={
          <>
            <Link to="/models" className="btn hidden md:inline-flex">
              <Brain size={13} /> Model insights
            </Link>
            <button
              className="btn btn-primary"
              onClick={() => {
                setCompareIds(['024', '071'])
                navigate('/compare/024/071')
              }}
            >
              <Columns2 size={13} /> Compare 24 · 71
            </button>
          </>
        }
      >
        <div className="flex items-center gap-1.5" role="group" aria-label="Sort predictions">
          <ArrowUpDown size={12} className="text-ink-3" />
          {(['risk', 'rul', 'confidence'] as SortKey[]).map((k) => (
            <button key={k} onClick={() => setSort(k)} className={cn('chip', sort === k && 'chip-active')} aria-pressed={sort === k}>
              Sort · {k === 'rul' ? 'lowest RUL' : k === 'risk' ? 'highest risk' : 'confidence'}
            </button>
          ))}
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto px-4 py-3.5 md:px-6">
        <div className="grid gap-3 xl:grid-cols-[1.5fr_1fr]">
          <section className="card overflow-hidden" aria-label="RUL ranking">
            <div className="flex items-center justify-between border-b border-line px-4 py-3">
              <span className="panel-title">Fleet ranking · top 12</span>
              <span className="text-[10px] text-ink-3">{stats.criticalCount + stats.warningCount} engines below the 60-cycle line</span>
            </div>
            <div className="divide-y divide-line/60">
              {ranked.map((e, i) => {
                const p = getPrediction(e.id)!
                return (
                  <button
                    key={e.id}
                    onClick={() => openExplorer(e.id)}
                    className="group flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-surface-2"
                    style={{ animation: `rise-in 0.3s ease-out ${i * 35}ms both` }}
                  >
                    <span className="stat-num w-8 shrink-0 text-[11px] text-ink-3">{String(i + 1).padStart(2, '0')}</span>
                    <span className="stat-num w-10 shrink-0 text-[13px] font-semibold">#{e.id}</span>
                    <span className="hidden w-16 shrink-0 sm:block">
                      <EngineBadge status={e.status} />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="relative block h-1.5 overflow-hidden rounded-full bg-white/5">
                        <span
                          className="absolute left-0 top-0 block h-full rounded-full transition-[width] duration-700"
                          style={{ width: `${Math.min(100, (e.rul / 260) * 100)}%`, background: e.rul < 20 ? 'var(--crit)' : e.rul < 60 ? 'var(--warn)' : 'var(--accent)' }}
                        />
                      </span>
                      <span className="mt-1 block text-[9.5px] text-ink-3">cycle {e.currentCycle} · conf {Math.round(p.confidence * 100)}%</span>
                    </span>
                    <span className="shrink-0 text-right">
                      <span className="stat-num block text-[14px] font-semibold" style={{ color: e.rul < 20 ? 'var(--crit)' : e.rul < 60 ? 'var(--warn)' : 'var(--ink1)' }}>
                        {e.rul}
                      </span>
                      <span className="block text-[9px] text-ink-3">cycles</span>
                    </span>
                    <ArrowRight size={13} className="shrink-0 text-ink-3 transition-transform group-hover:translate-x-0.5" />
                  </button>
                )
              })}
            </div>
          </section>

          <div className="space-y-3">
            <section className="card p-4" aria-label="RUL distribution">
              <span className="panel-title">RUL distribution · 100 engines</span>
              <div className="mt-3 h-[150px]">
                <RangeBars bars={dist.map((d) => ({ range: d.bucket, mae: d.count, count: d.count }))} />
              </div>
              <p className="mt-1 text-[10px] text-ink-3">Count of engines per remaining-life bucket. Tail-heavy: most units retain long horizons; {dist[0].count} sit under 25 cycles.</p>
            </section>

            <section className="card border-accent/25 p-4" aria-label="Why this prediction">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-accent">
                  <Sparkles size={12} /> Why this prediction?
                </span>
                <button className="text-[10px] font-medium text-ink-3 hover:text-ink-1" onClick={() => navigate(`/engine/${focusEngine.id}?explain=1`)}>
                  open twin →
                </button>
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="stat-num text-[24px] font-bold">{focusPrediction.rul}</span>
                <span className="text-[11px] text-ink-3">cycles for Engine #{focusEngine.id}</span>
                <span className="stat-num ml-auto text-[11px] font-semibold text-info">{Math.round(focusPrediction.confidence * 100)}% conf</span>
              </div>
              <div className="mt-3 space-y-2">
                {focusPrediction.attributions.slice(0, 4).map((a) => (
                  <div key={a.sensorId} className="flex items-center gap-2.5">
                    <span className="w-[110px] shrink-0 truncate text-[10.5px] text-ink-2">
                      <span className="stat-num mr-1.5 text-[9px] text-ink-3">{a.sensorId}</span>
                      {SENSOR_MAP[a.sensorId]?.short}
                    </span>
                    <span className="h-[6px] flex-1 overflow-hidden rounded-full bg-white/5">
                      <span className="block h-full rounded-full bg-gradient-to-r from-accent/40 to-accent" style={{ width: `${a.weight * 100}%` }} />
                    </span>
                    <span className="stat-num w-8 text-right text-[10px] text-ink-3">{a.weight.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <p className="mt-3 border-t border-line pt-2.5 text-[9.5px] leading-relaxed text-ink-3">
                MODEL OUTPUT — attribution from integrated gradients over the 30-cycle input window. Raw sensor trajectories and derived variance metrics feed the model; this panel explains its decision.
              </p>
            </section>

            <section className="card p-4" aria-label="Fleet-wide signal importance">
              <span className="panel-title">Fleet-wide signal importance</span>
              <div className="mt-3 space-y-1.5">
                {metrics.featureImportance.slice(0, 5).map((a) => (
                  <div key={a.sensorId} className="flex items-center gap-2.5">
                    <span className="w-[110px] shrink-0 truncate text-[10.5px] text-ink-2">
                      <span className="stat-num mr-1.5 text-[9px] text-ink-3">{a.sensorId}</span>
                      {SENSOR_MAP[a.sensorId]?.short}
                    </span>
                    <span className="h-[6px] flex-1 overflow-hidden rounded-full bg-white/5">
                      <span className="block h-full rounded-full bg-info/70" style={{ width: `${a.weight * 100}%` }} />
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="card overflow-hidden" aria-label="Compare engines">
              <div className="px-4 py-3">
                <span className="panel-title">Comparison preview</span>
                <div className="mt-3 grid grid-cols-3 items-center gap-2 text-center">
                  <span className="stat-num text-[12px] font-semibold">#{compare[0].id}</span>
                  <span className="text-[9px] uppercase tracking-widest text-ink-3">metric</span>
                  <span className="stat-num text-[12px] font-semibold">#{compare[1].id}</span>
                  {(
                    [
                      ['Health', compare[0].health, compare[1].health, '%'],
                      ['RUL', compare[0].rul, compare[1].rul, ''],
                      ['Risk', compare[0].risk, compare[1].risk, '%']
                    ] as const
                  ).map(([label, a, b, unit]) => (
                    <div key={label} className="col-span-3 grid grid-cols-3 items-center border-t border-line/60 pt-2">
                      <span className="stat-num text-[13px] font-semibold">{a}{unit}</span>
                      <span className="text-[9px] uppercase tracking-widest text-ink-3">{label}</span>
                      <span className="stat-num text-[13px] font-semibold text-ink-2">{b}{unit}</span>
                    </div>
                  ))}
                </div>
              </div>
              <button
                className="btn btn-primary m-4 mt-0 w-full"
                onClick={() => {
                  setCompareIds(['024', '071'])
                  navigate('/compare/024/071')
                }}
              >
                <Columns2 size={13} /> Open split-screen comparison
              </button>
            </section>
          </div>
        </div>

        <div className="mt-5">
          <AICommandBar contextLabel="the predictions" />
        </div>
      </div>
    </div>
  )
}
