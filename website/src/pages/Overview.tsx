import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowRight, TrendingDown, TrendingUp, Radio, Cpu, CloudLightning, ChevronRight, Sparkles, AlertTriangle } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import { RadialScore, Sparkline } from '@/components/charts/Sparkline'
import FleetMapChart from '@/components/charts/FleetMapChart'
import { MultiLine } from '@/components/charts/modelCharts'
import { api } from '@/services/api'
import type { Engine, FleetStats } from '@/types/engine'
import { STATUS_META } from '@/types/engine'
import { useAppStore } from '@/store/useAppStore'
import { useNow } from '@/lib/hooks'
import { fmtTime } from '@/lib/format'
import { cn } from '@/lib/utils'
import { getEngines, cohortCurves, rulDistribution, getFleetStats } from '@/data/db'

const COHORT_COLORS = { critical: 'var(--crit)', warning: 'var(--warn)', nominal: 'var(--ok)' } as const

function Card({ className, children, ...rest }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('card card-hover edge-lit flex flex-col overflow-hidden', className)} {...rest}>
      {children}
    </div>
  )
}

function CardHead({ label, action }: { label: string; action?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-4 pt-4">
      <span className="panel-title">{label}</span>
      {action}
    </div>
  )
}

export default function Overview() {
  const [stats, setStats] = useState<FleetStats | null>(null)
  const [engines, setEngines] = useState<Engine[]>([])
  const [dist, setDist] = useState<{ bucket: string; count: number }[]>([])
  const [curves, setCurves] = useState<{ status: 'critical' | 'warning' | 'nominal'; label: string; points: { x: number; y: number }[] }[]>([])
  const openExplorer = useAppStore((s) => s.openExplorer)
  const lastSync = useAppStore((s) => s.lastSync)
  const navigate = useNavigate()
  const now = useNow(1000)

  useEffect(() => {
    let alive = true
    api.getFleetStats().then((s) => alive && setStats(s))
    api.getFleet().then((f) => alive && setEngines(f))
    api.getRulDistribution().then((d) => alive && setDist(d))
    api.getCohortCurves().then((c) => alive && setCurves(c as typeof curves))
    return () => {
      alive = false
    }
  }, [])

  const attention = useMemo(() => [...(engines.length ? engines : getEngines())].sort((a, b) => b.risk - a.risk).slice(0, 4), [engines])
  const fleetHealthSpark = useMemo(() => {
    const es = engines.length ? engines : getEngines()
    return Array.from({ length: 40 }, (_, i) => {
      const idx = es.length - 1 - (39 - i)
      const e = es[Math.abs(idx) % es.length]
      return e.health
    }).map((v, i, arr) => arr.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1))
  }, [engines])
  const rulDist = dist.length ? dist : rulDistribution()
  const cohortData = curves.length
    ? curves
    : (cohortCurves() as { status: 'critical' | 'warning' | 'nominal'; label: string; points: { x: number; y: number }[] }[])
  const s = stats ?? getFleetStats()
  const healthColor = s.health >= 75 ? 'var(--ok)' : s.health >= 55 ? 'var(--warn)' : 'var(--crit)'
  const nominalShare = (s.nominalCount / s.monitored) * 100

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Mission Overview"
        title="Bento Command Center"
        sub="Fleet-wide health, risk and degradation — synthesized across 100 turbofan units."
        meta={
          <span className="flex items-center gap-1.5 rounded-full border border-ok/25 bg-ok/10 px-2 py-0.5 text-[10px] font-semibold text-ok">
            <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-ok" />
            LIVE
          </span>
        }
        actions={
          <>
            <Link to="/predictions" className="btn hidden md:inline-flex">
              <Sparkles size={13} className="text-accent" /> Run RUL Prediction
            </Link>
            <Link to="/fleet" className="btn btn-primary">
              Fleet Map <ArrowRight size={13} />
            </Link>
          </>
        }
      />

      <div className="flex-1 overflow-y-auto px-4 pb-6 pt-4 md:px-6">
        <div className="grid auto-rows-auto grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-12">
          <Card className="col-span-1 md:col-span-2 lg:col-span-5 lg:row-span-2 animate-rise-in" style={{ animationDelay: '40ms' }}>
            <CardHead label="Fleet Health" action={<span className="font-mono text-[10px] text-ink-3">roll-40-cycle window</span>} />
            <div className="relative flex flex-1 flex-col items-center justify-center px-4 pb-4 pt-1">
              {/* ambient bloom behind the gauge */}
              <div
                className="pointer-events-none absolute h-[240px] w-[240px] rounded-full blur-[60px]"
                style={{ background: healthColor, opacity: 0.13 }}
                aria-hidden
              />
              <RadialScore value={s.health} size={188} stroke={11} color={healthColor}>
                <span className="eyebrow">Composite index</span>
                <span className="stat-num font-display text-[46px] font-bold leading-none tracking-tight" style={{ color: healthColor }}>
                  {Math.round(s.health)}<span className="text-[20px]">%</span>
                </span>
                <span className={cn('mt-1.5 inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide', s.health >= 75 ? 'border-ok/30 bg-ok/10 text-ok' : 'border-warn/30 bg-warn/10 text-warn')}>
                  {s.health >= 75 ? 'Nominal' : 'Degraded'}
                </span>
              </RadialScore>
              <div className="mt-3 flex w-full items-center justify-between rounded-xl border border-line bg-surface-2/70 px-3.5 py-2.5 transition-colors hover:border-line-2">
                <div>
                  <div className="flex items-center gap-1 text-[11px] font-medium text-ink-2">
                    {s.healthDelta >= 0 ? <TrendingUp size={12} className="text-ok" /> : <TrendingDown size={12} className="text-warn" />}
                    <span className={cn('stat-num font-semibold', s.healthDelta >= 0 ? 'text-ok' : 'text-warn')}>
                      {s.healthDelta >= 0 ? '+' : ''}{s.healthDelta} pts
                    </span>
                    vs previous window
                  </div>
                  <div className="mt-1 text-[10.5px] text-ink-3">{s.nominalCount} / {s.monitored} engines operating normally</div>
                </div>
                <div className="h-10 w-24" aria-hidden>
                  <Sparkline values={fleetHealthSpark} color={healthColor} strokeWidth={1.4} />
                </div>
              </div>
              <div className="mt-2.5 grid w-full grid-cols-3 gap-2">
                {[
                  ['Avg RUL', `${s.avgRul} cyc`],
                  ['Cycles ingested', s.totalCycles.toLocaleString()],
                  ['Units monitored', String(s.monitored)]
                ].map(([k, v]) => (
                  <div key={k} className="group rounded-xl border border-line bg-surface-2/60 px-2.5 py-2 transition-all duration-200 hover:-translate-y-0.5 hover:border-accent/25 hover:bg-surface-3/60">
                    <div className="eyebrow !tracking-[0.1em]">{k}</div>
                    <div className="stat-num mt-0.5 text-[14px] font-semibold text-ink-1 transition-colors group-hover:text-accent">{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          <Card className="col-span-1 lg:col-span-3 animate-rise-in" style={{ animationDelay: '90ms' }}>
            <CardHead label="System Status" action={<span className="font-mono text-[10px] text-ink-3">{fmtTime(lastSync)}</span>} />
            <div className="flex-1 space-y-2 px-4 py-3">
              {[
                { icon: Radio, name: 'Telemetry stream', state: 'CONNECTED' },
                { icon: Cpu, name: 'ML inference', state: 'READY' },
                { icon: CloudLightning, name: 'Prediction service', state: 'ONLINE' }
              ].map((row) => (
                <div key={row.name} className="flex items-center justify-between text-[11.5px]">
                  <span className="flex items-center gap-2 text-ink-2">
                    <row.icon size={13} className="text-ink-3" />
                    {row.name}
                  </span>
                  <span className="flex items-center gap-1.5 font-mono text-[10px] font-semibold text-ok">
                    <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-ok" />
                    {row.state}
                  </span>
                </div>
              ))}
              <div className="!mt-3 rounded-lg border border-ok/20 bg-ok/5 px-2.5 py-2 text-[10.5px] leading-relaxed text-ink-2">
                All services nominal. Last synchronization <span className="stat-num text-ink-1">{fmtTime(lastSync)}</span> · {Math.max(0, Math.round((now - lastSync) / 1000))}s ago.
              </div>
            </div>
          </Card>

          <Card className="col-span-1 md:col-span-2 lg:col-span-4 lg:row-span-2 animate-rise-in" style={{ animationDelay: '140ms' }}>
            <CardHead
              label="Requires Attention"
              action={
                <span className="rounded-full border border-crit/30 bg-crit/10 px-2 py-0.5 text-[10px] font-semibold text-crit">
                  {s.attentionCount} engines
                </span>
              }
            />
            <div className="flex-1 space-y-1.5 px-3 py-3">
              {attention.map((e, i) => (
                <button
                  key={e.id}
                  onClick={() => openExplorer(e.id)}
                  className="group relative flex w-full items-center gap-3 overflow-hidden rounded-xl border border-line bg-surface-2/50 px-3 py-2.5 text-left transition-all duration-300 ease-swift hover:-translate-y-px hover:border-line-2 hover:bg-surface-3/70 hover:shadow-card-hover"
                  style={{ animation: `rise-in 0.4s cubic-bezier(0.22,1,0.36,1) ${180 + i * 70}ms both` }}
                >
                  <span
                    className="absolute inset-y-0 left-0 w-[2px] opacity-70 transition-opacity group-hover:opacity-100"
                    style={{ background: STATUS_META[e.status].color }}
                    aria-hidden
                  />
                  <span className="stat-num pl-1.5 text-[13px] font-semibold text-ink-1">#{e.id}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[11px] text-ink-3">RUL {e.rul} · cycle {e.currentCycle}</span>
                    <span className="mt-1.5 block h-1 overflow-hidden rounded-full bg-white/5">
                      <span
                        className="block h-full rounded-full transition-all duration-500"
                        style={{ width: `${e.risk}%`, background: STATUS_META[e.status].color }}
                      />
                    </span>
                  </span>
                  <span className="stat-num text-[12px] font-semibold" style={{ color: STATUS_META[e.status].color }}>
                    {e.risk}%
                  </span>
                  <ChevronRight size={13} className="text-ink-3 transition-transform duration-300 group-hover:translate-x-0.5" />
                </button>
              ))}
            </div>
            <Link to="/fleet?status=attention" className="flex items-center justify-between border-t border-line px-4 py-2.5 text-[11.5px] font-medium text-ink-2 transition-colors hover:text-ink-1">
              Open watchlist on the fleet canvas
              <ArrowRight size={13} />
            </Link>
          </Card>

          <Card className="col-span-1 lg:col-span-3 animate-rise-in" style={{ animationDelay: '190ms' }}>
            <CardHead label="Average RUL" />
            <div className="flex flex-1 items-baseline gap-2 px-4 pt-2">
              <span className="stat-num font-display text-[30px] font-bold leading-none">{s.avgRul}</span>
              <span className="text-[11px] text-ink-3">cycles remaining</span>
            </div>
            <div className="mt-2 flex h-[46px] items-end gap-1 px-4 pb-2" aria-hidden>
              {rulDist.map((b, i) => (
                <div key={b.bucket} className="group relative flex-1" title={`${b.bucket}: ${b.count}`}>
                  <div
                    className="w-full rounded-sm transition-all"
                    style={{
                      height: `${8 + (b.count / Math.max(...rulDist.map((d) => d.count))) * 34}px`,
                      background: i === 0 ? 'var(--crit)' : i === 1 ? 'var(--warn)' : 'var(--accent)',
                      opacity: i > 1 ? 0.65 : 0.9
                    }}
                  />
                </div>
              ))}
            </div>
            <div className="px-4 pb-3 text-[10px] text-ink-3">
              <span className="stat-num text-crit">{rulDist[0].count}</span> engines under 25 cycles · <span className="stat-num text-warn">{rulDist[1].count}</span> under 50
            </div>
          </Card>

          <Card className="col-span-1 md:col-span-2 lg:col-span-5 lg:row-span-2 animate-rise-in" style={{ animationDelay: '240ms' }}>
            <CardHead
              label="Fleet Risk Map"
              action={
                <button className="flex items-center gap-1 text-[10.5px] font-medium text-ink-3 transition-colors hover:text-ink-1" onClick={() => navigate('/fleet')}>
                  Open canvas <ArrowRight size={12} />
                </button>
              }
            />
            <div className="relative h-[248px] flex-1 px-2 py-2">
              <FleetMapChart engines={engines.length ? engines : getEngines()} onEngineClick={(e) => openExplorer(e.id)} />
            </div>
            <div className="flex items-center justify-between border-t border-line px-4 py-2 text-[10px] text-ink-3">
              <span className="flex items-center gap-3">
                {(['nominal', 'warning', 'critical'] as const).map((st) => (
                  <span key={st} className="flex items-center gap-1.5">
                    <span className={cn('h-1.5 w-1.5 rounded-full', STATUS_META[st].dot)} />
                    {STATUS_META[st].label}
                  </span>
                ))}
              </span>
              <span>node size = operating cycles</span>
            </div>
          </Card>

          <Card className="col-span-1 md:col-span-2 lg:col-span-7 animate-rise-in" style={{ animationDelay: '290ms' }}>
            <CardHead
              label="Degradation Timeline"
              action={
                <span className="flex items-center gap-2.5 text-[9.5px] text-ink-3">
                  {cohortData.map((c) => (
                    <span key={c.status} className="flex items-center gap-1">
                      <span className="h-[3px] w-3.5 rounded-full" style={{ background: COHORT_COLORS[c.status] }} />
                      {c.label}
                    </span>
                  ))}
                </span>
              }
            />
            <div className="h-[172px] px-3 py-2">
              <MultiLine lines={cohortData.map((c) => ({ label: c.label, points: c.points, color: COHORT_COLORS[c.status] }))} />
            </div>
            <div className="border-t border-line px-4 py-2 text-[10.5px] leading-relaxed text-ink-3">
              Mean health index per cohort across normalized life progress. The critical cohort enters the failure window near <span className="stat-num text-ink-2">~60% life</span> — maintenance triggers fire long before.
            </div>
          </Card>

          <Card
            className="col-span-1 md:col-span-2 lg:col-span-7 animate-rise-in"
            style={{
              animationDelay: '340ms',
              backgroundImage:
                'linear-gradient(120deg, rgba(122,140,255,0.10) 0%, rgba(167,139,250,0.05) 42%, transparent 78%)'
            }}
          >
            <div className="pointer-events-none absolute -right-16 -top-20 h-56 w-56 rounded-full bg-accent/20 blur-[70px]" aria-hidden />
            <div className="relative flex items-center justify-between px-4 pt-4">
              <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-accent">
                <Sparkles size={12} />
                Model Insight
              </span>
              <span className="chip !text-[9px]">LSTM v2.4 · CONFIDENCE 87%</span>
            </div>
            <div className="relative px-4 pb-4 pt-3">
              <p className="max-w-[62ch] text-[12.5px] leading-relaxed text-ink-2">
                Fleet-wide degradation is stable, but <span className="font-semibold text-ink-1">{s.criticalCount} engines</span> show compounding wear signatures.
                Most relevant signals: <span className="text-ink-1">HPC Outlet Temp · Static Pressure · Fuel Ratio</span>. Three engines will cross the maintenance
                threshold within 15 cycles — led by <button className="font-semibold text-accent underline-offset-2 transition-colors hover:text-violet hover:underline" onClick={() => openExplorer('071')}>Engine #071</button>.
              </p>
              <div className="mt-3.5 flex flex-wrap gap-2">
                <button className="btn btn-primary h-8 text-[12px]" onClick={() => navigate('/telemetry')}>
                  Inspect signals
                </button>
                <button className="btn h-8 text-[12px]" onClick={() => navigate('/fleet?status=attention')}>
                  <AlertTriangle size={12} /> Open watchlist
                </button>
              </div>
            </div>
          </Card>
        </div>

        <div className="mt-5">
          <AICommandBar contextLabel="this fleet" />
        </div>
      </div>
    </div>
  )
}
