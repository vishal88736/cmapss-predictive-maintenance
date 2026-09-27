import { useMemo, useState } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { Sparkles, Check, ChevronLeft, ChevronRight } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import { getEngines, getAlerts } from '@/data/db'
import type { Alert } from '@/types/alert'
import { STATUS_META } from '@/types/engine'
import { EngineBadge } from '@/components/engine/engineBits'
import { useAppStore } from '@/store/useAppStore'
import { fmtTime, fmtAgo } from '@/lib/format'
import { cn } from '@/lib/utils'

const ALERT_STATUS_META: Record<'critical' | 'warning' | 'info', { label: string; color: string; bg: string; border: string; dot: string }> = {
  critical: { label: 'Critical', color: 'var(--crit)', bg: 'bg-crit/10', border: 'border-crit/30', dot: 'bg-crit' },
  warning: { label: 'Warning', color: 'var(--warn)', bg: 'bg-warn/10', border: 'border-warn/30', dot: 'bg-warn' },
  info: { label: 'Info', color: 'var(--info)', bg: 'bg-info/10', border: 'border-info/30', dot: 'bg-info' }
}

type SeverityFilter = 'all' | 'critical' | 'warning' | 'info'

export default function Alerts() {
  const [params] = useSearchParams()
  const [filter, setFilter] = useState<SeverityFilter>((params.get('severity') as SeverityFilter) ?? 'all')
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set())
const openExplorer = useAppStore((s) => s.openExplorer)
const navigate = useNavigate()
  const baseAlerts = useMemo(() => getAlerts(), [])
  const liveAlerts = useAppStore((s) => s.alerts)
  const allAlerts = useMemo(() => [...liveAlerts, ...baseAlerts].sort((a, b) => b.ts - a.ts), [liveAlerts, baseAlerts])
  const filtered = useMemo(() =>
    allAlerts.filter((a) => filter === 'all' || a.severity === filter).map((a) => ({
      ...a,
      isLive: a.live && !acknowledged.has(a.id)
    })),
    [allAlerts, filter, acknowledged]
  )

  const counts = useMemo(() => ({
    all: allAlerts.length,
    critical: allAlerts.filter((a) => a.severity === 'critical').length,
    warning: allAlerts.filter((a) => a.severity === 'warning').length,
    info: allAlerts.filter((a) => a.severity === 'info').length
  }), [allAlerts])

  const ack = (id: string) => {
    setAcknowledged((s) => {
      const n = new Set(s)
      n.add(id)
      return n
    })
  }

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="Event Stream"
        title="Alert Timeline"
        sub="Fleet-wide event log — severity, engine, and cycle context."
        meta={
          <span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 text-[10px] font-medium text-ink-2">
            {counts.all} events in the last 6h
          </span>
        }
        actions={
          <div className="flex gap-1.5" role="group" aria-label="Severity filter">
            {(['all', 'critical', 'warning', 'info'] as SeverityFilter[]).map((f) => (
              <button key={f} onClick={() => setFilter(f)} className={cn('chip', filter === f && 'chip-active')} aria-pressed={filter === f}>
                {f === 'all' ? `All (${counts.all})` : `${f.charAt(0).toUpperCase() + f.slice(1)} (${counts[f as keyof typeof counts]})`}
              </button>
            ))}
          </div>
        }
      >
        <div className="flex items-center justify-between text-[10px] text-ink-3">
          <span className="flex items-center gap-1">
            <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent" />
            LIVE feed
          </span>
          <span>New events stream in real time. Click an event to focus the engine.</span>
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto px-4 py-3.5 md:px-6">
        <div className="mx-auto max-w-3xl">
          {filtered.length === 0 ? (
            <div className="py-16 text-center text-[12px] text-ink-3">No events match this filter.</div>
          ) : (
            <ol className="relative space-y-4" role="list" aria-label="Alert timeline">
              <div className="absolute left-[20px] top-0 bottom-0 w-px bg-white/[0.07]" aria-hidden />
              {filtered.map((alert, i) => {
                const eng = getEngines().find((e) => e.id === alert.engineId)
                const meta = ALERT_STATUS_META[alert.severity]
                const isFirst = i === 0
                return (
                  <li key={alert.id} className="relative" style={{ animation: `rise-in 0.4s ease-out ${Math.min(i * 25, 300)}ms both` }}>
                    <div className="absolute left-0 top-0 flex h-full w-[40px] items-start justify-center" aria-hidden>
                      <div className={cn('flex h-full flex-col items-center', i < filtered.length - 1 && 'pt-2')}>
                        <span className={cn('relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full', alert.severity === 'critical' ? 'bg-crit' : alert.severity === 'warning' ? 'bg-warn' : 'bg-info')} />
                        <span className="h-full w-px bg-white/[0.07]" />
                      </div>
                    </div>
                    <article
                      className={cn('relative ml-[40px] rounded-xl border bg-surface-2/80 p-3.5 transition-colors hover:border-line-2', alert.isLive && 'border-accent/30 bg-accent/5')}
                      role="listitem"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-baseline gap-2">
                            <span className="panel-title">{alert.title}</span>
                            <span className={cn('rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide', alert.severity === 'critical' ? 'border-crit/30 bg-crit/10 text-crit' : alert.severity === 'warning' ? 'border-warn/30 bg-warn/10 text-warn' : 'border-line-2 bg-surface-3 text-ink-2')}>
                              {alert.severity.toUpperCase()}
                            </span>
                            {alert.isLive && (
                              <span className="rounded-full border border-accent/30 bg-accent/10 px-2 py-0.5 text-[9px] font-semibold text-accent animate-pulse-dot">LIVE</span>
                            )}
                          </div>
                          <div className="mt-1.5 text-[11.5px] leading-relaxed text-ink-2">{alert.detail}</div>
                          <div className="mt-2 flex flex-wrap items-center gap-3 text-[10px] text-ink-3">
                            <span className="flex items-center gap-1">
                              <span className="stat-num">Engine #{alert.engineId}</span>
                              <EngineBadge status={eng?.status ?? 'nominal'} size="sm" />
                            </span>
                            {alert.cycle && <span>cycle <span className="stat-num">{alert.cycle}</span></span>}
                            {alert.sensorId && <span>{alert.sensorId} excursion</span>}
                            <span>{fmtAgo(alert.ts)}</span>
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1.5">
                          {acknowledged.has(alert.id) ? (
                            <button className="chip !py-0.5 !text-[9px] text-ok" disabled>
                              <Check size={10} /> Acknowledged
                            </button>
                          ) : (
                            <button className="btn h-7 text-[11px]" onClick={() => ack(alert.id)}>
                              Acknowledge
                            </button>
                          )}
                          <button className="btn btn-primary h-7 text-[11px]" onClick={() => {
                            openExplorer(alert.engineId)
                            if (alert.cycle) navigate(`/engine/${alert.engineId}?cycle=${alert.cycle}`)
                          }}>
                            Focus engine <Sparkles size={11} />
                          </button>
                        </div>
                      </div>
                    </article>
                    {isFirst && <div className="absolute left-[16px] -top-2 w-8 h-8" />}
                  </li>
                )
              })}
            </ol>
          )}
        </div>
      </div>

      <div className="px-4 pb-4 md:px-6 md:pb-5">
        <AICommandBar contextLabel="the alert timeline" />
      </div>
    </div>
  )
}