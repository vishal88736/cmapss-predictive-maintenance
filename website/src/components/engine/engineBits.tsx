import { STATUS_META, type Engine, type EngineStatus } from '@/types/engine'
import { cn } from '@/lib/utils'

export function EngineBadge({ status, size = 'sm' }: { status: EngineStatus; size?: 'sm' | 'md' }) {
  const meta = STATUS_META[status]
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border font-semibold uppercase tracking-wide',
        size === 'sm' ? 'px-2 py-[3px] text-[9px]' : 'px-2.5 py-1 text-[10px]',
        meta.bg,
        meta.border,
        meta.text
      )}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', meta.dot)} aria-hidden />
      {meta.label}
      <span className="sr-only"> status</span>
    </span>
  )
}

export function EngineRow({ engine, onClick, active, compact }: { engine: Engine; onClick?: () => void; active?: boolean; compact?: boolean }) {
  const meta = STATUS_META[engine.status]
  return (
    <button
      onClick={onClick}
      className={cn(
        'group flex w-full items-center gap-3 rounded-lg border border-transparent px-2.5 py-2 text-left transition-all hover:border-line hover:bg-surface-2',
        active && 'border-accent/40 bg-accent/10'
      )}
    >
      <span className="stat-num w-9 shrink-0 text-[12px] font-semibold text-ink-1">#{engine.id}</span>
      <span className="relative hidden h-7 w-14 shrink-0 md:block" aria-hidden>
        <svg viewBox="0 0 56 28" className="h-full w-full">
          <polyline
            points={engine.rul > 0 ? '4,6 52,20' : '4,20 52,6'}
            fill="none"
            stroke={meta.color}
            strokeWidth={1.6}
            strokeLinecap="round"
            opacity={0.75}
          />
        </svg>
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-[11.5px] text-ink-2">
          <span className="stat-num">{engine.rul}</span> cycles left
          {!compact && <span className="text-ink-3"> · {engine.currentCycle} run</span>}
        </span>
      </span>
      <span className="flex shrink-0 items-center gap-2">
        <span className="hidden items-center gap-1 sm:flex">
          <span className="stat-num text-[11px] text-ink-3">{engine.health}%</span>
        </span>
        <EngineBadge status={engine.status} />
      </span>
    </button>
  )
}
