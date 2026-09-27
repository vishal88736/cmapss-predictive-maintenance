import { AlertTriangle, CheckCircle2, Info, X, Sparkles } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

const TONE = {
  info: { color: 'text-info', icon: Info, ring: 'shadow-[0_0_36px_-16px_rgba(76,194,255,0.9)]', edge: 'bg-info' },
  ok: { color: 'text-ok', icon: CheckCircle2, ring: 'shadow-[0_0_36px_-16px_rgba(61,220,151,0.9)]', edge: 'bg-ok' },
  warn: { color: 'text-warn', icon: AlertTriangle, ring: 'shadow-[0_0_36px_-16px_rgba(255,193,77,0.9)]', edge: 'bg-warn' },
  crit: { color: 'text-crit', icon: AlertTriangle, ring: 'shadow-[0_0_36px_-16px_rgba(255,93,108,0.9)]', edge: 'bg-crit' }
} as const

export default function Toasts() {
  const toasts = useAppStore((s) => s.toasts)
  const dismiss = useAppStore((s) => s.dismissToast)
  return (
    <div className="pointer-events-none fixed right-4 top-4 z-[95] flex w-[min(348px,calc(100vw-2rem))] flex-col gap-2" role="status" aria-live="polite">
      {toasts.map((t) => {
        const tone = TONE[t.tone ?? 'info']
        return (
          <div
            key={t.id}
            className={cn(
              'glass-strong pointer-events-auto relative flex items-start gap-2.5 overflow-hidden rounded-2xl p-3 animate-slide-left',
              tone.ring
            )}
          >
            {/* tone rail on the leading edge */}
            <span className={cn('absolute inset-y-0 left-0 w-[2px]', tone.edge)} />
            <span className={cn('mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg border border-line bg-surface-3/70', tone.color)}>
              <tone.icon size={13} />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-1">
                {t.tone === 'ok' && <Sparkles size={10} className="text-accent" />}
                {t.title}
              </div>
              {t.description && <div className="mt-0.5 text-[11.5px] leading-relaxed text-ink-2">{t.description}</div>}
              {t.actionLabel && t.action && (
                <button
                  className="btn btn-primary mt-2.5 h-7 px-2.5 text-[11px]"
                  onClick={() => {
                    t.action?.()
                    dismiss(t.id)
                  }}
                >
                  {t.actionLabel}
                </button>
              )}
            </div>
            <button className="icon-btn h-6 w-6 shrink-0" onClick={() => dismiss(t.id)} aria-label="Dismiss notification">
              <X size={12} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
