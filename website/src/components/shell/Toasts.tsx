import { AlertTriangle, CheckCircle2, Info, X, Sparkles } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

const TONE = {
  info: { color: 'text-info', icon: Info, border: 'border-info/25' },
  ok: { color: 'text-ok', icon: CheckCircle2, border: 'border-ok/25' },
  warn: { color: 'text-warn', icon: AlertTriangle, border: 'border-warn/25' },
  crit: { color: 'text-crit', icon: AlertTriangle, border: 'border-crit/25' }
} as const

export default function Toasts() {
  const toasts = useAppStore((s) => s.toasts)
  const dismiss = useAppStore((s) => s.dismissToast)
  return (
    <div className="pointer-events-none fixed right-4 top-4 z-[95] flex w-[min(340px,calc(100vw-2rem))] flex-col gap-2" role="status" aria-live="polite">
      {toasts.map((t) => {
        const tone = TONE[t.tone ?? 'info']
        return (
          <div key={t.id} className={cn('pointer-events-auto flex items-start gap-2.5 rounded-xl border bg-surface-2/95 p-3 shadow-pop backdrop-blur-lg animate-slide-left', tone.border)}>
            <tone.icon size={15} className={cn('mt-0.5 shrink-0', tone.color)} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-1">
                {t.tone === 'ok' && <Sparkles size={10} className="text-accent" />}
                {t.title}
              </div>
              {t.description && <div className="mt-0.5 text-[11.5px] leading-relaxed text-ink-2">{t.description}</div>}
              {t.actionLabel && t.action && (
                <button className="btn btn-primary mt-2 h-6 px-2 text-[11px]" onClick={() => { t.action?.(); dismiss(t.id) }}>
                  {t.actionLabel}
                </button>
              )}
            </div>
            <button className="icon-btn h-5 w-5 shrink-0" onClick={() => dismiss(t.id)} aria-label="Dismiss notification">
              <X size={11} />
            </button>
          </div>
        )
      })}
    </div>
  )
}
