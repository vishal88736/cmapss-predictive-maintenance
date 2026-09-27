import { cn } from '@/lib/utils'

interface Props {
  eyebrow: string
  title: string
  sub?: string
  actions?: React.ReactNode
  meta?: React.ReactNode
  className?: string
  children?: React.ReactNode
}

export default function PageHeader({ eyebrow, title, sub, actions, meta, className, children }: Props) {
  return (
    <header className={cn('sticky top-0 z-30 pt-safe', className)}>
      {/* blur + fade so content dissolves under the bar */}
      <div className="glass absolute inset-0 -z-10 rounded-none border-x-0 border-t-0" />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 px-4 pb-3 pt-3 md:px-6">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            {/* accent tick + eyebrow */}
            <span
              className="h-3 w-[2px] shrink-0 rounded-full"
              style={{ background: 'linear-gradient(180deg, #7A8CFF, #A78BFA)' }}
            />
            <div className="eyebrow">{eyebrow}</div>
          </div>

          <h1 className="mt-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-1 font-display text-[20px] font-semibold leading-tight tracking-tight text-ink-1 md:text-[24px]">
            <span className="truncate text-gradient">{title}</span>
            {meta}
          </h1>

          {sub && <p className="mt-1 hidden max-w-[76ch] text-[12px] leading-relaxed text-ink-3 md:block">{sub}</p>}
        </div>

        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>

      {children && (
        <div className="border-t border-line/70 bg-surface-1/40 px-4 py-2 backdrop-blur-md md:px-6">{children}</div>
      )}
    </header>
  )
}
