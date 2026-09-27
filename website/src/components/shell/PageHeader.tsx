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
    <header className={cn('sticky top-0 z-30 border-b border-line bg-bg/80 backdrop-blur-lg', className)}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 md:px-6">
        <div className="min-w-0 flex-1">
          <div className="eyebrow">{eyebrow}</div>
          <h1 className="mt-0.5 flex items-baseline gap-3 font-display text-[17px] font-semibold leading-tight text-ink-1 md:text-[19px]">
            <span className="truncate">{title}</span>
            {meta}
          </h1>
          {sub && <p className="mt-0.5 hidden text-[11.5px] text-ink-3 md:block">{sub}</p>}
        </div>
        {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      {children && <div className="border-t border-line/60 px-4 py-2 md:px-6">{children}</div>}
    </header>
  )
}
