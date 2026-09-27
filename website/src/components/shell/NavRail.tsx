import { useRef, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { Home, Orbit, Gauge, Activity, TrendingUp, AlertTriangle, Network, Settings2, Command as CommandIcon } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

interface RailItemProps {
  to?: string
  icon: React.ReactNode
  label: string
  shortcut?: string
  onClick?: () => void
  tooltipSide?: boolean
}

function RailItem({ to, icon, label, shortcut, onClick }: RailItemProps) {
  const [hover, setHover] = useState(false)
  const itemRef = useRef<HTMLAnchorElement | HTMLButtonElement>(null)
  const [tt, setTt] = useState<{ x: number; y: number } | null>(null)

  const content = (
    <>
      <span className="relative z-10 flex h-5 w-5 shrink-0 items-center justify-center">{icon}</span>
      <span className="nav-label pointer-events-none absolute left-[52px] z-10 whitespace-nowrap text-[12.5px] font-medium tracking-tight text-ink-2 opacity-0 transition-all duration-150 group-hover/rail:text-ink-1 group-hover/rail:opacity-100">
        {label}
      </span>
      {shortcut && (
        <span className="nav-label pointer-events-none absolute right-3 z-10 opacity-0 transition-opacity duration-150 group-hover/rail:opacity-70">
          <span className="kbd">{shortcut}</span>
        </span>
      )}
      {to && (
        <span className="absolute left-0 top-1/2 z-10 h-0 w-[2px] -translate-y-1/2 rounded-r bg-accent transition-all duration-200 nav-active-bar" />
      )}
    </>
  )

  const showTooltip = () => {
    const r = itemRef.current?.getBoundingClientRect()
    if (r) setTt({ x: r.right + 10, y: r.top + r.height / 2 })
    setHover(true)
  }
  const hideTooltip = () => {
    setHover(false)
    setTt(null)
  }

  return (
    <>
      {to ? (
        <NavLink
          ref={itemRef as React.Ref<HTMLAnchorElement>}
          to={to}
          onMouseEnter={showTooltip}
          onMouseLeave={hideTooltip}
          className={({ isActive }) =>
            cn(
              'group/item relative flex h-11 w-full items-center justify-center transition-colors hover:bg-white/5',
              isActive && 'bg-white/[0.06] text-ink-1 [&_.nav-active-bar]:h-6'
            )
          }
          aria-label={label}
        >
          {content}
        </NavLink>
      ) : (
        <button
          ref={itemRef as React.Ref<HTMLButtonElement>}
          onClick={onClick}
          onMouseEnter={showTooltip}
          onMouseLeave={hideTooltip}
          className="relative flex h-11 w-full items-center justify-center text-ink-2 transition-colors hover:bg-white/5 hover:text-ink-1"
          aria-label={label}
        >
          {content}
        </button>
      )}
      {tt && hover && (
        <div
          className="pointer-events-none fixed z-[100] -translate-y-1/2 rounded-md border border-line-2 bg-surface-3 px-2.5 py-1 text-[11px] font-medium text-ink-1 shadow-pop"
          style={{ left: tt.x, top: tt.y }}
          role="tooltip"
        >
          {label}
        </div>
      )}
    </>
  )
}

export default function NavRail() {
  const setPaletteOpen = useAppStore((s) => s.setPaletteOpen)
  const setSettingsOpen = useAppStore((s) => s.setSettingsOpen)
  const focusMode = useAppStore((s) => s.focusMode)
  const presentation = useAppStore((s) => s.presentation.active)

  if (focusMode || presentation) return null

  return (
    <nav
      className="group/rail relative z-40 hidden h-full shrink-0 flex-col border-r border-line bg-surface-1/80 backdrop-blur-md transition-[width] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] md:flex md:w-[64px] md:hover:w-[190px] xl:w-[68px]"
      aria-label="Primary navigation"
    >
      <div className="flex h-14 items-center justify-start overflow-hidden border-b border-line px-[19px]">
        <div className="relative flex h-7 w-7 shrink-0 items-center justify-center">
          <svg viewBox="0 0 32 32" className="h-7 w-7">
            <path d="M16 4 L26 25 L16 21 L6 25 Z" fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" />
            <circle cx="16" cy="15" r="2.4" fill="var(--info)" />
          </svg>
        </div>
        <span className="nav-label pointer-events-none absolute left-[52px] z-10 whitespace-nowrap font-display text-[13px] font-semibold tracking-tight text-ink-1 opacity-0 transition-opacity duration-150 group-hover/rail:opacity-100">
          AeroGuard
        </span>
      </div>

      <div className="mt-3 flex flex-1 flex-col gap-0.5 overflow-hidden px-0">
        <RailItem to="/" icon={<Home size={17} strokeWidth={1.8} />} label="Overview" />
        <RailItem to="/fleet" icon={<Orbit size={17} strokeWidth={1.8} />} label="Fleet Map" />
        <RailItem to="/engines" icon={<Gauge size={17} strokeWidth={1.8} />} label="Engines" />
        <RailItem to="/telemetry" icon={<Activity size={17} strokeWidth={1.8} />} label="Telemetry" />
        <RailItem to="/predictions" icon={<TrendingUp size={17} strokeWidth={1.8} />} label="Predictions" />
        <RailItem to="/alerts" icon={<AlertTriangle size={17} strokeWidth={1.8} />} label="Alerts" />
        <RailItem to="/models" icon={<Network size={17} strokeWidth={1.8} />} label="Models" />
      </div>

      <div className="flex flex-col gap-0.5 overflow-hidden border-t border-line py-2">
        <RailItem icon={<CommandIcon size={16} strokeWidth={1.8} />} label="Command" shortcut="⌘K" onClick={() => setPaletteOpen(true)} />
        <RailItem icon={<Settings2 size={16} strokeWidth={1.8} />} label="Settings" onClick={() => setSettingsOpen(true)} />
      </div>
    </nav>
  )
}
