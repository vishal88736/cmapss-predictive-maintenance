import { useRef, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  Home,
  Orbit,
  Gauge,
  Activity,
  TrendingUp,
  AlertTriangle,
  Network,
  Settings2,
  Command as CommandIcon,
  PanelLeftClose
} from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

interface RailItemProps {
  to?: string
  icon: React.ReactNode
  label: string
  shortcut?: string
  onClick?: () => void
  badge?: number
}

function RailItem({ to, icon, label, shortcut, onClick, badge }: RailItemProps) {
  const itemRef = useRef<HTMLAnchorElement | HTMLButtonElement>(null)
  const [tt, setTt] = useState<{ x: number; y: number } | null>(null)

  const content = (
    <>
      {/* active pill, sits behind the icon */}
      <span className="nav-active absolute inset-0 rounded-xl opacity-0 transition-all duration-300 ease-swift" />

      <span className="relative z-10 flex h-5 w-5 shrink-0 items-center justify-center transition-transform duration-300 ease-spring group-hover/item:scale-110">
        {icon}
      </span>

      <span className="nav-label pointer-events-none relative z-10 ml-3 whitespace-nowrap text-[12.5px] font-medium tracking-tight text-ink-3 transition-colors duration-200 group-hover/rail:text-ink-1">
        {label}
      </span>

      {badge !== undefined && badge > 0 && (
        <span className="nav-label pointer-events-none relative z-10 ml-auto whitespace-nowrap rounded-full bg-crit/15 px-1.5 py-0.5 font-mono text-[9px] font-semibold text-crit">
          {badge}
        </span>
      )}

      {shortcut && (
        <span className="nav-label pointer-events-none relative z-10 ml-auto whitespace-nowrap opacity-0 transition-opacity duration-200 group-hover/rail:opacity-60">
          <span className="kbd">{shortcut}</span>
        </span>
      )}
    </>
  )

  const showTooltip = () => {
    const r = itemRef.current?.getBoundingClientRect()
    if (r) setTt({ x: r.right + 10, y: r.top + r.height / 2 })
  }

  const base =
    'group/item relative mx-2 flex h-10 shrink-0 items-center rounded-xl px-3 text-ink-3 transition-colors duration-200 hover:text-ink-1'

  return (
    <>
      {to ? (
        <NavLink
          ref={itemRef as React.Ref<HTMLAnchorElement>}
          to={to}
          onMouseEnter={showTooltip}
          onMouseLeave={() => setTt(null)}
          className={({ isActive }) => cn(base, isActive && 'text-ink-1 [&_.nav-active]:opacity-100')}
          aria-label={label}
        >
          {content}
        </NavLink>
      ) : (
        <button
          ref={itemRef as React.Ref<HTMLButtonElement>}
          onClick={onClick}
          onMouseEnter={showTooltip}
          onMouseLeave={() => setTt(null)}
          className={base}
          aria-label={label}
        >
          {content}
        </button>
      )}
      {tt && (
        <div
          className="glass-strong pointer-events-none fixed z-[100] -translate-y-1/2 rounded-lg px-2.5 py-1 text-[11px] font-medium text-ink-1 shadow-pop"
          style={{ left: tt.x, top: tt.y }}
          role="tooltip"
        >
          {label}
          {shortcut && <span className="ml-2 font-mono text-[10px] text-ink-3">{shortcut}</span>}
        </div>
      )}
    </>
  )
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div className="nav-label mx-5 mb-1 mt-4 whitespace-nowrap text-[9.5px] font-semibold uppercase tracking-[0.16em] text-ink-3 opacity-0 transition-opacity duration-200 first:mt-2 group-hover/rail:opacity-100">
      {children}
    </div>
  )
}

function BrandMark() {
  return (
    <svg viewBox="0 0 32 32" className="h-[26px] w-[26px] shrink-0 overflow-visible">
      <defs>
        <linearGradient id="ag-brand" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#7A8CFF" />
          <stop offset="55%" stopColor="#A78BFA" />
          <stop offset="100%" stopColor="#4CC2FF" />
        </linearGradient>
      </defs>
      <path
        d="M16 3.5 L27 26 L16 21.4 L5 26 Z"
        fill="none"
        stroke="url(#ag-brand)"
        strokeWidth={2.1}
        strokeLinejoin="round"
      />
      <circle cx="16" cy="14.6" r="2.6" fill="url(#ag-brand)" />
      <circle cx="16" cy="14.6" r="5.6" fill="none" stroke="url(#ag-brand)" strokeWidth={0.8} opacity={0.35} />
    </svg>
  )
}

export default function NavRail() {
  const setPaletteOpen = useAppStore((s) => s.setPaletteOpen)
  const setSettingsOpen = useAppStore((s) => s.setSettingsOpen)
  const focusMode = useAppStore((s) => s.focusMode)
  const presentation = useAppStore((s) => s.presentation.active)
  const alertCount = useAppStore((s) => s.alerts.length)

  if (focusMode || presentation) return null

  return (
    <nav className="group/rail z-40 hidden shrink-0 py-3 pl-3 pr-0 md:block" aria-label="Primary navigation">
      <div className="glass flex h-full w-[60px] flex-col overflow-hidden rounded-2xl shadow-rail transition-[width] duration-300 ease-swift group-hover/rail:w-[206px]">
        {/* brand */}
        <div className="flex h-14 shrink-0 items-center border-b border-line px-5">
          <BrandMark />
          <span className="nav-label pointer-events-none ml-2.5 whitespace-nowrap font-display text-[13.5px] font-semibold tracking-tight text-ink-1 opacity-0 transition-opacity duration-200 group-hover/rail:opacity-100">
            AeroGuard
            <span className="ml-1.5 align-middle font-mono text-[9px] font-medium text-ink-3">v2.4</span>
          </span>
        </div>

        <div className="no-scrollbar flex flex-1 flex-col overflow-y-auto overflow-x-hidden pb-1 pt-1">
          <SectionLabel>Monitor</SectionLabel>
          <RailItem to="/" icon={<Home size={17} strokeWidth={1.8} />} label="Overview" />
          <RailItem to="/fleet" icon={<Orbit size={17} strokeWidth={1.8} />} label="Fleet Map" />
          <RailItem to="/engines" icon={<Gauge size={17} strokeWidth={1.8} />} label="Engines" />
          <RailItem to="/alerts" icon={<AlertTriangle size={17} strokeWidth={1.8} />} label="Alerts" badge={alertCount} />

          <SectionLabel>Intelligence</SectionLabel>
          <RailItem to="/telemetry" icon={<Activity size={17} strokeWidth={1.8} />} label="Telemetry" />
          <RailItem to="/predictions" icon={<TrendingUp size={17} strokeWidth={1.8} />} label="Predictions" />
          <RailItem to="/models" icon={<Network size={17} strokeWidth={1.8} />} label="Models" />
        </div>

        <div className="flex shrink-0 flex-col border-t border-line py-1">
          <RailItem icon={<CommandIcon size={16} strokeWidth={1.8} />} label="Command" shortcut="⌘K" onClick={() => setPaletteOpen(true)} />
          <RailItem icon={<Settings2 size={16} strokeWidth={1.8} />} label="Settings" onClick={() => setSettingsOpen(true)} />
        </div>

        <div className="nav-label flex h-8 shrink-0 items-center gap-2.5 overflow-hidden border-t border-line px-5 text-[10px] text-ink-3 opacity-0 transition-opacity duration-200 group-hover/rail:opacity-100">
          <PanelLeftClose size={12} />
          <span className="whitespace-nowrap">Collapse rail</span>
        </div>
      </div>
    </nav>
  )
}
