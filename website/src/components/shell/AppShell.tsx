import { useEffect, useRef } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import { Home, Orbit, TrendingUp, AlertTriangle, Command as CommandIcon } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import NavRail from './NavRail'
import AmbientBackground from './AmbientBackground'
import SystemStatus from './SystemStatus'
import Toasts from './Toasts'
import HelpOverlay from './HelpOverlay'
import SettingsPanel from './SettingsPanel'
import CommandPalette from '@/components/command/CommandPalette'
import EngineExplorer from '@/components/engine/EngineExplorer'
import Presentation from './Presentation'
import { useAppStore } from '@/store/useAppStore'
import { makeLiveAlert } from '@/data/generate'
import { cn } from '@/lib/utils'

function MobileTabBar() {
  const setPaletteOpen = useAppStore((s) => s.setPaletteOpen)
  const focusMode = useAppStore((s) => s.focusMode)
  const presentation = useAppStore((s) => s.presentation.active)
  if (focusMode || presentation) return null
  const items = [
    { to: '/', label: 'Overview', icon: Home },
    { to: '/fleet', label: 'Fleet', icon: Orbit },
    { to: '/predictions', label: 'Predict', icon: TrendingUp },
    { to: '/alerts', label: 'Alerts', icon: AlertTriangle }
  ]
  return (
    <nav
      className="fixed inset-x-0 bottom-0 z-40 flex justify-center px-3 pb-3 md:hidden"
      aria-label="Mobile navigation"
    >
      <div className="glass-strong flex w-full max-w-md items-stretch justify-between rounded-2xl p-1.5 pb-safe shadow-rail">
        {items.map((i) => (
          <NavLink
            key={i.to}
            to={i.to}
            className={({ isActive }) =>
              cn(
                'relative flex flex-1 flex-col items-center justify-center gap-1 rounded-xl px-2 py-1.5 text-[9.5px] font-medium transition-all duration-300 ease-swift',
                isActive ? 'text-ink-1' : 'text-ink-3'
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && <span className="nav-active absolute inset-0 rounded-xl" />}
                <i.icon size={17} strokeWidth={1.8} className="relative z-10" />
                <span className="relative z-10">{i.label}</span>
              </>
            )}
          </NavLink>
        ))}
        <button
          className="flex flex-1 flex-col items-center justify-center gap-1 rounded-xl px-2 py-1.5 text-[9.5px] font-medium text-ink-3 transition-colors hover:text-ink-1"
          onClick={() => setPaletteOpen(true)}
          aria-label="Open command palette"
        >
          <CommandIcon size={17} strokeWidth={1.8} />
          Search
        </button>
      </div>
    </nav>
  )
}

export default function AppShell() {
  const navigate = useNavigate()
  const location = useLocation()
  const store = useAppStore()
  const alertCounter = useRef(0)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)
      const s = useAppStore.getState()

      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        s.setPaletteOpen(!s.paletteOpen)
        return
      }
      if (typing) return

      if (e.key === 'Escape') {
        const any = s.paletteOpen || s.helpOpen || s.settingsOpen || s.explorerEngineId || s.focusMode || s.presentation.active
        if (any) {
          s.setPaletteOpen(false)
          s.setHelpOpen(false)
          s.setSettingsOpen(false)
          s.closeExplorer()
          s.setFocusMode(false)
          s.stopPresentation()
        }
        return
      }
      if (e.key === '?') {
        e.preventDefault()
        s.setHelpOpen(!s.helpOpen)
        return
      }
      if (e.key === '/') {
        e.preventDefault()
        s.setPaletteOpen(true)
        return
      }
      if (e.key === 'f' || e.key === 'F') {
        const onTwin = location.pathname.startsWith('/engine/')
        if (!onTwin) {
          navigate(`/engine/${s.lastEngineId ?? '024'}`)
          useAppStore.getState().setFocusMode(true)
        } else {
          s.toggleFocusMode()
        }
        return
      }
      if (e.key === 'e' || e.key === 'E') {
        if (s.lastEngineId) s.openExplorer(s.lastEngineId)
        return
      }
      if (e.key === 'p' || e.key === 'P') {
        if (!s.presentation.active) s.startPresentation(s.lastEngineId ?? '024')
        return
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [navigate, location.pathname])

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>
    const schedule = () => {
      timer = setTimeout(() => {
        const s = useAppStore.getState()
        if (s.liveFeed && !document.hidden && !s.presentation.active) {
          const alert = makeLiveAlert(alertCounter.current++)
          s.pushAlert(alert)
          s.tickSync()
          if (alert.severity !== 'info' || Math.random() > 0.4) {
            s.pushToast({
              title: `${alert.severity === 'critical' ? 'Critical' : alert.severity === 'warning' ? 'Warning' : 'Info'} · Engine #${alert.engineId}`,
              description: alert.title,
              tone: alert.severity === 'critical' ? 'crit' : alert.severity === 'warning' ? 'warn' : 'info',
              actionLabel: 'View engine',
              action: () => s.openExplorer(alert.engineId),
              duration: 6000
            })
          }
        }
        schedule()
      }, 50000 + Math.random() * 45000)
    }
    schedule()
    return () => clearTimeout(timer)
  }, [])

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-bg">
      <AmbientBackground />
      <NavRail />
      <main className={cn('relative z-10 min-w-0 flex-1', 'pb-20 md:pb-0')} id="main-content">
        <div key={location.pathname} className="h-full animate-fade-in">
          <Outlet />
        </div>
      </main>
      <MobileTabBar />
      <SystemStatus />
      <Toasts />
      <CommandPalette />
      <HelpOverlay />
      <SettingsPanel />
      <EngineExplorer />
      <Presentation />
    </div>
  )
}
