import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Home, Orbit, Gauge, Activity, TrendingUp, AlertTriangle, Network, Play, Columns2,
  Sparkles, Focus, Presentation, Settings2, Keyboard, ScanSearch, ArrowRightLeft,
  Radar, Thermometer, Gauge as GaugeIcon, Waves, RefreshCw, Zap
} from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { getEngines } from '@/data/db'
import { SENSOR_CHANNELS, SENSOR_MAP } from '@/data/sensors'
import type { Engine } from '@/types/engine'
import { STATUS_META } from '@/types/engine'

export type CommandGroup = 'recent' | 'actions' | 'navigation' | 'engines' | 'sensors' | 'intelligence'

export interface CommandItem {
  id: string
  title: string
  subtitle?: string
  group: CommandGroup
  keywords: string
  icon: React.ReactNode
  perform: () => void
  badge?: { label: string; tone: 'nominal' | 'warning' | 'critical' | 'info' }
}

const RECENTS_KEY = 'aeroguard-recents'

const pad = (n: string) => String(Number(n)).padStart(3, '0')

export function getRecents(): string[] {
  try {
    return JSON.parse(localStorage.getItem(RECENTS_KEY) || '[]')
  } catch {
    return []
  }
}

export function pushRecent(id: string) {
  const next = [id, ...getRecents().filter((r) => r !== id)].slice(0, 4)
  localStorage.setItem(RECENTS_KEY, JSON.stringify(next))
}

const STATUS_TONE: Record<string, 'nominal' | 'warning' | 'critical' | 'info'> = { nominal: 'info', warning: 'warning', critical: 'critical' }

export function useCommands(query: string): { items: CommandItem[]; dynamic: CommandItem[] } {
  const navigate = useNavigate()
  const store = useAppStore()

  const engines = useMemo(() => getEngines(), [])

  const intents = useMemo<CommandItem[]>(() => {
    const q = query.toLowerCase().trim()
    if (!q) return []
    const out: CommandItem[] = []
    const smart: CommandItem[] = []

    const compareMatch = q.match(/compare\s+(?:engine\s*)?#?(\d{1,3})\s*(?:and|vs|with|to|&)\s*(?:engine\s*)?#?(\d{1,3})/)
    if (compareMatch) {
      const a = pad(compareMatch[1])
      const b = pad(compareMatch[2])
      if (engines.find((e) => e.id === a) && engines.find((e) => e.id === b)) {
        smart.push({
          id: 'intent-compare',
          title: `Compare Engine #${a} and #${b}`,
          subtitle: 'Split-screen comparison with synchronized timelines',
          group: 'intelligence',
          keywords: 'compare engines diff versus',
          icon: <Columns2 size={15} />,
          perform: () => {
            store.setCompareIds([a, b])
            navigate(`/compare/${a}/${b}`)
          }
        })
      }
    }

    if (/(lowest|shortest|worst|minimum)\s+(rul|life|remaining)/.test(q)) {
      smart.push({
        id: 'intent-low-rul',
        title: 'Show engines by lowest RUL',
        subtitle: `Ranking — #${engines.slice().sort((a, b) => a.rul - b.rul)[0].unit} leads with ${engines.slice().sort((a, b) => a.rul - b.rul)[0].rul} cycles`,
        group: 'intelligence',
        keywords: 'lowest rul ranking worst engines remaining useful life',
        icon: <TrendingUp size={15} />,
        perform: () => navigate('/predictions?sort=rul&order=asc')
      })
    }

    if (/(critical|emergency|danger|failing)/.test(q)) {
      smart.push({
        id: 'intent-critical',
        title: 'Show critical engines',
        subtitle: `${engines.filter((e) => e.status === 'critical').length} engines in critical state`,
        group: 'intelligence',
        keywords: 'critical engines emergency failing danger',
        icon: <AlertTriangle size={15} />,
        perform: () => navigate('/fleet?status=critical')
      })
    }

    if (/(high.?risk|watchlist|attention|require)/.test(q)) {
      smart.push({
        id: 'intent-watchlist',
        title: 'Open the watchlist',
        subtitle: 'Fleet engines requiring attention, sorted by risk',
        group: 'intelligence',
        keywords: 'high risk watchlist attention require urgent',
        icon: <Radar size={15} />,
        perform: () => navigate('/fleet?status=attention')
      })
    }

    if (/(why|explain|reason|insight|cause)/.test(q)) {
      const em = q.match(/#?(\d{1,3})/)
      if (em) {
        const e = engines.find((en) => en.unit === Number(em[1]))
        if (e) {
          smart.push({
            id: 'intent-why',
            title: `Why is Engine #${e.id} at ${e.status} risk?`,
            subtitle: `Model explanation — top signals: ${e.topSensors.join(' · ')}`,
            group: 'intelligence',
            keywords: `why explain reason insight cause engine ${e.unit} risk`,
            icon: <Sparkles size={15} />,
            perform: () => {
              pushRecent(e.id)
              store.openExplorer(e.id)
              navigate(`/engine/${e.id}?explain=1`)
            }
          })
        }
      }
    }

    if (/(simulate|what.?if|scenario|experiment)/.test(q)) {
      const em = q.match(/#?(\d{1,3})/)
      const target = em ? engines.find((en) => en.unit === Number(em[1])) : engines.find((en) => en.unit === 24)
      if (target) {
        smart.push({
          id: 'intent-sim',
          title: `Create what-if scenario for Engine #${target.id}`,
          subtitle: 'Baseline vs scenario — RUL and risk impact',
          group: 'intelligence',
          keywords: 'simulate what if scenario experiment model',
          icon: <Zap size={15} />,
          perform: () => {
            pushRecent(target.id)
            navigate(`/engine/${target.id}?simulate=1`)
          }
        })
      }
    }

    if (/(show\s+)?alerts?/.test(q) && /(alert|alarm|event)/.test(q)) {
      smart.push({
        id: 'intent-alerts',
        title: 'Show recent alerts',
        subtitle: 'Event timeline across the fleet',
        group: 'intelligence',
        keywords: 'show alerts events alarms timeline',
        icon: <AlertTriangle size={15} />,
        perform: () => navigate('/alerts')
      })
    }

    const simNum = q.match(/simulate\s*\+?(\d{1,2})\s*%/)
    if (simNum) {
      smart.push({
        id: 'intent-sim-temp',
        title: `Simulate +${simNum[1]}% temperature on Engine #024`,
        subtitle: 'What-if scenario on the showcase engine',
        group: 'intelligence',
        keywords: 'simulate temperature percent scenario',
        icon: <Thermometer size={15} />,
        perform: () => navigate('/engine/024?simulate=1')
      })
    }

    out.push(...smart)
    return out
  }, [query, engines, navigate, store])

  const items = useMemo<CommandItem[]>(() => {
    const nav = (to: string, label: string, icon: React.ReactNode, kw: string, baseId: string): CommandItem => ({
      id: `nav-${baseId}`, title: label, group: 'navigation', keywords: kw, icon, perform: () => navigate(to)
    })
    const statics: CommandItem[] = [
      { id: 'act-compare', title: 'Compare Engines', subtitle: '024 vs 071 — synchronized telemetry', group: 'actions', keywords: 'compare engines split screen diff', icon: <Columns2 size={15} />, perform: () => { store.setCompareIds(['024', '071']); navigate('/compare/024/071') } },
      { id: 'act-predict', title: 'Run RUL Prediction', subtitle: 'Refresh inference for the showcase engine', group: 'actions', keywords: 'run rul prediction model inference', icon: <Play size={15} />, perform: () => { navigate('/predictions'); store.pushToast({ title: 'RUL inference refreshed', description: 'LSTM RUL Predictor v2.4 · 100 engines · RMSE 19.8', tone: 'ok' }) } },
      { id: 'act-sensor', title: 'Analyze Sensor', subtitle: 'Open the telemetry lab', group: 'actions', keywords: 'analyze sensor telemetry chart', icon: <ScanSearch size={15} />, perform: () => navigate('/telemetry') },
      { id: 'act-focus', title: 'Toggle Focus Mode', subtitle: 'Hide navigation, maximize the canvas', group: 'actions', keywords: 'focus mode zen hide navigation', icon: <Focus size={15} />, perform: () => { store.setFocusMode(false); store.toggleFocusMode(); navigate(`/engine/${store.lastEngineId ?? '024'}`) } },
      { id: 'act-present', title: 'Start Presentation', subtitle: 'Fullscreen guided walkthrough', group: 'actions', keywords: 'presentation demo fullscreen walkthrough judges', icon: <Presentation size={15} />, perform: () => store.startPresentation(store.lastEngineId ?? '024') },
      { id: 'act-refresh', title: 'Resync Telemetry', subtitle: 'Pull latest cycle data from the stream', group: 'actions', keywords: 'resync refresh data stream sync', icon: <RefreshCw size={15} />, perform: () => { store.tickSync(); store.pushToast({ title: 'Telemetry resynced', description: 'All 100 units up to date', tone: 'ok' }) } },
      nav('/', 'Overview', <Home size={15} />, 'overview home dashboard bento command center', 'overview'),
      nav('/fleet', 'Fleet Map', <Orbit size={15} />, 'fleet map engines spatial risk canvas', 'fleet'),
      nav('/engines', 'Engines', <Gauge size={15} />, 'engines list index units table', 'engines'),
      nav('/telemetry', 'Telemetry Lab', <Activity size={15} />, 'telemetry lab sensors signals charts', 'telemetry'),
      nav('/predictions', 'Predictions', <TrendingUp size={15} />, 'predictions rul ranking simulation what if', 'predictions'),
      nav('/alerts', 'Alerts', <AlertTriangle size={15} />, 'alerts events timeline warnings critical', 'alerts'),
      nav('/models', 'Model Insights', <Network size={15} />, 'model insights ml observability metrics drift lstm', 'models'),
      { id: 'sys-settings', title: 'Open Settings', group: 'actions', keywords: 'settings preferences units motion', icon: <Settings2 size={15} />, perform: () => store.setSettingsOpen(true) },
      { id: 'sys-shortcuts', title: 'Keyboard Shortcuts', subtitle: 'Show the shortcut map', group: 'actions', keywords: 'keyboard shortcuts help keys', icon: <Keyboard size={15} />, perform: () => store.setHelpOpen(true) }
    ]

    const engineItems: CommandItem[] = engines.map((e: Engine) => ({
      id: `eng-${e.id}`,
      title: `Engine #${e.id}`,
      subtitle: `RUL ${e.rul} · Health ${e.health}% · Risk ${e.risk}%`,
      group: 'engines',
      keywords: `engine ${e.unit} unit ${e.id} rul ${e.status}`,
      icon: <GaugeIcon size={15} style={{ color: STATUS_META[e.status].color }} />,
      badge: { label: STATUS_META[e.status].label, tone: STATUS_TONE[e.status] },
      perform: () => {
        pushRecent(e.id)
        store.openExplorer(e.id)
      }
    }))

    const sensorItems: CommandItem[] = SENSOR_CHANNELS.map((s) => ({
      id: `sensor-${s.id}`,
      title: `${s.id} — ${s.name}`,
      subtitle: `Analyze ${s.short} · baseline ${s.baseline} ${s.unit}`,
      group: 'sensors',
      keywords: `sensor ${s.id} ${s.name} ${s.short} ${s.family} channel`,
      icon: s.family === 'temp' ? <Thermometer size={15} /> : s.family === 'speed' ? <GaugeIcon size={15} /> : s.family === 'pressure' ? <Waves size={15} /> : <Activity size={15} />,
      perform: () => {
        store.setPrimarySensor(s.id)
        navigate('/telemetry')
      }
    }))

    return [...statics, ...engineItems, ...sensorItems]
  }, [engines, navigate, store])

  return { items, dynamic: intents }
}

export { SENSOR_MAP }
