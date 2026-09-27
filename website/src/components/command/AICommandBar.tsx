import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Sparkles, ArrowUp, X } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { useCommands, type CommandItem } from '@/commands/registry'
import { getEngines, getFleetStats } from '@/data/db'
import { cn } from '@/lib/utils'

interface Answer {
  text: string
  actions: CommandItem[]
}

const SUGGESTIONS = [
  'Which engines have the lowest RUL?',
  'Why is Engine 24 at high risk?',
  'Compare 24 and 71',
  'Show abnormal temperature patterns',
  'Simulate +5% temperature'
]

export default function AICommandBar({ contextLabel, className }: { contextLabel: string; className?: string }) {
  const [query, setQuery] = useState('')
  const [thinking, setThinking] = useState(false)
  const [answer, setAnswer] = useState<Answer | null>(null)
  const { dynamic } = useCommands(query)
  const navigate = useNavigate()
  const setPaletteOpen = useAppStore((s) => s.setPaletteOpen)
  const engines = useMemo(() => getEngines(), [])
  const stats = useMemo(() => getFleetStats(), [])

  const computeAnswer = (q: string): Answer => {
    const ql = q.toLowerCase()
    if (/(lowest|shortest|worst).*(rul|life)|rul.*low/.test(ql)) {
      const top = [...engines].sort((a, b) => a.rul - b.rul).slice(0, 3)
      return {
        text: `The shortest remaining useful lives are ${top.map((e) => `#${e.id} (${e.rul} cyc)`).join(', ')}. ${stats.criticalCount} engines are in critical state right now.`,
        actions: dynamic
      }
    }
    if (/(how many|count|total).*(engine|critical|warning)/.test(ql)) {
      return {
        text: `Fleet: ${stats.monitored} engines monitored — ${stats.nominalCount} nominal, ${stats.warningCount} warning, ${stats.criticalCount} critical. Average health ${stats.health}%, average RUL ${stats.avgRul} cycles.`,
        actions: []
      }
    }
    if (/abnormal|anomal|deviat|temperature pattern/.test(ql)) {
      const withAnoms = engines.filter((e) => e.anomalyCount > 0).slice(0, 3)
      return {
        text: withAnoms.length
          ? `${withAnoms.map((e) => `#${e.id}`).join(', ')} show significant sensor deviations in the recent window. The strongest signal is on ${withAnoms[0].topSensors[0]}.`
          : 'No significant deviations detected in the current window.',
        actions: dynamic.length ? dynamic : []
      }
    }
    if (dynamic.length) {
      return {
        text: 'I resolved this to an analytical action.',
        actions: dynamic
      }
    }
    return {
      text: `I couldn't map that to a specific analysis for ${contextLabel.toLowerCase()}. Try asking about engines (“engine 24”), RUL rankings, comparisons (“compare 24 and 71”), sensors, or alerts.`,
      actions: []
    }
  }

  const submit = () => {
    if (!query.trim() || thinking) return
    setThinking(true)
    const q = query
    setTimeout(() => {
      setAnswer(computeAnswer(q))
      setThinking(false)
    }, 550)
  }

  return (
    <div className={cn('relative', className)}>
      {answer && (
        <div className="absolute bottom-[calc(100%+10px)] left-0 right-0 z-30 animate-rise-in rounded-xl border border-accent/25 bg-surface-2/95 p-3.5 shadow-pop backdrop-blur-xl md:left-24 md:right-24">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-accent">
              <Sparkles size={11} /> Model Insight
            </span>
            <button className="icon-btn h-6 w-6" onClick={() => setAnswer(null)} aria-label="Dismiss answer">
              <X size={12} />
            </button>
          </div>
          <p className="text-[12.5px] leading-relaxed text-ink-1">{answer.text}</p>
          {answer.actions.length > 0 && (
            <div className="mt-2.5 flex flex-wrap gap-2">
              {answer.actions.slice(0, 2).map((a) => (
                <button
                  key={a.id}
                  className="btn btn-primary h-7 px-2.5 text-[11px]"
                  onClick={() => {
                    a.perform()
                    setAnswer(null)
                  }}
                >
                  {a.icon}
                  {a.title}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex items-center gap-2 overflow-x-auto rounded-xl border border-line bg-surface-2/80 px-2 py-1.5 shadow-card backdrop-blur-md no-scrollbar">
        <Sparkles size={13} className="ml-1 shrink-0 text-accent" aria-hidden />
        <div className="flex gap-1.5 no-scrollbar">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              onClick={() => {
                setQuery(s)
                setAnswer(null)
              }}
              className="chip whitespace-nowrap hover:border-accent/40 hover:text-ink-1"
            >
              {s}
            </button>
          ))}
        </div>
        <div className="ml-auto flex min-w-0 flex-1 items-center gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
            placeholder={`Ask about ${contextLabel.toLowerCase()}…`}
            aria-label={`Ask AI about ${contextLabel}`}
            className="h-7 min-w-[120px] flex-1 bg-transparent text-[12px] text-ink-1 outline-none placeholder:text-ink-3"
          />
          <button
            onClick={submit}
            disabled={!query.trim() || thinking}
            className="btn btn-primary h-7 shrink-0 gap-1 px-2.5 text-[11px]"
            aria-label="Ask"
          >
            {thinking ? (
              <span className="flex items-center gap-1.5">
                <span className="h-2 w-2 animate-pulse-dot rounded-full bg-accent" /> thinking…
              </span>
            ) : (
              <>
                Ask <ArrowUp size={11} />
              </>
            )}
          </button>
          <button onClick={() => setPaletteOpen(true)} className="kbd shrink-0" aria-label="Open command palette">⌘K</button>
        </div>
      </div>
    </div>
  )
}
