import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, CornerDownLeft, ArrowUpDown, Command as CommandIcon } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { useCommands, getRecents, pushRecent, type CommandItem } from '@/commands/registry'
import { fuzzySearch } from '@/lib/fuzzy'
import { getEngines } from '@/data/db'
import { cn } from '@/lib/utils'
import { STATUS_META } from '@/types/engine'

const GROUP_ORDER: CommandItem['group'][] = ['recent', 'intelligence', 'actions', 'navigation', 'engines', 'sensors']
const GROUP_LABELS: Record<CommandItem['group'], string> = {
  recent: 'Recent',
  intelligence: 'AI Intent',
  actions: 'Actions',
  navigation: 'Navigation',
  engines: 'Engines',
  sensors: 'Sensors'
}

const BADGE_TONE: Record<string, string> = {
  nominal: 'border-info/30 bg-info/10 text-info',
  warning: 'border-warn/30 bg-warn/10 text-warn',
  critical: 'border-crit/30 bg-crit/10 text-crit',
  info: 'border-line-2 bg-surface-3 text-ink-2'
}

export default function CommandPalette() {
  const open = useAppStore((s) => s.paletteOpen)
  const setOpen = useAppStore((s) => s.setPaletteOpen)
  const [query, setQuery] = useState('')
  const [sel, setSel] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const { items, dynamic } = useCommands(query)
  const engines = useMemo(() => getEngines(), [])
  const recents = useMemo(() => getRecents(), [open])

  const results = useMemo(() => {
    if (!query.trim()) {
      const recentItems: CommandItem[] = recents
        .map((id) => engines.find((e) => e.id === id))
        .filter(Boolean)
        .slice(0, 2)
        .map((e) => ({
          id: `recent-${e!.id}`,
          title: `Engine #${e!.id}`,
          subtitle: `RUL ${e!.rul} · Health ${e!.health}%`,
          group: 'recent' as const,
          keywords: '',
          icon: <CommandIcon size={15} />,
          perform: () => {
            pushRecent(e!.id)
            useAppStore.getState().openExplorer(e!.id)
          }
        }))
      const base = items.filter((i) => i.group === 'actions' || i.group === 'navigation').slice(0, 11)
      return [...recentItems, ...base]
    }
    const intentItems = dynamic
    const scored = fuzzySearch(query, items, (i) => `${i.title} ${i.subtitle ?? ''} ${i.keywords}`, 14)
    const scoredItems = scored.map((s) => s.item)
    const seen = new Set<string>()
    return [...intentItems, ...scoredItems].filter((i) => (seen.has(i.id) ? false : (seen.add(i.id), true))).slice(0, 16)
  }, [query, items, dynamic, recents, engines])

  const grouped = useMemo(() => {
    const groups: { group: CommandItem['group']; items: CommandItem[] }[] = []
    for (const g of GROUP_ORDER) {
      const gi = results.filter((r) => r.group === g)
      if (gi.length) groups.push({ group: g, items: gi })
    }
    return groups
  }, [results])

  const flat = useMemo(() => grouped.flatMap((g) => g.items), [grouped])

  useEffect(() => {
    if (open) {
      setQuery('')
      setSel(0)
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [open])

  useEffect(() => {
    setSel(0)
  }, [query])

  useEffect(() => {
    const el = listRef.current?.querySelector<HTMLElement>(`[data-idx="${sel}"]`)
    el?.scrollIntoView({ block: 'nearest' })
  }, [sel])

  if (!open) return null

  const run = (item: CommandItem) => {
    setOpen(false)
    item.perform()
  }

  return (
    <div className="fixed inset-0 z-[90] flex items-start justify-center px-4 pt-[14vh] pb-6" role="dialog" aria-modal="true" aria-label="Command palette">
      <div className="absolute inset-0 bg-black/55 backdrop-blur-[3px] animate-fade-in" onClick={() => setOpen(false)} />
      <div className="glass-strong edge-lit relative w-full max-w-[580px] overflow-hidden rounded-2xl shadow-pop animate-scale-in">
        <div className="flex items-center gap-3 border-b border-line px-4">
          <Search size={15} className="shrink-0 text-ink-3" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSel((s) => Math.min(s + 1, flat.length - 1))
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSel((s) => Math.max(s - 1, 0))
              } else if (e.key === 'Enter') {
                e.preventDefault()
                if (flat[sel]) run(flat[sel])
              } else if (e.key === 'Escape') {
                setOpen(false)
              } else if (e.key === 'Tab') {
                e.preventDefault()
              }
            }}
            placeholder="Search engines, sensors, actions… try “compare 24 and 71” or “lowest RUL”"
            className="h-12 w-full bg-transparent text-[13.5px] text-ink-1 outline-none placeholder:text-ink-3"
            aria-label="Search commands"
            spellCheck={false}
          />
          <span className="kbd shrink-0">esc</span>
        </div>

        <div ref={listRef} className="max-h-[46vh] overflow-y-auto overscroll-contain p-1.5" role="listbox" aria-label="Command results">
          {grouped.map((g) => (
            <div key={g.group} className="mb-1">
              <div className="px-2.5 pb-1 pt-2 text-[9.5px] font-semibold uppercase tracking-[0.16em] text-ink-3">{GROUP_LABELS[g.group]}</div>
              {g.items.map((item) => {
                const idx = flat.indexOf(item)
                const active = idx === sel
                return (
                  <button
                    key={item.id}
                    data-idx={idx}
                    role="option"
                    aria-selected={active}
                    onMouseEnter={() => setSel(idx)}
                    onClick={() => run(item)}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition-colors',
                      active ? 'bg-accent/15' : 'hover:bg-white/[0.04]'
                    )}
                  >
                    <span className={cn('flex h-7 w-7 shrink-0 items-center justify-center rounded-md border', active ? 'border-accent/40 bg-accent/15 text-[#aab4ff]' : 'border-line bg-surface-3 text-ink-2')}>
                      {item.icon}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[12.5px] font-medium text-ink-1">{item.title}</span>
                      {item.subtitle && <span className="block truncate text-[11px] text-ink-3">{item.subtitle}</span>}
                    </span>
                    {item.badge && (
                      <span className={cn('shrink-0 rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide', BADGE_TONE[item.badge.tone])}>
                        {item.badge.label}
                      </span>
                    )}
                    {active && <CornerDownLeft size={12} className="shrink-0 text-ink-3" />}
                  </button>
                )
              })}
            </div>
          ))}
          {!results.length && (
            <div className="flex flex-col items-center gap-1.5 px-4 py-10 text-center">
              <span className="text-[13px] text-ink-2">No matches for “{query}”</span>
              <span className="text-[11px] text-ink-3">Try “engine 24”, “temperature sensor”, “critical engines” or “compare 24 and 71”</span>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-line px-4 py-2 text-[10px] text-ink-3">
          <span className="flex items-center gap-1.5">
            <ArrowUpDown size={11} /> navigate
          </span>
          <span className="flex items-center gap-1.5">
            <span className="kbd">↵</span> run command
          </span>
        </div>
      </div>
    </div>
  )
}
