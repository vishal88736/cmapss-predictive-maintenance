import { X } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'

const SHORTCUTS: { keys: string[]; label: string; group: string }[] = [
  { keys: ['⌘', 'K'], label: 'Command palette', group: 'Global' },
  { keys: ['/'], label: 'Ask the AI bar', group: 'Global' },
  { keys: ['?'], label: 'This shortcut map', group: 'Global' },
  { keys: ['E'], label: 'Engine explorer (last engine)', group: 'Global' },
  { keys: ['F'], label: 'Focus mode on the digital twin', group: 'Global' },
  { keys: ['P'], label: 'Presentation mode', group: 'Global' },
  { keys: ['Esc'], label: 'Close panels & modes', group: 'Global' },
  { keys: ['[', ']'], label: 'Previous / next engine (twin)', group: 'Engine Twin' },
  { keys: ['1', '–', '7'], label: 'Jump between sections (palette)', group: 'Global' },
  { keys: ['←', '→'], label: 'Scrub timeline / advance slides', group: 'Engine Twin' },
  { keys: ['Enter'], label: 'Run selected command', group: 'Command Palette' },
  { keys: ['↑', '↓'], label: 'Move through results', group: 'Command Palette' }
]

export default function HelpOverlay() {
  const open = useAppStore((s) => s.helpOpen)
  const setOpen = useAppStore((s) => s.setHelpOpen)
  if (!open) return null
  return (
    <div className="fixed inset-0 z-[92] flex items-center justify-center p-4" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts">
      <div className="absolute inset-0 bg-black/55 backdrop-blur-[3px] animate-fade-in" onClick={() => setOpen(false)} />
      <div className="relative w-full max-w-[430px] overflow-hidden rounded-xl border border-line-2 bg-surface-2/95 p-5 shadow-pop animate-rise-in">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-display text-[15px] font-semibold text-ink-1">Keyboard first</h2>
            <p className="text-[11px] text-ink-3">The whole interface is reachable without a mouse.</p>
          </div>
          <button className="icon-btn" onClick={() => setOpen(false)} aria-label="Close shortcuts">
            <X size={15} />
          </button>
        </div>
        <div className="max-h-[52vh] space-y-1 overflow-y-auto pr-1">
          {SHORTCUTS.map((s) => (
            <div key={s.label} className="flex items-center justify-between gap-3 rounded-lg px-2 py-1.5 hover:bg-white/[0.03]">
              <span className="text-[12px] text-ink-2">{s.label}</span>
              <span className="flex shrink-0 items-center gap-1">
                {s.keys.map((k) => (
                  <span key={k} className="kbd">{k}</span>
                ))}
              </span>
            </div>
          ))}
        </div>
        <p className="mt-4 border-t border-line pt-3 text-[10px] leading-relaxed text-ink-3">
          Status is never communicated through color alone — every indicator pairs color with a label and icon. Motion respects prefers-reduced-motion.
        </p>
      </div>
    </div>
  )
}
