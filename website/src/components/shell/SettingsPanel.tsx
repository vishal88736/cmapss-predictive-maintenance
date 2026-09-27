import { X, Thermometer, Radio, Sparkles } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { cn } from '@/lib/utils'

function Toggle({ on, onChange, label, description, icon: Icon }: { on: boolean; onChange: (v: boolean) => void; label: string; description: string; icon: React.ElementType }) {
  return (
    <button className="flex w-full items-start gap-3 rounded-xl border border-line bg-surface-2 p-3.5 text-left transition-colors hover:border-line-2" onClick={() => onChange(!on)} role="switch" aria-checked={on}>
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-line bg-surface-3 text-ink-2">
        <Icon size={14} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-[12.5px] font-medium text-ink-1">{label}</span>
        <span className="mt-0.5 block text-[11px] leading-relaxed text-ink-3">{description}</span>
      </span>
      <span className={cn('relative mt-1 h-[18px] w-8 shrink-0 rounded-full transition-colors', on ? 'bg-accent/80' : 'bg-white/10')}>
        <span className={cn('absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white transition-all', on ? 'left-[16px]' : 'left-[2px]')} />
      </span>
    </button>
  )
}

export default function SettingsPanel() {
  const open = useAppStore((s) => s.settingsOpen)
  const setOpen = useAppStore((s) => s.setSettingsOpen)
  const tempUnit = useAppStore((s) => s.tempUnit)
  const setTempUnit = useAppStore((s) => s.setTempUnit)
  const liveFeed = useAppStore((s) => s.liveFeed)
  const setLiveFeed = useAppStore((s) => s.setLiveFeed)

  if (!open) return null
  return (
    <div className="fixed inset-0 z-[93]" role="dialog" aria-modal="true" aria-label="Settings">
      <div className="absolute inset-0 bg-black/45 animate-fade-in" onClick={() => setOpen(false)} />
      <div className="absolute bottom-0 right-0 top-0 flex w-full max-w-[340px] flex-col border-l border-line-2 bg-surface-1/95 shadow-pop backdrop-blur-xl animate-slide-left">
        <div className="flex items-center justify-between border-b border-line px-4 py-3.5">
          <h2 className="font-display text-[14px] font-semibold">Settings</h2>
          <button className="icon-btn" onClick={() => setOpen(false)} aria-label="Close settings">
            <X size={15} />
          </button>
        </div>
        <div className="flex-1 space-y-3 overflow-y-auto p-4">
          <div>
            <div className="panel-title mb-2">Display</div>
            <div className="flex gap-2" role="radiogroup" aria-label="Temperature unit">
              {(['R', 'K'] as const).map((u) => (
                <button
                  key={u}
                  className={cn('btn flex-1', tempUnit === u ? 'btn-primary' : '')}
                  onClick={() => setTempUnit(u)}
                  role="radio"
                  aria-checked={tempUnit === u}
                >
                  °{u} {u === 'R' ? '(Rankine)' : '(Kelvin)'}
                </button>
              ))}
            </div>
          </div>
          <Toggle
            on={liveFeed}
            onChange={setLiveFeed}
            label="Live telemetry simulation"
            description="Stream synthetic C-MAPSS cycles, generate live alerts and keep the clock ticking."
            icon={Radio}
          />
          <Toggle
            on={true}
            onChange={() => {}}
            label="Reduced motion (system)"
            description="Follows your OS prefers-reduced-motion setting automatically."
            icon={Sparkles}
          />
          <div className="!mt-5 rounded-xl border border-line bg-surface-2 p-3.5">
            <div className="panel-title mb-2">About</div>
            <p className="text-[11px] leading-relaxed text-ink-3">
              AeroGuard AI · v2.4<br />
              Telemetry synthesised from NASA C-MAPSS FD001 baselines (100 turbofan units, 14+ channels).
              Model metrics reflect a BiLSTM RUL predictor. All data is served behind a single service layer —
              point <code className="rounded bg-surface-3 px-1 font-mono text-[10px] text-ink-2">services/api.ts</code> at a real backend to go live.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
