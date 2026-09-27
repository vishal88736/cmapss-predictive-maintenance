import { useState } from 'react'
import { ChevronDown, Radio, Cpu, CloudLightning, Database } from 'lucide-react'
import { useAppStore } from '@/store/useAppStore'
import { useNow } from '@/lib/hooks'
import { fmtTime } from '@/lib/format'
import { cn } from '@/lib/utils'

const SERVICES = [
  { id: 'stream', label: 'Telemetry stream', state: 'CONNECTED', icon: Radio },
  { id: 'ml', label: 'ML inference', state: 'READY', icon: Cpu },
  { id: 'pred', label: 'Prediction service', state: 'ONLINE', icon: CloudLightning }
]

export default function SystemStatus() {
  const [open, setOpen] = useState(false)
  const lastSync = useAppStore((s) => s.lastSync)
  const liveFeed = useAppStore((s) => s.liveFeed)
  const now = useNow(1000)

  return (
    <div className="pointer-events-auto fixed bottom-4 left-4 z-40 hidden md:block">
      <div
        className={cn(
          'glass-strong w-[212px] animate-rise-in overflow-hidden rounded-2xl shadow-rail transition-all duration-300 ease-swift',
          open && 'w-[258px]'
        )}
      >
        <button
          className="flex w-full items-center justify-between px-3.5 pt-3 text-left"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          aria-label={`System status. Telemetry connected, ML ready, predictions online. Last sync ${fmtTime(lastSync)}. Click to ${open ? 'collapse' : 'expand'} diagnostics.`}
        >
          <span className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-ok shadow-glow-ok animate-pulse-dot" />
            <span className="panel-title">System</span>
          </span>
          <ChevronDown size={12} className={cn('text-ink-3 transition-transform duration-300 ease-swift', open && 'rotate-180')} />
        </button>
        <div className="space-y-1.5 px-3.5 pb-3 pt-2">
          {SERVICES.map((s) => (
            <div key={s.id} className="flex items-center justify-between text-[11px]">
              <span className="flex items-center gap-2 text-ink-2">
                <s.icon size={12} className="text-ink-3" />
                {s.label}
              </span>
              <span className="flex items-center gap-1.5 font-mono text-[10px] font-semibold text-ok">
                <span className="h-1.5 w-1.5 rounded-full bg-ok animate-pulse-dot" role="img" aria-label="status dot" />
                {s.state}
              </span>
            </div>
          ))}
          <div className="flex items-center justify-between border-t border-line pt-2 text-[10px] text-ink-3">
            <span className="flex items-center gap-1.5">
              <Radio size={10} className={liveFeed ? 'text-ok' : 'text-ink-3'} />
              {liveFeed ? 'Live simulation' : 'Feed paused'}
            </span>
            <span className="font-mono">{fmtTime(lastSync)}</span>
          </div>
        </div>
        {open && (
          <div className="animate-fade-in space-y-2 border-t border-line bg-surface-0/60 px-3.5 py-3 text-[10.5px]">
            <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
              {[
                ['Ingest rate', '48.2 cyc/s'],
                ['Buffer', 'healthy'],
                ['Model', 'LSTM v2.4'],
                ['Inference', '38 ms'],
                ['Units tracked', '100'],
                ['Uptime', '99.98%']
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between gap-2">
                  <span className="text-ink-3">{k}</span>
                  <span className="font-mono text-ink-2">{v}</span>
                </div>
              ))}
            </div>
            <div className="flex items-start gap-1.5 border-t border-line pt-2.5 text-ink-3">
              <Database size={11} className="mt-0.5 shrink-0" />
              <span>FD001 · synced {fmtTime(lastSync)} ({Math.max(0, Math.round((now - lastSync) / 1000))}s ago)</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
