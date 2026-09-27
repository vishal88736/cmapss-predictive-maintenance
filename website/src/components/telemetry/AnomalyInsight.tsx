import { X, Brain } from 'lucide-react'
import type { Anomaly } from '@/types/telemetry'
import { SENSOR_MAP } from '@/data/sensors'
import { fmtSensor } from '@/lib/format'
import { useAppStore } from '@/store/useAppStore'

export default function AnomalyInsight({ anomaly, onClose }: { anomaly: Anomaly; onClose: () => void }) {
  const tempUnit = useAppStore((s) => s.tempUnit)
  const ch = SENSOR_MAP[anomaly.sensorId]
  const tone = anomaly.severity === 'critical' ? 'crit' : anomaly.severity === 'warning' ? 'warn' : 'info'
  return (
    <div className="card animate-rise-in border-accent/25 p-4" role="region" aria-label="Sensor anomaly insight">
      <div className="mb-2 flex items-center justify-between">
        <span className={`flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-${tone}`}>
          <span className={`h-1.5 w-1.5 rounded-full bg-${tone}`} />
          {anomaly.sensorId} anomaly · cycle {anomaly.cycle}
        </span>
        <button className="icon-btn h-6 w-6" onClick={onClose} aria-label="Close anomaly insight">
          <X size={12} />
        </button>
      </div>
      <div className="grid grid-cols-2 gap-x-5 gap-y-2.5 text-[11.5px] sm:grid-cols-4">
        <div>
          <div className="eyebrow">Observed</div>
          <div className="stat-num mt-0.5 font-semibold text-ink-1">{fmtSensor(anomaly.sensorId, anomaly.observed, tempUnit)}</div>
        </div>
        <div>
          <div className="eyebrow">Expected range</div>
          <div className="stat-num mt-0.5 text-ink-2">
            {fmtSensor(anomaly.sensorId, anomaly.expectedLow, tempUnit, 1)}–{fmtSensor(anomaly.sensorId, anomaly.expectedHigh, tempUnit, 1)}
          </div>
        </div>
        <div>
          <div className="eyebrow">Deviation</div>
          <div className={`stat-num mt-0.5 font-semibold ${anomaly.deviation > 0 ? 'text-warn' : 'text-info'}`}>
            {anomaly.deviation > 0 ? '+' : ''}{anomaly.deviation}%
          </div>
        </div>
        <div>
          <div className="eyebrow">Channel</div>
          <div className="mt-0.5 text-ink-2">{ch?.short ?? anomaly.sensorId}</div>
        </div>
      </div>
      <div className="mt-3 space-y-1.5 border-t border-line pt-3 text-[11.5px] leading-relaxed">
        <div className="flex gap-2">
          <span className="w-[74px] shrink-0 text-ink-3">Pattern</span>
          <span className="text-ink-2">{anomaly.pattern}</span>
        </div>
        <div className="flex gap-2">
          <span className="w-[74px] shrink-0 text-ink-3">Effect</span>
          <span className="text-ink-2">{anomaly.effect}</span>
        </div>
      </div>
      <p className="mt-2.5 flex items-start gap-1.5 text-[10px] leading-relaxed text-ink-3">
        <Brain size={11} className="mt-px shrink-0" />
        Interpretation is MODEL-DERIVED from the learned operating envelope — not a direct sensor alarm. Correlation with degradation is supported by attribution scores, not claimed causality.
      </p>
    </div>
  )
}
