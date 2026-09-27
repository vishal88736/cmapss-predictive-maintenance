export type EngineStatus = 'nominal' | 'warning' | 'critical'

export type DegradationStage = 'healthy' | 'early' | 'degradation' | 'critical'

export interface Engine {
  id: string
  unit: number
  currentCycle: number
  rul: number
  health: number
  risk: number
  status: EngineStatus
  stage: DegradationStage
  faultMode: string
  dataset: string
  confidence: number
  topSensors: string[]
  sensorSnapshot: Record<string, number>
  anomalyCount: number
  lastSync: number
}

export interface FleetStats {
  health: number
  healthDelta: number
  nominalCount: number
  warningCount: number
  criticalCount: number
  attentionCount: number
  avgRul: number
  totalCycles: number
  monitored: number
}

export const STATUS_META: Record<EngineStatus, { label: string; color: string; text: string; bg: string; border: string; dot: string }> = {
  nominal: { label: 'Nominal', color: 'var(--ok)', text: 'text-ok', bg: 'bg-ok/10', border: 'border-ok/30', dot: 'bg-ok' },
  warning: { label: 'Warning', color: 'var(--warn)', text: 'text-warn', bg: 'bg-warn/10', border: 'border-warn/30', dot: 'bg-warn' },
  critical: { label: 'Critical', color: 'var(--crit)', text: 'text-crit', bg: 'bg-crit/10', border: 'border-crit/30', dot: 'bg-crit' }
}
