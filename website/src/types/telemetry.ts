import type { EngineStatus } from './engine'

export type SensorBehavior = 'stable' | 'drift' | 'variance' | 'degrading' | 'anomalous'

export type SensorFamily = 'temp' | 'pressure' | 'speed' | 'ratio' | 'bleed'

export interface SensorChannel {
  id: string
  name: string
  short: string
  unit: string
  decimals: number
  baseline: number
  behavior: SensorBehavior
  family: SensorFamily
  primary?: boolean
  simulateCoef: number
  simulateUnit?: string
  step: number
}

export interface SensorPoint {
  cycle: number
  v: number
}

export interface Anomaly {
  id: string
  engineId: string
  sensorId: string
  cycle: number
  observed: number
  expectedLow: number
  expectedHigh: number
  deviation: number
  pattern: string
  effect: string
  severity: Extract<EngineStatus, 'warning' | 'critical'> | 'info'
  interpretation: 'model' | 'rule'
}

export interface TelemetryBundle {
  engineId: string
  cycles: number[]
  sensors: Record<string, number[]>
  health: number[]
  expected: Record<string, [number, number]>
  anomalies: Anomaly[]
}

export interface TelemetrySnapshot {
  cycle: number
  values: Record<string, number>
  health: number
  rul: number
  deviations: Record<string, number>
}
