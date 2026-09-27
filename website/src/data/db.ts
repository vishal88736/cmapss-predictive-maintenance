import { SENSOR_CHANNELS } from './sensors'
import { ENGINES, TELEMETRY, PREDICTIONS, ALERTS, MODELS, MODEL_METRICS, fleetHealthAt, cohortCurves, rulDistribution, sensorAt } from './generate'
import type { Engine, FleetStats } from '@/types/engine'
import type { TelemetryBundle, TelemetrySnapshot, Anomaly } from '@/types/telemetry'
import type { Prediction, SimulationResult, SimulationOverrides } from '@/types/prediction'
import type { Alert } from '@/types/alert'
import type { ModelInfo, ModelMetrics } from '@/types/model'
import { SENSOR_MAP, SIMULATABLE_SENSORS } from './sensors'
import { clamp } from '@/lib/prng'

export function getEngines(): Engine[] {
  return ENGINES
}

export function getEngine(id: string): Engine | undefined {
  return ENGINES.find((e) => e.id === id || e.unit === Number(id))
}

export function getTelemetry(id: string): TelemetryBundle | undefined {
  const e = getEngine(id)
  return e ? TELEMETRY[e.id] : undefined
}

export function getPrediction(id: string): Prediction | undefined {
  const e = getEngine(id)
  return e ? PREDICTIONS[e.id] : undefined
}

export function getAlerts(): Alert[] {
  return ALERTS
}

export function getModels(): ModelInfo[] {
  return MODELS
}

export function getModelMetrics(id: string): ModelMetrics | undefined {
  return MODEL_METRICS[id]
}

export function getFleetStats(): FleetStats {
  const n = ENGINES.length
  const health = ENGINES.reduce((s, e) => s + e.health, 0) / n
  const avgRul = ENGINES.reduce((s, e) => s + e.rul, 0) / n
  const totalCycles = ENGINES.reduce((s, e) => s + e.currentCycle, 0)
  const prev = fleetHealthAt(40)
  return {
    health: Math.round(health * 10) / 10,
    healthDelta: Math.round((health - prev) * 10) / 10,
    nominalCount: ENGINES.filter((e) => e.status === 'nominal').length,
    warningCount: ENGINES.filter((e) => e.status === 'warning').length,
    criticalCount: ENGINES.filter((e) => e.status === 'critical').length,
    attentionCount: ENGINES.filter((e) => e.status === 'critical' || e.risk >= 78).length,
    avgRul: Math.round(avgRul),
    totalCycles,
    monitored: n
  }
}

export { cohortCurves, rulDistribution, sensorAt }

export function getSnapshot(engineId: string, cycle: number): TelemetrySnapshot | null {
  const e = getEngine(engineId)
  const b = getTelemetry(engineId)
  if (!e || !b) return null
  const c = clamp(Math.round(cycle), 0, e.currentCycle)
  const values: Record<string, number> = {}
  const deviations: Record<string, number> = {}
  for (const ch of SENSOR_CHANNELS) {
    const arr = b.sensors[ch.id]
    values[ch.id] = arr[c]
    const [lo, hi] = b.expected[ch.id]
    const mid = (lo + hi) / 2
    deviations[ch.id] = ((arr[c] - mid) / mid) * 100
  }
  const health = b.health[c]
  const rul = c <= e.currentCycle ? e.rul + (e.currentCycle - c) : Math.max(0, e.rul - (c - e.currentCycle))
  return { cycle: c, values, health: Math.round(health * 10) / 10, rul, deviations }
}

export function anomaliesIn(engineId: string, cycle: number, window: number): Anomaly[] {
  const b = getTelemetry(engineId)
  if (!b) return []
  return b.anomalies.filter((a) => Math.abs(a.cycle - cycle) <= window)
}

export function simulate(engineId: string, overrides: SimulationOverrides): SimulationResult | null {
  const e = getEngine(engineId)
  const b = getTelemetry(engineId)
  if (!e || !b) return null
  const contributions: { sensorId: string; deltaRul: number }[] = []
  let deltaRul = 0
  for (const ch of SIMULATABLE_SENSORS) {
    const target = overrides[ch.id]
    if (target == null) continue
    const current = e.sensorSnapshot[ch.id]
    const pctChange = ((target - current) / current) * 100
    const delta = ch.simulateCoef * pctChange
    if (Math.abs(delta) > 0.01) contributions.push({ sensorId: ch.id, deltaRul: Math.round(delta) })
    deltaRul += delta
  }
  contributions.sort((a, b2) => Math.abs(b2.deltaRul) - Math.abs(a.deltaRul))
  const scenarioRul = Math.round(clamp(e.rul + deltaRul, 1, 500))
  const scenarioHealth = Math.round(clamp(e.health - deltaRul * 0.32, 4, 99))
  const scenarioRisk = Math.round(clamp(e.risk - deltaRul * 0.9, 1, 99))
  const p = getPrediction(engineId)!
  const steps = Math.min(p.horizonCycles.length - 1, 48)
  const trajectoryCycles: number[] = []
  const baselineTrajectory: number[] = []
  const scenarioTrajectory: number[] = []
  for (let i = 0; i <= steps; i++) {
    const u = i / steps
    const c = Math.round(e.currentCycle + u * e.rul * 1.25)
    const base = e.health - (e.health - 13) * Math.pow(u, 2.3)
    const scale = scenarioRul / e.rul
    const su = clamp(u / Math.max(scale, 0.05), 0, 1.6)
    const scen = scenarioHealth - (scenarioHealth - 13) * Math.pow(Math.min(su, 1), 2.3)
    trajectoryCycles.push(c)
    baselineTrajectory.push(Math.max(4, base))
    scenarioTrajectory.push(Math.max(4, u <= 1 ? scen : Math.max(4, 13 - 9 * (su - 1))))
  }
  return {
    engineId: e.id,
    baselineRul: e.rul,
    scenarioRul,
    baselineRisk: e.risk,
    scenarioRisk,
    baselineHealth: e.health,
    scenarioHealth,
    trajectoryCycles,
    baselineTrajectory,
    scenarioTrajectory,
    contributions: contributions.slice(0, 6),
    estimated: true
  }
}

export const VI_CHANNEL_ID = 'VI'

export function sensorChannel(id: string) {
  if (id === VI_CHANNEL_ID) return VI_META
  return SENSOR_MAP[id]
}

const VI_META = {
  id: 'VI',
  name: 'Vibration Index',
  short: 'Vibration',
  unit: '',
  decimals: 2,
  baseline: 0.72,
  behavior: 'anomalous' as const,
  family: 'bleed' as const,
  step: 0.01,
  simulateCoef: -1.1,
  derived: true
}
