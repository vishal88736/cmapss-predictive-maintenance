import {
  getEngines, getEngine, getTelemetry, getPrediction, getAlerts, getModels,
  getModelMetrics, getFleetStats, cohortCurves, rulDistribution, getSnapshot,
  simulate as simulateSync, anomaliesIn, sensorAt
} from '@/data/db'
import type { Engine, FleetStats } from '@/types/engine'
import type { TelemetryBundle, TelemetrySnapshot, Anomaly } from '@/types/telemetry'
import type { Prediction, SimulationResult, SimulationOverrides } from '@/types/prediction'
import type { Alert } from '@/types/alert'
import type { ModelInfo, ModelMetrics } from '@/types/model'

const later = <T>(value: () => T, ms = 120): Promise<T> =>
  new Promise((resolve) => setTimeout(() => resolve(value()), ms * (0.6 + Math.random() * 0.8)))

export interface Api {
  getFleet(): Promise<Engine[]>
  getFleetStats(): Promise<FleetStats>
  getEngine(id: string): Promise<Engine | undefined>
  getTelemetry(id: string): Promise<TelemetryBundle | undefined>
  getPrediction(id: string): Promise<Prediction | undefined>
  getAlerts(): Promise<Alert[]>
  getModels(): Promise<ModelInfo[]>
  getModelMetrics(id: string): Promise<ModelMetrics | undefined>
  getCohortCurves(): Promise<{ status: string; label: string; points: { x: number; y: number }[] }[]>
  getRulDistribution(): Promise<{ bucket: string; count: number }[]>
  getSnapshot(engineId: string, cycle: number): TelemetrySnapshot | null
  getAnomalies(engineId: string, cycle: number, window: number): Anomaly[]
  getSensorValue(engineId: string, sensorId: string, cycle: number): number | null
  simulate(engineId: string, overrides: SimulationOverrides): Promise<SimulationResult | null>
}

export const api: Api = {
  getFleet: () => later(getEngines, 180),
  getFleetStats: () => later(getFleetStats, 100),
  getEngine: (id) => later(() => getEngine(id), 90),
  getTelemetry: (id) => later(() => getTelemetry(id), 220),
  getPrediction: (id) => later(() => getPrediction(id), 160),
  getAlerts: () => later(getAlerts, 140),
  getModels: () => later(getModels, 100),
  getModelMetrics: (id) => later(() => getModelMetrics(id), 200),
  getCohortCurves: () => later(cohortCurves, 150),
  getRulDistribution: () => later(rulDistribution, 130),
  getSnapshot,
  getAnomalies: anomaliesIn,
  getSensorValue: sensorAt,
  simulate: (engineId, overrides) => later(() => simulateSync(engineId, overrides), 600)
}
