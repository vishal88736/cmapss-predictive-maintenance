export type Subset = 'FD001' | 'FD002' | 'FD003' | 'FD004'
export const subsets: Subset[] = ['FD001', 'FD002', 'FD003', 'FD004']

// Source columns: unit, cycle, three operational settings, then sensor measurements.
// Readings are kept as source-number arrays to keep the shipped dataset small.
export interface Engine { engineId: number; readings: number[][] }
export interface Dataset {
  subset: Subset
  split: 'train'
  source: string
  sensorNames: string[]
  conditionNames: string[]
  engines: Engine[]
}
export interface Prediction {
  engineId: number
  cycle: number
  predictedRUL: number
  actualRUL?: number
  error?: number
}
export interface Experiment {
  modelName: string
  dataset: string
  configuration?: Record<string, string | number>
  metrics?: Record<string, number>
}
