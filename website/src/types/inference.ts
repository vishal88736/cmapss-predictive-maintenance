export type Subset = 'FD001' | 'FD002' | 'FD003' | 'FD004'
export const SUBSETS: Subset[] = ['FD001', 'FD002', 'FD003', 'FD004']
export type Page = 'Prediction' | 'Model' | 'Metrics' | 'About'
export type Step = 'uploaded' | 'validation' | 'preprocessing' | 'features' | 'inference' | 'prediction'
export type StepState = 'pending' | 'running' | 'completed' | 'failed'
export type Stages = Record<Step, StepState>
export interface BackendStatus {
  backendConnected: boolean
  models: { subset: Subset; ready: boolean; reason: string; missing: string[] }[]
  artifactDirectory: string
  artifactConfigured: boolean
  acceptedFormat: string
  maxUploadBytes: number
}
export interface UploadRecord {
  uploadId: string
  subset: Subset
  fileName: string
  format: string
  rowCount: number
  columnCount: number
  columns: string[]
  engineIds: number[]
  sensorColumns: string[]
  modelFeatures: string[]
  groundTruthFile?: string | null
  groundTruthAvailable: boolean
  stages: Stages
}
export interface EnginePrediction { engineId: number; cycle: number; predictedRUL: number; actualRUL?: number; absoluteError?: number }
export interface EvaluationRecord {
  id?: string
  subset: string
  modelName?: string
  metrics: Record<string, number>
  source: string
  sourceFile?: string
  cellId?: string
  executionCount?: number
  sourceSHA256?: string
  scope: string
  note?: string
  predictions: EnginePrediction[]
}
export interface PredictionResult {
  predictions: EnginePrediction[]
  evaluation?: EvaluationRecord | null
  subset: Subset
  modelName: string
  windowSelection: string
}
export interface InferenceJob {
  jobId: string
  status: 'running' | 'completed' | 'failed'
  stages: Stages
  fileName: string
  engineCount: number
  timestamp: string
  error?: string | null
  result?: PredictionResult | null
}
export interface Telemetry { engineId: number; readings: number[][]; sensorNames: string[]; conditionNames: string[] }
export interface ModelMetadata {
  name: string
  dataset: string
  inputFormat: string
  columns: string[]
  features: { name: string; description: string }[]
  output: string
  configuredWindowLength: number
  configuredRULCap: number
  preprocessing: string[]
  architecture: string[]
  sources: string[]
}
