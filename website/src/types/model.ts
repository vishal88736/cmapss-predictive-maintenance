import type { FeatureAttribution } from './prediction'

export type ModelStatus = 'active' | 'shadow' | 'archived'

export interface ModelInfo {
  id: string
  name: string
  version: string
  status: ModelStatus
  kind: string
  trainedOn: string
  params: string
  latency: number
  description: string
}

export interface ModelMetrics {
  rmse: number
  mae: number
  r2: number
  avgConfidence: number
  avp: { actual: number; predicted: number }[]
  residuals: { x: number; count: number }[]
  errorByRange: { range: string; mae: number; count: number }[]
  featureImportance: FeatureAttribution[]
  drift: { day: string; rmse: number; confidence: number; volume: number }[]
  confidenceDist: { x: number; count: number }[]
  trainingCurve: { epoch: number; loss: number; valLoss: number }[]
}
