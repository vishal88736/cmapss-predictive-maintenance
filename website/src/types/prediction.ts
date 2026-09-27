export interface FeatureAttribution {
  sensorId: string
  weight: number
}

export interface Prediction {
  engineId: string
  modelId: string
  rul: number
  confidence: number
  failureCycle: number
  horizonCycles: number[]
  horizonHealth: number[]
  horizonLow: number[]
  horizonHigh: number[]
  attributions: FeatureAttribution[]
  generatedAt: number
  insight: string
  mae: number
}

export interface SimulationOverrides {
  [sensorId: string]: number
}

export interface SimulationResult {
  engineId: string
  baselineRul: number
  scenarioRul: number
  baselineRisk: number
  scenarioRisk: number
  baselineHealth: number
  scenarioHealth: number
  trajectoryCycles: number[]
  baselineTrajectory: number[]
  scenarioTrajectory: number[]
  contributions: { sensorId: string; deltaRul: number }[]
  estimated: boolean
}
