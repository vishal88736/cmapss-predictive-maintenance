export type EngineStatus = 'NOMINAL' | 'EARLY' | 'WARNING' | 'CRITICAL';

export interface Engine {
  id: number; // 1..100
  label: string; // "ENGINE #024"
  totalCycles: number; // lifetime length 100-300
  currentCycle: number; // where we are now (== totalCycles for observed, failure beyond)
  rul: number; // remaining useful life at currentCycle
  health: number; // 0-100
  risk: number; // 0-100
  status: EngineStatus;
  confidence: number; // model confidence 0-100
  operatingSetting: { altitude: number; mach: number; throttle: number };
  lastSync: string;
  degradationOnset: number; // cycle where degradation begins
}

export interface SensorDef {
  key: string;
  name: string;
  unit: string;
  short: string;
  base: number;
  kind: 'stable' | 'drift' | 'variance' | 'degrade' | 'anomaly';
  expected: [number, number];
}

export interface TelemetryPoint {
  cycle: number;
  values: Record<string, number>;
  health: number;
  rulTrue: number;
  rulPred: number;
  anomaly: boolean;
  anomalySensors?: string[];
}

export interface EngineSeries {
  engineId: number;
  points: TelemetryPoint[];
  predicted: { cycle: number; rulPred: number; health: number; lo: number; hi: number }[];
  failureCycle: number;
}

export interface AlertItem {
  id: string;
  time: string;
  minutesAgo: number;
  severity: 'CRITICAL' | 'WARNING' | 'INFO';
  engineId: number;
  title: string;
  detail: string;
  cycle: number;
  acked: boolean;
}

export interface ModelInfo {
  id: string;
  name: string;
  version: string;
  status: 'ACTIVE' | 'STAGED' | 'ARCHIVED';
  rmse: number;
  mae: number;
  r2: number;
  latencyMs: number;
}

export interface FeatureAttribution {
  sensor: string;
  name: string;
  weight: number; // 0..1
}
