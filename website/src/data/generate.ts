import { Rng, clamp, lerp } from '@/lib/prng'
import { SENSOR_CHANNELS } from './sensors'
import type { Engine, DegradationStage, EngineStatus } from '@/types/engine'
import type { TelemetryBundle, Anomaly, SensorChannel } from '@/types/telemetry'
import type { Prediction, FeatureAttribution } from '@/types/prediction'
import type { Alert, AlertSeverity } from '@/types/alert'
import type { ModelInfo, ModelMetrics } from '@/types/model'

const RISK_ANCHORS: [number, number][] = [
  [0, 100], [3, 97], [5, 94], [8, 92], [10, 90], [15, 86], [20, 80],
  [25, 77], [30, 73], [40, 64], [50, 57], [60, 50], [80, 36], [100, 26],
  [130, 16], [160, 10], [200, 6], [260, 3.5], [320, 2]
]

export function riskFromRul(rul: number, health: number): number {
  let base = 50
  for (let i = 1; i < RISK_ANCHORS.length; i++) {
    const [x0, y0] = RISK_ANCHORS[i - 1]
    const [x1, y1] = RISK_ANCHORS[i]
    if (rul <= x1) {
      base = lerp(y0, y1, (rul - x0) / Math.max(x1 - x0, 1e-6))
      break
    }
  }
  if (rul > 320) base = 2
  return Math.round(clamp(base + (60 - health) * 0.12, 1, 99))
}

export function classify(rul: number, health: number): EngineStatus {
  if (rul < 20 || health < 50) return 'critical'
  if (rul < 60 || health < 76) return 'warning'
  return 'nominal'
}

export function stageOf(health: number): DegradationStage {
  if (health >= 85) return 'healthy'
  if (health >= 68) return 'early'
  if (health >= 45) return 'degradation'
  return 'critical'
}

interface EngineSpec {
  unit: number
  rul: number
  health: number
  cycles: number
  confidence?: number
  risk?: number
  anomalies?: { sensor: string; cycle: number; amp: number; severity: Anomaly['severity'] }[]
}

const STORY: Record<number, EngineSpec> = {
  24: { unit: 24, rul: 37, health: 71, cycles: 261, confidence: 0.87, risk: 68, anomalies: [
    { sensor: 'T24', cycle: 259, amp: 0.068, severity: 'warning' },
    { sensor: 'W31', cycle: 203, amp: 0.028, severity: 'info' }
  ] },
  71: { unit: 71, rul: 12, health: 43, cycles: 199, confidence: 0.93, risk: 91, anomalies: [
    { sensor: 'T30', cycle: 188, amp: 0.094, severity: 'critical' },
    { sensor: 'phi', cycle: 161, amp: 0.061, severity: 'warning' }
  ] },
  38: { unit: 38, rul: 44, health: 66, cycles: 244, confidence: 0.84, anomalies: [
    { sensor: 'Ps30', cycle: 236, amp: -0.045, severity: 'warning' }
  ] },
  52: { unit: 52, rul: 88, health: 81, cycles: 173, confidence: 0.9, anomalies: [
    { sensor: 'P15', cycle: 168, amp: 0.032, severity: 'info' }
  ] },
  7: { unit: 7, rul: 212, health: 94, cycles: 128, confidence: 0.91 }
}

const DRIFT_DIR: Record<string, number> = {
  T24: 1, T30: 1, T50: 1, P15: 0, P30: -1, Nf: -1, Nc: 1, epr: 0,
  Ps30: -1, phi: 1, NRf: -1, NRc: 1, BPR: -1, htBleed: 0, W31: 1, W32: 1
}
const DRIFT_MAX: Record<string, number> = {
  T24: 0.016, T30: 0.042, T50: 0.05, P15: 0.004, P30: 0.055, Nf: 0.011,
  Nc: 0.026, epr: 0.003, Ps30: 0.085, phi: 0.07, NRf: 0.009, NRc: 0.032,
  BPR: 0.055, htBleed: 0.012, W31: 0.13, W32: 0.16
}
const NOISE_BASE: Record<string, number> = {
  T24: 0.0011, T30: 0.0013, T50: 0.0012, P15: 0.0022, P30: 0.0011, Nf: 0.0007,
  Nc: 0.0008, epr: 0.0009, Ps30: 0.0018, phi: 0.0015, NRf: 0.0006, NRc: 0.0009,
  BPR: 0.0014, htBleed: 0.0028, W31: 0.003, W32: 0.0035
}
const VAR_GROW: Record<string, number> = {
  T24: 1.4, T30: 2.4, T50: 2.2, P15: 4.5, P30: 2.2, Nf: 1.1, Nc: 1.6, epr: 0.3,
  Ps30: 2.6, phi: 2.4, NRf: 0.8, NRc: 1.4, BPR: 2.0, htBleed: 3.5, W31: 2.8, W32: 3.0
}

const PATTERN_TEXT: Record<string, [string, string]> = {
  temp: ['Sustained positive temperature excursion', 'Accelerated thermal wear in compressor section'],
  pressure: ['Pressure margin erosion below expected band', 'Reduced compressor efficiency, higher fuel demand'],
  speed: ['Speed offset from corrected baseline', 'Rotor efficiency drift'],
  ratio: ['Ratio drifting outside operating envelope', 'Flow imbalance feeding degradation signal'],
  bleed: ['Coolant bleed expansion', 'Turbine cooling capacity degradation']
}

function gaussBump(t: number, c0: number, w: number): number {
  const d = (t - c0) / w
  return Math.exp(-0.5 * d * d)
}

function buildEngine(spec: EngineSpec): { engine: Engine; telemetry: TelemetryBundle; prediction: Prediction } {
  const id = String(spec.unit).padStart(3, '0')
  const rng = new Rng(`engine-${id}-v3`)
  const N = spec.cycles + 1
  const trueRul = Math.max(4, Math.round(spec.rul * rng.range(0.9, 1.12)))
  const lifeTotal = spec.cycles + trueRul
  const pNow = spec.cycles / lifeTotal
  const D = (100 - spec.health) / Math.pow(pNow, 1.55)
  const driftScale = clamp(0.55 + (1 - spec.health / 100) * 1.7 + rng.range(-0.15, 0.2), 0.35, 2.4)

  const cycles = Array.from({ length: N }, (_, i) => i)
  const health: number[] = new Array(N)
  const sensors: Record<string, number[]> = {}
  const status = classify(spec.rul, spec.health)

  for (const ch of SENSOR_CHANNELS) {
    sensors[ch.id] = new Array(N)
  }

  const baseAnoms = [...(spec.anomalies ?? [])]
  if (!spec.anomalies) {
    const chance = status === 'critical' ? 0.85 : status === 'warning' ? 0.6 : 0.16
    if (rng.bool(chance)) {
      const pool = SENSOR_CHANNELS.filter((c) => Math.abs(DRIFT_DIR[c.id]) === 1)
      const count = status === 'nominal' ? 1 : rng.int(1, 2)
      for (let i = 0; i < count; i++) {
        const ch = rng.pick(pool)
        const ampMag = status === 'critical' ? rng.range(0.05, 0.11) : status === 'warning' ? rng.range(0.025, 0.06) : rng.range(0.012, 0.024)
        const amp = DRIFT_DIR[ch.id] >= 0 ? ampMag : -ampMag
        baseAnoms.push({
          sensor: ch.id,
          cycle: Math.round(spec.cycles * rng.range(0.68, 0.97)),
          amp,
          severity: status === 'critical' ? 'critical' : status === 'warning' ? 'warning' : 'info'
        })
      }
    }
  }

  for (let t = 0; t < N; t++) {
    const x = t / lifeTotal
    health[t] = clamp(100 - D * Math.pow(x, 1.55) + rng.gauss(0, 0.5), 8, 100.2)
    for (const ch of SENSOR_CHANNELS) {
      const dir = DRIFT_DIR[ch.id]
      const drift = dir * DRIFT_MAX[ch.id] * driftScale * Math.pow(x, 1.8)
      const noiseSd = ch.baseline * NOISE_BASE[ch.id] * (1 + VAR_GROW[ch.id] * Math.pow(x, 3))
      let v = ch.baseline * (1 + drift) + rng.gauss(0, noiseSd)
      for (const a of baseAnoms) {
        if (a.sensor === ch.id) v *= 1 + a.amp * gaussBump(t, a.cycle, 3.2)
      }
      if (ch.id === 'P15' || ch.id === 'htBleed') v *= 1 + rng.gauss(0, 0.004 * (1 + 2 * x))
      sensors[ch.id][t] = v
    }
  }

  const w31 = sensors['W31']
  const w32 = sensors['W32']
  const vi: number[] = new Array(N)
  for (let t = 0; t < N; t++) {
    const x = t / lifeTotal
    const lo = Math.max(0, t - 9)
    let mean = 0
    for (let i = lo; i <= t; i++) mean += w31[i] + w32[i]
    mean /= t - lo + 1
    let variance = 0
    for (let i = lo; i <= t; i++) {
      const d = w31[i] + w32[i] - mean
      variance += d * d
    }
    variance /= t - lo + 1
    const sd = Math.sqrt(variance)
    vi[t] = clamp(0.58 + 0.5 * Math.pow(x, 2) + (sd / 2.6) * 0.34 + rng.gauss(0, 0.02), 0.3, 2.2)
    for (const a of baseAnoms) {
      if (a.sensor === 'W31' || a.sensor === 'W32') vi[t] *= 1 + Math.abs(a.amp) * 2.4 * gaussBump(t, a.cycle, 3.2)
    }
  }
  sensors['VI'] = vi

  const expected: Record<string, [number, number]> = {}
  for (const ch of SENSOR_CHANNELS) {
    const dir = DRIFT_DIR[ch.id]
    const drift = dir * DRIFT_MAX[ch.id] * driftScale * Math.pow(pNow, 1.8)
    const mid = ch.baseline * (1 + drift)
    const band = Math.abs(mid) * (NOISE_BASE[ch.id] * 4.5 + 0.004)
    expected[ch.id] = [mid - band, mid + band]
  }
  expected['VI'] = [0.6, 0.92]

  const anomalies: Anomaly[] = baseAnoms
    .filter((a) => a.cycle > 4 && a.cycle <= spec.cycles)
    .map((a, i) => {
      const ch = SENSOR_CHANNELS.find((c) => c.id === a.sensor)!
      const arr = sensors[a.sensor]
      const observed = arr[a.cycle]
      const [elo, ehi] = expectedAt(ch, a.cycle, lifeTotal, driftScale)
      const mid = (elo + ehi) / 2
      const dev = ((observed - mid) / mid) * 100
      const [pattern, effect] = PATTERN_TEXT[ch.family]
      return {
        id: `${id}-a${i}`,
        engineId: id,
        sensorId: a.sensor,
        cycle: a.cycle,
        observed,
        expectedLow: elo,
        expectedHigh: ehi,
        deviation: Math.round(dev * 10) / 10,
        pattern: a.amp < 0 ? 'Negative excursion below expected band' : pattern,
        effect,
        severity: a.severity,
        interpretation: 'model' as const
      }
    })

  const sensorSnapshot: Record<string, number> = {}
  for (const ch of SENSOR_CHANNELS) sensorSnapshot[ch.id] = +sensors[ch.id][N - 1].toFixed(ch.decimals + 2)
  sensorSnapshot['VI'] = +vi[N - 1].toFixed(3)

  const risk = spec.risk ?? riskFromRul(spec.rul, spec.health)
  const confidence = spec.confidence ?? clamp(0.96 - spec.rul * 0.0009 + rng.range(-0.04, 0.02), 0.62, 0.97)

  const engine: Engine = {
    id,
    unit: spec.unit,
    currentCycle: spec.cycles,
    rul: spec.rul,
    health: Math.round(spec.health),
    risk,
    status,
    stage: stageOf(spec.health),
    faultMode: spec.unit % 3 === 0 ? 'HPC Degradation' : spec.unit % 7 === 0 ? 'Fan Degradation' : 'HPC Degradation',
    dataset: 'FD001',
    confidence,
    topSensors: attributions(id).slice(0, 3).map((a) => a.sensorId),
    sensorSnapshot,
    anomalyCount: anomalies.filter((a) => a.severity !== 'info').length,
    lastSync: Date.now()
  }

  const prediction = buildPrediction(engine, health, anomalies, rng)

  const bundle: TelemetryBundle = {
    engineId: id,
    cycles,
    sensors,
    health: health.map((h) => +h.toFixed(2)),
    expected,
    anomalies
  }

  return { engine, telemetry: bundle, prediction }
}

function expectedAt(ch: SensorChannel, cycle: number, lifeTotal: number, driftScale: number): [number, number] {
  const x = cycle / lifeTotal
  const dir = DRIFT_DIR[ch.id]
  const drift = dir * DRIFT_MAX[ch.id] * driftScale * Math.pow(x, 1.8)
  const mid = ch.baseline * (1 + drift)
  const band = Math.abs(mid) * (NOISE_BASE[ch.id] * 4.5 + 0.004)
  return [mid - band, mid + band]
}

const ATTR_POOL = ['phi', 'Ps30', 'T30', 'W31', 'BPR', 'T50', 'Nc', 'T24', 'P30', 'W32', 'NRc', 'Nf']

function attributions(id: string): FeatureAttribution[] {
  if (id === '024') {
    return [
      { sensorId: 'T30', weight: 0.94 }, { sensorId: 'Ps30', weight: 0.81 }, { sensorId: 'phi', weight: 0.74 },
      { sensorId: 'W31', weight: 0.52 }, { sensorId: 'BPR', weight: 0.4 }
    ]
  }
  const rng = new Rng(`attr-${id}`)
  const shuffled = rng.shuffle(ATTR_POOL).slice(0, 5)
  let w = rng.range(0.88, 0.97)
  return shuffled.map((sid) => {
    const a = { sensorId: sid, weight: +w.toFixed(3) }
    w *= rng.range(0.62, 0.82)
    return a
  })
}

function buildPrediction(engine: Engine, health: number[], anomalies: Anomaly[], rng: Rng): Prediction {
  const h0 = health[health.length - 1]
  const rul = engine.rul
  const steps = Math.min(Math.max(14, Math.round(rul / 2)), 64)
  const overshoot = Math.round(rul * 0.22) + 4
  const horizonCycles: number[] = []
  const horizonHealth: number[] = []
  const horizonLow: number[] = []
  const horizonHigh: number[] = []
  const bandScale = 1.3 - engine.confidence
  const total = rul + overshoot
  for (let i = 0; i <= steps; i++) {
    const c = engine.currentCycle + Math.round((total * i) / steps)
    const u = clamp((c - engine.currentCycle) / rul, 0, 1.4)
    const hv = u <= 1 ? h0 - (h0 - 13) * Math.pow(u, 2.35) : Math.max(6, 13 - 7 * (u - 1))
    const band = (2.4 + 15 * Math.min(u, 1.1)) * bandScale * (0.6 + rng.next() * 0.12)
    horizonCycles.push(c)
    horizonHealth.push(Math.max(4, hv))
    horizonLow.push(Math.max(2, hv - band))
    horizonHigh.push(Math.min(102, hv + band))
  }
  const insight =
    engine.status === 'critical'
      ? `Engine #${engine.unit} shows compound degradation across multiple channels with a steep failure trajectory. Model recommends shop-floor inspection before cycle ${engine.currentCycle + Math.max(4, Math.round(rul * 0.6))}.`
      : engine.status === 'warning'
        ? `Engine #${engine.unit} is showing a sustained degradation pattern across recent operating cycles. Most relevant signals: ${engine.topSensors.join(' · ')}.`
        : `Engine #${engine.unit} remains inside the nominal degradation envelope. No maintenance action required within the predicted horizon.`

  return {
    engineId: engine.id,
    modelId: 'lstm-rul-v2.4',
    rul,
    confidence: engine.confidence,
    failureCycle: engine.currentCycle + rul,
    horizonCycles,
    horizonHealth,
    horizonLow,
    horizonHigh,
    attributions: attributions(engine.id),
    generatedAt: Date.now() - rng.int(60, 600) * 1000,
    insight,
    mae: Math.round((9 + (1 - engine.confidence) * 30 + rng.range(0, 6)) * 10) / 10
  }
}

function buildEngineSpecs(): EngineSpec[] {
  const specs: EngineSpec[] = Object.values(STORY)
  const rng = new Rng('fleet-dist-v3')
  const used = new Set(Object.keys(STORY).map(Number))
  const push = (n: number, gen: (unit: number) => EngineSpec) => {
    let added = 0
    while (added < n) {
      const unit = rng.int(1, 100)
      if (used.has(unit)) continue
      used.add(unit)
      specs.push(gen(unit))
      added++
    }
  }
  push(9, (unit) => ({ unit, rul: rng.int(2, 18), health: rng.range(33, 57), cycles: rng.int(180, 298) }))
  push(16, (unit) => ({ unit, rul: rng.int(26, 58), health: rng.range(57, 77), cycles: rng.int(150, 290) }))
  push(70, (unit) => ({ unit, rul: rng.int(70, 320), health: rng.range(78, 97), cycles: rng.int(100, 260) }))
  return specs
}

const ALL_SPECS = buildEngineSpecs()

export const ENGINES: Engine[] = []
export const TELEMETRY: Record<string, TelemetryBundle> = {}
export const PREDICTIONS: Record<string, Prediction> = {}

for (const spec of ALL_SPECS) {
  const built = buildEngine({ ...spec, unit: spec.unit })
  ENGINES.push(built.engine)
  TELEMETRY[built.engine.id] = built.telemetry
  PREDICTIONS[built.engine.id] = built.prediction
}
ENGINES.sort((a, b) => a.unit - b.unit)

const storyAlerts: { engineId: string; severity: AlertSeverity; title: string; detail: string; minsAgo: number; sensorId?: string; cycle?: number }[] = [
  { engineId: '071', severity: 'critical', title: 'RUL below threshold', detail: 'Predicted remaining useful life dropped below the 15-cycle alert threshold. Schedule immediate inspection.', minsAgo: 8, cycle: 199 },
  { engineId: '024', severity: 'warning', title: 'Temperature deviation', detail: 'LPC outlet temperature excursion detected at cycle 259, outside the expected operating band.', minsAgo: 13, sensorId: 'T24', cycle: 259 },
  { engineId: '052', severity: 'info', title: 'Telemetry pattern changed', detail: 'Bypass duct pressure variance shifted pattern. Within tolerance — monitoring continued.', minsAgo: 26, sensorId: 'P15', cycle: 168 },
  { engineId: '038', severity: 'warning', title: 'Increasing degradation', detail: 'HPC static pressure margin eroding faster than cohort baseline over the last 20 cycles.', minsAgo: 39, sensorId: 'Ps30', cycle: 236 }
]

function generateAlerts(): Alert[] {
  const now = Date.now()
  const alerts: Alert[] = storyAlerts.map((a, i) => ({
    id: `al-story-${i}`,
    ts: now - a.minsAgo * 60000,
    severity: a.severity,
    engineId: a.engineId,
    title: a.title,
    detail: a.detail,
    cycle: a.cycle,
    sensorId: a.sensorId
  }))
  const rng = new Rng('alerts-v3')
  const templates: Record<AlertSeverity, { title: string[]; sensor?: string }[]> = {
    critical: [
      { title: ['RUL below threshold', 'Health index critical'], sensor: undefined },
      { title: ['Sensor excursion beyond critical band', 'Variance explosion detected'], sensor: 'T30' }
    ],
    warning: [
      { title: ['Increasing degradation', 'Health decline accelerating', 'Temperature deviation'], sensor: 'T50' },
      { title: ['Pressure margin erosion', 'Fuel ratio drifting high'], sensor: 'Ps30' }
    ],
    info: [
      { title: ['Telemetry pattern changed', 'Cohort baseline updated'], sensor: 'P15' },
      { title: ['Scheduled model refresh applied', 'Sensor recalibration recorded'], sensor: undefined }
    ]
  }
  let i = 0
  for (const e of ENGINES) {
    if (['024', '071', '052', '038'].includes(e.id)) continue
    const sev: AlertSeverity = e.status === 'critical' ? (rng.bool(0.6) ? 'critical' : 'warning') : e.status === 'warning' ? (rng.bool(0.7) ? 'warning' : 'info') : 'info'
    if (sev === 'info' && !rng.bool(0.22)) continue
    if (e.status === 'nominal' && !rng.bool(0.18)) continue
    const t = rng.pick(templates[sev])
    const sensor = t.sensor ?? rng.pick(SENSOR_CHANNELS).id
    alerts.push({
      id: `al-${i++}`,
      ts: now - rng.int(45, 330) * 60000,
      severity: sev,
      engineId: e.id,
      title: rng.pick(t.title),
      detail: `${sensor} on Engine #${e.unit} deviated from the learned operating envelope around cycle ${Math.max(4, e.currentCycle - rng.int(2, 30))}.`,
      cycle: Math.max(4, e.currentCycle - rng.int(2, 30)),
      sensorId: sensor
    })
  }
  alerts.sort((a, b) => b.ts - a.ts)
  return alerts
}

export const ALERTS: Alert[] = generateAlerts()

export function makeLiveAlert(seed: number): Alert {
  const rng = new Rng(`live-${seed}-${Date.now() % 100000}`)
  const pool = ENGINES.filter((e) => e.status !== 'nominal')
  const e = rng.bool(0.75) && pool.length ? rng.pick(pool) : rng.pick(ENGINES)
  const sev: AlertSeverity = e.status === 'critical' ? 'critical' : e.status === 'warning' ? 'warning' : 'info'
  const sensor = rng.pick(SENSOR_CHANNELS)
  const titles: Record<AlertSeverity, string[]> = {
    critical: ['RUL re-estimated downward', 'Failure trajectory steepened'],
    warning: ['Degradation signal rising', 'Sensor near alert band'],
    info: ['Telemetry pattern changed', 'Variance expansion noted']
  }
  return {
    id: `al-live-${Date.now()}`,
    ts: Date.now(),
    severity: sev,
    engineId: e.id,
    title: rng.pick(titles[sev]),
    detail: `${sensor.name} on Engine #${e.unit} crossed a monitored threshold at cycle ${e.currentCycle}. ${sev === 'critical' ? 'Immediate review recommended.' : 'Model continues tracking.'}`,
    cycle: e.currentCycle,
    sensorId: sensor.id,
    live: true
  }
}

export const MODELS: ModelInfo[] = [
  {
    id: 'lstm-rul-v2.4', name: 'LSTM RUL Predictor', version: 'v2.4', status: 'active',
    kind: 'Bidirectional LSTM · 2×128 units', trainedOn: 'FD001 train split · 100 units',
    params: '412K', latency: 38, description: 'Sequence model over a 30-cycle window of 14 sensor channels, trained with piecewise-linear RUL targets (clip 130).'
  },
  {
    id: 'tcae-td3-v1.9', name: 'TCAE + TD3 Health Agent', version: 'v1.9', status: 'shadow',
    kind: 'Temporal Conv. Autoencoder + TD3 policy', trainedOn: 'FD001 + FD003 · 100 units',
    params: '1.1M', latency: 62, description: 'Representation model with a reinforcement-learning maintenance policy, gated by an LQR safety layer. Shadow-evaluating against the LSTM.'
  },
  {
    id: 'rf-baseline-v1.2', name: 'Random Forest Baseline', version: 'v1.2', status: 'archived',
    kind: '500-tree regression forest', trainedOn: 'FD001 · hand-crafted features',
    params: '—', latency: 12, description: 'Interpretable baseline using last-value features. Retained for regression comparison and onboarding.'
  }
]

function buildModelMetrics(model: ModelInfo): ModelMetrics {
  const rng = new Rng(`metrics-${model.id}`)
  const rmse = model.id.startsWith('lstm') ? 19.8 : model.id.startsWith('tcae') ? 17.2 : 24.6
  const mae = model.id.startsWith('lstm') ? 13.6 : model.id.startsWith('tcae') ? 11.9 : 17.4
  const r2 = model.id.startsWith('lstm') ? 0.91 : model.id.startsWith('tcae') ? 0.93 : 0.86
  const avp = Array.from({ length: 240 }, () => {
    const actual = clamp(Math.abs(rng.gauss(95, 70)), 2, 260)
    const predicted = Math.max(1, actual + rng.gauss(0, rmse * 0.85))
    return { actual: Math.round(actual), predicted: Math.round(predicted) }
  })
  const residuals = Array.from({ length: 25 }, (_, i) => {
    const x = -60 + i * 5
    const c = 240 * Math.exp(-0.5 * Math.pow((x - rng.gauss(0, 3)) / (rmse * 1.1), 2)) * (1 + rng.range(-0.12, 0.12))
    return { x, count: Math.max(1, Math.round(c)) }
  })
  const ranges = ['0–25', '26–50', '51–100', '101–150', '151–200', '200+']
  const baseMae = [7.8, 11.2, 15.4, 18.9, 21.6, 24.2]
  const errorByRange = ranges.map((range, i) => ({
    range,
    mae: +(baseMae[i] * (mae / 13.6) * rng.range(0.92, 1.08)).toFixed(1),
    count: rng.int(40, 320)
  }))
  const featureImportance = [
    { sensorId: 'phi', weight: 0.92 }, { sensorId: 'Ps30', weight: 0.84 }, { sensorId: 'T30', weight: 0.78 },
    { sensorId: 'W31', weight: 0.66 }, { sensorId: 'BPR', weight: 0.61 }, { sensorId: 'T50', weight: 0.52 },
    { sensorId: 'Nc', weight: 0.48 }, { sensorId: 'T24', weight: 0.42 }, { sensorId: 'P30', weight: 0.38 },
    { sensorId: 'W32', weight: 0.33 }, { sensorId: 'NRc', weight: 0.27 }, { sensorId: 'P15', weight: 0.18 },
    { sensorId: 'htBleed', weight: 0.15 }, { sensorId: 'Nf', weight: 0.12 }, { sensorId: 'epr', weight: 0.08 }, { sensorId: 'NRf', weight: 0.06 }
  ]
  const drift = Array.from({ length: 30 }, (_, i) => {
    const day = new Date(now0() - (29 - i) * 86400000)
    return {
      day: `${day.getMonth() + 1}/${day.getDate()}`,
      rmse: +(rmse * rng.range(0.88, 1.06)).toFixed(1),
      confidence: +(0.87 + rng.gauss(0, 0.02) - i * 0.0006).toFixed(3),
      volume: rng.int(820, 1400)
    }
  })
  const confidenceDist = Array.from({ length: 22 }, (_, i) => {
    const x = 0.5 + i * 0.025
    return { x: +x.toFixed(3), count: Math.max(1, Math.round(150 * Math.exp(-0.5 * Math.pow((x - 0.87) / 0.075, 2)) + rng.range(-8, 8))) }
  })
  const trainingCurve = Array.from({ length: 60 }, (_, i) => {
    const e = i + 1
    const loss = 280 * Math.exp(-e / 11) + 19 + rng.gauss(0, 1.5)
    const valLoss = 300 * Math.exp(-e / 10) + 21.5 + rng.gauss(0, 2)
    return { epoch: e, loss: +Math.max(14, loss).toFixed(1), valLoss: +Math.max(15, valLoss).toFixed(1) }
  })
  return { rmse, mae, r2, avgConfidence: 0.87, avp, residuals, errorByRange, featureImportance, drift, confidenceDist, trainingCurve }
}

function now0(): number {
  return Date.now()
}

export const MODEL_METRICS: Record<string, ModelMetrics> = Object.fromEntries(MODELS.map((m) => [m.id, buildModelMetrics(m)]))

export function fleetHealthAt(cyclesAgo: number): number {
  let sum = 0
  let n = 0
  for (const e of ENGINES) {
    const b = TELEMETRY[e.id]
    const idx = Math.max(0, b.health.length - 1 - cyclesAgo)
    sum += b.health[idx]
    n++
  }
  return sum / Math.max(n, 1)
}

export function cohortCurves(): { status: EngineStatus; label: string; points: { x: number; y: number }[] }[] {
  const groups: EngineStatus[] = ['critical', 'warning', 'nominal']
  return groups.map((status) => {
    const engines = ENGINES.filter((e) => e.status === status)
    const points = Array.from({ length: 11 }, (_, i) => {
      const x = i / 10
      let sum = 0
      for (const e of engines) {
        const b = TELEMETRY[e.id]
        const idx = clamp(Math.round(x * (b.health.length - 1)), 0, b.health.length - 1)
        sum += b.health[idx]
      }
      return { x: Math.round(x * 100), y: engines.length ? +(sum / engines.length).toFixed(1) : 0 }
    })
    return { status, label: status[0].toUpperCase() + status.slice(1), points }
  })
}

export function rulDistribution(): { bucket: string; count: number }[] {
  const buckets = [
    { bucket: '0–25', lo: 0, hi: 25 }, { bucket: '26–50', lo: 26, hi: 50 },
    { bucket: '51–100', lo: 51, hi: 100 }, { bucket: '101–150', lo: 101, hi: 150 },
    { bucket: '151–200', lo: 151, hi: 200 }, { bucket: '201+', lo: 201, hi: 1e9 }
  ]
  return buckets.map((b) => ({ bucket: b.bucket, count: ENGINES.filter((e) => e.rul >= b.lo && e.rul <= b.hi).length }))
}

export function sensorAt(engineId: string, sensorId: string, cycle: number): number | null {
  const b = TELEMETRY[engineId]
  if (!b) return null
  const arr = b.sensors[sensorId]
  if (!arr) return null
  const idx = clamp(Math.round(cycle), 0, arr.length - 1)
  return arr[idx]
}
