import type { AlertItem, Engine, EngineSeries, EngineStatus, ModelInfo, SensorDef, TelemetryPoint } from '../types';

export const SENSORS: SensorDef[] = [
  { key: 'T24', name: 'Temperature', unit: '°K', short: 'TEMP', base: 641.2, kind: 'degrade', expected: [628, 635] },
  { key: 'P30', name: 'Pressure', unit: 'psia', short: 'PRESS', base: 90.4, kind: 'degrade', expected: [88.1, 92.0] },
  { key: 'Nf', name: 'Fan Speed', unit: 'rpm', short: 'FAN', base: 2388, kind: 'drift', expected: [2375, 2398] },
  { key: 'Nc', name: 'Core Speed', unit: 'rpm', short: 'CORE', base: 9045, kind: 'degrade', expected: [9000, 9090] },
  { key: 'VIB', name: 'Vibration', unit: 'in/s', short: 'VIB', base: 0.82, kind: 'variance', expected: [0.6, 0.9] },
  { key: 'W31', name: 'HPT Flow', unit: 'lbm/s', short: 'FLOW', base: 38.4, kind: 'drift', expected: [37.5, 39.2] },
  { key: 'S7', name: 'Sensor 07 · Bleed Enthalpy', unit: '—', short: 'S07', base: 553.8, kind: 'stable', expected: [549, 558] },
  { key: 'S9', name: 'Sensor 09 · LPT Temp', unit: '°R', short: 'S09', base: 906.1, kind: 'degrade', expected: [895, 915] },
  { key: 'S11', name: 'Sensor 11 · Static Press', unit: 'psia', short: 'S11', base: 47.2, kind: 'variance', expected: [46.4, 47.9] },
  { key: 'S12', name: 'Sensor 12 · Fuel Flow', unit: 'pps', short: 'S12', base: 520.1, kind: 'drift', expected: [515, 525] },
  { key: 'S13', name: 'Sensor 13 · Corrected Fan', unit: 'rpm', short: 'S13', base: 2387.5, kind: 'stable', expected: [2380, 2395] },
  { key: 'S14', name: 'Sensor 14 · Bypass Ratio', unit: '—', short: 'S14', base: 8.41, kind: 'stable', expected: [8.3, 8.5] },
  { key: 'S4', name: 'Sensor 04 · LPT Outlet', unit: '°R', short: 'S04', base: 1400.2, kind: 'degrade', expected: [1390, 1412] },
  { key: 'S2', name: 'Sensor 02 · LPC Outlet', unit: '°R', short: 'S02', base: 644.3, kind: 'anomaly', expected: [638, 650] },
];

// Deterministic PRNG (mulberry32)
function rng(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pad = (n: number, w = 3) => String(n).padStart(w, '0');

function smoothNoise(rand: () => number, n: number, scale: number) {
  // random walk smoothed
  const out: number[] = [];
  let v = 0;
  for (let i = 0; i < n; i++) {
    v = v * 0.92 + (rand() - 0.5) * scale;
    out.push(v);
  }
  return out;
}

export interface FleetData {
  engines: Engine[];
  series: Map<number, EngineSeries>;
  alerts: AlertItem[];
  models: ModelInfo[];
}

let cache: FleetData | null = null;

export function getFleetData(): FleetData {
  if (cache) return cache;
  const engines: Engine[] = [];
  const series = new Map<number, EngineSeries>();
  const rand = rng(20260214);

  for (let id = 1; id <= 100; id++) {
    const r = rng(id * 7919 + 13);
    const totalCycles = 130 + Math.floor(r() * 170); // 130..300
    const severity = r(); // determines health archetype
    // Force a spread: ~72 nominal, ~12 early, ~9 warning, ~7 critical
    let status: EngineStatus;
    let health: number;
    let rul: number;
    if (id === 24) { status = 'WARNING'; health = 71; rul = 37; }
    else if (id === 71) { status = 'CRITICAL'; health = 43; rul = 12; }
    else if (id === 38) { status = 'WARNING'; health = 66; rul = 29; }
    else if (id === 52) { status = 'EARLY'; health = 78; rul = 58; }
    else if (severity < 0.70) { status = 'NOMINAL'; health = 82 + r() * 15; rul = 90 + r() * 110; }
    else if (severity < 0.82) { status = 'EARLY'; health = 74 + r() * 8; rul = 55 + r() * 40; }
    else if (severity < 0.93) { status = 'WARNING'; health = 58 + r() * 14; rul = 25 + r() * 30; }
    else { status = 'CRITICAL'; health = 32 + r() * 18; rul = 5 + r() * 18; }
    health = Math.round(Math.min(99, Math.max(28, health)));
    rul = Math.round(Math.min(totalCycles - 20, Math.max(4, rul)));
    const currentCycle = totalCycles - Math.round(r() * 8); // near end of observed
    const failureCycle = currentCycle + rul;
    const degradationOnset = Math.max(20, Math.round(currentCycle * (status === 'NOMINAL' ? 0.72 : status === 'EARLY' ? 0.58 : status === 'WARNING' ? 0.45 : 0.38) + (r() - 0.5) * 20));
    const risk = Math.round(Math.min(98, Math.max(4, 100 - health + (r() - 0.5) * 10 + (status === 'CRITICAL' ? 12 : status === 'WARNING' ? 6 : 0))));

    engines.push({
      id,
      label: `ENGINE #${pad(id)}`,
      totalCycles,
      currentCycle: id === 24 ? 261 : id === 71 ? 284 : currentCycle,
      rul: id === 24 ? 37 : id === 71 ? 12 : rul,
      health,
      risk,
      status,
      confidence: Math.round(78 + r() * 18),
      operatingSetting: {
        altitude: Math.round(34000 + r() * 4000),
        mach: +(0.72 + r() * 0.1).toFixed(2),
        throttle: Math.round(88 + r() * 10),
      },
      lastSync: `14:${String(20 + Math.floor(r() * 39)).padStart(2, '0')}:${String(Math.floor(r() * 60)).padStart(2, '0')}`,
      degradationOnset,
    });
  }

  // sort: keep id order but compute series per engine
  for (const e of engines) {
    const r = rng(e.id * 331 + 7);
    const n = e.currentCycle;
    const onset = e.degradationOnset;
    const noises: Record<string, number[]> = {};
    for (const s of SENSORS) {
      noises[s.key] = smoothNoise(r, n, s.kind === 'variance' ? 2.2 : 0.9);
    }
    const points: TelemetryPoint[] = [];
    const anomalyCycles = new Set<number>();
    // inject 2-4 anomaly bursts for degraded engines
    if (e.status !== 'NOMINAL') {
      const bursts = e.status === 'CRITICAL' ? 4 : e.status === 'WARNING' ? 3 : 2;
      for (let b = 0; b < bursts; b++) {
        const c = Math.round(n * (0.78 + r() * 0.2));
        anomalyCycles.add(Math.min(n - 1, Math.max(10, c + Math.floor((r() - 0.5) * 12))));
      }
    }
    for (let c = 1; c <= n; c++) {
      const prog = Math.max(0, (c - onset) / Math.max(1, n - onset)); // 0..1 degraded progress
      const expo = Math.pow(prog, 1.6);
      const values: Record<string, number> = {};
      for (const s of SENSORS) {
        const nz = noises[s.key][c - 1] ?? 0;
        let v = s.base;
        const driftAmp = s.base * 0.004;
        if (s.kind === 'stable') v += nz * driftAmp * 0.5;
        else if (s.kind === 'drift') v += prog * s.base * 0.012 + nz * driftAmp;
        else if (s.kind === 'variance') v += prog * s.base * 0.008 + nz * driftAmp * (1 + prog * 3);
        else if (s.kind === 'degrade') v += expo * s.base * (s.key === 'T24' ? 0.035 : s.key === 'P30' ? 0.028 : 0.02) + nz * driftAmp * (1 + prog * 1.6);
        else if (s.kind === 'anomaly') v += expo * s.base * 0.014 + nz * driftAmp * (1 + prog * 2.4);
        // anomaly spike
        if (anomalyCycles.has(c) && (s.key === 'T24' || s.key === 'P30' || s.key === 'S11' || s.key === 'VIB')) {
          v += s.base * (0.008 + r() * 0.012);
        }
        values[s.key] = +v.toFixed(s.key === 'VIB' || s.key === 'S14' ? 3 : 1);
      }
      const healthCurve = e.health + (100 - e.health) * Math.pow(1 - prog, 1.2) * 0.15 - (100 - e.health) * 0 + (1 - expo) * (100 - e.health) * 0;
      // simpler: health starts ~97 and decays to e.health with slight noise
      const h = Math.max(e.health - 2, Math.min(99, 97 - (97 - e.health) * expo + noises['T24'][c - 1] * -0.4));
      const rulTrue = (n - c) + e.rul;
      const rulPred = Math.max(0, Math.round(rulTrue + (r() - 0.5) * (6 + expo * 10)));
      const isAnom = anomalyCycles.has(c);
      points.push({
        cycle: c,
        values,
        health: +h.toFixed(1),
        rulTrue,
        rulPred,
        anomaly: isAnom,
        anomalySensors: isAnom ? ['T24', 'P30', 'S11'].slice(0, 1 + Math.floor(r() * 3)) : undefined,
      });
    }
    // predicted trajectory beyond current
    const predicted: EngineSeries['predicted'] = [];
    const lastH = points[points.length - 1].health;
    for (let k = 1; k <= e.rul + 25; k++) {
      const c = n + k;
      const decay = Math.min(1, k / Math.max(1, e.rul + 12));
      const h = Math.max(8, lastH - (lastH - 12) * Math.pow(decay, 1.1));
      const spread = 3 + decay * 14;
      predicted.push({ cycle: c, rulPred: Math.max(0, e.rul - k), health: +h.toFixed(1), lo: +(h - spread).toFixed(1), hi: +(h + spread * 0.6).toFixed(1) });
    }
    series.set(e.id, { engineId: e.id, points, predicted, failureCycle: n + e.rul });
  }

  // Alerts: derive from worst engines + timeline spread
  const alerts: AlertItem[] = [];
  const sorted = [...engines].sort((a, b) => a.rul - b.rul).slice(0, 18);
  const times = ['14:32', '14:27', '14:14', '14:01', '13:47', '13:31', '13:12', '12:58', '12:40', '12:21', '11:55', '11:30', '11:02', '10:41', '10:15', '09:52', '09:20', '08:55'];
  sorted.forEach((e, i) => {
    const sev = e.status === 'CRITICAL' ? 'CRITICAL' : e.status === 'WARNING' ? 'WARNING' : i % 3 === 0 ? 'INFO' : 'WARNING';
    alerts.push({
      id: `AL-${1000 + i}`,
      time: times[i % times.length],
      minutesAgo: i * 9 + 3,
      severity: sev,
      engineId: e.id,
      title: sev === 'CRITICAL' ? 'RUL below threshold' : sev === 'WARNING' ? (e.id === 24 ? 'Temperature deviation' : 'Increasing degradation') : 'Telemetry pattern changed',
      detail: sev === 'CRITICAL'
        ? `Predicted failure in ${e.rul} cycles · confidence ${e.confidence}%`
        : sev === 'WARNING'
          ? `Sustained drift across T24 / P30 / S11 · cycle ${e.currentCycle}`
          : `Variance shift detected in recent operating window`,
      cycle: Math.max(5, e.currentCycle - Math.floor(rand() * 6)),
      acked: false,
    });
  });
  // ensure spec example order
  alerts[0] = { ...alerts.find(a => a.engineId === 71) ?? alerts[0], time: '14:32', severity: 'CRITICAL', engineId: 71, title: 'RUL below threshold' };
  alerts[1] = { ...alerts.find(a => a.engineId === 24) ?? alerts[1], time: '14:27', severity: 'WARNING', engineId: 24, title: 'Temperature deviation' };

  const models: ModelInfo[] = [
    { id: 'lstm', name: 'LSTM RUL Predictor', version: 'v2.4', status: 'ACTIVE', rmse: 19.8, mae: 13.6, r2: 0.91, latencyMs: 42 },
    { id: 'gru', name: 'GRU Degradation Model', version: 'v1.9', status: 'STAGED', rmse: 21.4, mae: 14.9, r2: 0.89, latencyMs: 31 },
    { id: 'xgb', name: 'XGBoost Survival Head', version: 'v3.1', status: 'ARCHIVED', rmse: 23.1, mae: 16.2, r2: 0.86, latencyMs: 12 },
    { id: 'tf', name: 'Transformer Forecaster', version: 'v0.9', status: 'STAGED', rmse: 18.6, mae: 12.8, r2: 0.93, latencyMs: 88 },
  ];

  cache = { engines, series, alerts, models };
  return cache;
}

export function engineLabel(id: number) {
  return `ENGINE #${pad(id)}`;
}

export function statusColor(status: EngineStatus): string {
  switch (status) {
    case 'NOMINAL': return '#34D399';
    case 'EARLY': return '#22D3EE';
    case 'WARNING': return '#FBBF24';
    case 'CRITICAL': return '#F87171';
  }
}

export function severityColor(s: 'CRITICAL' | 'WARNING' | 'INFO' | EngineStatus): string {
  if (s === 'CRITICAL') return '#F87171';
  if (s === 'WARNING') return '#FBBF24';
  if (s === 'INFO' || s === 'EARLY') return '#22D3EE';
  return '#34D399';
}
