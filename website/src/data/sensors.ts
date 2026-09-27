import type { SensorChannel } from '@/types/telemetry'

export const SENSOR_CHANNELS: SensorChannel[] = [
  { id: 'T24', name: 'LPC Outlet Temperature', short: 'LPC Temp', unit: '°R', decimals: 2, baseline: 641.82, behavior: 'drift', family: 'temp', primary: true, simulateCoef: -1.25, step: 0.5 },
  { id: 'T30', name: 'HPC Outlet Temperature', short: 'HPC Temp', unit: '°R', decimals: 1, baseline: 1589.7, behavior: 'degrading', family: 'temp', primary: true, simulateCoef: -2.3, step: 1 },
  { id: 'T50', name: 'LPT Outlet Temperature', short: 'LPT Temp', unit: '°R', decimals: 1, baseline: 1400.6, behavior: 'degrading', family: 'temp', primary: true, simulateCoef: -1.9, step: 1 },
  { id: 'P15', name: 'Bypass Duct Pressure', short: 'Bypass Pres', unit: 'psia', decimals: 2, baseline: 21.61, behavior: 'variance', family: 'pressure', step: 0.1, simulateCoef: -0.6 },
  { id: 'P30', name: 'HPC Outlet Pressure', short: 'HPC Pres', unit: 'psia', decimals: 2, baseline: 554.36, behavior: 'degrading', family: 'pressure', primary: true, simulateCoef: 2.1, step: 0.5 },
  { id: 'Nf', name: 'Physical Fan Speed', short: 'Fan Speed', unit: 'rpm', decimals: 0, baseline: 2388.06, behavior: 'drift', family: 'speed', step: 1, simulateCoef: 0.8 },
  { id: 'Nc', name: 'Physical Core Speed', short: 'Core Speed', unit: 'rpm', decimals: 0, baseline: 9046.19, behavior: 'drift', family: 'speed', primary: true, simulateCoef: -1.05, step: 5 },
  { id: 'epr', name: 'Engine Pressure Ratio', short: 'EPR', unit: '', decimals: 3, baseline: 1.3, behavior: 'stable', family: 'ratio', step: 0.005, simulateCoef: 0.5 },
  { id: 'Ps30', name: 'HPC Outlet Static Pressure', short: 'Static Pres', unit: 'psia', decimals: 2, baseline: 47.47, behavior: 'degrading', family: 'pressure', primary: true, simulateCoef: 2.4, step: 0.1 },
  { id: 'phi', name: 'Fuel Flow / Ps30 Ratio', short: 'Fuel Ratio', unit: 'pps/psi', decimals: 2, baseline: 521.66, behavior: 'degrading', family: 'ratio', primary: true, simulateCoef: -2.75, step: 1 },
  { id: 'NRf', name: 'Corrected Fan Speed', short: 'Corr Fan', unit: 'rpm', decimals: 1, baseline: 2388.02, behavior: 'stable', family: 'speed', step: 1, simulateCoef: 0.7 },
  { id: 'NRc', name: 'Corrected Core Speed', short: 'Corr Core', unit: 'rpm', decimals: 1, baseline: 8138.62, behavior: 'drift', family: 'speed', step: 5, simulateCoef: -0.95 },
  { id: 'BPR', name: 'Bypass Ratio', short: 'Bypass Ratio', unit: '', decimals: 3, baseline: 8.4195, behavior: 'degrading', family: 'ratio', primary: true, simulateCoef: 1.65, step: 0.01 },
  { id: 'htBleed', name: 'Bleed Enthalpy', short: 'Bleed Enth', unit: '', decimals: 0, baseline: 392.0, behavior: 'variance', family: 'bleed', step: 0.5, simulateCoef: -0.9 },
  { id: 'W31', name: 'HPT Coolant Bleed', short: 'HPT Bleed', unit: 'lbm/s', decimals: 2, baseline: 39.06, behavior: 'degrading', family: 'bleed', step: 0.05, simulateCoef: -1.5 },
  { id: 'W32', name: 'LPT Coolant Bleed', short: 'LPT Bleed', unit: 'lbm/s', decimals: 2, baseline: 23.419, behavior: 'degrading', family: 'bleed', step: 0.05, simulateCoef: -1.45 }
]

export const SENSOR_MAP: Record<string, SensorChannel> = Object.fromEntries(SENSOR_CHANNELS.map((s) => [s.id, s]))

export const PRIMARY_SENSORS = SENSOR_CHANNELS.filter((s) => s.primary)

export const SIMULATABLE_SENSORS = SENSOR_CHANNELS.filter((s) => Math.abs(s.simulateCoef) >= 1)

export function sensorValue(id: string, v: number): string {
  const ch = SENSOR_MAP[id]
  if (!ch) return String(v)
  return `${v.toFixed(ch.decimals)}${ch.unit ? (ch.unit === '°R' ? ' °R' : ` ${ch.unit}`) : ''}`
}
