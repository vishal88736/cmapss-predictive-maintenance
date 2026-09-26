import { getFleetData } from '../data/generator';
import type { AlertItem, Engine, EngineSeries, ModelInfo } from '../types';

/**
 * Service abstraction — all UI reads through here so the real
 * C-MAPSS ML backend can replace the mock without touching components.
 */
export const api = {
  getEngines(): Engine[] {
    return getFleetData().engines;
  },
  getEngine(id: number): Engine | undefined {
    return getFleetData().engines.find((e) => e.id === id);
  },
  getSeries(id: number): EngineSeries | undefined {
    return getFleetData().series.get(id);
  },
  getAlerts(): AlertItem[] {
    return getFleetData().alerts;
  },
  getModels(): ModelInfo[] {
    return getFleetData().models;
  },
  fleetHealth(): { health: number; avgRul: number; nominal: number; highRisk: number } {
    const engines = getFleetData().engines;
    const health = Math.round(engines.reduce((s, e) => s + e.health, 0) / engines.length);
    const avgRul = Math.round(engines.reduce((s, e) => s + e.rul, 0) / engines.length);
    const nominal = engines.filter((e) => e.status === 'NOMINAL').length;
    const highRisk = engines.filter((e) => e.status === 'WARNING' || e.status === 'CRITICAL').length;
    return { health, avgRul, nominal, highRisk };
  },
};
