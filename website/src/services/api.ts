import type { Dataset, Prediction, Subset } from '../types/domain'

const cache = new Map<Subset, Promise<Dataset>>()
const apiBase = import.meta.env.VITE_API_BASE_URL || '/api'

export function loadDataset(subset: Subset): Promise<Dataset> {
  if (!cache.has(subset)) {
    cache.set(subset, fetch(`${import.meta.env.BASE_URL}data/${subset}.json`)
      .then(async response => {
        if (!response.ok) throw new Error(`Dataset could not be loaded (${response.status}).`)
        const data = await response.json() as Dataset
        if (data.subset !== subset || data.split !== 'train' || !Array.isArray(data.engines) || !Array.isArray(data.sensorNames)) {
          throw new Error('Unexpected dataset format.')
        }
        return data
      }).catch(error => { cache.delete(subset); throw error }))
  }
  return cache.get(subset)!
}

export async function checkModelConnection(): Promise<boolean> {
  try {
    const response = await fetch(`${apiBase}/status`, { signal: AbortSignal.timeout(3500) })
    if (!response.ok) return false
    const status = await response.json()
    return status.modelConnected === true
  } catch { return false }
}

// A Python inference service can implement POST /api/predict with this request/response contract.
// The frontend never substitutes a synthetic result if the service is absent or returns invalid data.
export async function runPrediction(subset: Subset, engineId: number, cycle: number): Promise<Prediction> {
  const response = await fetch(`${apiBase}/predict`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subset, split: 'train', engineId, cycle }),
  })
  if (!response.ok) throw new Error(`Inference request failed (${response.status}).`)
  const result = await response.json() as Prediction
  if (result.engineId !== engineId || result.cycle !== cycle || !Number.isFinite(result.predictedRUL)) {
    throw new Error('Inference service returned an invalid prediction.')
  }
  return result
}
