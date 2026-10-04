import type { BackendStatus, EvaluationRecord, InferenceJob, ModelMetadata, Subset, Telemetry, UploadRecord } from '../types/inference'

const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '')

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try { response = await fetch(`${base}${path}`, options) }
  catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw error
    throw new Error('Backend not connected. Start the Python API and try again.')
  }
  if (!response.headers.get('content-type')?.includes('application/json')) throw new Error('Backend not connected. Start the Python API and try again.')
  const data = await response.json()
  if (!response.ok) {
    const message = typeof data.detail === 'string' ? data.detail : data.detail?.message
    throw new Error(message || (response.status === 422 ? 'Input file does not match the expected C-MAPSS format.' : 'The request could not be completed.'))
  }
  return data as T
}

export const inferenceApi = {
  status: () => request<BackendStatus>('/status'),
  model: () => request<ModelMetadata>('/model'),
  metrics: () => request<{ records: EvaluationRecord[] }>('/metrics'),
  validate: (file: File, subset: Subset, truth?: File | null, signal?: AbortSignal) => {
    const form = new FormData()
    form.append('test_file', file); form.append('subset', subset)
    if (truth) form.append('rul_file', truth)
    return request<UploadRecord>('/uploads', { method: 'POST', body: form, signal })
  },
  discard: (id: string) => request(`/uploads/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  telemetry: (id: string, unit: number, signal?: AbortSignal) => request<Telemetry>(`/uploads/${encodeURIComponent(id)}/engines/${unit}`, { signal }),
  predict: (uploadId: string) => request<InferenceJob>('/predictions', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({uploadId}) }),
  job: (id: string, signal?: AbortSignal) => request<InferenceJob>(`/predictions/${encodeURIComponent(id)}`, {signal}),
}
