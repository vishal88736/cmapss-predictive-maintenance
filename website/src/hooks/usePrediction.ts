import { useCallback, useEffect, useRef, useState } from 'react'
import { inferenceApi } from '../services/inference'
import type { InferenceJob, Subset, UploadRecord } from '../types/inference'

export function usePrediction() {
  const [file, setFile] = useState<File | null>(null)
  const [truthFile, setTruthFile] = useState<File | null>(null)
  const [subset, setSubset] = useState<Subset>('FD001')
  const [upload, setUpload] = useState<UploadRecord | null>(null)
  const [validating, setValidating] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [job, setJob] = useState<InferenceJob | null>(null)
  const [history, setHistory] = useState<InferenceJob[]>([])
  const validationRequest = useRef<AbortController | null>(null)
  const previousUpload = useRef<string | null>(null)

  const validate = useCallback(async (test: File, selected: Subset, truth: File | null) => {
    validationRequest.current?.abort()
    const controller = new AbortController()
    validationRequest.current = controller
    setValidating(true); setError(''); setUpload(null); setJob(null)
    if (previousUpload.current) { inferenceApi.discard(previousUpload.current).catch(() => {}); previousUpload.current = null }
    try {
      const result = await inferenceApi.validate(test, selected, truth, controller.signal)
      if (!controller.signal.aborted) { setUpload(result); previousUpload.current = result.uploadId }
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'File validation failed.')
    } finally { if (!controller.signal.aborted) setValidating(false) }
  }, [])

  const chooseFile = (next: File) => {
    setFile(next)
    const match = next.name.match(/test_(FD00[1-4])\.txt$/i)
    const selected = match ? match[1].toUpperCase() as Subset : subset
    setSubset(selected)
    setTruthFile(null)
    validate(next, selected, null)
  }
  const chooseTruth = (next: File | null) => { setTruthFile(next); if (file) validate(file, subset, next) }
  const chooseSubset = (next: Subset) => { setSubset(next); if (file) validate(file, next, truthFile) }
  const clear = () => {
    validationRequest.current?.abort()
    if (previousUpload.current) inferenceApi.discard(previousUpload.current).catch(() => {})
    previousUpload.current = null
    setFile(null); setTruthFile(null); setUpload(null); setJob(null); setError(''); setValidating(false)
  }
  const run = async () => {
    if (!upload || submitting || job?.status === 'running') return
    setError(''); setSubmitting(true); setJob(null)
    try { setJob(await inferenceApi.predict(upload.uploadId)) }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Prediction could not be completed.') }
    finally { setSubmitting(false) }
  }

  useEffect(() => {
    if (job?.status !== 'running') return
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const current = await inferenceApi.job(job.jobId, controller.signal)
        if (controller.signal.aborted) return
        setJob(current)
        if (current.status === 'running') timer = setTimeout(poll, 700)
        else {
          setHistory(previous => [current, ...previous.filter(item => item.jobId !== current.jobId)].slice(0, 10))
          if (current.error) setError(current.error)
        }
      } catch (cause) {
        if (!controller.signal.aborted) { setError(cause instanceof Error ? cause.message : 'Could not read inference status.'); timer = setTimeout(poll, 1500) }
      }
    }
    timer = setTimeout(poll, 300)
    return () => { controller.abort(); clearTimeout(timer) }
  }, [job?.jobId, job?.status])
  useEffect(() => () => { validationRequest.current?.abort(); if (previousUpload.current) inferenceApi.discard(previousUpload.current).catch(() => {}) }, [])

  return {file, truthFile, subset, upload, validating, submitting, error, job, history, chooseFile, chooseTruth, chooseSubset, clear, run, retry:()=>{if(file)validate(file,subset,truthFile)}}
}
export type PredictionState = ReturnType<typeof usePrediction>
