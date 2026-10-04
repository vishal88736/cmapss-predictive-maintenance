import { useEffect, useMemo, useState } from 'react'
import { Bar, BarChart, Brush, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { inferenceApi } from '../services/inference'
import type { EnginePrediction, Telemetry, UploadRecord } from '../types/inference'

const tooltip = { background:'var(--surface)', border:'1px solid var(--border-strong)', borderRadius:6, color:'var(--text)', fontSize:14 }
const tick = {fill:'var(--text-muted)',fontSize:12}
export const formatNumber = (value: number) => value.toLocaleString(undefined, {maximumFractionDigits:2})

export function RULByEngineChart({ predictions }: {predictions: EnginePrediction[]}) {
  if(predictions.length<2)return null
  return <div className="result-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={predictions} margin={{left:0,right:16,top:12,bottom:16}}><CartesianGrid stroke="var(--chart-grid)" vertical={false}/><XAxis dataKey="engineId" tick={tick} tickLine={false} axisLine={false} label={{value:'Engine ID',position:'insideBottom',offset:-10,fill:'var(--text-muted)',fontSize:12}}/><YAxis tick={tick} tickLine={false} axisLine={false} width={50}/><Tooltip contentStyle={tooltip} labelFormatter={value=>`Engine ${value}`} formatter={value=>[`${formatNumber(Number(value))} cycles`,'Predicted RUL']}/><Bar dataKey="predictedRUL" fill="var(--chart-primary)" radius={[2,2,0,0]} isAnimationActive={false}/>{predictions.length>30&&<Brush dataKey="engineId" height={20} stroke="var(--border-strong)" fill="var(--surface-raised)"/>}</BarChart></ResponsiveContainer></div>
}

export function SensorHistory({ upload, selectedEngine, onSelectEngine }: {upload:UploadRecord; selectedEngine:number; onSelectEngine:(id:number)=>void}) {
  const [data,setData] = useState<Telemetry|null>(null)
  const [sensor,setSensor] = useState(upload.modelFeatures[0] || upload.sensorColumns[0])
  const [error,setError] = useState('')
  const [loading,setLoading] = useState(false)
  useEffect(()=>{
    const controller = new AbortController()
    setLoading(true);setError('');setData(null)
    inferenceApi.telemetry(upload.uploadId,selectedEngine,controller.signal).then(result=>{if(!controller.signal.aborted)setData(result)}).catch(cause=>{if(!controller.signal.aborted)setError(cause.message)}).finally(()=>{if(!controller.signal.aborted)setLoading(false)})
    return ()=>controller.abort()
  },[upload.uploadId,selectedEngine])
  const points = useMemo(()=>{
    const index=data?.sensorNames.indexOf(sensor) ?? -1
    return index<0?[]:data!.readings.map(row=>({cycle:row[0],value:row[index+4]}))
  },[data,sensor])
  return <section className="content-section sensor-section"><div className="section-heading"><div><span className="eyebrow">INPUT INSPECTION</span><h2>Engine sensor history</h2><p>The original test readings supplied to the model.</p></div><div className="chart-selectors"><label>Engine<select aria-label="Telemetry engine" value={selectedEngine} onChange={event=>onSelectEngine(Number(event.target.value))}>{upload.engineIds.map(id=><option key={id} value={id}>Engine {id}</option>)}</select></label><label>Sensor<select aria-label="Telemetry sensor" value={sensor} onChange={event=>setSensor(event.target.value)}>{upload.sensorColumns.map(name=><option key={name}>{name}</option>)}</select></label></div></div><div className="chart-frame">{error?<div className="inline-error" role="alert">{error}</div>:loading?<div className="chart-empty">Loading uploaded telemetry…</div>:points.length?<div className="sensor-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={points} margin={{top:16,right:20,bottom:16,left:5}}><CartesianGrid stroke="var(--chart-grid)" vertical={false}/><XAxis dataKey="cycle" tick={tick} tickLine={false} axisLine={false} label={{value:'Operational cycle',position:'insideBottom',offset:-10,fill:'var(--text-muted)',fontSize:12}}/><YAxis domain={['auto','auto']} tick={tick} tickLine={false} axisLine={false} width={65}/><Tooltip contentStyle={tooltip} labelFormatter={value=>`Cycle ${value}`} formatter={value=>[formatNumber(Number(value)),sensor]}/><Line dataKey="value" type="linear" stroke="var(--chart-primary)" strokeWidth={2} dot={false} isAnimationActive={false}/>{points.length>30&&<Brush dataKey="cycle" height={22} stroke="var(--border-strong)" fill="var(--surface-raised)"/>}</LineChart></ResponsiveContainer></div>:<div className="chart-empty">No telemetry available.</div>}</div><p className="small-note">Raw measurements, original units. Drag the chart brush to inspect a cycle range.</p></section>
}

export function EvaluationCharts({ predictions }: {predictions: EnginePrediction[]}) {
  const rows=predictions.filter(row=>row.actualRUL!==undefined).map(row=>({...row,residual:row.predictedRUL-row.actualRUL!}))
  if(!rows.length)return <div className="evaluation-empty"><h3>Actual vs. predicted RUL</h3><p>Ground-truth RUL unavailable for this evaluation.</p><span>No paired engine-level evaluation records are attached to this result.</span></div>
  return <div className="evaluation-chart-grid"><div><h3>Actual vs. predicted RUL</h3><div className="result-chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{top:10,right:20,bottom:20,left:10}}><CartesianGrid stroke="var(--chart-grid)"/><XAxis dataKey="actualRUL" type="number" name="Actual RUL" tick={tick} label={{value:'Actual RUL',position:'insideBottom',offset:-12,fill:'var(--text-muted)',fontSize:12}}/><YAxis dataKey="predictedRUL" type="number" name="Predicted RUL" tick={tick} width={60}/><Tooltip contentStyle={tooltip}/><Scatter data={rows} fill="var(--chart-primary)" isAnimationActive={false}/></ScatterChart></ResponsiveContainer></div></div><div><h3>Prediction residuals</h3><div className="result-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={rows} margin={{top:10,right:20,bottom:20,left:5}}><CartesianGrid stroke="var(--chart-grid)" vertical={false}/><XAxis dataKey="engineId" tick={tick} label={{value:'Engine ID',position:'insideBottom',offset:-12,fill:'var(--text-muted)',fontSize:12}}/><YAxis tick={tick} width={60}/><ReferenceLine y={0} stroke="var(--text-muted)"/><Tooltip contentStyle={tooltip} labelFormatter={value=>`Engine ${value}`} formatter={value=>[formatNumber(Number(value)),'Predicted − actual']}/><Bar dataKey="residual" fill="var(--chart-primary)" isAnimationActive={false}/></BarChart></ResponsiveContainer></div></div></div>
}
