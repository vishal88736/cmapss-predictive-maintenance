import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, ArrowUpRight, BarChart3, Cpu, Info, Monitor, Moon, Sun, Upload } from 'lucide-react'
import PredictionPage from './pages/PredictionPage'
import ModelPage from './pages/ModelPage'
import MetricsPage from './pages/MetricsPage'
import AboutPage from './pages/AboutPage'
import { usePrediction } from './hooks/usePrediction'
import { useTheme, type Theme } from './hooks/useTheme'
import { inferenceApi } from './services/inference'
import type { BackendStatus, EvaluationRecord, ModelMetadata, Page } from './types/inference'
import './index.css'

const navigation: {page:Page;icon:typeof Upload}[] = [{page:'Prediction',icon:Upload},{page:'Model',icon:Cpu},{page:'Metrics',icon:BarChart3},{page:'About',icon:Info}]

export default function App() {
  const [page,setPage]=useState<Page>('Prediction')
  const [status,setStatus]=useState<BackendStatus|null>(null)
  const [model,setModel]=useState<ModelMetadata|null>(null)
  const [records,setRecords]=useState<EvaluationRecord[]>([])
  const [connectionError,setConnectionError]=useState('')
  const [metricsError,setMetricsError]=useState('')
  const theme=useTheme()
  const prediction=usePrediction()
  const refresh=useCallback(()=>{
    inferenceApi.status().then(result=>{setStatus(result);setConnectionError('')}).catch(cause=>{setStatus(null);setConnectionError(cause.message)})
    inferenceApi.model().then(setModel).catch(()=>setModel(null))
    inferenceApi.metrics().then(result=>{setRecords(result.records);setMetricsError('')}).catch(cause=>{setRecords([]);setMetricsError(cause.message)})
  },[])
  useEffect(refresh,[refresh])
  const sessionEvaluations=useMemo(()=>prediction.history.filter(item=>item.status==='completed'&&item.result?.evaluation).map(item=>({...item.result!.evaluation!,id:item.jobId,source:`Uploaded test evaluation · ${item.fileName}`})),[prediction.history])
  const ThemeIcon=theme.theme==='system'?Monitor:theme.theme==='dark'?Moon:Sun
  const navigate=(next:Page)=>{setPage(next);window.scrollTo({top:0,behavior:'instant'})}

  return <div className="tool-shell"><a href="#main-content" className="skip-link">Skip to content</a><header className="app-header"><div className="header-inner"><button className="wordmark" onClick={()=>navigate('Prediction')} aria-label="C-MAPSS prediction home"><span className="wordmark-icon"><Activity size={22}/></span><span>C-MAPSS<small>ENGINE PROGNOSTICS</small></span></button><nav aria-label="Main navigation">{navigation.map(({page:next,icon:Icon})=><button key={next} className={page===next?'active':''} aria-current={page===next?'page':undefined} onClick={()=>navigate(next)}><Icon size={16}/><span>{next}</span></button>)}</nav><label className="theme-selector"><ThemeIcon size={17}/><select aria-label="Color theme" value={theme.theme} onChange={event=>theme.setTheme(event.target.value as Theme)}><option value="light">Light</option><option value="dark">Dark</option><option value="system">System</option></select></label></div></header><main id="main-content" className="main-content">{connectionError&&<div className="connection-notice"><span className="status-mark"/><span>{connectionError}</span><button onClick={refresh}>Reconnect</button></div>}{page==='Prediction'&&<PredictionPage state={prediction} status={status} refreshStatus={refresh}/>} {page==='Model'&&<ModelPage model={model} status={status}/>} {page==='Metrics'&&<MetricsPage records={records} sessionEvaluations={sessionEvaluations} model={model} error={metricsError}/>} {page==='About'&&<AboutPage/>}</main><footer className="app-footer"><div className="footer-inner"><div><strong>NASA C-MAPSS RUL Prediction</strong><span>Independent predictive-maintenance research</span></div><div className="footer-status"><span className={`status-mark ${status?'connected':''}`}/>{status?'Python API connected':'Python API not connected'}</div><a href="https://github.com/vishal88736/cmapss-predictive-maintenance" target="_blank" rel="noreferrer">Project source <ArrowUpRight size={14}/></a></div></footer></div>
}
