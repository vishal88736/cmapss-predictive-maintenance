import { useState, type ReactNode } from 'react'
import { Activity, ArrowDown, ArrowRight, ArrowUpRight, Database, GitBranch, ShieldCheck, Sparkles } from 'lucide-react'
import EngineStage from '../components/EngineStage'
import { Button } from '../components/ui/button'
import { useMotion } from '../components/Motion'

interface OverviewProps {
  subset: string
  engineId?: number
  cycle: number | null
  referenceRUL?: number
  sensorName?: string
  sensorValue?: number
  selectors: ReactNode
  sensorSelector: ReactNode
  cyclePicker: ReactNode
  lifecycle: ReactNode
  telemetry: ReactNode
  pipeline: ReactNode
  integrity: ReactNode
  navigate: (page: 'Engine Explorer' | 'Model' | 'Sensor Telemetry' | 'Dataset' | 'RUL Prediction') => void
}

const value = (n?: number | null) => n == null ? 'Not available' : Number.isInteger(n) ? n.toLocaleString() : Number(n.toFixed(4)).toString()

export default function ResearchOverview(props: OverviewProps) {
  const [view, setView] = useState<'lifecycle' | 'telemetry'>('lifecycle')
  const motion = useMotion()
  const scrollToData = () => document.getElementById('engine-workspace')?.scrollIntoView({ behavior: motion ? 'smooth' : 'instant', block: 'start' })

  return <>
    <section className="mission-hero">
      <div className="mission-grid" aria-hidden="true"/>
      <div className="mission-topline"><span className="research-pill"><span/> NASA C-MAPSS · RESEARCH PROJECT</span><span className="hero-project-type">AEROSPACE / MACHINE LEARNING</span></div>
      <div className="mission-title"><span className="mission-eyebrow">LOOK BEYOND THE NEXT CYCLE.</span><h1>Remaining<br/><em>Useful Life<span>.</span></em></h1></div>
      <EngineStage/>
      <div className="engine-callout"><span className="callout-cross">+</span><span>TURBOFAN ENGINE<small>CONCEPTUAL SYSTEM VIEW</small></span></div>
      <div className="mission-copy"><h2>Predicting the remaining life<br/>of turbofan engines.</h2><p>A research journey from NASA's simulated engine telemetry to degradation analysis and machine learning.</p><div className="mission-actions"><Button onClick={()=>props.navigate('Engine Explorer')}>Explore engine data <ArrowUpRight size={17}/></Button><Button variant="secondary" onClick={()=>props.navigate('Model')}>Inside the model <ArrowRight size={16}/></Button></div></div>
      <div className="mission-source"><div className="source-orb"><Database size={18}/></div><div><span>OBSERVATION SOURCE</span><strong>{props.subset} <small>/ TRAINING DATA</small></strong><p>Local C-MAPSS source files</p></div></div>
      <div className="mission-bottomline"><span><ShieldCheck size={13}/> Simulated engines. Real observations.</span><button onClick={scrollToData}>ENTER THE WORKSPACE <ArrowDown size={14}/></button></div>
    </section>

    <div className="research-ribbon">{['C-MAPSS data','Sensor telemetry','Degradation','Machine learning','Remaining useful life'].map((label,i)=><div key={label}><span className="ribbon-index">{String(i+1).padStart(2,'0')}</span><span>{label}</span>{i<4&&<ArrowRight size={13}/>}</div>)}</div>

    <section className="engine-workspace" id="engine-workspace">
      <div className="workspace-section-head"><div><span className="eyebrow">THE OBSERVATION LAB</span><h2>An engine's life.<br/><span>A clearer perspective.</span></h2></div><div><p>Explore a real unit, one cycle at a time.<br/>No synthetic telemetry. No invented predictions.</p><button className="text-action" onClick={()=>props.navigate('Engine Explorer')}>Full engine explorer <ArrowUpRight size={16}/></button></div></div>
      <div className="workspace-console">
        <div className="console-topbar"><div className="console-title"><span className="source-indicator"/> ENGINE WORKSPACE <span className="console-source">{props.engineId === undefined?'AWAITING DATA':`UNIT ${props.engineId}`}</span></div><div className="console-selectors">{props.selectors}</div></div>
        <div className="workspace-tabs" role="tablist" aria-label="Engine data view"><button role="tab" id="lifecycle-tab" aria-selected={view==='lifecycle'} aria-controls="workspace-panel" onClick={()=>setView('lifecycle')}><GitBranch size={15}/> Lifecycle reference</button><button role="tab" id="telemetry-tab" aria-selected={view==='telemetry'} aria-controls="workspace-panel" onClick={()=>setView('telemetry')}><Activity size={15}/> Sensor signal</button><span className="console-provenance"><ShieldCheck size={12}/> SOURCE-BACKED DATA</span></div>
        <div className="console-body" id="workspace-panel" role="tabpanel" aria-labelledby={view==='lifecycle'?'lifecycle-tab':'telemetry-tab'}>
          <div className="console-chart"><div className="console-chart-head"><div><span className="label">{view==='lifecycle'?'ENGINE CYCLE → REMAINING USEFUL LIFE':'ENGINE CYCLE → SENSOR MEASUREMENT'}</span><h3>{view==='lifecycle'?'Run-to-failure reference':props.sensorName || 'Sensor telemetry'}</h3></div>{view==='telemetry'?<div className="console-sensor-select">{props.sensorSelector}</div>:<span className="console-legend"><i/> Derived reference</span>}</div><div key={view} className="console-chart-view">{view==='lifecycle'?props.lifecycle:props.telemetry}</div>{props.cyclePicker}</div>
          <aside className="observation-snapshot"><span className="snapshot-kicker">SELECTED OBSERVATION</span><div className="snapshot-unit"><span>ENGINE UNIT</span><strong>{props.engineId===undefined?'Not available':String(props.engineId).padStart(2,'0')}</strong><span>{props.subset} · TRAIN</span></div><div className="snapshot-detail"><span>Operating cycle</span><strong>{value(props.cycle)}</strong></div><div className="snapshot-detail"><span>{view==='lifecycle'?'Reference RUL':props.sensorName || 'Sensor value'}</span><strong>{view==='lifecycle'?value(props.referenceRUL):value(props.sensorValue)}{view==='lifecycle'&&props.referenceRUL!==undefined&&<small>cycles</small>}</strong></div><p>{view==='lifecycle'?'Derived from this unit’s final recorded training cycle. This is a ground-truth reference, not model inference.':'The original sensor measurement at the selected cycle. No smoothing or simulated readings.'}</p><button onClick={()=>props.navigate(view==='lifecycle'?'RUL Prediction':'Sensor Telemetry')}>{view==='lifecycle'?'Explore inference':'Open telemetry'} <ArrowUpRight size={15}/></button></aside>
        </div>
        <div className="console-footer"><span><ShieldCheck size={12}/> Verified observations only</span><span>HOVER TO INSPECT · SELECT A CYCLE TO EXPLORE</span></div>
      </div>
    </section>

    <section className="research-tiles">
      <button className="research-tile telemetry-tile" onClick={()=>props.navigate('Sensor Telemetry')}><Activity size={22}/><span className="eyebrow">FOLLOW THE SIGNAL</span><h3>Every cycle<br/>tells a story.</h3><p>Inspect multivariate sensor histories and operating conditions from the selected C-MAPSS engine.</p><span className="tile-link">Explore sensor telemetry <ArrowUpRight size={17}/></span><div className="tile-orbit" aria-hidden="true"/></button>
      <button className="research-tile dataset-tile" onClick={()=>props.navigate('Dataset')}><Database size={22}/><span className="eyebrow">UNDERSTAND THE SOURCE</span><h3>Simulated flight.<br/>Measurable degradation.</h3><p>C-MAPSS records turbofan operating conditions and sensor measurements across run-to-failure training histories.</p><span className="tile-link">Understand the dataset <ArrowUpRight size={17}/></span><div className="tile-grid-art" aria-hidden="true"/></button>
    </section>

    <section className="pathway-section"><div className="workspace-section-head"><div><span className="eyebrow">INSIDE THE RESEARCH</span><h2>From raw signals<br/><span>to remaining life.</span></h2></div><button className="text-action" onClick={()=>props.navigate('Model')}>View model architecture <ArrowUpRight size={16}/></button></div><div className="pathway-card"><div className="pathway-card-label"><Sparkles size={15}/> THE IMPLEMENTED RESEARCH PATHWAY</div>{props.pipeline}<p><ShieldCheck size={15}/> The training code selects sensor features, normalizes by operating regime, and constructs engine-specific windows. Model inference is available only when an actual service is connected.</p></div></section>
    {props.integrity}
  </>
}
