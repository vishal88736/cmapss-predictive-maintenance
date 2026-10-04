import { Check, Circle, LoaderCircle, X } from 'lucide-react'
import type { Stages } from '../../types/inference'

export const pendingStages: Stages = { uploaded:'pending', validation:'pending', preprocessing:'pending', features:'pending', inference:'pending', prediction:'pending' }
const labels = { uploaded:'File uploaded', validation:'Data validation', preprocessing:'Preprocessing', features:'Feature preparation', inference:'Model inference', prediction:'RUL prediction' }

export function ProcessStatus({ stages }: {stages: Stages}) {
  return <ol className="process-status" aria-label="Backend processing status">{Object.entries(labels).map(([key,label])=>{
    const state = stages[key as keyof Stages]
    return <li key={key} data-state={state}><span className="process-symbol">{state==='completed'?<Check size={15}/>:state==='running'?<LoaderCircle size={15} className="spinner"/>:state==='failed'?<X size={15}/>:<Circle size={12}/>}</span><span>{label}<small>{state==='pending'?'Not started':state==='running'?'In progress':state==='failed'?'Failed':'Completed'}</small></span></li>
  })}</ol>
}
