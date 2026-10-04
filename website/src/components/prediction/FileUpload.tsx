import { useRef, useState } from 'react'
import { FileText, Upload, X } from 'lucide-react'

export function FileUpload({file, onSelect, onClear, disabled, validating}: {file:File|null; onSelect:(file:File)=>void; onClear:()=>void; disabled:boolean; validating:boolean}) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const choose = (files: FileList | null) => { if(files?.[0] && !disabled) onSelect(files[0]) }
  return <div className={`upload-zone ${dragging?'drag-active':''} ${file?'has-file':''}`} onDragOver={event=>{event.preventDefault();if(!disabled)setDragging(true)}} onDragLeave={()=>setDragging(false)} onDrop={event=>{event.preventDefault();setDragging(false);choose(event.dataTransfer.files)}}>
    <input ref={input} type="file" accept=".txt,text/plain" className="visually-hidden" aria-label="Upload C-MAPSS test file" disabled={disabled} onChange={event=>{choose(event.target.files);event.target.value=''}}/>
    {file?<><div className="upload-icon"><FileText size={27}/></div><strong className="upload-file-name">{file.name}</strong><p>{validating?'Validating the uploaded file…':'Test data selected'}</p><div className="upload-actions"><button className="button secondary" disabled={disabled} onClick={()=>input.current?.click()}>Replace file</button><button className="button text-button" disabled={disabled} onClick={onClear}><X size={15}/> Remove</button></div></>:<><div className="upload-icon"><Upload size={27}/></div><h3>Upload Test Data</h3><p>Drop your NASA C-MAPSS test file here or browse</p><button className="button secondary" disabled={disabled} onClick={()=>input.current?.click()}>Browse files</button><span className="format-note">Headerless, whitespace-separated .txt</span></>}
  </div>
}
