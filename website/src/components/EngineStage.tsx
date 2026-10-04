import { useEffect, useRef, useState } from 'react'
import { useMotion } from './Motion'
import type { EngineSceneController } from './engine/scene'

function TurbineFallback() {
  return <svg className="turbine-fallback" viewBox="0 0 600 600" aria-hidden="true">
    <defs>
      <radialGradient id="turbine-metal"><stop stopColor="#16202d"/><stop offset=".76" stopColor="#111c29"/><stop offset=".9" stopColor="#7294b4"/><stop offset="1" stopColor="#d9e8f4"/></radialGradient>
      <linearGradient id="turbine-blade"><stop stopColor="#1c3247"/><stop offset=".45" stopColor="#7496b6"/><stop offset="1" stopColor="#bdcee0"/></linearGradient>
    </defs>
    <circle cx="300" cy="300" r="229" fill="url(#turbine-metal)" stroke="#7c99b7" strokeWidth="2"/>
    <circle cx="300" cy="300" r="214" fill="#0a131f" stroke="#c4d3e1" strokeWidth="3"/>
    <g transform="translate(300 300)"><g className="engine-fan" fill="url(#turbine-blade)" stroke="#99b3ca" strokeWidth=".7">
      {Array.from({length:28},(_,i)=><path key={i} d="M-10 -35Q-2 -112 -68 -193Q-52 -207 -29 -210Q18 -133 17 -37Z" transform={`rotate(${i*360/28})`}/>)}
    </g></g>
    <circle cx="300" cy="300" r="39" fill="url(#turbine-metal)" stroke="#a5bdd2" strokeWidth="2"/>
    <ellipse cx="290" cy="290" rx="12" ry="16" fill="#cadbe8" opacity=".4"/>
  </svg>
}

export default function EngineStage() {
  const mount = useRef<HTMLDivElement>(null)
  const controller = useRef<EngineSceneController | null>(null)
  const motion = useMotion()
  const motionRef = useRef(motion)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    motionRef.current = motion
    controller.current?.setMotion(motion)
  }, [motion])

  useEffect(() => {
    let cancelled = false
    import('./engine/scene').then(({ createEngineScene }) => {
      if (cancelled || !mount.current) return
      try {
        controller.current = createEngineScene(mount.current, motionRef.current)
        setReady(true)
      } catch {
        // The detailed SVG remains available when the browser has no WebGL renderer.
        setReady(false)
      }
    }).catch(() => { if (!cancelled) setReady(false) })
    return () => { cancelled = true; controller.current?.dispose(); controller.current = null }
  }, [])

  return <div className={`engine-stage ${ready?'scene-ready':''}`} role="img" aria-label="Conceptual three-dimensional turbofan engine. Decorative animation, not measured engine telemetry.">
    <div className="stage-halo" aria-hidden="true"/>
    <div className="stage-orbit orbit-outer" aria-hidden="true"/>
    <div className="stage-orbit orbit-inner" aria-hidden="true"/>
    <TurbineFallback/>
    <div className="engine-canvas" ref={mount}/>
    <div className="stage-reticle reticle-top" aria-hidden="true"/><div className="stage-reticle reticle-bottom" aria-hidden="true"/>
  </div>
}
