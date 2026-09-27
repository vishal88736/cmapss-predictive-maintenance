/**
 * AmbientBackground — the atmospheric layer that sits behind the whole app.
 * Three slow-drifting aurora blobs + a parallax grid + film grain + vignette.
 * Purely decorative, never interactive, and disabled under reduced motion.
 */
import { useEffect, useState } from 'react'
import { useReducedMotion } from '@/lib/hooks'

export default function AmbientBackground() {
  const reduced = useReducedMotion()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const still = reduced || !mounted

  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 z-0 overflow-hidden bg-bg">
      {/* aurora blobs */}
      <div
        className="absolute -left-[18%] -top-[22%] h-[62vmax] w-[62vmax] rounded-full opacity-[0.5] blur-[110px] animate-aurora-a"
        style={{
          background: 'radial-gradient(circle at 40% 40%, rgba(122,140,255,0.42), rgba(122,140,255,0) 62%)'
        }}
      />
      <div
        className="absolute -right-[14%] top-[6%] h-[52vmax] w-[52vmax] rounded-full opacity-[0.42] blur-[120px] animate-aurora-b"
        style={{
          background: 'radial-gradient(circle at 55% 45%, rgba(167,139,250,0.36), rgba(167,139,250,0) 64%)'
        }}
      />
      <div
        className="absolute bottom-[-26%] left-[22%] h-[58vmax] w-[58vmax] rounded-full opacity-[0.34] blur-[130px] animate-aurora-c"
        style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(76,194,255,0.3), rgba(76,194,255,0) 66%)'
        }}
      />

      {/* fine grid, slowly panning for depth */}
      <div
        className="absolute inset-0 grid-texture opacity-70"
        style={{ maskImage: 'radial-gradient(ellipse 90% 70% at 50% 0%, black 20%, transparent 78%)' }}
      />

      {/* horizon glow at the top of the viewport */}
      <div
        className="absolute inset-x-0 top-0 h-64"
        style={{ background: 'linear-gradient(180deg, rgba(122,140,255,0.09), transparent 72%)' }}
      />

      {/* vignette + grain */}
      <div
        className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse 120% 90% at 50% 0%, transparent 40%, rgba(0,0,0,0.55) 100%)' }}
      />
      <div className="noise absolute inset-0" />

      {still && (
        <style>{`.animate-aurora-a,.animate-aurora-b,.animate-aurora-c{animation:none}`}</style>
      )}
    </div>
  )
}