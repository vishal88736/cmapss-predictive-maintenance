import { createContext, useContext, useEffect, useRef, useSyncExternalStore } from 'react'

const motionQuery = '(prefers-reduced-motion: reduce)'
const subscribe = (callback: () => void) => {
  const media = window.matchMedia(motionQuery)
  media.addEventListener('change', callback)
  return () => media.removeEventListener('change', callback)
}

export function useReducedMotion() {
  return useSyncExternalStore(subscribe, () => window.matchMedia(motionQuery).matches, () => true)
}

export const MotionContext = createContext(false)
export const useMotion = () => useContext(MotionContext)

/** Reveal sections once as they enter the viewport; data updates stay immediate. */
export function MotionPage({ children, enabled }: { children: React.ReactNode; enabled: boolean }) {
  const root = useRef<HTMLElement>(null)

  useEffect(() => {
    const sections = Array.from(root.current?.children || [])
    if (!enabled || !('IntersectionObserver' in window)) {
      sections.forEach(section => section.removeAttribute('data-reveal'))
      return
    }

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return
        entry.target.setAttribute('data-reveal', 'visible')
        observer.unobserve(entry.target)
      })
    }, { threshold: 0.06 })

    sections.forEach(section => {
      // Above-the-fold content uses the page entrance; only lower sections wait.
      if (section.getBoundingClientRect().top >= window.innerHeight) {
        section.setAttribute('data-reveal', 'pending')
        observer.observe(section)
      }
    })

    return () => observer.disconnect()
  }, [enabled])

  return <main ref={root} id="main-content" className="content">{children}</main>
}
