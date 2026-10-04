import { useEffect, useState } from 'react'
export type Theme = 'light' | 'dark' | 'system'

function initialTheme(): Theme {
  try { const stored = localStorage.getItem('cmapss-theme'); if (stored === 'light' || stored === 'dark' || stored === 'system') return stored }
  catch { /* Theme selection also works without persistent browser storage. */ }
  return 'system'
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  const [systemDark, setSystemDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches)
  const resolved = theme === 'system' ? systemDark ? 'dark' : 'light' : theme
  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)')
    const update = () => setSystemDark(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = resolved
    document.documentElement.style.colorScheme = resolved
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#111820' : '#f6f5f1')
    try { localStorage.setItem('cmapss-theme', theme) } catch { /* Session-only fallback. */ }
  }, [theme, resolved])
  return { theme, setTheme, resolved }
}
