/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)', foreground: 'var(--text)', surface: 'var(--surface)',
        border: 'var(--border)', primary: 'var(--primary)', muted: 'var(--text-muted)', warning: 'var(--warning)'
      },
      fontFamily: {
        display: ['Manrope', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace']
      }
    }
  },
  plugins: []
}
