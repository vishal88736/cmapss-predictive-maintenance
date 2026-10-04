/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#09141e', foreground: '#e7f0f4', surface: '#0e1e2a',
        border: '#29404f', primary: '#78cbdb', muted: '#839eac', warning: '#d8a56a'
      },
      fontFamily: {
        display: ['Space Grotesk', 'sans-serif'],
        mono: ['IBM Plex Mono', 'monospace']
      }
    }
  },
  plugins: []
}
