/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#08090B',
        surface: {
          1: '#0E1013',
          2: '#14171B',
          3: '#1B1F25',
          DEFAULT: '#0E1013'
        },
        line: 'rgba(255,255,255,0.07)',
        ink: {
          1: '#E9EBEE',
          2: '#9AA1AB',
          3: '#5E6670'
        },
        accent: '#7C8CFF',
        ok: '#34D399',
        warn: '#F5B84A',
        crit: '#F0625D',
        info: '#4CC2FF'
      },
      fontFamily: {
        ui: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        display: ['"Space Grotesk"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace']
      },
      boxShadow: {
        card: '0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.6)',
        pop: '0 24px 64px -16px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.06)',
        rail: '8px 0 32px -16px rgba(0,0,0,0.7)'
      },
      keyframes: {
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'rise-in': { from: { opacity: '0', transform: 'translateY(10px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        'slide-left': { from: { opacity: '0', transform: 'translateX(24px)' }, to: { opacity: '1', transform: 'translateX(0)' } },
        'pulse-dot': { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.35' } },
        'draw-line': { from: { strokeDashoffset: '1' }, to: { strokeDashoffset: '0' } },
        'shimmer': { from: { backgroundPosition: '200% 0' }, to: { backgroundPosition: '-200% 0' } },
        'blink': { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.2' } }
      },
      animation: {
        'fade-in': 'fade-in 0.25s ease-out both',
        'rise-in': 'rise-in 0.35s cubic-bezier(0.22,1,0.36,1) both',
        'slide-left': 'slide-left 0.3s cubic-bezier(0.22,1,0.36,1) both',
        'pulse-dot': 'pulse-dot 2.2s ease-in-out infinite',
        'blink': 'blink 1.2s steps(2) infinite'
      }
    }
  },
  plugins: []
}
