/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        void: '#08090B',
        panel: '#0E1013',
        raised: '#14171B',
        edge: '#1F242B',
        line: 'rgba(255,255,255,0.08)',
        success: '#34D399',
        warning: '#FBBF24',
        critical: '#F87171',
        info: '#22D3EE',
        muted2: '#8B949E',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(255,255,255,0.06), 0 18px 50px -20px rgba(0,0,0,0.8)',
        card: '0 12px 40px -18px rgba(0,0,0,0.8)',
      },
      animation: {
        'pulse-dot': 'pulseDot 1.8s ease-in-out infinite',
        'draw-line': 'drawLine 1.4s ease-out forwards',
        'slide-in': 'slideIn 0.28s cubic-bezier(0.32,0.72,0,1) forwards',
        'fade-up': 'fadeUp 0.35s ease-out forwards',
        'scan': 'scan 3.2s linear infinite',
      },
      keyframes: {
        pulseDot: {
          '0%,100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.45', transform: 'scale(0.82)' },
        },
        drawLine: {
          from: { strokeDashoffset: '1200' },
          to: { strokeDashoffset: '0' },
        },
        slideIn: {
          from: { transform: 'translateX(32px)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
        fadeUp: {
          from: { transform: 'translateY(10px)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
        scan: {
          from: { transform: 'translateX(-100%)' },
          to: { transform: 'translateX(220%)' },
        },
      },
    },
  },
  plugins: [],
}
