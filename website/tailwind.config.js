/** @type {import('tailwindcss').Config} */

// AeroGuard AI — design tokens
// Layered "deep space" surfaces, aurora accent spectrum, glass + glow elevation.
export const palette = {
  bg: '#06070A',
  s1: '#0B0D12',
  s2: '#11141A',
  s3: '#181C24',
  s4: '#21262F',
  line: 'rgba(255,255,255,0.07)',
  line2: 'rgba(255,255,255,0.12)',
  ink1: '#F2F4F8',
  ink2: '#A2A9B8',
  ink3: '#6A7280',
  accent: '#7A8CFF',
  violet: '#A78BFA',
  cyan: '#4CC2FF',
  ok: '#3DDC97',
  warn: '#FFC14D',
  crit: '#FF5D6C'
}

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: palette.bg,
        surface: {
          0: palette.s1,
          1: palette.s1,
          2: palette.s2,
          3: palette.s3,
          4: palette.s4,
          DEFAULT: palette.s1
        },
        line: palette.line,
        'line-2': palette.line2,
        ink: {
          1: palette.ink1,
          2: palette.ink2,
          3: palette.ink3
        },
        accent: palette.accent,
        violet: palette.violet,
        cyan: palette.cyan,
        ok: palette.ok,
        warn: palette.warn,
        crit: palette.crit,
        info: palette.cyan
      },
      fontFamily: {
        ui: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        display: ['"Space Grotesk"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace']
      },
      borderRadius: {
        '4xl': '2rem',
        '5xl': '2.75rem'
      },
      boxShadow: {
        /* layered "card" elevation: top hairline + ambient drop + accent bloom */
        card: 'inset 0 1px 0 0 rgba(255,255,255,0.045), 0 1px 2px -1px rgba(0,0,0,0.6), 0 12px 32px -18px rgba(0,0,0,0.85)',
        'card-hover': 'inset 0 1px 0 0 rgba(255,255,255,0.07), 0 2px 4px -2px rgba(0,0,0,0.6), 0 18px 44px -20px rgba(0,0,0,0.9)',
        pop: 'inset 0 1px 0 0 rgba(255,255,255,0.06), 0 32px 80px -20px rgba(0,0,0,0.92), 0 0 0 1px rgba(255,255,255,0.07)',
        rail: '0 24px 60px -28px rgba(0,0,0,0.95), inset 0 1px 0 0 rgba(255,255,255,0.05)',
        glow: '0 0 0 1px rgba(122,140,255,0.35), 0 0 28px -6px rgba(122,140,255,0.45)',
        'glow-ok': '0 0 0 1px rgba(61,220,151,0.35), 0 0 26px -8px rgba(61,220,151,0.5)',
        'glow-warn': '0 0 0 1px rgba(255,193,77,0.35), 0 0 26px -8px rgba(255,193,77,0.45)',
        'glow-crit': '0 0 0 1px rgba(255,93,108,0.4), 0 0 30px -6px rgba(255,93,108,0.5)'
      },
      transitionTimingFunction: {
        spring: 'cubic-bezier(0.34, 1.56, 0.64, 1)',
        swift: 'cubic-bezier(0.22, 1, 0.36, 1)'
      },
      keyframes: {
        'fade-in': { from: { opacity: '0' }, to: { opacity: '1' } },
        'rise-in': { from: { opacity: '0', transform: 'translateY(12px) scale(0.995)' }, to: { opacity: '1', transform: 'translateY(0) scale(1)' } },
        'slide-left': { from: { opacity: '0', transform: 'translateX(28px) scale(0.97)' }, to: { opacity: '1', transform: 'translateX(0) scale(1)' } },
        'scale-in': { from: { opacity: '0', transform: 'scale(0.94)' }, to: { opacity: '1', transform: 'scale(1)' } },
        'pulse-dot': { '0%,100%': { opacity: '1', transform: 'scale(1)' }, '50%': { opacity: '0.35', transform: 'scale(0.82)' } },
        'draw-line': { from: { strokeDashoffset: '1' }, to: { strokeDashoffset: '0' } },
        'shimmer': { from: { backgroundPosition: '200% 0' }, to: { backgroundPosition: '-200% 0' } },
        'blink': { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.2' } },
        /* slow drifting aurora blobs behind the app */
        'aurora-a': {
          '0%,100%': { transform: 'translate3d(-8%, -4%, 0) scale(1)' },
          '50%': { transform: 'translate3d(10%, 8%, 0) scale(1.18)' }
        },
        'aurora-b': {
          '0%,100%': { transform: 'translate3d(6%, 6%, 0) scale(1.1)' },
          '50%': { transform: 'translate3d(-12%, -8%, 0) scale(0.92)' }
        },
        'aurora-c': {
          '0%,100%': { transform: 'translate3d(0, 0, 0) scale(1)' },
          '50%': { transform: 'translate3d(-6%, 10%, 0) scale(1.22)' }
        },
        'grid-pan': {
          from: { backgroundPosition: '0 0' },
          to: { backgroundPosition: '56px 56px' }
        },
        /* specular sweep for primary CTAs and hover cards */
        sheen: {
          '0%': { transform: 'translateX(-120%) skewX(-18deg)' },
          '100%': { transform: 'translateX(320%) skewX(-18deg)' }
        },
        'gradient-pan': {
          '0%,100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' }
        },
        'border-spin': {
          from: { '--angle': '0deg' },
          to: { '--angle': '360deg' }
        }
      },
      animation: {
        'fade-in': 'fade-in 0.25s ease-out both',
        'rise-in': 'rise-in 0.45s cubic-bezier(0.22,1,0.36,1) both',
        'slide-left': 'slide-left 0.35s cubic-bezier(0.34,1.56,0.64,1) both',
        'scale-in': 'scale-in 0.22s cubic-bezier(0.22,1,0.36,1) both',
        'pulse-dot': 'pulse-dot 2.2s ease-in-out infinite',
        'blink': 'blink 1.2s steps(2) infinite',
        'aurora-a': 'aurora-a 26s ease-in-out infinite',
        'aurora-b': 'aurora-b 34s ease-in-out infinite',
        'aurora-c': 'aurora-c 44s ease-in-out infinite',
        'grid-pan': 'grid-pan 24s linear infinite',
        sheen: 'sheen 0.9s cubic-bezier(0.22,1,0.36,1)',
        'gradient-pan': 'gradient-pan 7s ease infinite'
      }
    }
  },
  plugins: []
}
