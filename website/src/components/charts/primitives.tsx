import React, { useMemo } from 'react';

export function pathFrom(values: number[], w: number, h: number, pad = 4): string {
  if (!values.length) return '';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values
    .map((v, i) => {
      const x = pad + (i / Math.max(1, values.length - 1)) * (w - pad * 2);
      const y = h - pad - ((v - min) / span) * (h - pad * 2);
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
}

export function Sparkline({ values, width = 120, height = 32, stroke = '#22D3EE', fill = true, dashed = false }: {
  values: number[]; width?: number; height?: number; stroke?: string; fill?: boolean; dashed?: boolean;
}) {
  const d = useMemo(() => pathFrom(values, width, height), [values, width, height]);
  const id = useMemo(() => `g${Math.floor(Math.random() * 1e9)}`, []);
  return (
    <svg width={width} height={height} className="overflow-visible" aria-hidden>
      {fill && (
        <>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.35} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <path d={`${d} L${width - 4},${height - 4} L4,${height - 4} Z`} fill={`url(#${id})`} stroke="none" />
        </>
      )}
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.6} strokeDasharray={dashed ? '5 4' : undefined} strokeLinecap="round" />
    </svg>
  );
}

export function RadialGauge({ value, size = 148, stroke = '#34D399', label }: { value: number; size?: number; stroke?: string; label?: string }) {
  const r = (size - 18) / 2;
  const c = 2 * Math.PI * r;
  const off = c - (value / 100) * c;
  return (
    <div className="relative inline-flex items-center justify-center" role="img" aria-label={`${label ?? 'Score'} ${value} percent`}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={10} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={stroke} strokeWidth={10} strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={off} style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.32,0.72,0,1)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <div className="text-4xl font-extrabold tracking-tight tabular-nums">{value}<span className="text-base font-semibold text-muted2">%</span></div>
        {label && <div className="text-[11px] tracking-[0.18em] text-muted2 font-semibold mt-1">{label}</div>}
      </div>
    </div>
  );
}

export function Bars({ values, color = '#22D3EE', height = 56 }: { values: number[]; color?: string; height?: number }) {
  const max = Math.max(...values, 1);
  return (
    <div className="flex items-end gap-[3px]" style={{ height }} aria-hidden>
      {values.map((v, i) => (
        <div key={i} className="flex-1 rounded-sm" style={{ height: `${(v / max) * 100}%`, background: color, opacity: 0.35 + (v / max) * 0.65 }} />
      ))}
    </div>
  );
}

export function Donut({ segments, size = 96 }: { segments: { value: number; color: string; label: string }[]; size?: number }) {
  const total = segments.reduce((s, x) => s + x.value, 0) || 1;
  let acc = 0;
  const r = (size - 14) / 2;
  const c = 2 * Math.PI * r;
  return (
    <svg width={size} height={size} className="-rotate-90" role="img" aria-label="Distribution chart">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={12} />
      {segments.map((s, i) => {
        const frac = s.value / total;
        const dash = frac * c;
        const off = -(acc / total) * c;
        acc += s.value;
        return <circle key={i} cx={size / 2} cy={size / 2} r={r} fill="none" stroke={s.color} strokeWidth={12} strokeDasharray={`${dash} ${c - dash}`} strokeDashoffset={off} strokeLinecap="butt" />;
      })}
    </svg>
  );
}
