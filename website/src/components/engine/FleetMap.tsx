import { useMemo, useState } from 'react';
import { ArrowUpDown, Search } from 'lucide-react';
import { statusColor } from '../../data/generator';
import { api } from '../../services/api';
import { useApp } from '../../store/AppContext';
import { FleetRiskMap } from './FleetRiskMap';

export function FleetMap() {
  const { setSelectedEngineId, openTwin, fleetQuery, setFleetQuery, fleetStatus, setFleetStatus } = useApp();
  const [mode, setMode] = useState<'map' | 'list'>('map');
  const [sort, setSort] = useState<'rul' | 'risk' | 'health'>('rul');
  const engines = useMemo(() => api.getEngines(), []);

  const filtered = engines
    .filter((e) => (fleetStatus === 'ALL' ? true : e.status === fleetStatus))
    .filter((e) => {
      const q = fleetQuery.trim().toLowerCase();
      if (!q) return true;
      return (`engine #${String(e.id).padStart(3, '0')} ${e.id} ${e.status}`.toLowerCase().includes(q));
    })
    .sort((a, b) => (sort === 'rul' ? a.rul - b.rul : sort === 'risk' ? b.risk - a.risk : a.health - b.health));

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2 rounded-lg border border-line bg-raised px-3 py-1.5">
          <Search size={14} className="text-muted2" />
          <input value={fleetQuery} onChange={(e) => setFleetQuery(e.target.value)} placeholder="Inline filter — e.g. “24”, “critical”…" className="w-[210px] bg-transparent text-[13px] outline-none placeholder:text-muted2" aria-label="Filter fleet" />
        </div>
        {['ALL', 'NOMINAL', 'EARLY', 'WARNING', 'CRITICAL'].map((s) => (
          <button key={s} onClick={() => setFleetStatus(s)} aria-pressed={fleetStatus === s}
            className={`rounded-full border px-3 py-1 font-mono text-[11px] ${fleetStatus === s ? 'border-white/40 bg-white/10 text-white' : 'border-line text-muted2 hover:text-white'}`}>{s}</button>
        ))}
        <div className="ml-auto flex rounded-lg border border-line p-0.5 text-[12px]">
          <button onClick={() => setMode('map')} className={`rounded-md px-3 py-1 ${mode === 'map' ? 'bg-white/10 text-white' : 'text-muted2'}`} aria-pressed={mode === 'map'}>Map</button>
          <button onClick={() => setMode('list')} className={`rounded-md px-3 py-1 ${mode === 'list' ? 'bg-white/10 text-white' : 'text-muted2'}`} aria-pressed={mode === 'list'}>List view</button>
        </div>
      </div>

      {mode === 'map' ? (
        <div className="surface rounded-2xl p-4">
          <FleetRiskMap />
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {filtered.slice(0, 8).map((e) => (
              <button key={e.id} onClick={() => setSelectedEngineId(e.id)} className="rounded-xl border border-white/[0.07] bg-raised p-2.5 text-left transition hover:border-white/20" aria-label={`Inspect ${e.label}`}>
                <div className="flex items-center gap-1.5">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: statusColor(e.status) }} aria-hidden />
                  <span className="font-mono text-[11px] font-bold">#{String(e.id).padStart(3, '0')}</span>
                  <span className="ml-auto font-mono text-[10px] text-muted2">RUL {e.rul}</span>
                </div>
                <div className="mt-1 text-[11px] text-muted2">Risk {e.risk}% · Health {e.health}%</div>
              </button>
            ))}
          </div>
          <div className="mt-2 font-mono text-[10px] text-muted2">Showing {filtered.length} / 100 engines · hover node for RUL · click for Engine Explorer</div>
        </div>
      ) : (
        <div className="surface overflow-hidden rounded-2xl" role="table" aria-label="Fleet list">
          <div className="flex items-center gap-2 border-b border-line bg-raised px-4 py-2 font-mono text-[10px] tracking-widest text-muted2">
            <span className="w-24">ENGINE</span><span className="w-20">STATUS</span>
            <button className="flex w-16 items-center gap-1 hover:text-white" onClick={() => setSort('rul')}>RUL<ArrowUpDown size={11} /></button>
            <button className="hidden w-16 items-center gap-1 hover:text-white sm:flex" onClick={() => setSort('risk')}>RISK<ArrowUpDown size={11} /></button>
            <button className="hidden w-20 items-center gap-1 hover:text-white md:flex" onClick={() => setSort('health')}>HEALTH<ArrowUpDown size={11} /></button>
            <span className="ml-auto">ACTION</span>
          </div>
          <div className="max-h-[480px] overflow-y-auto">
            {filtered.map((e) => (
              <div key={e.id} className="flex items-center gap-2 border-b border-white/[0.04] px-4 py-2 text-[13px] hover:bg-white/[0.02]" role="row">
                <span className="w-24 font-mono font-bold">#{String(e.id).padStart(3, '0')}</span>
                <span className="flex w-20 items-center gap-1.5 text-[12px]" style={{ color: statusColor(e.status) }}>
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: statusColor(e.status) }} aria-hidden />{e.status}
                </span>
                <span className="w-16 font-mono tabular-nums">{e.rul}</span>
                <span className="hidden w-16 font-mono tabular-nums sm:block">{e.risk}%</span>
                <span className="hidden w-20 font-mono tabular-nums md:block">{e.health}%</span>
                <span className="ml-auto flex gap-1.5">
                  <button onClick={() => setSelectedEngineId(e.id)} className="rounded-md border border-line px-2 py-1 text-[11px] text-muted2 hover:text-white">Inspect</button>
                  <button onClick={() => openTwin(e.id)} className="rounded-md bg-white px-2 py-1 text-[11px] font-bold text-black">Twin</button>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
