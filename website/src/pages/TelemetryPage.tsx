import { useApp } from '../store/AppContext';
import { TelemetryCanvas } from '../components/telemetry/TelemetryCanvas';
import { AIInsight } from '../components/ai/AIInsight';
import { api } from '../services/api';

export function TelemetryPage() {
  const { twinEngineId, setTwinEngineId } = useApp();
  return (
    <div className="grid gap-3 pb-24 lg:grid-cols-12">
      <div className="lg:col-span-8">
        <div className="mb-3 flex items-center gap-2">
          <span className="font-mono text-[11px] text-muted2">ENGINE</span>
          <select value={twinEngineId} onChange={(e) => setTwinEngineId(Number(e.target.value))} className="rounded-lg border border-line bg-raised px-2.5 py-1.5 font-mono text-[12px] font-bold" aria-label="Telemetry engine">
            {api.getEngines().map((e) => <option key={e.id} value={e.id}>#{String(e.id).padStart(3, '0')} · {e.status}</option>)}
          </select>
        </div>
        <TelemetryCanvas engineId={twinEngineId} />
      </div>
      <div className="lg:col-span-4"><AIInsight engineId={twinEngineId} /></div>
    </div>
  );
}
