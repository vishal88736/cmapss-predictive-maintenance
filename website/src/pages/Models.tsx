import { useMemo, useState } from 'react'
import { Sparkles, Brain, ChevronLeft, ChevronRight, Radio, Zap, Database, Layers } from 'lucide-react'
import PageHeader from '@/components/shell/PageHeader'
import AICommandBar from '@/components/command/AICommandBar'
import { ScatterChart, Histogram, RangeBars, AttributionBars, DualLine, MultiLine } from '@/components/charts/modelCharts'
import { getModels, getModelMetrics } from '@/data/db'
import { SENSOR_MAP } from '@/data/sensors'
import { cn } from '@/lib/utils'

export default function Models() {
  const models = useMemo(() => getModels(), [])
  const [modelId, setModelId] = useState(models[0].id)
  const model = models.find((m) => m.id === modelId)!
  const metrics = getModelMetrics(modelId)!
  const nextModel = models[(models.findIndex((m) => m.id === modelId) + 1) % models.length]
  const prevModel = models[(models.findIndex((m) => m.id === modelId) - 1 + models.length) % models.length]

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        eyebrow="ML Observability"
        title="Model Insights"
        sub="LSTM RUL Predictor v2.4 · 100 units · piecewise-linear targets (clip 130) · bidirectional 2×128"
        meta={
          <span className="chip !py-0.5 !text-[9px] border-ok/30 bg-ok/10 text-ok">
            ACTIVE
          </span>
        }
        actions={
          <div className="flex gap-1.5" role="group" aria-label="Select model">
            {models.map((m) => (
              <button key={m.id} onClick={() => setModelId(m.id)} className={cn('chip', m.id === modelId && 'chip-active')} aria-pressed={m.id === modelId}>
                {m.name}
              </button>
            ))}
          </div>
        }
      >
        <div className="flex items-center justify-between text-[10px] text-ink-3">
          <span>{model.kind} · {model.params} params · {model.latency} ms inference</span>
          <span>Trained on {model.trainedOn}</span>
        </div>
      </PageHeader>

      <div className="flex-1 overflow-y-auto px-4 py-3.5 md:px-6">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <MetricTile label="RMSE" value={metrics.rmse.toFixed(1)} unit="cycles" tone="info" />
          <MetricTile label="MAE" value={metrics.mae.toFixed(1)} unit="cycles" tone="info" />
          <MetricTile label="R²" value={metrics.r2.toFixed(2)} unit="" tone="ok" />
          <MetricTile label="Avg Conf." value={Math.round(metrics.avgConfidence * 100)} unit="%" tone="accent" />
        </div>

        <div className="mt-4 grid gap-3 xl:grid-cols-2">
          <section className="card p-4" aria-label="Actual vs Predicted">
            <span className="panel-title">Actual vs Predicted</span>
            <div className="mt-3 h-[220px]">
              <ScatterChart points={metrics.avp.map((p) => ({ x: p.actual, y: p.predicted }))} xLabel="Actual RUL" yLabel="Predicted RUL" color="var(--accent)" />
            </div>
            <p className="mt-1 text-[9.5px] text-ink-3">Diagonal = perfect prediction. Spread below diagonal shows conservative estimates for high-RUL engines.</p>
          </section>

          <section className="card p-4" aria-label="Residual distribution">
            <span className="panel-title">Residual distribution</span>
            <div className="mt-3 h-[220px]">
              <Histogram bins={metrics.residuals} color="var(--warn)" xLabel="Residual (cycles)" />
            </div>
            <p className="mt-1 text-[9.5px] text-ink-3">Slight negative skew — model tends to slightly over-estimate remaining life on severely degraded units.</p>
          </section>

          <section className="card p-4" aria-label="Error by RUL range">
            <span className="panel-title">Error by RUL range</span>
            <div className="mt-3 h-[220px]">
              <RangeBars bars={metrics.errorByRange} />
            </div>
            <p className="mt-1 text-[9.5px] text-ink-3">MAE rises with RUL range — piecewise clipping at 130 cycles compresses the training signal for healthy engines.</p>
          </section>

          <section className="card p-4" aria-label="Feature importance">
            <span className="panel-title">Feature importance (global)</span>
            <div className="mt-3 space-y-1">
              <AttributionBars items={metrics.featureImportance} nameOf={(id) => SENSOR_MAP[id]?.short ?? id} compact />
            </div>
          </section>
        </div>

        <div className="mt-4 grid gap-3 xl:grid-cols-2">
          <section className="card p-4" aria-label="Model drift over 30 days">
            <span className="panel-title">Prediction drift (30 days)</span>
            <div className="mt-3 h-[200px]">
              <DualLine
                series={[
                  { pts: metrics.drift.map((d, i) => ({ x: i, y: d.rmse })), color: 'var(--crit)', dashed: false },
                  { pts: metrics.drift.map((d, i) => ({ x: i, y: d.confidence * 20 })), color: 'var(--info)', dashed: true }
                ]}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-[9.5px] text-ink-3">
              <span className="flex items-center gap-1"><span className="h-[3px] w-5 rounded bg-crit" /> RMSE (cycles)</span>
              <span className="flex items-center gap-1"><span className="h-[3px] w-5 rounded border-t-2 border-dashed border-info" /> Confidence ×20</span>
            </div>
          </section>

          <section className="card p-4" aria-label="Confidence distribution">
            <span className="panel-title">Confidence distribution</span>
            <div className="mt-3 h-[200px]">
              <Histogram bins={metrics.confidenceDist} color="var(--info)" xLabel="Confidence" />
            </div>
            <p className="mt-1 text-[9.5px] text-ink-3">Peak at 0.87 — the model is well-calibrated on the validation split. Long tail toward 0.6 for ambiguous cases.</p>
          </section>

          <section className="card p-4" aria-label="Training curve">
            <span className="panel-title">Training curve</span>
            <div className="mt-3 h-[200px]">
              <DualLine
                series={[
                  { pts: metrics.trainingCurve.map((d) => ({ x: d.epoch, y: d.loss })), color: 'var(--accent)', dashed: false },
                  { pts: metrics.trainingCurve.map((d) => ({ x: d.epoch, y: d.valLoss })), color: 'var(--crit)', dashed: true }
                ]}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-[9.5px] text-ink-3">
              <span className="flex items-center gap-1"><span className="h-[3px] w-5 rounded bg-accent" /> Train loss</span>
              <span className="flex items-center gap-1"><span className="h-[3px] w-5 rounded border-t-2 border-dashed border-crit" /> Val loss</span>
            </div>
          </section>

          <section className="card p-4" aria-label="Model metadata">
            <span className="panel-title">Model metadata</span>
            <div className="mt-3 grid gap-2 text-[11px]">
              {[
                ['Architecture', model.kind],
                ['Training data', model.trainedOn],
                ['Parameters', model.params],
                ['Inference latency', `${model.latency} ms`],
                ['Status', model.status.toUpperCase()],
                ['Description', model.description]
              ].map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="w-[120px] shrink-0 text-ink-3">{k}</span>
                  <span className="text-ink-2">{v}</span>
                </div>
              ))}
            </div>
            <p className="mt-3 text-[9.5px] leading-relaxed text-ink-3">
              Point the <code className="rounded bg-surface-3 px-1 font-mono text-[10px] text-ink-2">services/api.ts</code> layer at a live backend (model serving + telemetry ingestion) to replace this sandbox.
            </p>
          </section>
        </div>

        <div className="mt-5">
          <AICommandBar contextLabel="model observability" />
        </div>
      </div>
    </div>
  )
}

function MetricTile({ label, value, unit, tone }: { label: string; value: string | number; unit: string; tone: 'info' | 'ok' | 'accent' }) {
  const colors: Record<string, string> = { info: 'var(--info)', ok: 'var(--ok)', accent: 'var(--accent)' }
  return (
    <div className="card p-4">
      <div className="eyebrow">{label}</div>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="stat-num font-display text-[28px] font-bold" style={{ color: colors[tone] }}>{value}</span>
        <span className="text-[11px] text-ink-3">{unit}</span>
      </div>
    </div>
  )
}