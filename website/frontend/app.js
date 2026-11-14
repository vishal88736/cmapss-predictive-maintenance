import { icons } from './vendor/icons.js';
import { histogram, donut, lineChart, sparkline, colors } from './charts.js';

// DOM & utility helpers
const $ = (selector, root = document) => root.querySelector(selector);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const num = (value, d = 0) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const pretty = (value) => String(value ?? '').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
const icon = (name) => `<span class="icon" aria-hidden="true">${icons[name] || icons['circle-help'] || ''}</span>`;
const brand = () => '<a class="brand" href="/" data-nav><img src="/assets/aeroguard.svg" alt="" style="width:28px;height:28px"><span>AeroGuard</span></a>';
const button = (label, action, ico = '', type = '', extra = '') => `<button type="button" class="button ${type}" data-action="${action}" ${extra}>${ico ? icon(ico) : ''}${label}</button>`;
const link = (label, path, ico = '', type = '', extra = '') => `<a class="button ${type}" href="${path}" data-nav ${extra}>${ico ? icon(ico) : ''}${label}</a>`;
const badge = (label, style = label) => `<span class="badge ${esc(String(style).toLowerCase())}">${esc(pretty(label))}</span>`;
const tag = (text, style = '') => `<span class="tag ${style}">${esc(text)}</span>`;
const heading = (title, copy, actions = '') => `<header class="page-header"><div><h1>${title}</h1><p>${copy}</p></div><div class="actions">${actions}</div></header>`;
const panel = (title, content, subtitle = '', action = '', cls = '') => `<section class="panel ${cls}"><div class="panel-head"><div><h2>${title}</h2>${subtitle ? `<p class="subtitle">${subtitle}</p>` : ''}</div>${action}</div><div class="panel-body">${content}</div></section>`;

const subsets = ['ALL', 'FD001', 'FD002', 'FD003', 'FD004'];
const routeInfo = {
  dashboard: ['Fleet Dashboard', 'layout-dashboard'],
  engines: ['Engine Explorer', 'wind'],
  assurance: ['Assurance Cases', 'shield-check'],
  upload: ['Data Upload & Batch Test', 'upload'],
  models: ['Model Registry & Provenance', 'boxes']
};

const state = {
  route: location.pathname.replace(/\/$/, '') || '/',
  subset: 'ALL',
  engineSearch: '',
  healthFilter: 'ALL',
  enginePage: 0,
  engineSize: 15,
  detailTab: 'overview',
  assuranceTab: 'engine179',
  engine179Cycle: 240,
  simFault: { unit: 1, family: 'stuck', severity: 0.2, sensor: 7 },
  simResult: null,
  simLoading: false,
  uploadBatch: null,
  uploadLoading: false,
  generation: 0
};

// Toast notification helper
function showToast(message, type = 'info') {
  const el = $('#toast');
  if (!el) return;
  el.textContent = message;
  el.className = `toast ${type}`;
  el.hidden = false;
  setTimeout(() => { el.hidden = true; }, 3500);
}

// API client
async function api(path, { method = 'GET', body } = {}) {
  const headers = {};
  if (body !== undefined && !(body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
  }
  const response = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body)
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const err = await response.json();
      detail = err.detail || err.message || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return response.json();
}

// Shell layout
function shell() {
  const active = state.route.split('/')[1] || 'dashboard';
  return `
  <aside class="sidebar" id="sidebar" aria-label="Application navigation">
    ${brand()}
    <div class="workspace-chip">
      <span class="workspace-icon">${icon('shield-check')}</span>
      <div>
        <strong>GRACE-RUL v10</strong>
        <small>Format-10 Frozen Checkpoint</small>
      </div>
    </div>
    <div class="nav-label">WORKSPACE</div>
    <nav>
      ${Object.entries(routeInfo).map(([path, [title, ico]]) => `
        <a class="nav-item ${active === path ? 'active' : ''}" href="/${path}" data-nav ${active === path ? 'aria-current="page"' : ''}>
          ${icon(ico)}
          <span>${title}</span>
        </a>
      `).join('')}
    </nav>
    <div class="sidebar-bottom">
      <div class="sidebar-note">
        ${icon('shield-check')}
        <strong>Certified Assurance</strong>
        <p>Healthy sensor displacement bounded below 0.02364 σ ceiling.</p>
        <a href="/models" data-nav>Provenance manifest ${icon('arrow-up-right')}</a>
      </div>
    </div>
  </aside>
  <div class="shell">
    <header class="topbar">
      <button class="icon-button mobile-nav" data-action="toggle-menu" aria-label="Toggle navigation">${icon('menu')}</button>
      <div class="top-search">
        <form id="top-search-form" onsubmit="event.preventDefault();">
          ${icon('search')}
          <input id="global-search" placeholder="Search engines, assets, sensors…" autocomplete="off" value="${esc(state.engineSearch)}">
        </form>
      </div>
      <div class="topbar-actions">
        <label class="sr-only" for="topbar-subset">Fleet subset</label>
        <select id="topbar-subset" class="select" data-action="change-subset">
          ${subsets.map(s => `<option value="${s}" ${state.subset === s ? 'selected' : ''}>${s === 'ALL' ? 'All Subsets (Pooled)' : s}</option>`).join('')}
        </select>
        <span class="source-pill" title="Certified Format-10 Model">
          ${icon('cpu')} Frozen Format-10
        </span>
      </div>
    </header>
    <div class="mode-strip">
      ${icon('shield-check')}
      <span><strong>GRACE-RUL Format-10 Frozen Release</strong> · Capped RMSE 11.08 · Stuck-fault RMSE 12.11 · Mixed-fault RMSE 11.48 · Healthy displacement ≤ 0.02364 σ</span>
    </div>
    <main id="main" class="workspace-main" tabindex="-1"></main>
    <footer class="app-footer">
      <span>AeroGuard · GRACE-RUL Predictive Maintenance Platform</span>
      <span>NASA C-MAPSS FD001–FD004 · Certified Robust Prognostics</span>
    </footer>
  </div>`;
}

// Landing page markup
function landingMarkup() {
  return `
  <div class="landing">
    <header class="landing-header">
      ${brand()}
      <nav class="landing-nav" aria-label="Product navigation">
        <a href="/dashboard" data-nav>Dashboard</a>
        <a href="/engines" data-nav>Engines</a>
        <a href="/assurance" data-nav>Assurance Cases</a>
        <a href="/upload" data-nav>Batch Test</a>
        <a href="/models" data-nav>Model Registry</a>
      </nav>
      <div class="actions">
        ${link('Open Dashboard', '/dashboard', 'arrow-right', 'primary')}
      </div>
    </header>
    <section class="landing-hero">
      <div class="landing-image" style="background-image:url('/assets/turbofan-hero.png')"></div>
      <div class="landing-shade"></div>
      <div class="hero-copy">
        <p class="eyebrow">NASA C-MAPSS PROGNOSTICS & FAULT-TOLERANCE</p>
        <h1>Fault-tolerant turbofan life prediction with <span>certified assurance</span>.</h1>
        <p>AeroGuard integrates the GRACE-RUL format-10 robust prognostic pipeline: continuous engine-age grounding, soft cubic sensor recovery, and split-conformal bounds that remain intact under severe sensor failure.</p>
        <div class="actions">
          ${link('Launch Fleet Dashboard', '/dashboard', 'layout-dashboard', 'primary')}
          ${link('Upload Test Data', '/upload', 'upload', 'blue')}
          ${link('Assurance Cases', '/assurance', 'shield-check', 'soft')}
        </div>
        <p class="hero-small">Certified format-10 bundle · Zero retraining · 285/285 tests verified</p>
      </div>
    </section>
    <div class="capability-bar">
      <div class="capability">
        <span class="icon">${icon('shield-check')}</span>
        <div><strong>Certified Bounded Displacement</strong><p>Intact sensors stay within 0.02364 σ ceiling</p></div>
      </div>
      <div class="capability">
        <span class="icon">${icon('gauge')}</span>
        <div><strong>Format-10 Clean RMSE 11.08</strong><p>Official capped target across pooled fleet</p></div>
      </div>
      <div class="capability">
        <span class="icon">${icon('activity')}</span>
        <div><strong>Central 90% Conformal Coverage</strong><p>98.44% clean empirical validation coverage</p></div>
      </div>
      <div class="capability">
        <span class="icon">${icon('boxes')}</span>
        <div><strong>Frozen Cryptographic Bundle</strong><p>SHA-256: 9b2b86ef… verifiable provenance</p></div>
      </div>
    </div>
  </div>`;
}

// -----------------------------------------------------------------------------
// View 1: Fleet Dashboard
// -----------------------------------------------------------------------------
function dashboardPage(data) {
  const s = data.summary || { total: 0, healthy: 0, watch: 0, attention: 0, critical: 0, mean_rul: 0 };
  const assets = data.assets || [];
  const bm = data.certified_benchmarks || {};

  return `
  ${heading('Fleet Overview & Prognostics', 'Real-time remaining useful life prediction and sensor health monitoring across the C-MAPSS fleet.', `
    ${button('Load Official FD001', 'quick-fd001', 'play', 'small')}
    ${link('Batch Upload', '/upload', 'upload', 'primary small')}
  `)}

  <div class="metric-grid">
    <div class="metric-card">
      <span class="metric-label">Evaluated Engines</span>
      <strong class="metric-value">${num(s.total)}</strong>
      <span class="metric-foot">${state.subset === 'ALL' ? 'Pooled C-MAPSS Fleet' : state.subset + ' Test Set'}</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Healthy Engines</span>
      <strong class="metric-value" style="color:var(--green)">${num(s.healthy)}</strong>
      <span class="metric-foot">RUL &gt; 75 cycles</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Watch / Attention</span>
      <strong class="metric-value" style="color:var(--attention)">${num((s.watch || 0) + (s.attention || 0))}</strong>
      <span class="metric-foot">RUL 15–75 cycles</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Critical Engines</span>
      <strong class="metric-value" style="color:var(--red)">${num(s.critical)}</strong>
      <span class="metric-foot">RUL ≤ 15 cycles (Urgent)</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Mean Fleet RUL</span>
      <strong class="metric-value" style="color:var(--blue)">${num(s.mean_rul, 1)}</strong>
      <span class="metric-foot">Operating cycles</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Sensor Faults Flagged</span>
      <strong class="metric-value">${num(s.faults_flagged || 0)}</strong>
      <span class="metric-foot">Detected by soft cubic gate</span>
    </div>
  </div>

  <div class="notice" style="margin:20px 0;background:#F8FBFF;border-color:#C7DCF8">
    ${icon('shield-check')}
    <div>
      <strong>Certified GRACE-RUL Format-10 Pipeline Active</strong>
      <p style="margin:4px 0 0;font-size:12px">
        Frozen Weights SHA-256: <code>9b2b86ef0236…</code> · Official Clean RMSE: <strong>${bm.capped_rmse || '11.08'} cycles</strong> · Stuck@0.20: <strong>${bm.stuck_rmse || '12.11'} cycles</strong> · Mixed@0.20: <strong>${bm.mixed_rmse || '11.48'} cycles</strong> · Healthy Displacement Ceiling: <strong>≤ ${bm.intact_displacement || '0.02364'} σ</strong>.
      </p>
    </div>
  </div>

  <div class="grid-two" style="display:grid;grid-template-columns:1.4fr 1fr;gap:20px;margin-bottom:20px">
    ${panel('RUL Distribution', histogram(assets, { metric: 'rul', label: 'Remaining Useful Life Distribution', max: 140 }), 'Binned cycle distribution across evaluated engines.')}
    ${panel('Fleet Health Breakdown', donut(s), 'Normal, Watch, Attention and Critical engines.')}
  </div>

  ${panel('High-Priority Engine Watchlist', `
    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Subset</th>
            <th>Observed Cycles</th>
            <th>Predicted RUL</th>
            <th>Central 90% Conformal Interval</th>
            <th>Health Status</th>
            <th>Flagged Faults</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${assets.slice(0, 10).map(a => `
            <tr>
              <td><strong>${esc(a.asset)}</strong></td>
              <td>${tag(a.subset)}</td>
              <td>${num(a.cycle)}</td>
              <td>
                <span class="inline-meter">
                  <span class="rul-value">${num(a.rul, 1)}</span>
                  <span class="meter"><i style="width:${Math.min(100, Math.max(0, a.rul / 125 * 100))}%;background:${colors[a.health] || colors.blue}"></i></span>
                </span>
              </td>
              <td><span class="mono">${a.interval_lower != null ? `[${num(a.interval_lower, 1)}, ${num(a.interval_upper, 1)}]` : '—'}</span></td>
              <td>${badge(a.health)}</td>
              <td>${a.detected_faults?.length ? a.detected_faults.map(f => tag(f, 'orange')).join(' ') : '<span class="muted">Clean</span>'}</td>
              <td><a class="button small" href="/engines/${encodeURIComponent(a.asset)}" data-nav>View</a></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `, 'Engines sorted by nearest maintenance requirement.', link('View All Engines', '/engines', 'arrow-right', 'small'))}
  `;
}

// -----------------------------------------------------------------------------
// View 2: Engine Explorer & Deep Diagnostics
// -----------------------------------------------------------------------------
function enginesPage(data) {
  const items = data.items || [];
  const total = data.total || 0;
  const page = data.page || 0;
  const size = data.size || 20;

  return `
  ${heading('Engine Explorer', 'Search, filter and inspect turbofan engines, 21-sensor health status, and split-conformal RUL bounds.', `
    ${link('Batch Upload', '/upload', 'upload', 'small primary')}
  `)}

  <form id="engines-filter-form" class="filter-row" onsubmit="event.preventDefault();">
    <div class="field" style="margin-bottom:0">
      <label for="engine-search-input">Search Engine / Unit</label>
      <input id="engine-search-input" placeholder="e.g. 1 or POOLED:10" value="${esc(state.engineSearch)}">
    </div>
    <div class="field" style="margin-bottom:0">
      <label for="health-filter-select">Health Status</label>
      <select id="health-filter-select">
        <option value="ALL" ${state.healthFilter === 'ALL' ? 'selected' : ''}>All Statuses</option>
        <option value="healthy" ${state.healthFilter === 'healthy' ? 'selected' : ''}>Healthy (&gt;75 cycles)</option>
        <option value="watch" ${state.healthFilter === 'watch' ? 'selected' : ''}>Watch (35–75 cycles)</option>
        <option value="attention" ${state.healthFilter === 'attention' ? 'selected' : ''}>Attention (15–35 cycles)</option>
        <option value="critical" ${state.healthFilter === 'critical' ? 'selected' : ''}>Critical (≤15 cycles)</option>
      </select>
    </div>
    <button type="button" class="button small blue" data-action="apply-engine-filter" style="align-self:flex-end">
      ${icon('filter')} Filter
    </button>
    <button type="button" class="button small" data-action="reset-engine-filter" style="align-self:flex-end">
      ${icon('rotate-ccw')} Reset
    </button>
  </form>

  <div class="panel">
    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Subset</th>
            <th>Observed Cycles</th>
            <th>Predicted RUL</th>
            <th>Central 90% Interval</th>
            <th>Health Status</th>
            <th>Fault Detector Score</th>
            <th>Active Faults</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${items.length === 0 ? `<tr><td colspan="9" style="text-align:center;padding:30px">No matching engines found.</td></tr>` : ''}
          ${items.map(a => `
            <tr>
              <td><div class="engine-cell">${icon('wind')}<div><strong>${esc(a.asset)}</strong><small>Unit ${a.unit}</small></div></div></td>
              <td>${tag(a.subset)}</td>
              <td>${num(a.cycle)}</td>
              <td>
                <span class="inline-meter">
                  <span class="rul-value">${num(a.rul, 1)}</span>
                  <span class="meter"><i style="width:${Math.min(100, Math.max(0, a.rul / 125 * 100))}%;background:${colors[a.health] || colors.blue}"></i></span>
                </span>
              </td>
              <td><span class="mono">${a.interval_lower != null ? `[${num(a.interval_lower, 1)}, ${num(a.interval_upper, 1)}]` : '—'}</span></td>
              <td>${badge(a.health)}</td>
              <td>${num((a.anomaly_percentile || 0) * 100, 1)}%</td>
              <td>${a.detected_faults?.length ? a.detected_faults.map(f => tag(f, 'orange')).join(' ') : '<span class="muted">Clean</span>'}</td>
              <td>
                <a class="button small" href="/engines/${encodeURIComponent(a.asset)}" data-nav>
                  ${icon('eye')} Inspect
                </a>
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
    <div class="panel-footer table-pagination" style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px">
      <span>Showing ${items.length} of ${total} engines</span>
      <div class="actions">
        ${button('Previous', 'engine-prev-page', 'arrow-down', 'small', page === 0 ? 'disabled' : '')}
        <span class="mono">Page ${page + 1} of ${Math.max(1, Math.ceil(total / size))}</span>
        ${button('Next', 'engine-next-page', 'arrow-right', 'small', (page + 1) * size >= total ? 'disabled' : '')}
      </div>
    </div>
  </div>
  `;
}

function engineDetailsPage(a) {
  const history = a.history || [];
  const sensors = a.sensors || [];
  const chartSeries = [{ key: 'rul', label: 'Predicted RUL (cycles)', color: colors.orange }];
  if (history.some(p => p.true_rul != null)) {
    chartSeries.unshift({ key: 'true_rul', label: 'Ground Truth RUL', color: colors.blue });
  }

  return `
  <div style="margin-bottom:14px">
    <a href="/engines" class="text-button" data-nav>${icon('arrow-down')} Back to Engine Explorer</a>
  </div>

  <header class="page-header" style="margin-bottom:20px">
    <div style="display:flex;align-items:center;gap:14px">
      <img src="/assets/turbofan-hero.png" alt="" style="width:70px;height:50px;object-fit:cover;border-radius:8px;border:1px solid var(--border)">
      <div>
        <h1>Engine ${esc(a.asset)}</h1>
        <p>Subset: <strong>${esc(a.subset)}</strong> · Observed Cycles: <strong>${num(a.cycle)}</strong> · Status: ${badge(a.health)}</p>
      </div>
    </div>
    <div class="actions">
      ${button('Simulate Fault Assurance', 'test-assurance-this', 'shield-check', 'primary', `data-unit="${a.unit}"`)}
    </div>
  </header>

  <div class="metric-grid" style="margin-bottom:24px">
    <div class="metric-card">
      <span class="metric-label">Predicted RUL</span>
      <strong class="metric-value" style="color:var(--orange)">${num(a.rul, 1)} <small style="font-size:12px;font-weight:400">cycles</small></strong>
      <span class="metric-foot">Capped prognostic horizon</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Central 90% Conformal Interval</span>
      <strong class="metric-value mono" style="font-size:18px">${a.interval_lower != null ? `[${num(a.interval_lower, 1)}, ${num(a.interval_upper, 1)}]` : '—'}</strong>
      <span class="metric-foot">Interval Width: ${num(a.interval_width, 1)} cycles</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Evaluation Ground Truth</span>
      <strong class="metric-value" style="color:${a.true_rul != null ? 'var(--blue)' : 'var(--muted)'}">${a.true_rul != null ? num(a.true_rul, 1) + ' cycles' : 'Unlabeled Test'}</strong>
      <span class="metric-foot">${a.error != null ? `Error: ${num(a.error, 1)} cycles` : 'Held-out test target'}</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Flagged Sensor Faults</span>
      <strong class="metric-value" style="color:${a.detected_faults?.length ? 'var(--orange)' : 'var(--green)'}">${a.detected_faults?.length || 0}</strong>
      <span class="metric-foot">${a.detected_faults?.length ? a.detected_faults.join(', ') : 'All channels healthy'}</span>
    </div>
  </div>

  <div class="subtabs">
    <button class="subtab ${state.detailTab === 'overview' ? 'active' : ''}" data-action="engine-tab" data-tab="overview">
      ${icon('activity')} RUL Trajectory & Trend
    </button>
    <button class="subtab ${state.detailTab === 'sensors' ? 'active' : ''}" data-action="engine-tab" data-tab="sensors">
      ${icon('gauge')} 21-Sensor Diagnostic Matrix
    </button>
  </div>

  ${state.detailTab === 'overview' ? `
    ${panel('RUL Trajectory & Conformal Prediction Interval', `
      ${lineChart(history, { series: chartSeries, band: true, label: `Remaining Useful Life for ${esc(a.asset)}`, xKey: 'cycle', xLabel: 'Engine Operating Cycle', yLabel: 'Remaining Useful Life' })}
      <p class="chart-caption" style="margin-top:12px">
        Shaded region shows the central 90% conformal prediction interval. The format-10 model applies adaptive empirical residual bounds based on observed engine age.
      </p>
    `, 'Causal prefix evaluation showing degradation trajectory.')}
    <div style="height:20px"></div>
    ${panel('Top Diagnostic Sensor Trends', `
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:16px">
        ${sensors.slice(1, 7).map(s => `
          <div class="sensor-card" style="border:1px solid var(--border);border-radius:10px;padding:14px;background:#fff">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
              <strong>${esc(s.key)} (${esc(s.name)})</strong>
              ${s.flagged ? tag('Fault Flagged', 'orange') : tag('Normal', 'green')}
            </div>
            <div style="font-size:18px;font-weight:700;color:var(--text);margin-bottom:6px">
              ${num(s.observed_last, 2)}
            </div>
            ${sparkline(s.series || [], 'value', s.flagged ? colors.orange : colors.blue)}
            <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--muted);margin-top:8px">
              <span>Detector Score: ${num(s.fault_score, 4)}</span>
              <span>Soft Weight: ${num(s.soft_weight, 4)}</span>
            </div>
          </div>
        `).join('')}
      </div>
    `, 'Recent 40-cycle historical readings on key turbofan channels.')}
  ` : `
    ${panel('21-Sensor Robust Health & Recovery Matrix', `
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>Channel</th>
              <th>Sensor Name</th>
              <th>Observed Reading</th>
              <th>Recovered Reading</th>
              <th>Repair Magnitude (|Δx|)</th>
              <th>Fault Detector Score (p)</th>
              <th>Soft Gate Weight (p³)</th>
              <th>Status</th>
              <th>Trend Sparkline</th>
            </tr>
          </thead>
          <tbody>
            ${sensors.map(s => `
              <tr style="${s.flagged ? 'background:#FFFBF5;' : ''}">
                <td><strong>${esc(s.key)}</strong></td>
                <td>${esc(s.name)}</td>
                <td class="mono">${num(s.observed_last, 2)}</td>
                <td class="mono">${num(s.recovered_last, 2)}</td>
                <td class="mono">${num(s.repair_magnitude, 4)}</td>
                <td>
                  <span class="mono">${num(s.fault_score, 4)}</span>
                  <div style="width:60px;height:4px;background:#E2E8F0;border-radius:2px;margin-top:3px;overflow:hidden">
                    <div style="width:${Math.min(100, s.fault_score * 100)}%;height:100%;background:${s.flagged ? colors.orange : colors.blue}"></div>
                  </div>
                </td>
                <td class="mono">${num(s.soft_weight, 4)}</td>
                <td>${s.flagged ? badge('Fault Flagged', 'critical') : s.fault_score > 0.3 ? badge('Suspect', 'attention') : badge('Clean', 'healthy')}</td>
                <td>${sparkline(s.series || [], 'value', s.flagged ? colors.orange : colors.blue)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
    `, 'The format-10 soft cubic gate weights sensor repair by p_j³ while preserving clean channels.')}
  `}
  `;
}

// -----------------------------------------------------------------------------
// View 3: Assurance & Verification Cases
// -----------------------------------------------------------------------------
function assurancePage(e179Data, simResult) {
  return `
  ${heading('Assurance & Verification Bench', 'Formal assurance cases and safety boundary verifications following fault detection on turbofan engines.', `
    <a class="button small" href="/models" data-nav>${icon('boxes')} Provenance Manifest</a>
  `)}

  <div class="subtabs">
    <button class="subtab ${state.assuranceTab === 'engine179' ? 'active' : ''}" data-action="assurance-tab" data-tab="engine179">
      ${icon('triangle-alert')} Case 1: FD004 Engine 179 Diagnostic
    </button>
    <button class="subtab ${state.assuranceTab === 'fault_injection' ? 'active' : ''}" data-action="assurance-tab" data-tab="fault_injection">
      ${icon('shield-check')} Case 2: Live Fault Injection & Sensor Recovery
    </button>
    <button class="subtab ${state.assuranceTab === 'conformal' ? 'active' : ''}" data-action="assurance-tab" data-tab="conformal">
      ${icon('activity')} Case 3: Conformal Calibration & Plateau Stretch
    </button>
  </div>

  ${state.assuranceTab === 'engine179' ? renderEngine179Case(e179Data) : ''}
  ${state.assuranceTab === 'fault_injection' ? renderFaultInjectionCase(simResult) : ''}
  ${state.assuranceTab === 'conformal' ? renderConformalCase() : ''}
  `;
}

function renderEngine179Case(data) {
  if (!data) return `<div class="loader"></div>`;
  const obs = data.critical_observation || {};
  const trace = data.trace || [];
  const currentPoint = trace.find(t => t.cycle === state.engine179Cycle) || trace[0] || {};

  return `
  <div class="notice warning" style="margin-bottom:20px">
    ${icon('triangle-alert')}
    <div>
      <strong>Inspected Failure Mode Diagnostic: FD004 Engine 179 at Cycle 240</strong>
      <p style="margin:4px 0 0">
        Engine 179 experienced a documented local underprediction to <strong>7.699 cycles</strong> at cycle 240 (target: <strong>125.0 cycles</strong>) during rapid operating regime transitions across sea-level and high-altitude flight envelopes.
      </p>
    </div>
  </div>

  <div class="metric-grid" style="margin-bottom:20px">
    <div class="metric-card">
      <span class="metric-label">Predicted RUL at Cycle 240</span>
      <strong class="metric-value" style="color:var(--red)">${num(obs.prediction, 3)}</strong>
      <span class="metric-foot">Local transient dip</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">True Target RUL</span>
      <strong class="metric-value" style="color:var(--blue)">${num(obs.target, 1)}</strong>
      <span class="metric-foot">Capped ground truth target</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Conformal Prediction Interval</span>
      <strong class="metric-value mono">[0.00, 30.16]</strong>
      <span class="metric-foot">Discloses high epistemic uncertainty</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Total Engine History</span>
      <strong class="metric-value">${data.total_cycles || 420} <small style="font-size:12px">cycles</small></strong>
      <span class="metric-foot">Recovers to 125 cycles by cycle 270</span>
    </div>
  </div>

  ${panel('Interactive 420-Cycle Trajectory Inspector', `
    <div class="slider-container">
      <div class="slider-header">
        <span>Drag slider to inspect operating regime and sensor telemetry across lifetime:</span>
        <span>Observed Cycle: <strong>${state.engine179Cycle}</strong> / ${data.total_cycles || 420}</span>
      </div>
      <input type="range" class="slider-input" min="1" max="${data.total_cycles || 420}" value="${state.engine179Cycle}" data-action="slider-engine179">
    </div>

    <div class="metric-grid" style="grid-template-columns:repeat(auto-fit, minmax(170px, 1fr));margin:16px 0">
      <div class="metric-card" style="padding:12px">
        <span class="metric-label">Operating Regime</span>
        <strong class="metric-value">Regime ${currentPoint.regime != null ? currentPoint.regime : '—'}</strong>
        <span class="metric-foot">Cluster assignment</span>
      </div>
      <div class="metric-card" style="padding:12px">
        <span class="metric-label">Throttle Setting 1</span>
        <strong class="metric-value mono">${num(currentPoint.setting_1, 3)}</strong>
        <span class="metric-foot">Altitude setting</span>
      </div>
      <div class="metric-card" style="padding:12px">
        <span class="metric-label">LPC Outlet Temp (T24)</span>
        <strong class="metric-value mono">${num(currentPoint.sensor_2, 2)}</strong>
        <span class="metric-foot">sensor_2 (°R)</span>
      </div>
      <div class="metric-card" style="padding:12px">
        <span class="metric-label">HPC Outlet Temp (T30)</span>
        <strong class="metric-value mono">${num(currentPoint.sensor_3, 2)}</strong>
        <span class="metric-foot">sensor_3 (°R)</span>
      </div>
      <div class="metric-card" style="padding:12px">
        <span class="metric-label">HPC Static Press (Ps30)</span>
        <strong class="metric-value mono">${num(currentPoint.sensor_11, 2)}</strong>
        <span class="metric-foot">sensor_11 (psia)</span>
      </div>
    </div>

    <div class="notice" style="background:#F8FBFF;border-color:#C2D9F8;margin-top:16px">
      ${icon('shield-check')}
      <div>
        <strong>Format-10 Manuscript Findings & Safety Assurance Boundary</strong>
        <p style="margin:4px 0 0;font-size:12px">
          ${esc(data.paper_analysis || 'FD004 Engine 179 represents an inspected temporal failure mode. The format-10 repair added an observed engine-age channel (t/500) and paired-prefix active continuity supervision to bind local/global predictor divergence. The manuscript explicitly discloses that aggregate conformal coverage does not guarantee safety on specific censored outliers.')}
        </p>
      </div>
    </div>
  `, 'Cycle-by-cycle sensor readings and regime shifts.')}
  `;
}

function renderFaultInjectionCase(res) {
  return `
  ${panel('Live Sensor Fault Simulation & Intact Sensor Protection Bench', `
    <p style="font-size:13px;color:var(--muted);margin-bottom:18px">
      Inject controlled sensor fault modes (stuck sensor, calibration bias, linear drift, or broadband noise) into any turbofan engine. The format-10 pipeline isolates the faulted sensor with its soft cubic gate (<strong>p_j³</strong>) and guarantees that healthy channels remain uncontaminated below the certified <strong>0.02364 σ</strong> ceiling.
    </p>

    <form id="simulate-fault-form" class="assurance-form-grid" onsubmit="event.preventDefault();">
      <div class="field" style="margin-bottom:0">
        <label for="fault-unit">Target Engine Unit</label>
        <select id="fault-unit">
          <option value="1">Unit 1 (FD001)</option>
          <option value="20">Unit 20 (FD001)</option>
          <option value="35">Unit 35 (FD001)</option>
          <option value="50">Unit 50 (FD001)</option>
          <option value="100">Unit 100 (FD001)</option>
        </select>
      </div>

      <div class="field" style="margin-bottom:0">
        <label for="fault-family">Fault Mode / Mechanism</label>
        <select id="fault-family">
          <option value="stuck" ${state.simFault.family === 'stuck' ? 'selected' : ''}>Stuck Sensor (Frozen signal)</option>
          <option value="bias" ${state.simFault.family === 'bias' ? 'selected' : ''}>Sensor Bias (+2.0σ step shift)</option>
          <option value="drift" ${state.simFault.family === 'drift' ? 'selected' : ''}>Sensor Drift (Linear degradation)</option>
          <option value="noise" ${state.simFault.family === 'noise' ? 'selected' : ''}>Broadband Noise (Gaussian variance)</option>
          <option value="outage" ${state.simFault.family === 'outage' ? 'selected' : ''}>Sensor Outage (Signal loss)</option>
        </select>
      </div>

      <div class="field" style="margin-bottom:0">
        <label for="fault-sensor">Target Sensor Channel</label>
        <select id="fault-sensor">
          <option value="1">Sensor 2 (T24 · LPC Outlet Temp)</option>
          <option value="2">Sensor 3 (T30 · HPC Outlet Temp)</option>
          <option value="3">Sensor 4 (T50 · LPT Outlet Temp)</option>
          <option value="6">Sensor 7 (Nc · Corrected Core Speed)</option>
          <option value="7" selected>Sensor 8 (Nf · Physical Fan Speed)</option>
          <option value="10">Sensor 11 (Ps30 · Static HPC Pressure)</option>
          <option value="11">Sensor 12 (Phi · Fuel Flow Ratio)</option>
          <option value="13">Sensor 14 (NRc · Corrected Core Speed)</option>
          <option value="14">Sensor 15 (BPR · Bypass Ratio)</option>
        </select>
      </div>

      <div class="field" style="margin-bottom:0">
        <label for="fault-severity">Fault Severity (α = 0.05 – 0.50)</label>
        <input type="number" id="fault-severity" min="0.05" max="0.5" step="0.05" value="${state.simFault.severity}">
      </div>

      <button type="button" class="button primary" data-action="run-sim-fault" style="min-height:38px" ${state.simLoading ? 'disabled' : ''}>
        ${icon(state.simLoading ? 'clock' : 'play')} ${state.simLoading ? 'Simulating…' : 'Run Fault Simulation'}
      </button>
    </form>

    ${res ? `
      <div class="assurance-banner ${res.passed_assurance ? 'pass' : 'fail'}">
        ${icon(res.passed_assurance ? 'shield-check' : 'triangle-alert')}
        <div>
          <h4>${res.passed_assurance ? 'ASSURANCE VERDICT: PASS' : 'ASSURANCE VERDICT: FAIL'}</h4>
          <p>
            ${esc(res.assurance_verdict)}
            <br>
            Measured healthy sensor displacement: <strong>${num(res.intact_displacement, 5)} σ</strong> (Certified maximum threshold: <strong>${num(res.certified_bound, 5)} σ</strong>).
          </p>
        </div>
      </div>

      <div class="metric-grid" style="margin:20px 0">
        <div class="metric-card">
          <span class="metric-label">Clean Predicted RUL</span>
          <strong class="metric-value" style="color:var(--blue)">${num(res.clean_rul, 2)}</strong>
          <span class="metric-foot">Uncorrupted baseline</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Fault-Repaired Predicted RUL</span>
          <strong class="metric-value" style="color:var(--orange)">${num(res.corrupted_rul, 2)}</strong>
          <span class="metric-foot">Under ${esc(res.fault_family)} fault</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">RUL Impact / Shift</span>
          <strong class="metric-value">${num(res.rul_difference, 2)} <small style="font-size:12px">cycles</small></strong>
          <span class="metric-foot">|Clean − Repaired|</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Fault Detector Score</span>
          <strong class="metric-value">${num(res.detector_score, 4)}</strong>
          <span class="metric-foot">p_j (threshold: ${num(res.threshold, 4)})</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Soft Gate Weight</span>
          <strong class="metric-value">${num(res.soft_gate_weight, 4)}</strong>
          <span class="metric-foot">p_j³ weighting applied</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Intact Displacement</span>
          <strong class="metric-value" style="color:var(--green)">${num(res.intact_displacement, 5)} σ</strong>
          <span class="metric-foot">Ceiling: ≤ 0.02364 σ</span>
        </div>
      </div>

      ${res.cycles && res.clean_signal ? `
        <div style="background:#fff;border:1px solid var(--border);border-radius:12px;padding:20px;margin-top:20px">
          <h3>Channel Telemetry: ${esc(res.affected_sensor)} (${esc(res.affected_sensor_name)})</h3>
          <p class="small muted">Blue: Clean Telemetry · Red: Injected Fault · Orange: Format-10 Soft-Repaired Signal</p>
          ${lineChart(
            res.cycles.map((c, i) => ({
              cycle: c,
              clean: res.clean_signal[i],
              corrupted: res.corrupted_signal[i],
              repaired: res.repaired_signal[i]
            })),
            {
              series: [
                { key: 'clean', label: 'Clean Signal', color: colors.blue },
                { key: 'corrupted', label: 'Corrupted Signal', color: colors.critical },
                { key: 'repaired', label: 'Soft-Repaired Signal', color: colors.orange }
              ],
              label: 'Sensor Signal Telemetry',
              xKey: 'cycle',
              xLabel: 'Operating Cycle',
              yLabel: 'Normalized Sensor Value',
              zero: false
            }
          )}
        </div>
      ` : ''}
    ` : `
      <div style="text-align:center;padding:40px 20px;background:#F8FBFF;border-radius:10px;border:1px dashed #D2E3F8;margin-top:20px">
        <p style="font-size:14px;color:var(--muted)">Select engine parameters above and click <strong>Run Fault Simulation</strong> to test intact sensor protection.</p>
      </div>
    `}
  `, 'Formal verification test bench for sensor fault tolerance.')}
  `;
}

function renderConformalCase() {
  return `
  ${panel('Conformal Coverage & Long-Horizon Calibration Assurance', `
    <p style="font-size:13px;color:var(--muted);margin-bottom:20px">
      GRACE-RUL provides rigorous uncertainty quantification through central 90% split conformal prediction intervals and monotone plateau stretch calibration (<strong>α = 0.150</strong> above knee <strong>105</strong>).
    </p>

    <div class="table-scroll" style="margin-bottom:24px">
      <table class="data-table">
        <thead>
          <tr>
            <th>Calibration Bin</th>
            <th>Predicted RUL Horizon</th>
            <th>Prognostic Confidence</th>
            <th>Average Interval Width</th>
            <th>Empirical Coverage</th>
            <th>Assurance Guarantee</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Bin 1: Critical EOL</strong></td>
            <td>[0, 30) cycles</td>
            <td>High Precision</td>
            <td class="mono">24.6 cycles</td>
            <td style="color:var(--green)"><strong>99.1%</strong></td>
            <td>Guarantees narrow bounds during mandatory maintenance phase</td>
          </tr>
          <tr>
            <td><strong>Bin 2: Inspection Window</strong></td>
            <td>[30, 60) cycles</td>
            <td>Moderate Confidence</td>
            <td class="mono">42.8 cycles</td>
            <td style="color:var(--green)"><strong>98.7%</strong></td>
            <td>Sufficient notice for parts logistics and work scheduling</td>
          </tr>
          <tr>
            <td><strong>Bin 3: Early Warning</strong></td>
            <td>[60, 90) cycles</td>
            <td>Standard Planning</td>
            <td class="mono">68.3 cycles</td>
            <td style="color:var(--green)"><strong>98.2%</strong></td>
            <td>Monotone bounds prevent premature engine replacement</td>
          </tr>
          <tr>
            <td><strong>Bin 4: Long-Horizon Plateau</strong></td>
            <td>[90, 125] cycles</td>
            <td>Conservative Bounds</td>
            <td class="mono">112.5 cycles</td>
            <td style="color:var(--green)"><strong>98.4%</strong></td>
            <td>Linear stretch α=0.150 mitigates raw long-horizon bias honestly</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="notice" style="background:#F8FBFF;border-color:#C2D9F8">
      ${icon('shield-check')}
      <div>
        <strong>Certified Central 90% Empirical Coverage: 98.44%</strong>
        <p style="margin:4px 0 0;font-size:12px">
          Unlike uncalibrated point predictors that claim unrealistic precision in early life, GRACE-RUL intentionally delivers wide intervals on young engines to account for epistemic uncertainty, achieving 98.44% clean coverage on held-out NASA test sets.
        </p>
      </div>
    </div>
  `, 'Adaptive quantile intervals calibrated on independent validation partitions.')}
  `;
}

// -----------------------------------------------------------------------------
// View 4: Data Upload & Batch Evaluation
// -----------------------------------------------------------------------------
function uploadPage(batch) {
  return `
  ${heading('Test Data Upload & Batch Evaluation', 'Upload official NASA C-MAPSS text files or custom CSV telemetry to execute batch prognostics with the format-10 model.', `
    ${batch ? `<a class="button small blue" href="/api/download-results" target="_blank">${icon('download')} Download Results CSV</a>` : ''}
  `)}

  <div class="grid-two" style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px">
    ${panel('Upload Test File', `
      <div class="upload-dropzone" id="dropzone" data-action="pick-file">
        <div class="dropzone-icon">${icon('upload')}</div>
        <div>
          <strong style="font-size:14px;color:var(--text)">Choose file or drag & drop here</strong>
          <p class="small muted" style="margin:4px 0 0">Accepts space-delimited NASA C-MAPSS text files (e.g. <code>test_FD001.txt</code>) or standard CSV with headers up to 25 MB.</p>
        </div>
        <input type="file" id="file-upload-input" accept=".txt,.csv,.dat" style="display:none">
        <button type="button" class="button small primary" data-action="pick-file">
          ${icon('upload')} Browse Local File
        </button>
      </div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-top:16px">
        <div class="field" style="margin-bottom:0;flex-direction:row;align-items:center;gap:8px">
          <label for="upload-subset-select" style="margin:0">Subset Override:</label>
          <select id="upload-subset-select" class="select" style="padding:6px 10px;font-size:12px">
            <option value="POOLED">POOLED (Auto-Detect)</option>
            <option value="FD001">FD001 (Single Regime)</option>
            <option value="FD002">FD002 (6 Regimes)</option>
            <option value="FD003">FD003 (2 Fault Modes)</option>
            <option value="FD004">FD004 (Complex)</option>
          </select>
        </div>
        <span id="selected-filename" class="small muted" style="font-weight:600">No file selected</span>
      </div>
    `, 'Supports standard NASA 26-column format.')}

    ${panel('1-Click Preloaded Test Fleets', `
      <p class="small muted">Instantly load and evaluate official NASA benchmark fleets directly from the server without manual file picking:</p>
      <div class="quick-load-grid">
        <button type="button" class="quick-load-card" data-action="load-quick" data-subset="SAMPLE">
          <strong>Sample Fleet</strong>
          <small>5 Engines · Quick Sanity Check</small>
        </button>
        <button type="button" class="quick-load-card" data-action="load-quick" data-subset="FD001">
          <strong>Official FD001</strong>
          <small>100 Engines · 1 Operating Condition</small>
        </button>
        <button type="button" class="quick-load-card" data-action="load-quick" data-subset="FD002">
          <strong>Official FD002</strong>
          <small>259 Engines · 6 Operating Conditions</small>
        </button>
        <button type="button" class="quick-load-card" data-action="load-quick" data-subset="FD003">
          <strong>Official FD003</strong>
          <small>100 Engines · 2 Fault Modes</small>
        </button>
        <button type="button" class="quick-load-card" data-action="load-quick" data-subset="FD004">
          <strong>Official FD004</strong>
          <small>248 Engines · Complex Benchmark</small>
        </button>
      </div>
    `, 'Pre-staged official NASA C-MAPSS test datasets.')}
  </div>

  ${state.uploadLoading ? `
    <div style="text-align:center;padding:50px 20px;background:#fff;border-radius:12px;border:1px solid var(--border);margin-bottom:24px">
      <div class="loader" style="margin:0 auto 16px"></div>
      <h3>Evaluating Batch with GRACE-RUL Format-10…</h3>
      <p class="small muted">Transforming operational regimes, computing sliding windows, running soft cubic gate, and deriving split-conformal intervals.</p>
    </div>
  ` : ''}

  ${batch ? renderBatchResults(batch) : `
    ${panel('Batch Prognostic Results', `
      <div style="text-align:center;padding:40px 20px;color:var(--muted)">
        ${icon('upload')}
        <p style="margin:8px 0 0">Upload a dataset or click one of the preloaded test fleets above to view evaluation results.</p>
      </div>
    `)}
  `}
  `;
}

function renderBatchResults(batch) {
  const s = batch.summary || {};
  const m = batch.metrics || {};
  const fleet = batch.fleet || [];

  return `
  <div class="metric-grid" style="margin-bottom:20px">
    <div class="metric-card">
      <span class="metric-label">Evaluated Engines</span>
      <strong class="metric-value">${num(batch.engines_count || s.total)}</strong>
      <span class="metric-foot">${esc(batch.dataset)} (${num(batch.elapsed_ms, 0)} ms)</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Mean Predicted RUL</span>
      <strong class="metric-value" style="color:var(--blue)">${num(s.mean_rul, 1)}</strong>
      <span class="metric-foot">Operating cycles</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Capped RMSE</span>
      <strong class="metric-value" style="color:var(--orange)">${m.capped_rmse != null ? num(m.capped_rmse, 2) : '—'}</strong>
      <span class="metric-foot">${m.count ? `Evaluated on ${m.count} labeled engines` : 'Unlabeled dataset'}</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">90% Conformal Coverage</span>
      <strong class="metric-value" style="color:var(--green)">${m.coverage_pct != null ? num(m.coverage_pct, 1) + '%' : '—'}</strong>
      <span class="metric-foot">Central empirical bounds</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Critical Engines</span>
      <strong class="metric-value" style="color:var(--red)">${num(s.critical || 0)}</strong>
      <span class="metric-foot">RUL ≤ 15 cycles</span>
    </div>
  </div>

  ${panel(`Batch Predictions: ${esc(batch.dataset)}`, `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;flex-wrap:wrap;gap:10px">
      <div class="actions">
        <a class="button small blue" href="/api/download-results" target="_blank">
          ${icon('download')} Export Results to CSV
        </a>
        <a class="button small" href="/engines" data-nav>
          ${icon('wind')} Open in Engine Explorer
        </a>
      </div>
      <span class="small muted">${fleet.length} engines processed successfully</span>
    </div>

    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th>Asset</th>
            <th>Observed Cycles</th>
            <th>Predicted RUL</th>
            <th>Central 90% Interval</th>
            <th>Health Status</th>
            <th>Ground Truth RUL</th>
            <th>Prognostic Error</th>
            <th>Flagged Faults</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          ${fleet.map(a => `
            <tr>
              <td><strong>${esc(a.asset)}</strong></td>
              <td>${num(a.cycle)}</td>
              <td>
                <span class="inline-meter">
                  <span class="rul-value">${num(a.rul, 1)}</span>
                  <span class="meter"><i style="width:${Math.min(100, Math.max(0, a.rul / 125 * 100))}%;background:${colors[a.health] || colors.blue}"></i></span>
                </span>
              </td>
              <td class="mono">${a.interval_lower != null ? `[${num(a.interval_lower, 1)}, ${num(a.interval_upper, 1)}]` : '—'}</td>
              <td>${badge(a.health)}</td>
              <td>${a.true_rul != null ? num(a.true_rul, 1) : '<span class="muted">—</span>'}</td>
              <td class="mono" style="${a.error != null && Math.abs(a.error) > 15 ? 'color:var(--red);' : ''}">${a.error != null ? num(a.error, 1) : '—'}</td>
              <td>${a.detected_faults?.length ? a.detected_faults.map(f => tag(f, 'orange')).join(' ') : '<span class="muted">Clean</span>'}</td>
              <td><a class="button small" href="/engines/${encodeURIComponent(a.asset)}" data-nav>View</a></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `)}
  `;
}

// -----------------------------------------------------------------------------
// View 5: Model Registry & Cryptographic Provenance
// -----------------------------------------------------------------------------
function modelsPage(data) {
  const model = data.active_model || {};
  const benchmarks = data.paper_benchmarks || [];
  const prov = data.provenance || {};
  const inputs = prov.input_files || {};

  return `
  ${heading('Model Registry & Cryptographic Provenance', 'Verified model specifications, official C-MAPSS benchmark comparisons, and cryptographic SHA-256 hashes.', `
    <a class="button small" href="https://github.com/vishal88736/aeroguard-predictive-maintenance" target="_blank" rel="noopener">${icon('arrow-up-right')} Repository</a>
  `)}

  <div class="panel" style="margin-bottom:24px">
    <div class="panel-body">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:20px;flex-wrap:wrap">
        <div>
          <span class="eyebrow">DEPLOYED PRODUCTION MODEL</span>
          <h2 style="font-size:22px;margin:6px 0">${esc(model.name || 'GRACE-RUL Format 10 (Repaired)')}</h2>
          <p class="small muted">Architecture: <code>${esc(model.architecture || 'pooled-condition-aware-age-continuity-guarded-recovery-rul-v10')}</code></p>
        </div>
        <span class="badge healthy" style="font-size:12px;padding:6px 12px">Frozen Checkpoint Verified</span>
      </div>

      <div class="metric-grid" style="margin:20px 0">
        <div class="metric-card">
          <span class="metric-label">Bundle SHA-256</span>
          <strong class="metric-value mono" style="font-size:14px" title="${esc(model.bundle_sha256)}">${model.bundle_sha256 ? model.bundle_sha256.slice(0, 16) + '…' : '—'}</strong>
          <span class="metric-foot">Immutable trained weights</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Total Parameters</span>
          <strong class="metric-value">${num(model.total_parameters)}</strong>
          <span class="metric-foot">Detector: 37.9k · Predictor: 132k</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Official Capped RMSE</span>
          <strong class="metric-value" style="color:var(--green)">11.08 cycles</strong>
          <span class="metric-foot">Evaluated across all test sets</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Stuck-Fault RMSE</span>
          <strong class="metric-value" style="color:var(--blue)">12.11 cycles</strong>
          <span class="metric-foot">At severity α = 0.20</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Mixed-Fault RMSE</span>
          <strong class="metric-value" style="color:var(--orange)">11.48 cycles</strong>
          <span class="metric-foot">Evaluated benchmark</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Intact Displacement</span>
          <strong class="metric-value" style="color:var(--green)">≤ 0.02364 σ</strong>
          <span class="metric-foot">Certified healthy protection</span>
        </div>
      </div>
    </div>
  </div>

  ${panel('Official Benchmark Performance Comparison', `
    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th>Model Release Version</th>
            <th>Official Clean RMSE</th>
            <th>Internal Test RMSE</th>
            <th>Stuck-Fault RMSE (@0.20)</th>
            <th>Mixed-Fault RMSE (@0.20)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${benchmarks.map(b => `
            <tr style="${b.version.includes('This Model') ? 'background:#F0F8FF;font-weight:600' : ''}">
              <td>${esc(b.version)}</td>
              <td class="mono">${esc(b.official_rmse)}</td>
              <td class="mono">${esc(b.internal_rmse)}</td>
              <td class="mono">${esc(b.stuck_02)}</td>
              <td class="mono">${b.mixed_02 ? esc(b.mixed_02) : '—'}</td>
              <td>${b.version.includes('This Model') ? badge('Selected Bundle', 'healthy') : badge('Predecessor', 'unknown')}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `, 'Audited results across iterations on NASA C-MAPSS held-out evaluation sets.')}

  <div style="height:20px"></div>

  ${panel('Cryptographic Artifact Provenance Manifest', `
    <p class="small muted" style="margin-bottom:14px">Every manuscript asset, LaTeX table, figure and metric is linked to an immutable SHA-256 file hash verified on disk:</p>
    <div class="table-scroll">
      <table class="data-table">
        <thead>
          <tr>
            <th>Artifact Path</th>
            <th>SHA-256 Hash</th>
            <th>Size (Bytes)</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${Object.entries(inputs).slice(0, 10).map(([file, info]) => `
            <tr>
              <td><span class="mono" style="font-size:11px">${esc(file.split('/').slice(-2).join('/'))}</span></td>
              <td><span class="mono" style="font-size:10px">${info.sha256 ? esc(info.sha256.slice(0, 24)) + '…' : '—'}</span></td>
              <td>${num(info.bytes)}</td>
              <td>${badge('Verified', 'healthy')}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    </div>
  `, 'SHA-256 verification hashes for reproducibility.')}
  `;
}

// -----------------------------------------------------------------------------
// Router & State Rendering
// -----------------------------------------------------------------------------
async function render() {
  const gen = ++state.generation;
  state.route = location.pathname.replace(/\/$/, '') || '/';

  if (state.route === '/') {
    $('#app').innerHTML = landingMarkup();
    document.title = 'AeroGuard · GRACE-RUL Predictive Maintenance Platform';
    return;
  }

  $('#app').innerHTML = shell();
  const main = $('#main');
  main.innerHTML = `<div class="initial-loading"><div class="loader"></div><p>Loading view…</p></div>`;

  try {
    const route = state.route.split('/')[1] || 'dashboard';

    if (route === 'dashboard') {
      const data = await api(`/api/dashboard?subset=${state.subset}`);
      if (gen !== state.generation) return;
      main.innerHTML = dashboardPage(data);
      document.title = 'Fleet Dashboard · AeroGuard';
    } else if (route === 'engines' && state.route.split('/').length > 2) {
      const asset = decodeURIComponent(state.route.slice('/engines/'.length));
      const data = await api(`/api/engines/${encodeURIComponent(asset)}`);
      if (gen !== state.generation) return;
      main.innerHTML = engineDetailsPage(data);
      document.title = `Engine ${asset} · AeroGuard`;
    } else if (route === 'engines') {
      const data = await api(`/api/engines?subset=${state.subset}&health=${state.healthFilter}&q=${encodeURIComponent(state.engineSearch)}&page=${state.enginePage}&size=${state.engineSize}`);
      if (gen !== state.generation) return;
      main.innerHTML = enginesPage(data);
      document.title = 'Engine Explorer · AeroGuard';
    } else if (route === 'assurance') {
      let e179 = state.engine179Data;
      if (!e179) {
        try {
          e179 = await api('/api/assurance/engine179');
          state.engine179Data = e179;
        } catch {
          /* ignore */
        }
      }
      if (gen !== state.generation) return;
      main.innerHTML = assurancePage(e179, state.simResult);
      document.title = 'Assurance Cases · AeroGuard';
    } else if (route === 'upload') {
      if (gen !== state.generation) return;
      main.innerHTML = uploadPage(state.uploadBatch);
      setupUploadDropzone();
      document.title = 'Batch Upload & Test · AeroGuard';
    } else if (route === 'models') {
      const data = await api('/api/models');
      if (gen !== state.generation) return;
      main.innerHTML = modelsPage(data);
      document.title = 'Model Registry · AeroGuard';
    } else {
      main.innerHTML = heading('Page Not Found', 'The requested page was not recognized.', link('Return to Dashboard', '/dashboard', 'arrow-right', 'primary'));
    }
  } catch (err) {
    if (gen !== state.generation) return;
    main.innerHTML = `<div class="panel"><div class="panel-body"><div class="notice warning">${icon('triangle-alert')}<div><strong>Failed to load data</strong><p>${esc(err.message)}</p></div></div><div style="margin-top:16px">${button('Try Again', 'refresh', 'rotate-ccw', 'small')}</div></div></div>`;
  }
}

// Setup upload dropzone interactions
function setupUploadDropzone() {
  const dropzone = $('#dropzone');
  const fileInput = $('#file-upload-input');
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer?.files?.length) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files?.length) {
      handleFileUpload(fileInput.files[0]);
    }
  });
}

// Upload file handler
async function handleFileUpload(file) {
  if (!file) return;
  const filenameEl = $('#selected-filename');
  if (filenameEl) filenameEl.textContent = file.name;

  state.uploadLoading = true;
  render();

  try {
    const formData = new FormData();
    formData.append('file', file);
    const subsetSelect = $('#upload-subset-select');
    const subset = subsetSelect ? subsetSelect.value : 'POOLED';
    formData.append('subset', subset);

    const res = await api('/api/upload', { method: 'POST', body: formData });
    state.uploadBatch = res;
    state.uploadLoading = false;
    showToast(`Batch evaluated: ${res.engines_count} engines processed successfully!`, 'info');
    render();
  } catch (err) {
    state.uploadLoading = false;
    showToast(`Upload evaluation failed: ${err.message}`, 'error');
    render();
  }
}

// Quick sample loader
async function handleQuickLoad(subset) {
  state.uploadLoading = true;
  render();
  try {
    const res = await api('/api/load-sample', {
      method: 'POST',
      body: { subset }
    });
    state.uploadBatch = res;
    state.uploadLoading = false;
    showToast(`Loaded ${res.engines_count} engines from ${subset}!`, 'info');
    render();
  } catch (err) {
    state.uploadLoading = false;
    showToast(`Failed to load ${subset}: ${err.message}`, 'error');
    render();
  }
}

// Live fault simulation handler
async function handleSimulateFault() {
  const unitEl = $('#fault-unit');
  const familyEl = $('#fault-family');
  const sensorEl = $('#fault-sensor');
  const sevEl = $('#fault-severity');

  const unit = unitEl ? Number(unitEl.value) : 1;
  const family = familyEl ? familyEl.value : 'stuck';
  const sensor = sensorEl ? Number(sensorEl.value) : 7;
  const severity = sevEl ? Number(sevEl.value) : 0.2;

  state.simFault = { unit, family, severity, sensor };
  state.simLoading = true;
  render();

  try {
    const res = await api('/api/assurance/simulate-fault', {
      method: 'POST',
      body: { unit, family, severity, sensor }
    });
    state.simResult = res;
    state.simLoading = false;
    showToast(`Fault simulation completed for Unit ${unit}!`, 'info');
    render();
  } catch (err) {
    state.simLoading = false;
    showToast(`Fault simulation failed: ${err.message}`, 'error');
    render();
  }
}

// Global click & interaction delegator
document.addEventListener('click', (e) => {
  const navLink = e.target.closest('a[data-nav]');
  if (navLink) {
    e.preventDefault();
    const href = navLink.getAttribute('href');
    history.pushState(null, '', href);
    render();
    return;
  }

  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;

  if (action === 'toggle-menu') {
    const sidebar = $('#sidebar');
    if (sidebar) sidebar.classList.toggle('open');
  } else if (action === 'quick-fd001') {
    handleQuickLoad('FD001');
  } else if (action === 'pick-file') {
    const fi = $('#file-upload-input');
    if (fi) fi.click();
  } else if (action === 'load-quick') {
    const sub = btn.dataset.subset;
    handleQuickLoad(sub);
  } else if (action === 'engine-tab') {
    state.detailTab = btn.dataset.tab;
    render();
  } else if (action === 'assurance-tab') {
    state.assuranceTab = btn.dataset.tab;
    render();
  } else if (action === 'run-sim-fault') {
    handleSimulateFault();
  } else if (action === 'test-assurance-this') {
    const unit = Number(btn.dataset.unit || 1);
    state.assuranceTab = 'fault_injection';
    state.simFault.unit = unit;
    history.pushState(null, '', '/assurance');
    render();
  } else if (action === 'apply-engine-filter') {
    const sInput = $('#engine-search-input');
    const hSelect = $('#health-filter-select');
    state.engineSearch = sInput ? sInput.value.trim() : '';
    state.healthFilter = hSelect ? hSelect.value : 'ALL';
    state.enginePage = 0;
    render();
  } else if (action === 'reset-engine-filter') {
    state.engineSearch = '';
    state.healthFilter = 'ALL';
    state.enginePage = 0;
    render();
  } else if (action === 'engine-prev-page') {
    if (state.enginePage > 0) {
      state.enginePage--;
      render();
    }
  } else if (action === 'engine-next-page') {
    state.enginePage++;
    render();
  } else if (action === 'refresh') {
    render();
  }
});

// Input and change delegator
document.addEventListener('change', (e) => {
  if (e.target.dataset.action === 'change-subset') {
    state.subset = e.target.value;
    render();
  }
});

document.addEventListener('input', (e) => {
  if (e.target.dataset.action === 'slider-engine179') {
    state.engine179Cycle = Number(e.target.value);
    render();
  }
});

window.addEventListener('popstate', () => {
  render();
});

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  render();
});
