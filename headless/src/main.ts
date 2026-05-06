import { invoke } from './api';

// ─── State ────────────────────────────────────────────────────────────────────
let trajectoryPath: string | null = null;
let topologyPath:   string | null = null;
let isLoaded    = false;
let totalFrames = 0;
let activeToolId:  string | null = null;
let simMode:       'md' | 'qmm'  = 'md';
let umbrellaLoaded = false;
let qmmTopoPath:   string | null  = null;
let cvRstBlocks:   any[]          = [];
let cvRstPath:     string | null  = null;

// ─── Tool registry ────────────────────────────────────────────────────────────
//
// Every tool is declared once here. Adding a new tool in the future means:
//   1. Add an entry to TOOLS
//   2. Add a case to runTool()
// No HTML changes needed.

interface Tool {
  id:       string;
  label:    string;
  desc:     string;
  category: string;
  invoke:   string;           // Rust command name
  chartType: 'line' | 'heatmap' | 'scatter' | 'bar' | 'pmf' | 'ramachandran' | 'prs' | 'cluster' | 'geometry' | 'dihedral_ts' | 'sasa' | 'none';
  clickAction?: 'frame' | 'residue' | 'pair';
  map3d?:      string;
  xLabel?:  string;
  yLabel?:  string;
  // For tools with configurable parameters (wizard-style tools added later)
  params?:  ToolParam[];
}

interface ToolParam {
  id:      string;
  label:   string;
  type:    'number' | 'text' | 'select';
  default: string;
  options?: string[];  // for type: 'select'
}

const CATEGORIES = [
  { id: 'structural',      label: 'Structural' },
  { id: 'correlation',     label: 'Correlation & Dynamics' },
  { id: 'nma',             label: 'Normal Mode Analysis' },
  { id: 'network',         label: 'Network Analysis' },
  { id: 'thermodynamics',  label: 'Thermodynamics' },
  { id: 'quantumchem',   label: 'Quantum Chemistry' },
] as const;

const TOOLS: Tool[] = [
  // ── Structural ──────────────────────────────────────────────────────────────
  {
    id: 'rmsd', label: 'RMSD', category: 'structural',
    desc: 'Root-mean-square deviation from reference frame. Tracks global conformational drift.',
    invoke: 'run_rmsd', chartType: 'line', xLabel: 'Frame', yLabel: 'RMSD (Å)', clickAction: 'frame'
  },
  {
    id: 'rmsf', label: 'RMSF', category: 'structural',
    desc: 'Per-residue root-mean-square fluctuation. Identifies flexible and rigid regions.',
    invoke: 'run_rmsf', chartType: 'line', xLabel: 'Residue', yLabel: 'RMSF (Å)', clickAction: 'residue', map3d: 'bfactor'
  },
  {
    id: 'rg', label: 'Radius of Gyration', category: 'structural',
    desc: 'Structural compactness per frame. Tracks global folding and unfolding events.',
    invoke: 'run_radius_of_gyration', chartType: 'line', xLabel: 'Frame', yLabel: 'Rg (Å)', clickAction: 'frame'
  },
  {
    id: 'sasa', label: 'SASA', category: 'structural',
    desc: 'Shrake-Rupley solvent accessible surface area. Per-residue mean/std and total per frame. Maps to 3D via bfactor. Click a residue bar to view in 3D.',
    invoke: 'run_sasa', chartType: 'sasa', clickAction: 'residue', map3d: 'bfactor',
    params: [
      { id: 'probe', label: 'Probe radius (Å)', type: 'number', default: '1.4' },
    ],
  },
  {
    id: 'pca', label: 'PCA', category: 'structural',
    desc: 'Principal component analysis of Cα covariance. Decomposes collective motions into essential modes.',
    invoke: 'run_pca', chartType: 'scatter', xLabel: 'PC1', yLabel: 'PC2', clickAction: 'frame'
  },
  {
    id: 'dihedrals', label: 'Ramachandran / Dihedrals', category: 'structural',
    desc: 'Backbone φ/ψ angles over the trajectory. Ramachandran plot with per-residue mean positions. Click a residue to see its full dihedral time series.',
    invoke: 'run_dihedrals', chartType: 'ramachandran', clickAction: 'residue',
  },
  {
    id: 'clustering', label: 'Clustering / MSM', category: 'structural',
    desc: 'K-means clustering of frames in PCA space. Optionally builds a Markov State Model (MSM) with PCCA macrostate assignment and implied timescales. Requires PCA.',
    invoke: 'run_clustering', chartType: 'cluster', clickAction: 'frame',
    params: [
      { id: 'n_clusters', label: 'Clusters (k)',         type: 'number', default: '4'      },
      { id: 'method',     label: 'Method',               type: 'select', default: 'kmeans',
        options: ['kmeans', 'msm'] },
      { id: 'lag',        label: 'MSM lag (frames)',     type: 'number', default: '10'     },
      { id: 'n_macro',    label: 'PCCA macrostates',     type: 'number', default: '4'      },
    ],
  },
  {
    id: 'dihedral_ts', label: 'Dihedral Time Series', category: 'structural',
    desc: 'φ and ψ angle time series for selected residues. Shows rotameric transitions over the trajectory. Requires Ramachandran analysis to be run first.',
    invoke: '', chartType: 'dihedral_ts', clickAction: 'frame',
    params: [
      { id: 'atom_indices', label: 'Atom indices (comma-separated)', type: 'text', default: '0' },
    ],
  },
  // ── Correlation & Dynamics ───────────────────────────────────────────────────
  {
    id: 'dccm', label: 'DCCM', category: 'correlation',
    desc: 'Dynamic cross-correlation matrix. Reveals correlated and anti-correlated residue motion pairs.',
    invoke: 'run_dccm', chartType: 'heatmap', clickAction: 'pair'
  },
  {
    id: 'hbond', label: 'H-Bond Lifetime', category: 'correlation',
    desc: 'Hydrogen bond occupancy and lifetime over the trajectory. Requires full-atom topology.',
    invoke: 'run_hbond', chartType: 'bar', xLabel: 'Donor–Acceptor pair', yLabel: 'Occupancy',
  },
  {
    id: 'contacts', label: 'Contact Map', category: 'correlation',
    desc: 'Residue–residue contact frequency matrix over the trajectory.',
    invoke: 'run_contacts', chartType: 'heatmap', clickAction: 'pair'
  },
  {
    id: 'mutual_info', label: 'Mutual Information', category: 'correlation',
    desc: 'Normalised mutual information matrix. Captures non-linear coupling between residues — complements DCCM which only detects linear correlations.',
    invoke: 'run_mutual_information', chartType: 'heatmap', clickAction: 'pair',
  },
  // ── Normal Mode Analysis ─────────────────────────────────────────────────────
  {
    id: 'nma_enm', label: 'Elastic Network Model', category: 'nma',
    desc: 'Gaussian/anisotropic network model normal modes from mean structure. Requires Cα only.',
    invoke: 'run_enm', chartType: 'line', xLabel: 'Mode', yLabel: 'Frequency', clickAction: 'residue', map3d: 'bfactor',
    params: [
      { id: 'cutoff',    label: 'Cutoff (Å)', type: 'number', default: '7.5' },
      { id: 'n_modes',   label: 'Modes',       type: 'number', default: '20'  },
      { id: 'model',     label: 'Model',        type: 'select', default: 'ANM',
        options: ['ANM', 'GNM'] },
    ],
  },
  {
    id: 'nma_overlap', label: 'Mode–Trajectory Overlap', category: 'nma',
    desc: 'Cumulative overlap of NMA modes with observed trajectory fluctuations.',
    invoke: 'run_nma_overlap', chartType: 'bar', xLabel: 'Mode', yLabel: 'Overlap', clickAction: 'residue'
  },
  // ── Network Analysis ─────────────────────────────────────────────────────────
  {
    id: 'net_community', label: 'Community Detection', category: 'network',
    desc: 'Girvan-Newman community detection on the DCCM-derived residue network.',
    invoke: 'run_communities', chartType: 'none',
    params: [
      { id: 'threshold', label: 'Edge threshold', type: 'number', default: '0.6' },
    ],
  },
  {
    id: 'prs', label: 'Perturbation Response Scanning', category: 'network',
    desc: 'PRS: apply a unit perturbation at each residue and measure the response across the network. Identifies allosteric drivers (high effectiveness) and sensors (high sensitivity).',
    invoke: 'run_prs', chartType: 'prs', clickAction: 'pair', map3d: 'bfactor',
  },
  {
    id: 'net_betweenness', label: 'Betweenness Centrality', category: 'network',
    desc: 'Identifies hub residues by betweenness centrality in the correlation network.',
    invoke: 'run_betweenness', chartType: 'bar', xLabel: 'Residue', yLabel: 'Centrality', clickAction: 'residue', map3d: 'bfactor'
  },
  {
    id: 'net_paths', label: 'Optimal Paths', category: 'network',
    desc: 'Shortest communication paths between residue pairs in the allosteric network.',
    invoke: 'run_optimal_paths', chartType: 'none',
    params: [
      { id: 'source', label: 'Source residue', type: 'number', default: '1'  },
      { id: 'sink',   label: 'Sink residue',   type: 'number', default: '50' },
    ],
  },
  // ── Thermodynamics ───────────────────────────────────────────────────────────
  {
    id: 'fes', label: 'Free Energy Surface', category: 'thermodynamics',
    desc: '2D free energy landscape projected onto PC1/PC2. Requires PCA to be run first.',
    invoke: 'run_fes', chartType: 'heatmap',
  },
  {
    id: 'entropy', label: 'Conformational Entropy', category: 'thermodynamics',
    desc: 'Quasi-harmonic entropy estimate from the covariance matrix eigenspectrum.',
    invoke: 'run_entropy', chartType: 'none',
  },
  // ── Quantum Chemistry ────────────────────────────────────────────────────────────
  {
    id: 'umbrella_viewer', label: 'Window Trajectory', category: 'quantumchem',
    desc: 'Scroll or play through all umbrella sampling window restart files as a trajectory. Use arrow keys or the slider to step through windows and observe bond breaking/forming events.',
    invoke: '', chartType: 'none',
  },
  {
    id: 'mbar', label: 'MBAR / PMF', category: 'quantumchem',
    desc: 'MBAR analysis of umbrella sampling windows. Outputs PMF, ΔG‡, and KDE overlap histogram. Load QM/MM windows first.',
    invoke: 'run_mbar', chartType: 'pmf',
    params: [
      { id: 'fc',    label: 'Force constant (kcal/mol/Å²)', type: 'number', default: '300.0' },
      { id: 'temp',  label: 'Temperature (K)',               type: 'number', default: '300.0' },
      { id: 'n_bins', label: 'PMF bins',                      type: 'number', default: '29'    },
      { id: 'n_boot', label: 'Bootstrap replicates',          type: 'number', default: '50'    },
    ],
  },
  {
    id: 'geometry', label: 'Distance / Angle Monitor', category: 'quantumchem',
    desc: 'Compute interatomic distances and angles. Paste a server-side cv.rst path in the sidebar to auto-populate r1/r2/RC, or enter 0-based atom index pairs manually. For umbrella source, results plot vs. window CV and can be overlaid on the PMF.',
    invoke: 'run_geometry_series', chartType: 'geometry', clickAction: 'frame',
    params: [
      { id: 'pairs',    label: 'Distance pairs (i,j;i,j… 0-based)',    type: 'text',   default: ''           },
      { id: 'triplets', label: 'Angle triplets (i,j,k;i,j,k… 0-based)', type: 'text',  default: ''           },
      { id: 'source',   label: 'Source',                                 type: 'select', default: 'trajectory',
        options: ['trajectory', 'umbrella'] },
    ],
  },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
const $ = (sel: string) => document.querySelector(sel) as HTMLElement;
const $$ = (sel: string) => document.querySelectorAll(sel) as NodeListOf<HTMLElement>;

function log(msg: string, level: 'info' | 'success' | 'error' | 'warn' = 'info') {
  const logEl = $('#log')!;
  const ts     = new Date().toLocaleTimeString('en-US', { hour12: false });
  const prefix = { info: '·', success: '✓', error: '✗', warn: '!' }[level];
  const line   = document.createElement('div');
  line.className = `log-line log-${level}`;
  line.innerHTML = `<span class="log-ts">${ts}</span><span class="log-pfx">${prefix}</span><span>${msg}</span>`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function updateLoadState() {
  const hasTraj = !!trajectoryPath;
  const isReady = isLoaded;

  const loadBtn = $('#load-btn') as HTMLButtonElement;
  loadBtn.disabled = !hasTraj;
  loadBtn.className = hasTraj ? 'btn btn-primary' : 'btn btn-primary disabled';

  const vizBtn = $('#visualize-btn') as HTMLButtonElement;
  vizBtn.disabled = !isReady;
  vizBtn.className = isReady ? 'Btn btn-primary' : 'btn btn-primary disabled';

  const noLoadNeeded = ['mbar','umbrella_viewer','geometry','dihedral_ts'];
  $$('.tool-run-btn').forEach(btn => {
    const noLoad = noLoadNeeded.includes(activeToolId ?? '');
    (btn as HTMLButtonElement).disabled = !isLoaded && !noLoad;
    btn.classList.toggle('disabled', !isLoaded && !noLoad);
  });

  const dot  = $('#status-dot')!;
  const text = $('#status-text')!;
  if (isLoaded) {
    dot.className  = 'status-dot loaded';
    text.textContent = 'Loaded';
  } else if (hasTraj && !!topologyPath) {
    dot.className  = 'status-dot ready';
    text.textContent = 'Ready to load';
  } else if (hasTraj) {
    dot.className  = 'status-dot partial';
    text.textContent = 'Topology optional';
  } else {
    dot.className  = 'status-dot idle';
    text.textContent = 'No files selected';
  }
}

// ─── Tool list rendering ──────────────────────────────────────────────────────

function renderToolList(categoryId: string) {
  const listEl  = $('#tool-list')!;
  const tools   = TOOLS.filter(t => t.category === categoryId);
  listEl.innerHTML = '';

  for (const tool of tools) {
    const item = document.createElement('div');
    item.className = 'tool-item' + (tool.id === activeToolId ? ' active' : '');
    item.dataset.toolId = tool.id;
    item.innerHTML = `
      <div class="tool-item-name">${tool.label}</div>
      <div class="tool-item-desc">${tool.desc}</div>`;
    item.addEventListener('click', () => selectTool(tool.id));
    listEl.appendChild(item);
  }
}

function selectTool(toolId: string) {
  activeToolId = toolId;
  const tool = TOOLS.find(t => t.id === toolId)!;

  // Update sidebar selection highlight
  $$('.tool-item').forEach(el => {
    el.classList.toggle('active', (el as HTMLElement).dataset.toolId === toolId);
  });

  // Build the tool detail panel (right side, top zone)
  const detailEl = $('#tool-detail')!;
  const hasParams = tool.params && tool.params.length > 0;

  const paramsHtml = hasParams ? `
    <div class="param-grid">
      ${tool.params!.map(p => `
        <div class="param-row">
          <label class="param-lbl" for="param-${p.id}">${p.label}</label>
          ${p.type === 'select'
            ? `<select class="param-input" id="param-${p.id}">
                ${p.options!.map(o => `<option${o === p.default ? ' selected' : ''}>${o}</option>`).join('')}
               </select>`
            : `<input class="param-input" id="param-${p.id}" type="${p.type}" value="${p.default}" />`
          }
        </div>`).join('')}
    </div>` : '';

  const map3dHtml = tool.map3d ? `
    <button class="tool-map3d-btn disabled" id="map3d-btn" disabled
            title="Run the analysis first">⬡ Map to 3D</button>` : '';

  // These tools operate on QM/MM data, not the MD trajectory
  const noLoadNeededTools = ['mbar','umbrella_viewer','geometry','dihedral_ts'];
  const needsLoad = !noLoadNeededTools.includes(tool.id);

  detailEl.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-name">${tool.label}</div>
        <div class="detail-desc">${tool.desc}</div>
      </div>
      <div style="display:flex;gap:6px;align-items:center;flex-shrink:0">
        ${map3dHtml}
        <button class="tool-run-btn${needsLoad && !isLoaded ? ' disabled' : ''}" id="run-btn"
          ${needsLoad && !isLoaded ? 'disabled' : ''}>Run</button>
      </div>
    </div>
    ${paramsHtml}`;

  showChartPlaceholder(tool.label);
  $('#run-btn')?.addEventListener('click', () => runTool(tool));
  $('#map3d-btn')?.addEventListener('click', () => {
    if (tool.map3d) openAnalysisViewer({ mode: tool.map3d }, tool.label);
  });
}

// ─── Chart display ────────────────────────────────────────────────────────────

function showChartPlaceholder(toolName: string) {
  const area = $('#chart-display')!;
  area.innerHTML = `
    <div class="chart-placeholder">
      <div class="placeholder-icon">◈</div>
      <div class="placeholder-name">${toolName}</div>
      <div class="placeholder-hint">Press Run to execute analysis</div>
    </div>`;
}

function showChartLoading(toolName: string) {
  const area = $('#chart-display')!;
  area.innerHTML = `
    <div class="chart-placeholder">
      <div class="placeholder-spin"></div>
      <div class="placeholder-name">${toolName}</div>
      <div class="placeholder-hint">Computing…</div>
    </div>`;
}

function renderLineChart(data: number[], title: string, xLabel: string, yLabel: string, onPointClick?: (index: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w  = area.clientWidth  || 600;
  const svgH = (area.clientHeight || 340) - TITLE_H;
  const pad  = { top: 24, right: 24, bottom: 58, left: 62 };
  const iw   = w    - pad.left - pad.right;
  const ih   = svgH - pad.top  - pad.bottom;

  const min = Math.min(...data), max = Math.max(...data), range = max - min || 1;
  const xs  = (i: number) => (i / (data.length - 1)) * iw;
  const ys  = (v: number) => ih - ((v - min) / range) * ih;
  const pts = data.map((v, i) => `${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(' ');

  const yTicks = 5;
  const yGrid = Array.from({ length: yTicks }, (_, i) => {
    const v = min + (range * i) / (yTicks - 1);
    const y = ys(v);
    return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-8" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v.toFixed(2)}</text>`;
  }).join('');

  const nTicks = Math.min(8, data.length);
  const xGrid = Array.from({ length: nTicks }, (_, i) => {
    const idx = Math.round((i / (nTicks - 1)) * (data.length - 1));
    const x   = xs(idx);
    return `<line x1="${x.toFixed(1)}" y1="0" x2="${x.toFixed(1)}" y2="${ih}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="${x.toFixed(1)}" y="${(ih + 18).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${idx}</text>`;
  }).join('');

  const hitRects = onPointClick ? data.map((_, i) => {
    const x = xs(i);
    return `<rect x="${(x-5).toFixed(1)}" y="0" width="10" height="${ih}"
              fill="transparent" data-i="${i}" style="cursor:pointer"/>`;
  }).join('') : '';
  const lcHint = onPointClick
    ? `<text x="${iw}" y="-6" text-anchor="end" fill="#3d4245" font-size="9">click → 3D view</text>` : '';

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">${title}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block" id="line-svg">
      <g transform="translate(${pad.left},${pad.top})">
        ${yGrid}${xGrid}
        <polyline points="${pts}" fill="none" stroke="#00c4a7" stroke-width="1.5" stroke-linejoin="round"/>
        <line id="hover-line" x1="0" y1="0" x2="0" y2="${ih}"
              stroke="#00c4a7" stroke-width="0.5" stroke-dasharray="3,3" opacity="0" pointer-events="none"/>
        <circle id="hover-dot" cx="0" cy="0" r="4" fill="#00c4a7" opacity="0" pointer-events="none"/>
        ${hitRects}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-46" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="11">${yLabel}</text>
        <text x="${iw/2}" y="${(ih + 42).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="11">${xLabel}</text>
        ${lcHint}
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onPointClick) {
    const svg = document.getElementById('line-svg')!;
    const hLine = document.getElementById('hover-line')!;
    const hDot  = document.getElementById('hover-dot')!;
    svg.addEventListener('mousemove', (e) => {
      const rect = svg.getBoundingClientRect();
      const idx  = Math.max(0, Math.min(data.length - 1,
        Math.round((e.clientX - rect.left - pad.left) / iw * (data.length - 1))));
      const x = xs(idx).toFixed(1), y = ys(data[idx]).toFixed(1);
      hLine.setAttribute('x1', x); hLine.setAttribute('x2', x); hLine.setAttribute('opacity', '1');
      hDot.setAttribute('cx', x);  hDot.setAttribute('cy', y);  hDot.setAttribute('opacity', '1');
    });
    svg.addEventListener('mouseleave', () => {
      hLine.setAttribute('opacity','0'); hDot.setAttribute('opacity','0');
    });
    svg.querySelectorAll<SVGRectElement>('rect[data-i]').forEach(r =>
      r.addEventListener('click', () => onPointClick(parseInt(r.dataset.i!, 10)))
    );
  }
}

function renderScatterChart(points: [number,number][], title: string, xLabel: string, yLabel: string, onPointClick?: (index: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w    = area.clientWidth  || 600;
  const svgH = (area.clientHeight || 340) - TITLE_H;
  const pad  = { top: 24, right: 24, bottom: 58, left: 62 };
  const iw   = w    - pad.left - pad.right;
  const ih   = svgH - pad.top  - pad.bottom;

  const pxs = points.map(p => p[0]), pys = points.map(p => p[1]);
  const xmin = Math.min(...pxs), xmax = Math.max(...pxs), xr = xmax - xmin || 1;
  const ymin = Math.min(...pys), ymax = Math.max(...pys), yr = ymax - ymin || 1;

  const sx = (v: number) => ((v - xmin) / xr) * iw;
  const sy = (v: number) => ih - ((v - ymin) / yr) * ih;

  const xTicks = 5;
  const xTickSvg = Array.from({ length: xTicks }, (_, i) => {
    const v = xmin + (xr * i) / (xTicks - 1);
    const x = sx(v);
    return `<line x1="${x.toFixed(1)}" y1="0" x2="${x.toFixed(1)}" y2="${ih}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="${x.toFixed(1)}" y="${(ih + 18).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${v.toFixed(1)}</text>`;
  }).join('');

  const yTicks = 5;
  const yTickSvg = Array.from({ length: yTicks }, (_, i) => {
    const v = ymin + (yr * i) / (yTicks - 1);
    const y = sy(v);
    return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-8" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v.toFixed(1)}</text>`;
  }).join('');

  const dots = points.map(([x, y], i) =>
    `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="4" fill="#5b8dee"
       fill-opacity="0.5" data-i="${i}" style="cursor:${onPointClick ? 'pointer' : 'default'}"/>`
  ).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">${title}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block">
      <g transform="translate(${pad.left},${pad.top})">
        ${yTickSvg}${xTickSvg}${dots}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-46" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="11">${yLabel}</text>
        <text x="${iw/2}" y="${(ih + 42).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="11">${xLabel}</text>
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onPointClick) {
    area.querySelectorAll<SVGCircleElement>('circle[data-i]').forEach(dot => {
      dot.addEventListener('click', () => onPointClick(parseInt(dot.dataset.i!, 10)));
      dot.addEventListener('mouseenter', () => dot.setAttribute('fill-opacity', '1'));
      dot.addEventListener('mouseleave', () => dot.setAttribute('fill-opacity', '0.5'));
    });
  }
}

function renderHeatmap(matrix: number[][], title: string, onCellClick?: (i: number, j: number) => void) {
  const area = $('#chart-display')!;
  const n    = matrix.length;

  const TITLE_H  = 36;
  const CB_W     = 20;
  const CB_GAP   = 10;
  const CB_LABEL = 40;
  const dpr      = window.devicePixelRatio || 1;
  const availW   = area.clientWidth  || 600;
  const availH   = (area.clientHeight || 480) - TITLE_H;
  const cssSize  = Math.min(availW - CB_W - CB_GAP - CB_LABEL, availH);
  const pxSize   = Math.round(cssSize * dpr);

  let lo = Infinity, hi = -Infinity;
  matrix.forEach(row => row.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));

  const colorScale = (v: number): [number,number,number] => {
    const t = (v - lo) / (hi - lo || 1);
    if (t < 0.5) {
      const s = t * 2;
      return [Math.round(s*255), Math.round(s*255), 255];
    } else {
      const s = (t - 0.5) * 2;
      const b = Math.round((1 - s) * 255);
      return [255, b, b];
    }
  };

  const canvas = document.createElement('canvas');
  canvas.width  = pxSize;
  canvas.height = pxSize;
  const ctx  = canvas.getContext('2d')!;
  const cell = pxSize / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const [r, g, b] = colorScale(matrix[i][j]);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(Math.round(j * cell), Math.round(i * cell),
                   Math.ceil(cell),      Math.ceil(cell));
    }
  }
  canvas.style.cssText = `
    display:block; flex-shrink:0;
    width:${cssSize}px; height:${cssSize}px;
    border-radius:3px;`;

  const cbH   = cssSize;
  const cbSvgW = CB_W + CB_GAP + CB_LABEL;
  const stops = Array.from({ length: 21 }, (_, i) => {
    const t   = i / 20;
    const v   = hi - t * (hi - lo);
    const [r, g, b] = colorScale(v);
    return `<stop offset="${(t * 100).toFixed(0)}%" stop-color="rgb(${r},${g},${b})"/>`;
  }).join('');

  const nTicks = 5;
  const ticks  = Array.from({ length: nTicks }, (_, i) => {
    const t   = i / (nTicks - 1);
    const v   = hi - t * (hi - lo);
    const y   = t * cbH;
    return `<line x1="${CB_W}" y1="${y.toFixed(1)}" x2="${CB_W + 5}" y2="${y.toFixed(1)}" stroke="#555" stroke-width="1"/>
            <text x="${CB_W + 8}" y="${(y + 4).toFixed(1)}" fill="#7a7f85" font-size="10">${v.toFixed(2)}</text>`;
  }).join('');

  const colorbarSvg = `
    <svg width="${cbSvgW}" height="${cbH}" style="flex-shrink:0;display:block">
      <defs>
        <linearGradient id="cb-grad" x1="0" y1="0" x2="0" y2="1" gradientUnits="objectBoundingBox">
          ${stops}
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${CB_W}" height="${cbH}" fill="url(#cb-grad)" rx="2"/>
      <line x1="${CB_W}" y1="0" x2="${CB_W}" y2="${cbH}" stroke="#444" stroke-width="0.5"/>
      ${ticks}
    </svg>`;

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">${title}</span>
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-family:var(--font-mono);font-size:9px;color:var(--muted)">${n}×${n}</span>
        <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
      </div>
    </div>
    <div class="heatmap-wrap">
      <div style="display:flex;align-items:flex-start;gap:${CB_GAP}px;">
        <div id="heatmap-canvas-slot"></div>
        ${colorbarSvg}
      </div>
    </div>`;

  area.querySelector('#heatmap-canvas-slot')!.appendChild(canvas);
  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onCellClick) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const col  = Math.floor((e.clientX - rect.left) / rect.width  * n);
      const row  = Math.floor((e.clientY - rect.top)  / rect.height * n);
      if (col >= 0 && col < n && row >= 0 && row < n) onCellClick(row, col);
    });
  }
}

function renderBarChart(data: number[], _labels: string[], title: string, xLabel: string, yLabel: string, onBarClick?: (index: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w    = area.clientWidth  || 600;
  const svgH = (area.clientHeight || 340) - TITLE_H;
  const pad  = { top: 24, right: 24, bottom: 58, left: 62 };
  const iw   = w    - pad.left - pad.right;
  const ih   = svgH - pad.top  - pad.bottom;

  const max = Math.max(...data, 1e-9);
  const bw  = iw / data.length;

  const bars = data.map((v, i) => {
    const bh = (v / max) * ih;
    return `<rect x="${(i * bw + bw * 0.1).toFixed(1)}" y="${(ih - bh).toFixed(1)}"
              width="${(bw * 0.8).toFixed(1)}" height="${bh.toFixed(1)}"
              fill="#00c4a7" fill-opacity="0.85" rx="1" data-i="${i}"
              style="cursor:${onBarClick ? 'pointer' : 'default'}"/>`;
  }).join('');

  const yTicks = 5;
  const yTickSvg = Array.from({ length: yTicks }, (_, i) => {
    const v = (max * i) / (yTicks - 1);
    const y = ih - (v / max) * ih;
    return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-8" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v.toFixed(2)}</text>`;
  }).join('');

  const nXLabels = Math.min(10, data.length);
  const xLabelSvg = Array.from({ length: nXLabels }, (_, i) => {
    const idx = Math.round((i / (nXLabels - 1 || 1)) * (data.length - 1));
    const x   = (idx + 0.5) * bw;
    return `<text x="${x.toFixed(1)}" y="${(ih + 18).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${idx}</text>`;
  }).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">${title}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block">
      <g transform="translate(${pad.left},${pad.top})">
        ${yTickSvg}${bars}${xLabelSvg}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-46" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="11">${yLabel}</text>
        <text x="${iw/2}" y="${(ih + 42).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="11">${xLabel}</text>
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onBarClick) {
    area.querySelectorAll<SVGRectElement>('rect[data-i]').forEach(r => {
      r.addEventListener('click', () => onBarClick(parseInt(r.dataset.i!, 10)));
      r.addEventListener('mouseenter', () => r.setAttribute('fill-opacity', '1'));
      r.addEventListener('mouseleave', () => r.setAttribute('fill-opacity', '0.85'));
    });
  }
}

// ─── Ramachandran chart ───────────────────────────────────────────────────────

function renderRamachandranChart(result: any, onResidueClick?: (atomIdx: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 580;
  const svgH    = (area.clientHeight || 540) - TITLE_H;
  const pad     = { top: 24, right: 32, bottom: 48, left: 52 };
  const iw      = w - pad.left - pad.right;
  const ih      = svgH - pad.top - pad.bottom;
  const { density, residues, mode, n_frames } = result;

  const toSvg = (phi: number, psi: number): [string, string] => [
    ((phi + 180) / 360 * iw).toFixed(1),
    (ih - (psi + 180) / 360 * ih).toFixed(1),
  ];

  const BINS = density.length;
  const cellW = iw / BINS, cellH = ih / BINS;
  let heatCells = '';
  for (let yi = 0; yi < BINS; yi++) {
    for (let xi = 0; xi < BINS; xi++) {
      const v = density[yi][xi] as number;
      if (v < 0.004) continue;
      const t = Math.pow(v, 0.35);
      const rr = Math.round(t*60), gg = Math.round(120+t*80), bb = Math.round(130+t*40);
      const x = (xi*cellW).toFixed(1), y = (ih-(yi+1)*cellH).toFixed(1);
      heatCells += `<rect x="${x}" y="${y}" width="${(cellW+0.6).toFixed(1)}" height="${(cellH+0.6).toFixed(1)}" fill="rgb(${rr},${gg},${bb})" opacity="${(t*0.85).toFixed(2)}"/>`;
    }
  }

  const regionPath = (pts: readonly (readonly [number,number])[]) =>
    pts.map(([p,s],i) => `${i===0?'M':'L'}${toSvg(p,s).join(',')}`).join(' ') + ' Z';
  const helixPts  = [[-145,-60],[-30,-60],[-30,20],[-145,20]] as const;
  const sheetPts  = [[-170,90],[-55,90],[-55,180],[-170,180]] as const;
  const sheet2Pts = [[-170,-180],[-55,-180],[-55,-155],[-170,-155]] as const;
  const lhPts     = [[30,20],[80,20],[80,80],[30,80]] as const;

  const nRes = (residues as any[]).length;
  const dots = (residues as any[]).map((r, k) => {
    if (!r) return '';
    const phi = r.phi_mean, psi = r.psi_mean;
    if (phi == null || psi == null || !isFinite(phi) || !isFinite(psi)) return '';
    const [cx,cy] = toSvg(phi, psi);
    const hue = Math.round(k/nRes*300);
    const title = `${r.res_name??'?'} ${r.res_seq??'?'}  phi=${phi.toFixed(1)}  psi=${psi.toFixed(1)}  sigma=${(r.phi_std??0).toFixed(1)}`;
    return `<circle cx="${cx}" cy="${cy}" r="5" fill="hsl(${hue},80%,62%)" stroke="#0d0e0f" stroke-width="0.8" opacity="0.92" data-atom="${r.atom_idx}" style="cursor:${onResidueClick?'pointer':'default'}"><title>${title}</title></circle>`;
  }).join('');

  const xTicks = [-180,-90,0,90,180].map(v => {
    const [x] = toSvg(v, 0);
    return `<line x1="${x}" y1="0" x2="${x}" y2="${ih}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="${x}" y="${(ih+16).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${v}</text>`;
  }).join('');
  const yTicks = [-180,-90,0,90,180].map(v => {
    const [,y] = toSvg(0, v);
    return `<line x1="0" y1="${y}" x2="${iw}" y2="${y}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-6" y="${(+y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v}</text>`;
  }).join('');

  const [aLx,aLy] = toSvg(-87,-20), [bLx,bLy] = toSvg(-112,135), [lLx,lLy] = toSvg(55,50);
  const [z0x]     = toSvg(0,-180);
  const [,z0y]    = toSvg(-180,0);
  const nAlpha = (residues as any[]).filter(r=>r?.phi_mean!=null&&r.phi_mean>=-145&&r.phi_mean<=-30&&r.psi_mean>=-60&&r.psi_mean<=20).length;
  const nBeta  = (residues as any[]).filter(r=>r?.phi_mean!=null&&r.phi_mean>=-170&&r.phi_mean<=-55&&r.psi_mean>=90).length;
  const nOther = nRes - nAlpha - nBeta;
  const modeLabel = mode==='backbone'?'Backbone φ/ψ':'Cα pseudo-dihedral';
  const hint = onResidueClick ? `<text x="${iw}" y="-4" text-anchor="end" fill="#3d4245" font-size="9">click residue → dihedral time series</text>` : '';

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">Ramachandran · ${modeLabel} · ${n_frames} frames · ${nRes} residues</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block">
      <g transform="translate(${pad.left},${pad.top})">
        ${heatCells}
        <path d="${regionPath(helixPts)}"  fill="#00c4a7" fill-opacity="0.07" stroke="#00c4a7" stroke-width="0.8" stroke-opacity="0.5"/>
        <path d="${regionPath(sheetPts)}"  fill="#5b8dee" fill-opacity="0.07" stroke="#5b8dee" stroke-width="0.8" stroke-opacity="0.5"/>
        <path d="${regionPath(sheet2Pts)}" fill="#5b8dee" fill-opacity="0.07" stroke="#5b8dee" stroke-width="0.8" stroke-opacity="0.5"/>
        <path d="${regionPath(lhPts)}"     fill="#e09a2e" fill-opacity="0.05" stroke="#e09a2e" stroke-width="0.8" stroke-opacity="0.3"/>
        ${xTicks}${yTicks}
        <line x1="${z0x}" y1="0" x2="${z0x}" y2="${ih}" stroke="#444" stroke-width="0.5" stroke-dasharray="3,4"/>
        <line x1="0" y1="${z0y}" x2="${iw}" y2="${z0y}" stroke="#444" stroke-width="0.5" stroke-dasharray="3,4"/>
        ${dots}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-38" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="11">ψ (°)</text>
        <text x="${iw/2}" y="${(ih+40).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="11">φ (°)</text>
        <text x="${aLx}" y="${aLy}" text-anchor="middle" fill="#00c4a7" font-size="11" font-family="monospace" opacity="0.75">α</text>
        <text x="${bLx}" y="${bLy}" text-anchor="middle" fill="#5b8dee" font-size="11" font-family="monospace" opacity="0.75">β</text>
        <text x="${lLx}" y="${lLy}" text-anchor="middle" fill="#e09a2e" font-size="9"  font-family="monospace" opacity="0.5">LH</text>
        <rect x="${iw-132}" y="4" width="132" height="50" rx="3" fill="#141618" opacity="0.8"/>
        <circle cx="${iw-122}" cy="16" r="4" fill="#00c4a7" opacity="0.8"/>
        <text x="${iw-114}" y="20" fill="#7a7f85" font-size="9" font-family="monospace">alpha: ${nAlpha}</text>
        <circle cx="${iw-122}" cy="30" r="4" fill="#5b8dee" opacity="0.8"/>
        <text x="${iw-114}" y="34" fill="#7a7f85" font-size="9" font-family="monospace">beta:  ${nBeta}</text>
        <circle cx="${iw-122}" cy="44" r="4" fill="#7a7f85" opacity="0.5"/>
        <text x="${iw-114}" y="48" fill="#7a7f85" font-size="9" font-family="monospace">other: ${nOther}</text>
        ${hint}
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onResidueClick) {
    area.querySelectorAll<SVGCircleElement>('circle[data-atom]').forEach(dot => {
      dot.addEventListener('click', () => onResidueClick(parseInt(dot.dataset.atom!, 10)));
      dot.addEventListener('mouseenter', () => { dot.setAttribute('r','7'); dot.setAttribute('stroke-width','1.5'); });
      dot.addEventListener('mouseleave', () => { dot.setAttribute('r','5'); dot.setAttribute('stroke-width','0.8'); });
    });
  }
}

// ─── PRS chart ────────────────────────────────────────────────────────────────

function renderPrsChart(result: any, onPairClick?: (i:number,j:number)=>void, onResidueClick?: (i:number)=>void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const BAR_H   = 100;
  const w       = area.clientWidth  || 620;
  const totalH  = (area.clientHeight || 560) - TITLE_H;
  const hmH     = totalH - BAR_H * 2 - 8;
  const { matrix, effectiveness, sensitivity } = result;
  const n = matrix.length;

  const dpr = window.devicePixelRatio || 1;
  const hmCSS = Math.min(w - 20, hmH);
  const hmPX  = Math.round(hmCSS * dpr);
  const canvas = document.createElement('canvas');
  canvas.width = hmPX; canvas.height = hmPX;
  canvas.style.cssText = `display:block;width:${hmCSS}px;height:${hmCSS}px;border-radius:2px;`;
  const ctx = canvas.getContext('2d')!;
  const cell = hmPX / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = (matrix[i] as number[])[j];
      ctx.fillStyle = `rgb(${Math.round(v*255)},${Math.round((1-v)*100)},${Math.round((1-v)*180)})`;
      ctx.fillRect(Math.round(j*cell), Math.round(i*cell), Math.ceil(cell), Math.ceil(cell));
    }
  }
  if (onPairClick) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const col  = Math.floor((e.clientX-rect.left)/rect.width  * n);
      const row  = Math.floor((e.clientY-rect.top) /rect.height * n);
      if (col>=0&&col<n&&row>=0&&row<n) onPairClick(row, col);
    });
  }

  const makeBar = (data: number[], title: string, color: string) => {
    const PL=52, PR=16, PT=14, PB=22;
    const bw = (w-PL-PR)/data.length;
    const max = Math.max(...data, 1e-9);
    const ih  = BAR_H - PT - PB;
    const bars = data.map((v,i) =>
      `<rect x="${(PL+i*bw+bw*0.05).toFixed(1)}" y="${(PT+ih*(1-v/max)).toFixed(1)}"
        width="${(bw*0.9).toFixed(1)}" height="${(ih*v/max).toFixed(1)}"
        fill="${color}" opacity="0.85" data-i="${i}" rx="1"
        style="cursor:${onResidueClick?'pointer':'default'}"/>`
    ).join('');
    const nxt = Math.min(8, data.length);
    const xL  = Array.from({length:nxt}, (_,k) => {
      const idx = Math.round(k/(nxt-1||1)*(data.length-1));
      return `<text x="${(PL+(idx+0.5)*bw).toFixed(1)}" y="${(BAR_H-4).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">${idx}</text>`;
    }).join('');
    return `<svg width="${w}" height="${BAR_H}" style="display:block">
      <text x="${PL}" y="10" fill="#7a7f85" font-size="9" font-family="monospace">${title}</text>
      ${bars}${xL}
    </svg>`;
  };

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">Perturbation Response Scanning (${n}×${n})</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <div style="display:flex;justify-content:center;padding:4px 0 2px">
      <div id="prs-canvas-slot"></div>
    </div>
    ${makeBar(effectiveness, 'Effectiveness  (how much perturbation at i propagates outward)', '#e09a2e')}
    ${makeBar(sensitivity,   'Sensitivity    (how much residue j responds to all perturbations)', '#5b8dee')}`;

  area.querySelector('#prs-canvas-slot')!.appendChild(canvas);
  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onResidueClick) {
    area.querySelectorAll<SVGRectElement>('rect[data-i]').forEach(r => {
      r.addEventListener('click', () => onResidueClick(parseInt(r.dataset.i!, 10)));
      r.addEventListener('mouseenter', () => r.setAttribute('opacity','1'));
      r.addEventListener('mouseleave', () => r.setAttribute('opacity','0.85'));
    });
  }
}

// ─── Cluster chart ────────────────────────────────────────────────────────────

function renderClusterChart(result: any, onFrameClick?: (frame: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const BAR_H   = 80;
  const w       = area.clientWidth  || 620;
  const svgH    = (area.clientHeight || 500) - TITLE_H - BAR_H;
  const pad     = { top: 20, right: 24, bottom: 44, left: 52 };
  const iw      = w - pad.left - pad.right;
  const ih      = svgH - pad.top - pad.bottom;
  const { assignments, centers, populations, method, n_clusters, implied_timescales: its } = result;
  const nFrames = assignments.length;
  const pts     = (window as any).__lastPcaPts as [number,number][] | null ?? null;

  if (!pts || pts.length !== nFrames) {
    area.innerHTML = `<div class="chart-placeholder">
      <div class="placeholder-icon">◈</div>
      <div class="placeholder-name">Clustering complete — run PCA first for scatter view</div>
      <div class="placeholder-hint">${n_clusters} clusters · ${method}</div></div>`;
    return;
  }

  const xmin = Math.min(...pts.map((p: any) => p[0])), xmax = Math.max(...pts.map((p: any) => p[0]));
  const ymin = Math.min(...pts.map((p: any) => p[1])), ymax = Math.max(...pts.map((p: any) => p[1]));
  const xr = (xmax-xmin)||1, yr = (ymax-ymin)||1;
  const sx = (v: number) => ((v-xmin)/xr * iw).toFixed(1);
  const sy = (v: number) => (ih - (v-ymin)/yr * ih).toFixed(1);
  const palette = (k: number) => `hsl(${(k/n_clusters)*300},70%,55%)`;
  const pcca: number[][] = result.pcca_membership ?? [];

  const dots = pts.map(([x, y]: [number,number], f: number) => {
    const k = assignments[f] as number;
    const opacity = pcca.length > 0 ? (0.4 + 0.6*(pcca[f]?.[k]??0.5)).toFixed(2) : '0.7';
    return `<circle cx="${sx(x)}" cy="${sy(y)}" r="3.5" fill="${palette(k)}"
      opacity="${opacity}" data-f="${f}" style="cursor:${onFrameClick?'pointer':'default'}"/>`;
  }).join('');

  const centroids = centers.map((c: [number,number], k: number) => {
    const cx = sx(c[0]), cy = sy(c[1]), col = palette(k);
    return `<line x1="${(+cx-6).toFixed(1)}" y1="${(+cy-6).toFixed(1)}" x2="${(+cx+6).toFixed(1)}" y2="${(+cy+6).toFixed(1)}" stroke="${col}" stroke-width="2.5"/>
            <line x1="${(+cx-6).toFixed(1)}" y1="${(+cy+6).toFixed(1)}" x2="${(+cx+6).toFixed(1)}" y2="${(+cy-6).toFixed(1)}" stroke="${col}" stroke-width="2.5"/>
            <text x="${(+cx+8).toFixed(1)}" y="${(+cy+4).toFixed(1)}" fill="${col}" font-size="9" font-family="monospace">C${k}</text>`;
  }).join('');

  const itsText = its?.length > 0 ? its.slice(0,3).map((t: number, i: number) => `ITS${i+2}=${t.toFixed(1)} fr`).join('  ') : '';

  const bw   = (w-52-24) / n_clusters;
  const bars = populations.map((p: number, k: number) => {
    const bh = p * (BAR_H - 26);
    return `<rect x="${(52+k*bw+bw*0.1).toFixed(1)}" y="${(BAR_H-26-bh).toFixed(1)}"
      width="${(bw*0.8).toFixed(1)}" height="${bh.toFixed(1)}" fill="${palette(k)}" opacity="0.85" rx="1"/>
      <text x="${(52+(k+0.5)*bw).toFixed(1)}" y="${(BAR_H-6).toFixed(1)}" text-anchor="middle"
        fill="#7a7f85" font-size="9" font-family="monospace">${(p*100).toFixed(0)}%</text>`;
  }).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">${method==='msm'?'MSM / PCCA':'K-means'} · ${n_clusters} clusters${itsText?' · '+itsText:''}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block">
      <g transform="translate(${pad.left},${pad.top})">
        ${dots}${centroids}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-38" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">PC2</text>
        <text x="${iw/2}" y="${(ih+36).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">PC1</text>
        ${onFrameClick?`<text x="${iw}" y="-4" text-anchor="end" fill="#3d4245" font-size="9">click frame → 3D view</text>`:''}
      </g>
    </svg>
    <svg width="${w}" height="${BAR_H}" style="display:block;border-top:1px solid #2e3235">
      <text x="52" y="10" fill="#7a7f85" font-size="9" font-family="monospace">cluster populations</text>
      ${bars}
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onFrameClick) {
    area.querySelectorAll<SVGCircleElement>('circle[data-f]').forEach(dot => {
      dot.addEventListener('click', () => onFrameClick(parseInt(dot.dataset.f!, 10)));
      dot.addEventListener('mouseenter', () => dot.setAttribute('opacity','1'));
      dot.addEventListener('mouseleave', () => {
        const f = parseInt(dot.dataset.f!, 10), k = assignments[f];
        dot.setAttribute('opacity', pcca.length>0 ? (0.4+0.6*(pcca[f]?.[k]??0.5)).toFixed(2) : '0.7');
      });
    });
  }
}

// ─── Geometry chart ───────────────────────────────────────────────────────────

function renderGeometryChart(result: any, onFrameClick?: (frame: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 620;
  const svgH    = (area.clientHeight || 420) - TITLE_H;
  const pad     = { top: 20, right: 24, bottom: 48, left: 62 };
  const iw      = w - pad.left - pad.right;
  const ih      = svgH - pad.top - pad.bottom;
  const { series, n_frames, source } = result;
  if (!series || series.length === 0) { showChartPlaceholder('Geometry Monitor'); return; }

  const allVals: number[] = (series as any[]).flatMap((s: any) => s.values.filter(isFinite));
  const ymin = Math.min(...allVals), ymax = Math.max(...allVals);
  const yr   = (ymax-ymin) || 1;
  const sy   = (v: number) => ih - ((v-ymin)/yr * ih);
  const sx   = (i: number) => (i / (n_frames-1||1)) * iw;
  const palette = ['#00c4a7','#5b8dee','#e09a2e','#e05c5c','#a78dee'];

  const lines = (series as any[]).map((s: any, si: number) => {
    const isComposite = s.kind === 'composite';
    const col   = isComposite ? '#e2e4e6' : palette[si % palette.length];
    const dash  = isComposite ? 'stroke-dasharray="5,3"' : '';
    const width = isComposite ? '2' : '1.5';
    const pts = (s.values as number[]).map((v, i) =>
      isFinite(v) ? `${sx(i).toFixed(1)},${sy(v).toFixed(1)}` : null
    ).filter(Boolean).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="${width}" ${dash} stroke-linejoin="round" opacity="0.9"/>`;
  }).join('');

  const yTicks = Array.from({length:5}, (_,i) => {
    const v = ymin + yr*i/4, y = sy(v);
    return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-6" y="${(y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v.toFixed(2)}</text>`;
  }).join('');
  const nxt = Math.min(8, n_frames);
  const xTicks = Array.from({length:nxt}, (_,i) => {
    const idx = Math.round(i/(nxt-1||1)*(n_frames-1));
    return `<text x="${sx(idx).toFixed(1)}" y="${(ih+16).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${idx}</text>`;
  }).join('');

  // Vertical legend box
  const LEG_ROW_H = 14, LEG_PAD = 6, LEG_W = 160;
  const LEG_H = (series as any[]).length * LEG_ROW_H + LEG_PAD * 2;
  const LEG_X = iw - LEG_W - 4, LEG_Y = 4;
  const legendRows = (series as any[]).map((s: any, si: number) => {
    const isComposite = s.kind === 'composite';
    const col  = isComposite ? '#e2e4e6' : palette[si % palette.length];
    const dash = isComposite ? 'stroke-dasharray="4,2"' : '';
    const ry   = LEG_Y + LEG_PAD + si*LEG_ROW_H + LEG_ROW_H/2;
    return `<line x1="${LEG_X+LEG_PAD}" y1="${ry}" x2="${LEG_X+LEG_PAD+16}" y2="${ry}" stroke="${col}" stroke-width="1.5" ${dash}/>
            <text x="${LEG_X+LEG_PAD+20}" y="${ry+3.5}" fill="${col}" font-size="9" font-family="monospace">${s.label}</text>`;
  }).join('');
  const legend = `<rect x="${LEG_X}" y="${LEG_Y}" width="${LEG_W}" height="${LEG_H}"
    rx="3" fill="#141618" opacity="0.82" stroke="#2e3235" stroke-width="0.5"/>
    ${legendRows}`;

  const hitRects = onFrameClick ? Array.from({length:n_frames}, (_,i) =>
    `<rect x="${(sx(i)-5).toFixed(1)}" y="0" width="10" height="${ih}" fill="transparent" data-f="${i}" style="cursor:pointer"/>`
  ).join('') : '';

  const overlayBtn = source === 'umbrella'
    ? `<button class="chart-export-btn" id="pmf-overlay-btn" style="margin-right:8px">Overlay on PMF</button>` : '';

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">Geometry Monitor · ${source} · ${(series as any[]).length} series</span>
      <div style="display:flex;gap:4px;align-items:center">
        ${overlayBtn}
        <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
      </div>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block">
      <g transform="translate(${pad.left},${pad.top})">
        ${yTicks}${xTicks}${lines}${hitRects}${legend}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-48" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="11">Distance (Å) / Angle (°)</text>
        <text x="${iw/2}" y="${(ih+40).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="11">${source==='umbrella'?'Window index':'Frame'}</text>
        ${onFrameClick?`<line id="geo-hover" x1="0" y1="0" x2="0" y2="${ih}" stroke="#00c4a7" stroke-width="0.5" stroke-dasharray="3,3" opacity="0" pointer-events="none"/>`:''}
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  document.getElementById('pmf-overlay-btn')?.addEventListener('click', () => {
    const mbarData = (window as any).__lastMbarResult;
    if (!mbarData) { alert('Run MBAR / PMF first.'); return; }
    renderPmfChart(mbarData, undefined, result);
  });
  if (onFrameClick) {
    const svg = area.querySelector('svg')!;
    const hl  = document.getElementById('geo-hover');
    svg.addEventListener('mousemove', (e) => {
      const rect = svg.getBoundingClientRect();
      const x = sx(Math.max(0, Math.min(n_frames-1, Math.round((e.clientX-rect.left-pad.left)/iw*(n_frames-1))))).toFixed(1);
      hl?.setAttribute('x1',x); hl?.setAttribute('x2',x); hl?.setAttribute('opacity','1');
    });
    svg.addEventListener('mouseleave', () => hl?.setAttribute('opacity','0'));
    area.querySelectorAll<SVGRectElement>('rect[data-f]').forEach(r =>
      r.addEventListener('click', () => onFrameClick(parseInt(r.dataset.f!,10)))
    );
  }
}

// ─── Dihedral time-series chart ───────────────────────────────────────────────

function renderDihedralTimeSeries(seriesData: any[], onFrameClick?: (frame: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 620;
  const panelH  = Math.floor(((area.clientHeight||500) - TITLE_H) / 2);
  const pad     = { top: 18, right: 24, bottom: 36, left: 52 };
  const iw      = w - pad.left - pad.right;
  const ih      = panelH - pad.top - pad.bottom;
  const nFrames = seriesData[0]?.phi?.length ?? 0;
  if (nFrames === 0) { showChartPlaceholder('Dihedral Time Series'); return; }

  const palette = ['#00c4a7','#5b8dee','#e09a2e','#e05c5c','#a78dee'];
  const sx = (i: number) => (i/(nFrames-1||1)) * iw;

  const makePanel = (angleKey: 'phi'|'psi', yLabel: string, svgY: number) => {
    const toY = (deg: number) => ih - ((deg+180)/360 * ih);
    const band = (lo: number, hi: number, col: string) => {
      const y1 = Math.min(toY(lo),toY(hi)), h = Math.abs(toY(lo)-toY(hi));
      return `<rect x="0" y="${y1.toFixed(1)}" width="${iw}" height="${h.toFixed(1)}" fill="${col}" opacity="0.06"/>`;
    };
    const bands = [band(-145,-30,'#00c4a7'), band(90,180,'#5b8dee'), band(-180,-155,'#5b8dee')].join('');
    const lines = seriesData.map((s: any, si: number) => {
      const col = palette[si % palette.length];
      const pts = (s[angleKey] as number[]).map((v, i) =>
        isFinite(v) ? `${sx(i).toFixed(1)},${toY(v).toFixed(1)}` : null
      ).filter(Boolean).join(' ');
      return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.2" stroke-linejoin="round" opacity="0.85"/>`;
    }).join('');
    const yTicks = [-180,-90,0,90,180].map(v => {
      const y = toY(v);
      return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
              <text x="-6" y="${(y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="9">${v}</text>`;
    }).join('');
    const xTicks = angleKey==='psi' ? Array.from({length:Math.min(6,nFrames)}, (_,i) => {
      const idx = Math.round(i/(Math.min(6,nFrames)-1||1)*(nFrames-1));
      return `<text x="${sx(idx).toFixed(1)}" y="${(ih+14).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">${idx}</text>`;
    }).join('') : '';
    return `<g transform="translate(${pad.left},${svgY+pad.top})">
      ${bands}${yTicks}${lines}
      <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
      <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
      <text x="${-(ih/2)}" y="-38" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">${yLabel} (°)</text>
      ${xTicks}
    </g>`;
  };

  const hitRects = onFrameClick ? Array.from({length:nFrames}, (_,i) =>
    `<rect x="${(pad.left+sx(i)-4).toFixed(1)}" y="0" width="8" height="${TITLE_H+panelH*2}" fill="transparent" data-f="${i}" style="cursor:pointer"/>`
  ).join('') : '';

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">Dihedral Time Series · ${seriesData.map((s:any)=>`${s.res_name} ${s.res_seq}`).join(', ')}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${panelH*2}" viewBox="0 0 ${w} ${panelH*2}" style="display:block">
      ${makePanel('phi','φ',0)}
      ${makePanel('psi','ψ',panelH)}
      <text x="${pad.left+iw/2}" y="${panelH*2-4}" text-anchor="middle" fill="#7a7f85" font-size="10">Frame</text>
      ${hitRects}
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onFrameClick) {
    area.querySelectorAll<SVGRectElement>('rect[data-f]').forEach(r =>
      r.addEventListener('click', () => onFrameClick(parseInt(r.dataset.f!,10)))
    );
  }
}

// ─── SASA chart ───────────────────────────────────────────────────────────────

function renderSasaChart(result: any, onResidueClick?: (atomIdx: number) => void) {
  const area   = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 620;
  const totalH  = (area.clientHeight || 520) - TITLE_H;
  const BAR_H   = Math.floor(totalH * 0.62);
  const LINE_H  = totalH - BAR_H;
  const { per_residue_mean, per_residue_std, total_per_frame, res_labels } = result;
  const n_res = per_residue_mean.length, n_frames = total_per_frame.length;

  const PL=58, PR=16, PT=20, PB=36;
  const bw  = (w-PL-PR) / n_res;
  const ih_b = BAR_H-PT-PB;
  const maxV = Math.max(...per_residue_mean, 1e-9);

  const bars = per_residue_mean.map((v: number, i: number) => {
    const sigma = per_residue_std[i] ?? 0;
    const frac  = v / maxV;
    const r = Math.round(frac*220), g = Math.round((1-Math.abs(frac-0.5)*2)*150), b = Math.round((1-frac)*210);
    const bh  = frac * ih_b;
    const bx  = (PL + i*bw + bw*0.1).toFixed(1);
    const by  = (PT + ih_b - bh).toFixed(1);
    const bwi = (bw*0.8).toFixed(1);
    const eH  = Math.min(sigma/maxV*ih_b, bh*0.5);
    const eY  = (PT + ih_b - bh - eH).toFixed(1);
    const eX  = (PL + (i+0.5)*bw).toFixed(1);
    return `<rect x="${bx}" y="${by}" width="${bwi}" height="${bh.toFixed(1)}"
      fill="rgb(${r},${g},${b})" opacity="0.85" rx="1" data-i="${i}"
      style="cursor:${onResidueClick?'pointer':'default'}"/>
      <line x1="${eX}" y1="${eY}" x2="${eX}" y2="${(PT+ih_b-bh+eH).toFixed(1)}"
        stroke="rgb(${r},${g},${b})" stroke-width="1" opacity="0.6"/>`;
  }).join('');

  const step  = Math.max(1, Math.ceil(n_res/20));
  const xTicks = per_residue_mean.map((_: number, i: number) => {
    if (i % step !== 0) return '';
    const x = (PL+(i+0.5)*bw).toFixed(1);
    return `<text x="${x}" y="${(PT+ih_b+14).toFixed(1)}" text-anchor="middle"
      fill="#7a7f85" font-size="8"
      transform="rotate(-35,${x},${(PT+ih_b+14).toFixed(1)})">${res_labels[i]??i}</text>`;
  }).join('');

  const yTicks = Array.from({length:5}, (_,i) => {
    const v = maxV*i/4, y = (PT+ih_b-(v/maxV)*ih_b).toFixed(1);
    return `<line x1="${PL}" y1="${y}" x2="${w-PR}" y2="${y}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="${(PL-4).toFixed(1)}" y="${(+y+4).toFixed(1)}" text-anchor="end"
              fill="#7a7f85" font-size="9">${v.toFixed(0)}</text>`;
  }).join('');

  const L2={t:12,b:28,l:PL,r:PR};
  const iw2=w-L2.l-L2.r, ih2=LINE_H-L2.t-L2.b;
  const tMin=Math.min(...total_per_frame), tMax=Math.max(...total_per_frame), tRange=(tMax-tMin)||1;
  const tsx = (i: number) => (i/(n_frames-1||1))*iw2;
  const tsy = (v: number) => ih2 - ((v-tMin)/tRange*ih2);
  const linePts = total_per_frame.map((v: number, i: number) =>
    `${tsx(i).toFixed(1)},${tsy(v).toFixed(1)}`).join(' ');
  const tyTick = [tMin,(tMin+tMax)/2,tMax].map(v =>
    `<text x="${(L2.l-4).toFixed(1)}" y="${(tsy(v)+4).toFixed(1)}" text-anchor="end"
      fill="#7a7f85" font-size="9">${v.toFixed(0)}</text>`
  ).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">SASA · ${n_res} residues · ${n_frames} frames</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${BAR_H}" viewBox="0 0 ${w} ${BAR_H}" style="display:block">
      ${yTicks}${bars}${xTicks}
      <line x1="${PL}" y1="${PT}" x2="${PL}" y2="${PT+ih_b}" stroke="#555" stroke-width="1"/>
      <line x1="${PL}" y1="${PT+ih_b}" x2="${w-PR}" y2="${PT+ih_b}" stroke="#555" stroke-width="1"/>
      <text x="${(PL-40).toFixed(1)}" y="${(PT+ih_b/2).toFixed(1)}"
        transform="rotate(-90,${(PL-40).toFixed(1)},${(PT+ih_b/2).toFixed(1)})"
        text-anchor="middle" fill="#7a7f85" font-size="10">SASA (Å²)</text>
      ${onResidueClick?`<text x="${w-PR}" y="${PT-4}" text-anchor="end" fill="#3d4245" font-size="9">click → 3D view</text>`:''}
    </svg>
    <svg width="${w}" height="${LINE_H}" viewBox="0 0 ${w} ${LINE_H}"
         style="display:block;border-top:1px solid #2e3235">
      <g transform="translate(${L2.l},${L2.t})">
        ${tyTick}
        <polyline points="${linePts}" fill="none" stroke="#00c4a7" stroke-width="1.5"/>
        <line x1="0" y1="0" x2="0" y2="${ih2}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih2}" x2="${iw2}" y2="${ih2}" stroke="#555" stroke-width="1"/>
        <text x="${iw2/2}" y="${(ih2+20).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">Frame</text>
        <text x="${(-ih2/2).toFixed(1)}" y="-40" transform="rotate(-90)"
          text-anchor="middle" fill="#7a7f85" font-size="10">Total SASA (Å²)</text>
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onResidueClick) {
    area.querySelectorAll<SVGRectElement>('rect[data-i]').forEach(bar => {
      bar.addEventListener('click', () => onResidueClick(parseInt(bar.dataset.i!, 10)));
      bar.addEventListener('mouseenter', () => bar.setAttribute('opacity','1'));
      bar.addEventListener('mouseleave', () => bar.setAttribute('opacity','0.85'));
    });
  }
}

// ─── Analysis 3D viewer ───────────────────────────────────────────────────────
// Opens analysis_viewer.html in a new browser tab, passing params as query string.
function openAnalysisViewer(params: Record<string, string | number>, title: string) {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)]))
  ).toString();
  window.open(`/analysis_viewer.html?${qs}`, '_blank');
  log(`3D viewer: ${title}`, 'success');
}

// ─── PMF chart ────────────────────────────────────────────────────────────────

function renderPmfChart(result: any, onWindowClick?: (windowIdx: number) => void, geoOverlay?: any) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 700;
  const totalH  = (area.clientHeight || 500) - TITLE_H;
  const histH   = Math.floor(totalH * 0.28);
  const pmfH    = totalH - histH;
  const PL = 62, PR = 22, PT = 14, PB = 52;
  const iw = w - PL - PR;

  const { bin_centers: bc, pmf, pmf_err,
          delta_g: dg, delta_g_err: dge,
          window_val0: wv, kde_x, kde_y, n_windows } = result;

  const cvMin   = Math.min(...wv) - 0.12;
  const cvMax   = Math.max(...wv) + 0.12;
  const cvRange = cvMax - cvMin || 1;
  const sx = (v: number) => ((v - cvMin) / cvRange * iw);

  const col = (i: number) => `hsl(${(i / n_windows) * 300},70%,55%)`;

  const kdeMax  = Math.max(...kde_y.flat().map(Number), 1e-9);
  const histIH  = histH - PT - 4;
  const histLines = kde_x.map((xs: number[], wi: number) => {
    const ys: number[] = kde_y[wi];
    if (!xs.length) return '';
    const pts = xs.map((x: number, j: number) =>
      `${sx(x).toFixed(1)},${(histIH - (ys[j] / kdeMax) * histIH).toFixed(1)}`).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${col(wi)}" stroke-width="1" opacity="0.7"/>`;
  }).join('');
  const vlines_h = wv.map((v: number, i: number) =>
    `<line x1="${sx(v).toFixed(1)}" y1="0" x2="${sx(v).toFixed(1)}" y2="${histIH}"
       stroke="${col(i)}" stroke-width="0.8" stroke-dasharray="2,2" opacity="0.45"/>`
  ).join('');

  const pmfIH   = pmfH - PT - PB;
  const pmfVals = pmf.filter((v: number) => isFinite(v));
  const pmfMin  = Math.min(...pmfVals);
  const pmfMax  = Math.max(...pmfVals);
  const pmfR    = pmfMax - pmfMin || 1;
  const sy = (v: number) => pmfIH - ((v - pmfMin) / pmfR * pmfIH);

  const errUpper = bc.map((x: number, i: number) =>
    `${sx(x).toFixed(1)},${sy(pmf[i] + (pmf_err[i] || 0)).toFixed(1)}`).join(' ');
  const errLower = [...bc].reverse().map((x: number, i: number) => {
    const ri = bc.length - 1 - i;
    return `${sx(x).toFixed(1)},${sy(pmf[ri] - (pmf_err[ri] || 0)).toFixed(1)}`;
  }).join(' ');
  const pmfLine = bc.map((x: number, i: number) =>
    `${sx(x).toFixed(1)},${sy(pmf[i]).toFixed(1)}`).join(' ');

  const dgIdx   = pmf.reduce((mi: number, v: number, i: number) =>
    isFinite(v) && v > (pmf[mi] ?? -Infinity) ? i : mi, 0);
  const dgAnnX  = sx(bc[dgIdx]).toFixed(1);
  const dgAnnY  = (sy(pmfMax) - 22).toFixed(1);

  const yTicks  = Array.from({ length: 5 }, (_, i) => {
    const v = pmfMin + pmfR * i / 4;
    const y = sy(v).toFixed(1);
    return `<line x1="0" y1="${y}" x2="${iw}" y2="${y}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-8" y="${(+y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="9">${v.toFixed(1)}</text>`;
  }).join('');

  const nXT = Math.min(10, wv.length);
  const xTicks = Array.from({ length: nXT }, (_, i) => {
    const v = wv[Math.round(i / (nXT - 1) * (wv.length - 1))];
    return `<text x="${sx(v).toFixed(1)}" y="${(pmfIH + 16).toFixed(1)}"
              text-anchor="middle" fill="#7a7f85" font-size="9">${v.toFixed(2)}</text>`;
  }).join('');

  const hitDots = bc.map((x: number, i: number) =>
    `<circle cx="${sx(x).toFixed(1)}" cy="${sy(pmf[i]).toFixed(1)}" r="5"
       fill="transparent" data-i="${i}"
       style="cursor:${onWindowClick ? 'pointer' : 'default'}"/>`
  ).join('');

  const vlines_p = wv.map((v: number, i: number) =>
    `<line x1="${sx(v).toFixed(1)}" y1="0" x2="${sx(v).toFixed(1)}" y2="${pmfIH}"
       stroke="${col(i)}" stroke-width="0.7" stroke-dasharray="2,2" opacity="0.35"/>`
  ).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">PMF · Umbrella Sampling (MBAR)</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${histH}" viewBox="0 0 ${w} ${histH}" style="display:block;border-bottom:1px solid #2e3235">
      <g transform="translate(${PL},${PT})">
        ${histLines}${vlines_h}
        <text x="-8" y="${(histIH/2).toFixed(0)}" transform="rotate(-90,-8,${(histIH/2).toFixed(0)})"
              text-anchor="middle" fill="#7a7f85" font-size="9">density</text>
      </g>
    </svg>
    <svg width="${w}" height="${pmfH}" viewBox="0 0 ${w} ${pmfH}" style="display:block" id="pmf-svg">
      <g transform="translate(${PL},${PT})">
        ${yTicks}${vlines_p}
        <polygon points="${errUpper} ${errLower}" fill="#00c4a7" opacity="0.12"/>
        <polyline points="${pmfLine}" fill="none" stroke="#00c4a7" stroke-width="2" stroke-linejoin="round"/>
        ${hitDots}
        <rect x="${(+dgAnnX - 100).toFixed(1)}" y="${(+dgAnnY - 13).toFixed(1)}" width="200" height="16"
              rx="2" fill="#141618" opacity="0.85"/>
        <text x="${dgAnnX}" y="${dgAnnY}" text-anchor="middle" fill="#00c4a7"
              font-size="11" font-family="monospace">
          ΔG‡ = ${dg.toFixed(2)} ± ${dge.toFixed(2)} kcal/mol
        </text>
        ${xTicks}
        <line x1="0" y1="0" x2="0" y2="${pmfIH}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${pmfIH}" x2="${iw}" y2="${pmfIH}" stroke="#555" stroke-width="1"/>
        <text x="${-(pmfIH/2)}" y="-48" transform="rotate(-90)" text-anchor="middle"
              fill="#7a7f85" font-size="10">PMF (kcal/mol)</text>
        <text x="${iw/2}" y="${(pmfIH + PB - 6).toFixed(1)}" text-anchor="middle"
              fill="#7a7f85" font-size="10">Reaction Coordinate</text>
        ${onWindowClick ? `<text x="${iw}" y="-2" text-anchor="end" fill="#3d4245" font-size="9">click curve → 3D view</text>` : ''}
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);

  // Cache for geometry overlay button
  (window as any).__lastMbarResult = result;

  // Geometry overlay panel
  if (geoOverlay && geoOverlay.series?.length > 0) {
    const geo = geoOverlay;
    const GEO_H = 120, GEO_PL = PL, GEO_PT = 10, GEO_PB = 30;
    const geoIH = GEO_H - GEO_PT - GEO_PB, geoIW = iw;
    const allVals: number[] = geo.series.flatMap((s: any) => s.values.filter(isFinite));
    const gMin = Math.min(...allVals), gMax = Math.max(...allVals), gRange = (gMax-gMin)||1;
    const geoPalette = ['#00c4a7','#e09a2e','#e05c5c','#5b8dee'];
    const gsy = (v: number) => geoIH - ((v-gMin)/gRange * geoIH);

    const gLines = geo.series.map((s: any, si: number) => {
      const isComposite = s.kind === 'composite';
      const col  = isComposite ? '#e2e4e6' : geoPalette[si % geoPalette.length];
      const dash = isComposite ? 'stroke-dasharray="4,2"' : '';
      const nWin = geo.n_frames;
      const pts  = s.values.map((v: number, i: number) => {
        const x = (i/(nWin-1||1)) * geoIW;
        return isFinite(v) ? `${x.toFixed(1)},${gsy(v).toFixed(1)}` : null;
      }).filter(Boolean).join(' ');
      return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.5" ${dash}/>`;
    }).join('');

    // Vertical legend box in overlay
    const gLegRowH = 12, gLegPad = 5, gLegW = 150;
    const gLegH = geo.series.length * gLegRowH + gLegPad * 2;
    const gLegX = geoIW - gLegW - 2, gLegY = 2;
    const gLegRows = geo.series.map((s: any, si: number) => {
      const isComposite = s.kind === 'composite';
      const col  = isComposite ? '#e2e4e6' : geoPalette[si % geoPalette.length];
      const dash = isComposite ? 'stroke-dasharray="4,2"' : '';
      const ry   = gLegY + gLegPad + si*gLegRowH + gLegRowH/2;
      return `<line x1="${gLegX+gLegPad}" y1="${ry}" x2="${gLegX+gLegPad+14}" y2="${ry}" stroke="${col}" stroke-width="1.5" ${dash}/>
              <text x="${gLegX+gLegPad+18}" y="${ry+3}" fill="${col}" font-size="8" font-family="monospace">${s.label}</text>`;
    }).join('');
    const gLegend = `<rect x="${gLegX}" y="${gLegY}" width="${gLegW}" height="${gLegH}"
      rx="2" fill="#141618" opacity="0.82" stroke="#2e3235" stroke-width="0.5"/>${gLegRows}`;

    const gTicks = [gMin,(gMin+gMax)/2,gMax].map(v =>
      `<text x="-4" y="${(gsy(v)+3).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="8">${v.toFixed(1)}</text>`
    ).join('');

    const geoSvg = document.createElementNS('http://www.w3.org/2000/svg','svg');
    geoSvg.setAttribute('width', String(w));
    geoSvg.setAttribute('height', String(GEO_H));
    geoSvg.setAttribute('viewBox', `0 0 ${w} ${GEO_H}`);
    geoSvg.style.cssText = 'display:block;border-top:1px solid #2e3235';
    geoSvg.innerHTML = `<g transform="translate(${GEO_PL},${GEO_PT})">
      ${gLines}${gTicks}${gLegend}
      <line x1="0" y1="0" x2="0" y2="${geoIH}" stroke="#555" stroke-width="1"/>
      <line x1="0" y1="${geoIH}" x2="${geoIW}" y2="${geoIH}" stroke="#555" stroke-width="1"/>
      <text x="${-(geoIH/2)}" y="-40" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="9">Geometry</text>
    </g>`;
    area.appendChild(geoSvg);
  }

  if (onWindowClick) {
    area.querySelectorAll<SVGCircleElement>('circle[data-i]').forEach(dot => {
      dot.addEventListener('click', () => {
        const i    = parseInt(dot.dataset.i!, 10);
        const bcV  = bc[i];
        let nearWin = 0, minD = Infinity;
        wv.forEach((v: number, wi: number) => {
          const d = Math.abs(v - bcV);
          if (d < minD) { minD = d; nearWin = wi; }
        });
        onWindowClick(nearWin);
      });
      dot.addEventListener('mouseenter', () => dot.setAttribute('r', '7'));
      dot.addEventListener('mouseleave', () => dot.setAttribute('r', '5'));
    });
  }
}

// ─── Tool runner ──────────────────────────────────────────────────────────────

async function runTool(tool: Tool) {
  if (tool.id === 'viewer3d') {
    openVisualizer(); return;
  }

  // Collect params if any
  const params: Record<string, string | number> = {};
  if (tool.params) {
    for (const p of tool.params) {
      const el = document.getElementById(`param-${p.id}`) as HTMLInputElement | HTMLSelectElement;
      params[p.id] = p.type === 'number' ? parseFloat(el?.value ?? p.default) : (el?.value ?? p.default);
    }
  }

  log(`Running ${tool.label}…`);
  showChartLoading(tool.label);

  const runBtn = $('#run-btn') as HTMLButtonElement;
  if (runBtn) { runBtn.disabled = true; runBtn.innerHTML = '<span class="spinner"></span> Running…'; }

  // Build click callbacks — each opens a self-contained analysis viewer tab
  const frameClick   = (i: number) =>
    openAnalysisViewer({ mode: 'frame',   frame: i }, `${tool.label} · Frame ${i}`);
  const residueClick = (i: number) =>
    openAnalysisViewer({ mode: 'residue', index: i }, `${tool.label} · Residue ${i+1}`);
  const pairClick    = (i: number, j: number) =>
    openAnalysisViewer({ mode: 'pair', i, j }, `${tool.label} · Residues ${i+1} & ${j+1}`);

  const lineClick    = tool.clickAction === 'frame'   ? frameClick
                     : tool.clickAction === 'residue' ? residueClick : undefined;
  const scatterClick = tool.clickAction === 'frame'   ? frameClick   : undefined;
  const barClick     = tool.clickAction === 'residue' ? residueClick : undefined;
  const heatClick    = tool.clickAction === 'pair'    ? pairClick    : undefined;

  try {
    // Tools with no Rust command (pure UI actions) — run directly and return
    if (!tool.invoke) {
      if (tool.id === 'umbrella_viewer') openUmbrellaViewer();
      else if (tool.id === 'dihedral_ts') {
        const idxStr = (document.getElementById('param-atom_indices') as HTMLInputElement)?.value ?? '0';
        const atomIndices = idxStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
        try {
          const series = await invoke<any[]>('get_dihedral_time_series', { atom_indices: atomIndices });
          renderDihedralTimeSeries(series, (frame: number) =>
            openAnalysisViewer({ mode: 'frame', frame }, `Dihedrals · Frame ${frame}`)
          );
          log(`Dihedral time series: ${series.map((s: any) => `${s.res_name} ${s.res_seq}`).join(', ')}`, 'success');
        } catch (e) {
          log(`Dihedral time series failed: ${e}`, 'error');
          showChartPlaceholder(tool.label);
        }
      } else {
        showChartPlaceholder(tool.label);
      }
      if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Run'; }
      return;
    }

    // Build invoke params — geometry needs special pre-processing
    let invokeParams: any = params;
    if (tool.id === 'geometry') {
      const sourceVal  = (params.source as string) ?? 'trajectory';
      const pairsStr   = (params.pairs   as string) ?? '';
      const tripletStr = (params.triplets as string) ?? '';
      const manualPairs: [number,number][]           = pairsStr.split(';').map(s => s.trim()).filter(Boolean)
        .map(s => s.split(',').map(Number) as [number,number]);
      const manualTriplets: [number,number,number][] = tripletStr.split(';').map(s => s.trim()).filter(Boolean)
        .map(s => s.split(',').map(Number) as [number,number,number]);
      const allPairs:   [number,number][] = [...manualPairs];
      const allLabels:  string[]          = manualPairs.map(() => '');
      const composites:    [number,number][] = [];
      const compWeights:   number[]          = [];
      const compLabels:    string[]          = [];

      if (sourceVal === 'umbrella' && cvRstBlocks.length > 0) {
        const cvCol = parseInt(($('#cv-col') as HTMLInputElement)?.value ?? '1', 10);
        const block = cvRstBlocks[cvCol - 1] ?? cvRstBlocks[0];
        if (block?.iat?.length >= 2) {
          const iat = (block.iat as number[]).map((n: number) => n - 1);
          if (iat.length === 4) {
            const r1Idx = allPairs.length;
            allPairs.push([iat[0], iat[1]]); allLabels.push('r1 (breaking)');
            const r2Idx = allPairs.length;
            allPairs.push([iat[2], iat[3]]); allLabels.push('r2 (forming)');
            composites.push([r1Idx, r2Idx]);
            compWeights.push((block.rstwt as number[])?.[1] ?? -1.0);
            compLabels.push('RC (r1−r2)');
          } else if (iat.length === 2) {
            allPairs.push([iat[0], iat[1]]);
            allLabels.push(block.comment || `d (col ${cvCol})`);
          }
        }
      }
      if (allPairs.length === 0 && manualTriplets.length === 0) {
        log('No atom pairs to measure — enter a cv.rst path or manual pairs.', 'warn');
        showChartPlaceholder(tool.label);
        if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Run'; }
        return;
      }
      invokeParams = {
        pairs: allPairs, triplets: manualTriplets, source: sourceVal,
        labels: allLabels, composites,
        composite_weights: compWeights, composite_labels: compLabels,
      };
    }

    const result: any = await invoke(tool.invoke, invokeParams);
    log(result.message ?? `${tool.label} complete.`, 'success');

    // Store bfactor data so Map to 3D works immediately
    if (tool.map3d === 'bfactor' && Array.isArray(result.data)) {
      try { await invoke('set_bfactors', { values: result.data }); } catch (_) {}
    }

    const map3dBtn = document.getElementById('map3d-btn') as HTMLButtonElement | null;

    switch (tool.chartType) {
      case 'line':
        renderLineChart(result.data, tool.label, tool.xLabel ?? 'Index', tool.yLabel ?? 'Value', lineClick);
        break;
      case 'heatmap':
        renderHeatmap(result.data, tool.label, heatClick);
        break;
      case 'scatter': {
        const pts = result.data.projections ?? result.data;
        (window as any).__lastPcaPts = pts;  // cache for cluster chart
        renderScatterChart(pts, tool.label, tool.xLabel ?? 'X', tool.yLabel ?? 'Y', scatterClick);
        if (result.data.explained_variance) {
          log(`PC1 ${(result.data.explained_variance[0]*100).toFixed(1)}%  PC2 ${(result.data.explained_variance[1]*100).toFixed(1)}%`);
        }
        break;
      }
      case 'bar':
        renderBarChart(result.data, [], tool.label, tool.xLabel ?? 'Index', tool.yLabel ?? 'Value', barClick);
        break;
      case 'sasa': {
        const sasaClick = (atomIdx: number) =>
          openAnalysisViewer({ mode: 'residue', index: atomIdx }, `SASA · Residue ${atomIdx}`);
        renderSasaChart(result.data, sasaClick);
        if (result.data.per_residue_mean) {
          try { await invoke('set_bfactors', { values: result.data.per_residue_mean }); } catch (_) {}
        }
        break;
      }
      case 'cluster': {
        renderClusterChart(result.data, (frame: number) =>
          openAnalysisViewer({ mode: 'cluster', cluster_frame: frame }, `Cluster · Frame ${frame}`)
        );
        const pop = (result.data.populations as number[]).map((p: number, k: number) => `C${k}=${(p*100).toFixed(0)}%`).join(' ');
        log(`${result.data.n_clusters} clusters · ${pop}`, 'success');
        break;
      }
      case 'geometry': {
        renderGeometryChart(result.data,
          result.data.source === 'trajectory'
            ? (frame: number) => openAnalysisViewer({ mode: 'frame', frame }, `Geometry · Frame ${frame}`)
            : undefined
        );
        break;
      }
      case 'ramachandran': {
        const residues = result.data?.residues ?? [];
        if (residues.length === 0) {
          log('No residue dihedral data — check atom selection includes backbone atoms.', 'warn');
          showChartPlaceholder(tool.label);
          break;
        }
        const ramaClick = (atomIdx: number) =>
          openAnalysisViewer({ mode: 'dihedral', index: atomIdx }, `Dihedrals · ${tool.label}`);
        renderRamachandranChart(result.data, ramaClick);
        log(`Mode: ${result.data.mode} · ${residues.length} residues`, 'success');
        break;
      }
      case 'prs': {
        const prsPairClick = (i: number, j: number) =>
          openAnalysisViewer({ mode: 'pair', i, j }, `PRS · Residues ${i+1} & ${j+1}`);
        const prsResClick = (i: number) =>
          openAnalysisViewer({ mode: 'residue', index: i }, `PRS · Residue ${i+1}`);
        renderPrsChart(result.data, prsPairClick, prsResClick);
        if (result.data.effectiveness) {
          try { await invoke('set_bfactors', { values: result.data.effectiveness }); } catch (_) {}
        }
        break;
      }
      case 'pmf': {
        (window as any).__lastMbarResult = result.data;
        const onWinClick = (winIdx: number) =>
          openAnalysisViewer(
            { mode: 'umbrella', window: winIdx },
            `Window ${winIdx} · CV=${result.data.window_val0[winIdx]?.toFixed(3) ?? '?'}`
          );
        renderPmfChart(result.data, onWinClick);
        log(`ΔG‡ = ${result.data.delta_g.toFixed(2)} ± ${result.data.delta_g_err.toFixed(2)} kcal/mol`, 'success');
        break;
      }
      case 'none':
        if (tool.id === 'umbrella_viewer') {
          openUmbrellaViewer();
        } else {
          showChartPlaceholder(tool.label);
        }
        break;
    }

    // Unlock Map to 3D button
    if (map3dBtn && tool.map3d) {
      map3dBtn.disabled = false;
      map3dBtn.classList.remove('disabled');
    }

  } catch (e) {
    log(`${tool.label} failed: ${e}`, 'error');
    showChartPlaceholder(tool.label);
  } finally {
    if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Run'; }
  }
}

// ─── Export CSV ───────────────────────────────────────────────────────────────
// Requests the CSV string from the server and triggers a browser download.
// No server-side file path needed — the data comes back as a string and
// the browser handles the "save file" dialog via a temporary object URL.
async function exportCsv() {
  log('Exporting CSV…');
  try {
    const csv = await invoke<string>('export_csv');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'analysis_results.csv';
    a.click();
    URL.revokeObjectURL(url);
    log('CSV downloaded.', 'success');
  } catch (e) {
    log(`Export failed: ${e}`, 'error');
  }
}

// ─── Viewer launchers ─────────────────────────────────────────────────────────
// All three open a new browser tab at the appropriate HTML page. The tab
// connects to the same server and same AppData state as the main window.

function openUmbrellaViewer() {
  if (!umbrellaLoaded) {
    log('Scan & load umbrella windows first (QM/MM sidebar).', 'warn');
    return;
  }
  window.open('/umbrella_viewer.html', '_blank');
  log('Umbrella viewer opened.', 'success');
}

function openVisualizer() {
  log('Opening 3D Visualizer…');
  window.open(`/visualizer.html?frames=${totalFrames}`, '_blank');
  log('Visualizer opened.', 'success');
}

// ─── Main UI ──────────────────────────────────────────────────────────────────

async function initUI() {
  const app = document.querySelector('#app')!;

  app.innerHTML = `
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg0: #0d0e0f; --bg1: #141618; --bg2: #1c1f21; --bg3: #242729;
    --border: #2e3235; --border-hi: #3d4245;
    --accent: #00c4a7; --accent-dim: #00856e; --accent2: #5b8dee;
    --text: #e2e4e6; --muted: #7a7f85;
    --danger: #e05c5c; --warn: #e09a2e; --success: #4ec97b;
    --font-sans: 'IBM Plex Sans', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
  }

  body { background: var(--bg0); color: var(--text); font-family: var(--font-sans); font-size: 13px; line-height: 1.5; }

  /* ── Shell: header + (sidebar | main) ── */
  .shell {
    display: grid;
    grid-template-columns: 280px 1fr;
    grid-template-rows: 44px 1fr;
    height: 100vh;
    overflow: hidden;
  }

  /* ── Header ── */
  .header {
    grid-column: 1 / -1;
    background: var(--bg1);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 18px;
    gap: 0;
    height: 44px;
  }
  .header-logo { font-family: var(--font-mono); font-size: 12px; font-weight: 500; color: var(--accent); letter-spacing: .05em; text-transform: uppercase; flex-shrink: 0; }
  .header-sep  { width: 1px; height: 18px; background: var(--border); margin: 0 14px; flex-shrink: 0; }
  .header-sub  { color: var(--muted); font-size: 10px; letter-spacing: .07em; text-transform: uppercase; flex-shrink: 0; }
  .header-cat { color: var(--muted); height: 15px; width: auto; display: block; flex-shrink: 0; margin-left: 10px; opacity: 0.65; transition: opacity .15s; }
  .header-cat:hover { opacity: 1; }

  .cat-nav {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin-left: 20px;
    height: 100%;
    flex: 1;
    overflow-x: auto;
    scrollbar-width: none;
  }
  .cat-nav::-webkit-scrollbar { display: none; }
  .cat-tab {
    display: flex;
    align-items: center;
    padding: 0 14px;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: color .15s, border-color .15s;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
    letter-spacing: .04em;
    flex-shrink: 0;
  }
  .cat-tab:hover  { color: var(--text); }
  .cat-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .status-pill { margin-left: auto; flex-shrink: 0; display: flex; align-items: center; gap: 6px; background: var(--bg2); border: 1px solid var(--border); border-radius: 100px; padding: 3px 10px 3px 8px; font-size: 10px; color: var(--muted); }
  .status-dot  { width: 6px; height: 6px; border-radius: 50%; background: var(--muted); transition: background .3s; }
  .status-dot.idle    { background: var(--muted); }
  .status-dot.partial { background: var(--warn); }
  .status-dot.ready   { background: var(--warn); }
  .status-dot.loaded  { background: var(--success); }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--bg1);
    border-right: 1px solid var(--border);
    display: grid;
    grid-template-rows: auto 1fr auto;
    overflow: hidden;
  }

  .sb-files {
    padding: 14px;
    display: flex; flex-direction: column; gap: 10px;
    border-bottom: 1px solid var(--border);
    overflow-y: auto;
    max-height: 60vh;
    flex-shrink: 0;
  }
  .sb-lbl   { font-family: var(--font-mono); font-size: 9px; font-weight: 500; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 6px; }

  /* Path inputs replace native file-picker buttons */
  .path-input {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
    color: var(--text); font-family: var(--font-mono); font-size: 10px;
    padding: 6px 9px; width: 100%; outline: none; transition: border-color .15s;
  }
  .path-input:focus { border-color: var(--accent-dim); }
  .path-input.set   { border-color: var(--accent-dim); color: var(--accent); }
  .path-hint { font-size: 9px; color: var(--muted); margin-top: 3px; }

  .sel-in {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
    color: var(--text); font-family: var(--font-mono); font-size: 11px;
    padding: 6px 9px; width: 100%; outline: none; transition: border-color .15s;
  }
  .sel-in:focus { border-color: var(--accent-dim); }

  .btn-primary {
    background: var(--accent); color: #000; font-family: var(--font-sans);
    font-size: 11px; font-weight: 500; border: none; border-radius: 3px;
    padding: 7px 12px; width: 100%; cursor: pointer; transition: background .15s;
  }
  .btn-primary:hover { background: #00ddc0; }

  .mode-tab {
    flex: 1; padding: 5px 0;
    font-family: var(--font-mono); font-size: 10px; font-weight: 500;
    letter-spacing: .04em; text-transform: uppercase;
    background: var(--bg2); border: 1px solid var(--border); color: var(--muted);
    cursor: pointer; transition: all .15s;
  }
  .mode-tab:first-child { border-radius: 3px 0 0 3px; }
  .mode-tab:last-child  { border-radius: 0 3px 3px 0; border-left: none; }
  .mode-tab.active { background: var(--accent); border-color: var(--accent); color: #000; }
  .btn-primary.disabled, .btn-primary:disabled { background: var(--bg3); color: var(--muted); cursor: not-allowed; }

  /* ── Tool browser ── */
  .sb-tools { display: flex; flex-direction: column; overflow: hidden; }
  .sb-tools-header {
    padding: 10px 14px 6px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .sb-tools-label { font-family: var(--font-mono); font-size: 9px; font-weight: 500; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
  .sb-tools-count { font-family: var(--font-mono); font-size: 9px; color: var(--border-hi); }

  #tool-list { flex: 1; overflow-y: auto; padding: 4px 0; }

  .tool-item {
    padding: 8px 14px; cursor: pointer;
    border-left: 2px solid transparent;
    transition: background .12s, border-color .12s;
  }
  .tool-item:hover  { background: var(--bg2); }
  .tool-item.active { background: var(--bg2); border-left-color: var(--accent); }
  .tool-item-name   { font-size: 12px; font-weight: 500; color: var(--text); }
  .tool-item-desc   { font-size: 10px; color: var(--muted); margin-top: 2px; line-height: 1.4; }

  /* ── Sidebar footer ── */
  .sb-footer { padding: 10px 14px; border-top: 1px solid var(--border); font-size: 9px; color: var(--muted); line-height: 1.7; flex-shrink: 0; }
  .sb-cat-bg { position: absolute; right: 12px; top: 0; height: 100%; width: auto; color: #1D9E75; opacity: 0.35; pointer-events: none; }

  /* ── Main panel ── */
  .main {
    background: var(--bg0);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto 140px;
    overflow: hidden;
  }

  #tool-detail {
    padding: 14px 18px;
    border-bottom: 1px solid var(--border);
    background: var(--bg1);
    flex-shrink: 0;
    min-height: 56px;
  }
  .detail-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
  .detail-name   { font-family: var(--font-mono); font-size: 11px; font-weight: 500; color: var(--accent); text-transform: uppercase; letter-spacing: .08em; }
  .detail-desc   { font-size: 11px; color: var(--muted); margin-top: 3px; line-height: 1.5; max-width: 560px; }

  .tool-run-btn {
    flex-shrink: 0;
    background: var(--accent); color: #000;
    font-family: var(--font-mono); font-size: 11px; font-weight: 500;
    border: none; border-radius: 3px; padding: 6px 16px;
    cursor: pointer; transition: background .15s;
    display: inline-flex; align-items: center; gap: 6px;
  }
  .tool-run-btn:hover { background: #00ddc0; }
  .tool-run-btn.disabled, .tool-run-btn:disabled { background: var(--bg3); color: var(--muted); cursor: not-allowed; }
  .tool-map3d-btn {
    flex-shrink:0; background:transparent; color:var(--accent);
    border:1px solid var(--accent-dim); font-family:var(--font-mono);
    font-size:11px; font-weight:500; border-radius:3px; padding:6px 12px;
    cursor:pointer; transition:all .15s;
  }
  .tool-map3d-btn:hover:not(:disabled) { background:var(--accent-dim); color:#000; }
  .tool-map3d-btn.disabled, .tool-map3d-btn:disabled { opacity:0.3; cursor:not-allowed; }

  .param-grid { display: flex; flex-wrap: wrap; gap: 8px 16px; margin-top: 10px; }
  .param-row  { display: flex; align-items: center; gap: 6px; }
  .param-lbl  { font-family: var(--font-mono); font-size: 10px; color: var(--muted); white-space: nowrap; }
  .param-input {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
    color: var(--text); font-family: var(--font-mono); font-size: 11px;
    padding: 4px 7px; outline: none; width: 80px; transition: border-color .15s;
  }
  .param-input:focus { border-color: var(--accent-dim); }
  select.param-input { width: auto; cursor: pointer; }

  #chart-display {
    background: var(--bg0);
    overflow: hidden;
    position: relative;
    display: flex;
    flex-direction: column;
  }

  .chart-placeholder {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 8px; color: var(--muted);
  }
  .placeholder-icon { font-size: 28px; opacity: .25; }
  .placeholder-name { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
  .placeholder-hint { font-size: 10px; color: var(--border-hi); }

  .welcome {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 32px 48px;
    max-width: 680px;
    margin: 0 auto;
    width: 100%;
  }
  .welcome-logo {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    color: var(--accent);
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .welcome-tagline {
    font-size: 11px;
    color: var(--muted);
    margin-bottom: 28px;
    letter-spacing: .02em;
  }
  .welcome-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 5px;
    overflow: hidden;
    width: 100%;
    margin-bottom: 24px;
  }
  .welcome-step {
    background: var(--bg1);
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .ws-num  { font-family: var(--font-mono); font-size: 9px; color: var(--accent); letter-spacing: .08em; }
  .ws-title{ font-size: 12px; font-weight: 500; color: var(--text); }
  .ws-desc { font-size: 10px; color: var(--muted); line-height: 1.5; }
  .welcome-ref {
    width: 100%;
    background: var(--bg1);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 14px 16px;
  }
  .wr-title { font-family: var(--font-mono); font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-bottom: 10px; }
  .wr-grid  { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 24px; }
  .wr-row   { display: flex; gap: 8px; align-items: baseline; }
  .wr-code  { font-family: var(--font-mono); font-size: 10px; color: #5b8dee; white-space: nowrap; }
  .wr-desc  { font-size: 10px; color: var(--muted); }

  @keyframes spin { to { transform: rotate(360deg); } }
  .placeholder-spin {
    width: 22px; height: 22px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  .spinner {
    display: inline-block; width: 10px; height: 10px;
    border: 1.5px solid transparent; border-top-color: currentColor;
    border-radius: 50%; animation: spin .6s linear infinite;
  }

  .chart-title-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px 4px;
    flex-shrink: 0;
  }
  .chart-title-text { font-family: var(--font-mono); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
  .chart-export-btn {
    background: transparent; border: 1px solid var(--border); border-radius: 3px;
    color: var(--muted); font-family: var(--font-mono); font-size: 10px;
    padding: 3px 9px; cursor: pointer; transition: all .15s;
  }
  .chart-export-btn:hover { border-color: var(--border-hi); color: var(--text); }

  .export-bar {
    padding: 8px 16px; border-top: 1px solid var(--border);
    background: var(--bg1); display: flex; align-items: center; gap: 10px;
  }
  .export-btn {
    background: transparent; border: 1px solid var(--border); border-radius: 3px;
    color: var(--muted); font-family: var(--font-mono); font-size: 10px;
    padding: 4px 10px; cursor: pointer; transition: all .15s;
  }
  .export-btn:hover { border-color: var(--border-hi); color: var(--text); }

  .log-panel {
    border-top: 1px solid var(--border); background: var(--bg1);
    padding: 8px 14px; min-height: 0; overflow-y: auto;
    font-family: var(--font-mono); font-size: 10px;
    display: flex; flex-direction: column;
  }
  .log-line { display: flex; gap: 8px; align-items: baseline; padding: 1px 0; }
  .log-ts   { color: #3d4245; min-width: 62px; }
  .log-pfx  { min-width: 10px; }
  .log-info    .log-pfx { color: var(--muted); }
  .log-success .log-pfx { color: var(--success); }
  .log-error   .log-pfx { color: var(--danger); }
  .log-error span:last-child { color: var(--danger); }
  .log-warn    .log-pfx { color: var(--warn); }

  .detail-empty { font-size: 11px; color: var(--muted); padding: 4px 0; }

  /* Progress bar — shown during long analysis runs via WebSocket events */
  #progress-bar {
    position: fixed; top: 0; left: 0; height: 2px;
    background: var(--accent); transition: width .15s linear;
    z-index: 100; pointer-events: none;
  }
  #progress-bar.hidden { opacity: 0; transition: opacity .4s, width .15s; }
</style>

<div id="progress-bar" class="hidden" style="width:0%"></div>

<div class="shell">

  <!-- Header with category navigation -->
  <header class="header">
    <span class="header-logo">ATMOS</span>
    <span class="header-sep"></span>
    <svg class="header-cat" viewBox="0 0 70 120" fill="none"
         stroke="currentColor" stroke-width="3"
         stroke-linecap="round" stroke-linejoin="round"
         aria-label="cat" role="img">
      <circle  cx="46" cy="34" r="16"/>
      <ellipse cx="46" cy="64" rx="21" ry="25"/>
      <path d="M33 23L26 4L40 17"/>
      <path d="M59 23L66 4L52 17"/>
      <path d="M25 83C4 99,7 117,30 112"/>
    </svg>
    <nav class="cat-nav" id="cat-nav">
      ${CATEGORIES.map((c, i) => `<button class="cat-tab${i === 0 ? ' active' : ''}" data-cat="${c.id}">${c.label}</button>`).join('')}
    </nav>
    <div class="status-pill">
      <span class="status-dot idle" id="status-dot"></span>
      <span id="status-text">No files selected</span>
    </div>
  </header>

  <!-- Sidebar -->
  <aside class="sidebar">

    <!-- File / selection / load — MD / QM-MM mode switch -->
    <div class="sb-files">

      <div style="display:flex;margin-bottom:6px;">
        <button class="mode-tab active" data-mode="md">MD</button>
        <button class="mode-tab"        data-mode="qmm">QM/MM</button>
      </div>

      <!-- MD panel -->
      <div id="panel-md" style="display:flex;flex-direction:column;gap:8px;">
        <div>
          <div class="sb-lbl">Trajectory</div>
          <input class="path-input" id="traj-path" type="text"
                 placeholder="/scratch/you/run/traj.nc" spellcheck="false"/>
          <div class="path-hint">.nc .ncrst .xtc .dcd .trr</div>
        </div>
        <div>
          <div class="sb-lbl">Topology <span style="color:#3d4245">(optional)</span></div>
          <input class="path-input" id="topo-path" type="text"
                 placeholder="/scratch/you/run/system.pdb" spellcheck="false"/>
          <div class="path-hint">.pdb</div>
        </div>
        <div>
          <div class="sb-lbl">Atom Selection</div>
          <input class="sel-in" type="text" id="sel" value="name CA" spellcheck="false"/>
          <div style="margin-top:4px;font-size:9px;color:var(--muted)">
            e.g. <code style="color:#5b8dee">name CA</code> &nbsp;
                 <code style="color:#5b8dee">(resid >= 1) and (resid <= 50)</code>
          </div>
        </div>
        <div style="display:flex;gap:6px;align-items:flex-end;">
          <button class="btn-primary disabled" id="load-btn" disabled style="flex:1">Load &amp; Index</button>
          <div style="flex-shrink:0">
            <div class="sb-lbl" style="font-size:8px;margin-bottom:2px">Stride</div>
            <input class="sel-in" type="number" id="stride-input" value="1" min="1" step="1"
                   style="width:54px;padding:4px 6px;font-size:11px"
                   title="Load every Nth frame — e.g. 10 keeps 10% of frames"/>
          </div>
        </div>
        <button class="btn-primary disabled" id="visualize-btn" disabled>Visualize Trajectory</button>
      </div>

      <!-- QM/MM umbrella panel (hidden by default) -->
      <div id="panel-qmm" style="display:none;flex-direction:column;gap:8px;">
        <div>
          <div class="sb-lbl">CV file pattern</div>
          <input class="sel-in" id="cv-pattern" type="text"
                 value="../{window}/step5.00_equilibration.cv" spellcheck="false"/>
          <div style="margin-top:3px;font-size:9px;color:var(--muted)">
            <code style="color:#5b8dee">{window}</code> = zero-padded window index
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          <div>
            <div class="sb-lbl">Windows</div>
            <input class="sel-in" id="n-windows" type="number" value="30" min="2"/>
          </div>
          <div>
            <div class="sb-lbl">CV column</div>
            <input class="sel-in" id="cv-col" type="number" value="2" min="0"/>
          </div>
          <div>
            <div class="sb-lbl">CV min</div>
            <input class="sel-in" id="val-min" type="number" value="-0.800" step="0.001"/>
          </div>
          <div>
            <div class="sb-lbl">CV max</div>
            <input class="sel-in" id="val-max" type="number" value="2.100" step="0.001"/>
          </div>
        </div>
        <div>
          <div class="sb-lbl">Restart pattern <span style="color:#3d4245">(optional)</span></div>
          <input class="sel-in" id="rst-pattern" type="text"
                 value="../{window}/step5.00.ncrst" spellcheck="false"/>
          <div style="margin-top:3px;font-size:9px;color:var(--muted)">For 3D visualisation of each window</div>
        </div>
        <div>
          <div class="sb-lbl">Topology <span style="color:#3d4245">(optional — for ncrst viz)</span></div>
          <input class="path-input" id="qmm-topo-path" type="text"
                 placeholder="/scratch/you/run/system.pdb" spellcheck="false"/>
          <div class="path-hint">.pdb</div>
        </div>
        <div>
          <div class="sb-lbl">CV restraint file <span style="color:#3d4245">(optional)</span></div>
          <input class="path-input" id="cv-rst-path" type="text"
                 placeholder="/scratch/you/run/cv.rst" spellcheck="false"/>
          <div class="path-hint">.rst .cv.rst NMR.def</div>
          <div id="cv-rst-status" style="font-size:9px;color:var(--muted);margin-top:3px;display:none"></div>
        </div>
        <button class="btn-primary" id="load-umbrella-btn">Scan &amp; Load Windows</button>
        <div class="path-hint" id="umbrella-status">No windows loaded</div>
      </div>

    </div>

    <!-- Tool browser (category controlled by header nav) -->
    <div class="sb-tools">
      <div class="sb-tools-header">
        <span class="sb-tools-label" id="tools-section-label">Structural</span>
        <span class="sb-tools-count" id="tools-count"></span>
      </div>
      <div id="tool-list"></div>
    </div>

    <!-- Footer -->
    <div class="sb-footer">
      rust · chemfiles · rayon · ndarray<br>
      <span style="color:#3d4245">load once · analyse in parallel</span>
      <svg class="sb-cat-bg" viewBox="0 0 70 120" fill="none"
           stroke="currentColor" stroke-width="3"
           stroke-linecap="round" stroke-linejoin="round"
           aria-hidden="true">
        <circle  cx="46" cy="34" r="16"/>
        <ellipse cx="46" cy="64" rx="21" ry="25"/>
        <path d="M33 23L26 4L40 17"/>
        <path d="M59 23L66 4L52 17"/>
        <path d="M25 83C4 99,7 117,30 112"/>
      </svg>
    </div>

  </aside>

  <!-- Main panel -->
  <main class="main">

    <!-- Tool detail / param bar -->
    <div id="tool-detail">
      <div class="detail-empty">← Select a tool from the sidebar</div>
    </div>

    <!-- Shared chart canvas -->
    <div id="chart-display">
      <div class="welcome">
        <div class="welcome-logo">ATMOS</div>
        <div class="welcome-tagline">High-performance molecular dynamics analysis · Rust + chemfiles + rayon</div>
        <div class="welcome-steps">
          <div class="welcome-step">
            <div class="ws-num">01 · FILES</div>
            <div class="ws-title">Enter file paths</div>
            <div class="ws-desc">Paste the full server-side paths to your trajectory (.nc, .xtc, .dcd) and optional topology (.pdb) in the sidebar. Files are read directly on the cluster.</div>
          </div>
          <div class="welcome-step">
            <div class="ws-num">02 · SELECT</div>
            <div class="ws-title">Define atom selection</div>
            <div class="ws-desc">Enter a chemfiles selection string. The default <code style="color:#5b8dee">name CA</code> selects Cα atoms. All analyses run on this subset — load once, analyse many times.</div>
          </div>
          <div class="welcome-step">
            <div class="ws-num">03 · ANALYSE</div>
            <div class="ws-title">Pick a tool and run</div>
            <div class="ws-desc">Use the category tabs above to browse tools. Select any tool in the left panel, adjust parameters if shown, and press Run. Results appear here.</div>
          </div>
        </div>
        <div class="welcome-ref">
          <div class="wr-title">Selection syntax reference</div>
          <div class="wr-grid">
            <div class="wr-row"><span class="wr-code">name CA</span><span class="wr-desc">Cα backbone atoms</span></div>
            <div class="wr-row"><span class="wr-code">name CA C N O</span><span class="wr-desc">Full backbone</span></div>
            <div class="wr-row"><span class="wr-code">(resid >= 1) and (resid <= 50)</span><span class="wr-desc">Residue range</span></div>
            <div class="wr-row"><span class="wr-code">resname ALA GLY SER</span><span class="wr-desc">By residue type</span></div>
            <div class="wr-row"><span class="wr-code">(name CA) and (resname ALA)</span><span class="wr-desc">Combined filter</span></div>
            <div class="wr-row"><span class="wr-code">index 0:499</span><span class="wr-desc">First 500 atoms (0-based)</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- Export bar -->
    <div class="export-bar">
      <span style="color:var(--muted);font-family:var(--font-mono);font-size:9px;text-transform:uppercase;letter-spacing:.08em">Export</span>
      <button class="export-btn" id="export-btn">Save CSV</button>
      <input class="path-input" id="batch-export-dir" type="text"
             placeholder="/scratch/you/results" spellcheck="false"
             style="flex:1;min-width:0;font-size:10px;padding:3px 6px"
             title="Server-side directory for batch export"/>
      <button class="export-btn" id="batch-export-btn" title="Export all cached analyses to the directory above">Batch Export</button>
    </div>

    <!-- Log -->
    <div class="log-panel" id="log"></div>

  </main>
</div>`;

  // ── Trajectory path input ─────────────────────────────────────────────────
  $('#traj-path')!.addEventListener('input', () => {
    const val = ($('#traj-path') as HTMLInputElement).value.trim();
    trajectoryPath = val || null;
    if (val) ($('#traj-path') as HTMLElement).classList.add('set');
    else     ($('#traj-path') as HTMLElement).classList.remove('set');
    updateLoadState();
  });

  // ── Topology path input ───────────────────────────────────────────────────
  $('#topo-path')!.addEventListener('input', () => {
    const val = ($('#topo-path') as HTMLInputElement).value.trim();
    topologyPath = val || null;
    updateLoadState();
  });

  // ── Load & Index ──────────────────────────────────────────────────────────
  $('#load-btn')!.addEventListener('click', async () => {
    if (!trajectoryPath) return;
    const btn = $('#load-btn') as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Loading…';
    log('Indexing trajectory into memory — this may take a minute for large files…');

    // Show a "still working" heartbeat every 10 seconds while load runs
    const heartbeat = setInterval(() => {
      const bar = document.getElementById('progress-bar');
      const pct = bar ? parseFloat(bar.style.width) || 0 : 0;
      if (pct > 0 && pct < 100) log(`Loading… ${pct.toFixed(0)}% of frames read`);
    }, 10_000);

    try {
      const msg: string = await invoke('load_trajectory', {
        path:          trajectoryPath,
        top_path:      topologyPath ?? null,
        selection_str: ($('#sel') as HTMLInputElement).value,
        stride:        parseInt(($('#stride-input') as HTMLInputElement).value ?? '1', 10) || 1,
      });
      const match = msg.match(/\d+/);
      if (match) totalFrames = parseInt(match[0]);
      isLoaded = true;
      log(msg, 'success');
      updateLoadState();
    } catch (e) {
      log(`Load error: ${e}`, 'error');
    } finally {
      clearInterval(heartbeat);
      btn.disabled = false;
      btn.textContent = 'Load & Index';
    }
  });

  $('#visualize-btn')!.addEventListener('click', () => {
    if (isLoaded) openVisualizer();
    else log('Please load and index the trajectory first.', 'warn');
  });

  // ── Category nav tabs → rebuild tool list ────────────────────────────────
  function switchCategory(catId: string) {
    activeToolId = null;
    $$('.cat-tab').forEach(tab => {
      tab.classList.toggle('active', (tab as HTMLElement).dataset.cat === catId);
    });
    const cat = CATEGORIES.find(c => c.id === catId);
    const tools = TOOLS.filter(t => t.category === catId);
    $('#tools-section-label')!.textContent = cat?.label ?? '';
    $('#tools-count')!.textContent = `${tools.length} tool${tools.length !== 1 ? 's' : ''}`;
    renderToolList(catId);
    $('#tool-detail')!.innerHTML = '<div class="detail-empty">← Select a tool from the list</div>';
  }

  $('#cat-nav')!.addEventListener('click', (e) => {
    const tab = (e.target as HTMLElement).closest('.cat-tab') as HTMLElement | null;
    if (tab?.dataset.cat) switchCategory(tab.dataset.cat);
  });

  // ── Mode tabs (MD / QM-MM) ────────────────────────────────────────────────
  document.querySelectorAll<HTMLButtonElement>('.mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      simMode = tab.dataset.mode as 'md' | 'qmm';
      document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const md  = $('#panel-md')  as HTMLElement;
      const qmm = $('#panel-qmm') as HTMLElement;
      md.style.display  = simMode === 'md'  ? 'flex' : 'none';
      qmm.style.display = simMode === 'qmm' ? 'flex' : 'none';
    });
  });

  // ── QM/MM topology path ───────────────────────────────────────────────────
  $('#qmm-topo-path')?.addEventListener('change', async () => {
    const val = ($('#qmm-topo-path') as HTMLInputElement).value.trim();
    if (!val) return;
    qmmTopoPath = val;
    log(`QM/MM topology: ${val.split('/').pop()}`);
    try {
      const msg: string = await invoke('set_qmm_topology', { path: qmmTopoPath });
      log(msg, 'success');
    } catch (e) {
      log(`Topology error: ${e}`, 'error');
    }
  });

  // ── cv.rst restraint file path ────────────────────────────────────────────
  $('#cv-rst-path')?.addEventListener('change', async () => {
    const val = ($('#cv-rst-path') as HTMLInputElement).value.trim();
    if (!val) { cvRstBlocks = []; cvRstPath = null; return; }
    cvRstPath = val;
    log(`Parsing cv.rst: ${val.split('/').pop()}`);
    try {
      cvRstBlocks = await invoke<any[]>('parse_cv_rst', { path: cvRstPath });
      const statusEl = $('#cv-rst-status') as HTMLElement;
      statusEl.style.display = 'block';
      statusEl.style.color   = 'var(--success)';
      statusEl.textContent   = `${cvRstBlocks.length} block${cvRstBlocks.length !== 1 ? 's' : ''}: ${cvRstBlocks.map((b: any) => b.cv_label).join(', ')}`;
      log(`cv.rst: ${cvRstBlocks.length} blocks parsed`, 'success');
    } catch (e) {
      log(`cv.rst parse failed: ${e}`, 'error');
      cvRstBlocks = [];
    }
  });

  // ── Umbrella window loader ─────────────────────────────────────────────────
  $('#load-umbrella-btn')?.addEventListener('click', async () => {
    const cvPattern = ($('#cv-pattern')  as HTMLInputElement).value.trim();
    const nWindows  = parseInt(($('#n-windows') as HTMLInputElement).value, 10);
    const cvCol     = parseInt(($('#cv-col')     as HTMLInputElement).value, 10);
    const valMin    = parseFloat(($('#val-min')  as HTMLInputElement).value);
    const valMax    = parseFloat(($('#val-max')  as HTMLInputElement).value);
    const rstPat    = ($('#rst-pattern') as HTMLInputElement).value.trim() || null;
    if (!cvPattern) { log('Enter a CV file pattern.', 'warn'); return; }

    const btn = $('#load-umbrella-btn') as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Scanning…';
    log(`Scanning ${nWindows} umbrella windows…`);

    try {
      const msg: string = await invoke('load_umbrella_windows', {
        cv_pattern:  cvPattern,
        n_windows:   nWindows,
        val_min:     valMin,
        val_max:     valMax,
        cv_col:      cvCol,
        rst_pattern: rstPat,
      });
      umbrellaLoaded = true;
      const statusEl = $('#umbrella-status') as HTMLElement;
      statusEl.textContent = msg;
      statusEl.style.color = 'var(--success)';
      log(msg, 'success');
    } catch (e) {
      log(`Window load failed: ${e}`, 'error');
      ($('#umbrella-status') as HTMLElement).textContent = `Error: ${e}`;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Scan & Load Windows';
    }
  });

  // ── Export button ──────────────────────────────────────────────────────────
  $('#export-btn')!.addEventListener('click', exportCsv);

  // ── Batch export ───────────────────────────────────────────────────────────
  // Sends all cached analyses to a server-side directory path.
  // Also serializes the current chart SVG and sends it alongside the CSV data.
  // No file dialog — the user types the cluster path directly in the export bar.
  $('#batch-export-btn')!.addEventListener('click', async () => {
    const dir = ($('#batch-export-dir') as HTMLInputElement).value.trim();
    if (!dir) {
      log('Enter a server-side export directory path first (e.g. /scratch/you/results).', 'warn');
      return;
    }
    const btn = $('#batch-export-btn') as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Exporting…';
    log(`Batch exporting to ${dir}…`);

    // Collect SVG strings from all visible charts
    const svgData: string[] = [];
    document.querySelectorAll<SVGSVGElement>('#chart-display svg').forEach(svg => {
      svgData.push(new XMLSerializer().serializeToString(svg));
    });

    try {
      const written: string[] = await invoke('batch_export', {
        dir,
        svg_data: svgData.length > 0 ? svgData : null,
      });
      log(`Batch export complete — ${written.length} files written to ${dir}: ${written.join(', ')}`, 'success');
    } catch (e) {
      log(`Batch export failed: ${e}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Batch Export';
    }
  });

  // ── Progress bar (driven by WebSocket progress events) ────────────────────
  // Opens a WebSocket and listens for { tool, pct } messages from the server.
  // The progress bar is shown at the top of the page during long analyses.
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const progressWs = new WebSocket(`${proto}://${location.host}/api/events`);
  let   progressHideTimer: ReturnType<typeof setTimeout> | null = null;

  progressWs.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data as string);
      // Progress events have shape { tool, pct }
      if (typeof msg.pct === 'number') {
        const bar = document.getElementById('progress-bar')!;
        bar.style.width = `${msg.pct.toFixed(1)}%`;
        bar.classList.remove('hidden');
        if (progressHideTimer) clearTimeout(progressHideTimer);
        if (msg.pct >= 99.9) {
          progressHideTimer = setTimeout(() => {
            bar.classList.add('hidden');
            bar.style.width = '0%';
          }, 600);
        }
      }
    } catch {}
  };
  progressWs.onclose = () => {};   // silent reconnect handled by events.ts if needed

  // ── Initial render ────────────────────────────────────────────────────────
  switchCategory(CATEGORIES[0].id);
  log('ATMOS ready.', 'success');
  updateLoadState();
}

window.addEventListener('DOMContentLoaded', initUI);
