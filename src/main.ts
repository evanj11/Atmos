import { open, save } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';

// ─── State ────────────────────────────────────────────────────────────────────
let trajectoryPath: string | null = null;
let topologyPath:   string | null = null;
let isLoaded    = false;
let totalFrames = 0;
let activeToolId:  string | null = null;
let simMode:       'md' | 'qmm'  = 'md';
let umbrellaLoaded = false;
let qmmTopoPath:   string | null  = null;
// Parsed cv.rst blocks — set when user loads a cv.rst file
let cvRstBlocks:   any[]  = [];
let cvRstPath:     string | null = null;

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
  chartType: 'line' | 'heatmap' | 'scatter' | 'bar' | 'pmf' | 'ramachandran' | 'prs' | 'cluster' | 'geometry' | 'dihedral_ts' | 'sasa' | 'membrane' | 'none';
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
  { id: 'membrane',        label: 'Membrane' },
  { id: 'thermodynamics',  label: 'Thermodynamics' },
  { id: 'quantumchem',     label: 'Quantum Chemistry' },
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
    desc: 'Shrake-Rupley solvent accessible surface area. Per-residue mean/std and total per frame. Maps to 3D via bfactor coloring. Click a residue to view in 3D.',
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
      { id: 'nClusters', label: 'Clusters (k)',         type: 'number', default: '4'       },
      { id: 'method',    label: 'Method',               type: 'select', default: 'kmeans',
        options: ['kmeans', 'msm'] },
      { id: 'lag',       label: 'MSM lag (frames)',     type: 'number', default: '10'      },
      { id: 'nMacro',    label: 'PCCA macrostates',     type: 'number', default: '4'       },
    ],
  },
  {
    id: 'dihedral_ts', label: 'Dihedral Time Series', category: 'structural',
    desc: 'φ and ψ angle time series for selected residues. Shows rotameric transitions over the trajectory. Requires Ramachandran analysis to be run first.',
    invoke: '', chartType: 'dihedral_ts', clickAction: 'frame',
    params: [
      { id: 'atomIndices', label: 'Atom indices (comma-separated)', type: 'text', default: '0' },
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
    params: [
      { id: 'bins', label: 'Histogram bins (0 = auto)', type: 'number', default: '0' },
    ],
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
    desc: 'PRS: apply a unit perturbation at each residue and measure the response across the network. Identifies allosteric drivers (high effectiveness) and sensors (high sensitivity). Click a pair to view in 3D.',
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
  // ── Membrane ─────────────────────────────────────────────────────────────────
  {
    id: 'membrane', label: 'Bilayer Analysis', category: 'membrane',
    desc: 'Membrane thickness, area per lipid (APL), leaflet z-density, and lipid order parameters (|SCD|). Load phosphate atoms (e.g. name P) for headgroup-based leaflet assignment. For SCD, include acyl-chain carbons.',
    invoke: 'run_membrane', chartType: 'membrane',
    params: [
      { id: 'headgroup', label: 'Headgroup atom name(s)', type: 'text',   default: 'P'  },
      { id: 'normal',    label: 'Membrane normal (0=x 1=y 2=z)', type: 'number', default: '2' },
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
      { id: 'nBins', label: 'PMF bins',                      type: 'number', default: '29'    },
      { id: 'nBoot', label: 'Bootstrap replicates',          type: 'number', default: '50'    },
    ],
  },
  {
    id: 'geometry', label: 'Distance / Angle Monitor', category: 'quantumchem',
    desc: 'Compute interatomic distances and angles. Load a cv.rst file in the QM/MM sidebar to auto-populate r1/r2/RC from the CV column — or enter 0-based atom index pairs manually. For umbrella source, results plot vs. window CV and can be overlaid on the PMF.',
    invoke: 'run_geometry_series', chartType: 'geometry', clickAction: 'frame',
    params: [
      { id: 'pairs',    label: 'Extra distance pairs (i,j;i,j… 0-based)',  type: 'text', default: ''      },
      { id: 'triplets', label: 'Angle triplets (i,j,k;i,j,k… 0-based)',   type: 'text', default: ''      },
      { id: 'source',   label: 'Source',                          type: 'select', default: 'trajectory',
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

  const saveBtn = $('#save-project-btn') as HTMLButtonElement;
  if (saveBtn) saveBtn.disabled = !isReady;

  const noLoadNeededTools = ['mbar','umbrella_viewer','geometry','dihedral_ts'];
  $$('.tool-run-btn').forEach(btn => {
    const noLoad = noLoadNeededTools.includes(activeToolId ?? '');
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
  // These tools work without a loaded MD trajectory
  const needsLoad = tool.id !== 'mbar'
                 && tool.id !== 'umbrella_viewer'
                 && tool.id !== 'geometry'
                 && tool.id !== 'dihedral_ts';

  detailEl.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-name">${tool.label}</div>
        <div class="detail-desc">${tool.desc}</div>
      </div>
      <div style="display:flex;gap:6px;align-items:center;flex-shrink:0">
        ${map3dHtml}
        <button class="tool-cancel-btn" id="cancel-btn" aria-label="Cancel">✕ Cancel</button>
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
  const TITLE_H = 36;  // height of .chart-title-bar
  const w  = area.clientWidth  || 600;
  // Subtract the title bar so the SVG only occupies the space below it.
  // pad.bottom must fit: tick labels (14px) + axis label (11px) + gap = 58px.
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

  // Axis tick labels for scatter
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

  // ── Size: largest square that fits in the available space ──────────────────
  // Reserve: title bar 36px, colorbar strip 20px + labels ~32px = ~52px right
  const TITLE_H  = 36;
  const CB_W     = 20;   // colorbar strip width in CSS px
  const CB_GAP   = 10;   // gap between matrix and colorbar
  const CB_LABEL = 40;   // space for colorbar tick labels to the right
  const dpr      = window.devicePixelRatio || 1;
  const availW   = area.clientWidth  || 600;
  const availH   = (area.clientHeight || 480) - TITLE_H;
  // Matrix fits in (availW - CB_W - CB_GAP - CB_LABEL) × availH
  const cssSize  = Math.min(availW - CB_W - CB_GAP - CB_LABEL, availH);
  const pxSize   = Math.round(cssSize * dpr);

  let lo = Infinity, hi = -Infinity;
  matrix.forEach(row => row.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));

  // Blue(−1) → White(0) → Red(+1) diverging palette
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

  // Draw matrix onto canvas at device-pixel resolution
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

  // ── SVG colorbar — drawn next to the matrix ────────────────────────────────
  // The gradient runs top→bottom: hi (red) at top, lo (blue) at bottom.
  // 5 evenly-spaced tick marks with value labels.
  const cbH   = cssSize;
  const cbSvgW = CB_W + CB_GAP + CB_LABEL;

  // Build gradient stops (21 steps = smooth enough)
  const stops = Array.from({ length: 21 }, (_, i) => {
    const t   = i / 20;                            // 0 = top (hi), 1 = bottom (lo)
    const v   = hi - t * (hi - lo);               // value at this position
    const [r, g, b] = colorScale(v);
    return `<stop offset="${(t * 100).toFixed(0)}%" stop-color="rgb(${r},${g},${b})"/>`;
  }).join('');

  // Tick marks at 5 positions
  const nTicks = 5;
  const ticks  = Array.from({ length: nTicks }, (_, i) => {
    const t   = i / (nTicks - 1);                  // 0=top, 1=bottom
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

  // ── Assemble: title bar, then [matrix canvas + colorbar] side by side ──────
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

  // Y-axis ticks
  const yTicks = 5;
  const yTickSvg = Array.from({ length: yTicks }, (_, i) => {
    const v = (max * i) / (yTicks - 1);
    const y = ih - (v / max) * ih;
    return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-8" y="${(y + 4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v.toFixed(2)}</text>`;
  }).join('');

  // X-axis index labels (show up to 10 evenly spaced)
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

function renderRamachandranChart(
  result: any,
  onResidueClick?: (atomIdx: number) => void,
) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 580;
  const svgH    = (area.clientHeight || 540) - TITLE_H;
  const pad     = { top: 24, right: 32, bottom: 48, left: 52 };
  const iw      = w    - pad.left - pad.right;
  const ih      = svgH - pad.top  - pad.bottom;
  const { density, residues, mode, n_frames } = result;

  // phi: -180->x=0, +180->x=iw   psi: -180->y=ih, +180->y=0 (SVG y down)
  const toSvg = (phi: number, psi: number): [string, string] => [
    ((phi + 180) / 360 * iw).toFixed(1),
    (ih - (psi + 180) / 360 * ih).toFixed(1),
  ];

  // Density heatmap
  const BINS  = density.length;
  const cellW = iw / BINS;
  const cellH = ih / BINS;
  let heatCells = '';
  for (let yi = 0; yi < BINS; yi++) {
    for (let xi = 0; xi < BINS; xi++) {
      const v = density[yi][xi] as number;
      if (v < 0.004) continue;
      const t = Math.pow(v, 0.35);
      const rr = Math.round(t * 60);
      const gg = Math.round(120 + t * 80);
      const bb = Math.round(130 + t * 40);
      const x = (xi * cellW).toFixed(1);
      const y = (ih - (yi + 1) * cellH).toFixed(1);
      heatCells += `<rect x="${x}" y="${y}" width="${(cellW+0.6).toFixed(1)}" height="${(cellH+0.6).toFixed(1)}" fill="rgb(${rr},${gg},${bb})" opacity="${(t*0.85).toFixed(2)}"/>`;
    }
  }

  // Allowed region paths
  const regionPath = (pts: readonly (readonly number[])[]) =>
    pts.map(([p,s], i) => `${i===0?'M':'L'}${toSvg(p,s).join(',')}`).join(' ') + ' Z';
  const helixPts  = [[-145,-60],[-30,-60],[-30,20],[-145,20]] as const;
  const sheetPts  = [[-170,90], [-55,90], [-55,180],[-170,180]] as const;
  const sheet2Pts = [[-170,-180],[-55,-180],[-55,-155],[-170,-155]] as const;
  const lhPts     = [[30,20],[80,20],[80,80],[30,80]] as const;

  // Residue mean dots
  const nRes = (residues as any[]).length;
  const dots = (residues as any[]).map((r, k) => {
    if (!r) return '';
    const phi = r.phi_mean, psi = r.psi_mean;
    if (phi == null || psi == null || !isFinite(phi) || !isFinite(psi)) return '';
    const [cx, cy] = toSvg(phi, psi);
    const hue = Math.round(k / nRes * 300);
    const title = `${r.res_name ?? '?'} ${r.res_seq ?? '?'}  phi=${phi.toFixed(1)}  psi=${psi.toFixed(1)}  sigma=${(r.phi_std ?? 0).toFixed(1)}`;
    return `<circle cx="${cx}" cy="${cy}" r="5" fill="hsl(${hue},80%,62%)" stroke="#0d0e0f" stroke-width="0.8" opacity="0.92" data-atom="${r.atom_idx}" style="cursor:${onResidueClick?'pointer':'default'}"><title>${title}</title></circle>`;
  }).join('');

  // Ticks
  const xTicks = [-180,-90,0,90,180].map(v => {
    const [x] = toSvg(v, 0);
    return `<line x1="${x}" y1="0" x2="${x}" y2="${ih}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="${x}" y="${(ih+16).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${v}</text>`;
  }).join('');
  const yTicks = [-180,-90,0,90,180].map(v => {
    const [, y] = toSvg(0, v);
    return `<line x1="0" y1="${y}" x2="${iw}" y2="${y}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-6" y="${(+y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v}</text>`;
  }).join('');

  // Region label centres
  const [aLx, aLy] = toSvg(-87, -20);
  const [bLx, bLy] = toSvg(-112, 135);
  const [lLx, lLy] = toSvg(55, 50);
  const [z0x]      = toSvg(0, -180);
  const [, z0y]    = toSvg(-180, 0);

  // Stats
  const nAlpha = (residues as any[]).filter(r => r?.phi_mean != null && r.phi_mean >= -145 && r.phi_mean <= -30 && r.psi_mean >= -60 && r.psi_mean <= 20).length;
  const nBeta  = (residues as any[]).filter(r => r?.phi_mean != null && r.phi_mean >= -170 && r.phi_mean <= -55 && r.psi_mean >= 90).length;
  const nOther = nRes - nAlpha - nBeta;
  const modeLabel = mode === 'backbone' ? 'Backbone φ/ψ' : 'Cα pseudo-dihedral';
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
      dot.addEventListener('mouseenter', () => { dot.setAttribute('r', '7'); dot.setAttribute('stroke-width', '1.5'); });
      dot.addEventListener('mouseleave', () => { dot.setAttribute('r', '5'); dot.setAttribute('stroke-width', '0.8'); });
    });
  }
}

// ─── PRS chart ────────────────────────────────────────────────────────────────
//
// Three-panel display: heatmap of the N×N response matrix (main panel) plus
// two bar charts for effectiveness and sensitivity below it.
// Clicking a heatmap cell opens the pair viewer; clicking a bar opens the
// residue viewer with the effectiveness/sensitivity value mapped to bfactor.

function renderPrsChart(
  result: any,
  onPairClick?:    (i: number, j: number) => void,
  onResidueClick?: (i: number) => void,
) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const BAR_H   = 100;
  const w       = area.clientWidth  || 620;
  const totalH  = (area.clientHeight || 560) - TITLE_H;
  const hmH     = totalH - BAR_H * 2 - 8;
  const { matrix, effectiveness, sensitivity } = result;
  const n = matrix.length;

  // ── Heatmap ───────────────────────────────────────────────────────────────
  const dpr      = window.devicePixelRatio || 1;
  const hmCSS    = Math.min(w - 20, hmH);
  const hmPX     = Math.round(hmCSS * dpr);
  const canvas   = document.createElement('canvas');
  canvas.width   = hmPX; canvas.height = hmPX;
  canvas.style.cssText = `display:block;width:${hmCSS}px;height:${hmCSS}px;border-radius:2px;`;
  const ctx = canvas.getContext('2d')!;
  const cell = hmPX / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = (matrix[i] as number[])[j];
      const r = Math.round(v * 255);
      const g = Math.round((1-v) * 100);
      const b = Math.round((1-v) * 180);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(Math.round(j*cell), Math.round(i*cell), Math.ceil(cell), Math.ceil(cell));
    }
  }
  if (onPairClick) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const col  = Math.floor((e.clientX - rect.left) / rect.width  * n);
      const row  = Math.floor((e.clientY - rect.top)  / rect.height * n);
      if (col >= 0 && col < n && row >= 0 && row < n) onPairClick(row, col);
    });
  }

  // ── Bar helpers ───────────────────────────────────────────────────────────
  const makeBar = (data: number[], title: string, color: string) => {
    const PL = 52, PR = 16, PT = 14, PB = 22;
    const bw  = (w - PL - PR) / data.length;
    const max = Math.max(...data, 1e-9);
    const ih  = BAR_H - PT - PB;
    const bars = data.map((v, i) =>
      `<rect x="${(PL + i*bw + bw*0.05).toFixed(1)}" y="${(PT + ih*(1-v/max)).toFixed(1)}"
        width="${(bw*0.9).toFixed(1)}" height="${(ih*v/max).toFixed(1)}"
        fill="${color}" opacity="0.85" data-i="${i}" rx="1"
        style="cursor:${onResidueClick ? 'pointer' : 'default'}"/>`
    ).join('');
    const nxt = Math.min(8, data.length);
    const xLbls = Array.from({length: nxt}, (_, k) => {
      const idx = Math.round(k/(nxt-1||1)*(data.length-1));
      const x   = (PL + (idx+0.5)*bw).toFixed(1);
      return `<text x="${x}" y="${(BAR_H-4).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">${idx}</text>`;
    }).join('');
    return `<svg width="${w}" height="${BAR_H}" style="display:block">
      <text x="${PL}" y="10" fill="#7a7f85" font-size="9" font-family="monospace">${title}</text>
      ${bars}${xLbls}
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
      r.addEventListener('mouseenter', () => r.setAttribute('opacity', '1'));
      r.addEventListener('mouseleave', () => r.setAttribute('opacity', '0.85'));
    });
  }
}

// ─── Cluster chart ────────────────────────────────────────────────────────────
//
// Two-panel view: top = PCA scatter coloured by cluster assignment with centroid
// markers; bottom = population bar chart.  If MSM was run, implied timescales
// are shown as a legend entry and PCCA membership is visualised via dot opacity.

function renderClusterChart(result: any, onFrameClick?: (frame: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const BAR_H   = 80;
  const w       = area.clientWidth  || 620;
  const svgH    = (area.clientHeight || 500) - TITLE_H - BAR_H;
  const pad     = { top: 20, right: 24, bottom: 44, left: 52 };
  const iw      = w    - pad.left - pad.right;
  const ih      = svgH - pad.top  - pad.bottom;
  const { assignments, centers, populations, method, n_clusters,
          implied_timescales: its } = result;

  const nFrames = assignments.length;
// @ts-ignore
  const _pca    = result._pca_pts as [number,number][] | undefined;

  // We'll re-fetch PCA from the most recent scatter chart data — stored as a
  // module-level variable so the cluster chart can colour the same points.
  const pts     = (window as any).__lastPcaPts as [number,number][] | null ?? null;

  if (!pts || pts.length !== nFrames) {
    // Fallback: simple population bar chart if PCA points not available
    area.innerHTML = `<div class="chart-placeholder">
      <div class="placeholder-icon">◈</div>
      <div class="placeholder-name">Clustering complete — run PCA first for scatter view</div>
      <div class="placeholder-hint">${n_clusters} clusters · ${method}</div></div>`;
    return;
  }

  const xmin = Math.min(...pts.map(p=>p[0])), xmax = Math.max(...pts.map(p=>p[0]));
  const ymin = Math.min(...pts.map(p=>p[1])), ymax = Math.max(...pts.map(p=>p[1]));
  const xr = (xmax-xmin)||1, yr = (ymax-ymin)||1;
  const sx = (v: number) => ((v-xmin)/xr * iw).toFixed(1);
  const sy = (v: number) => (ih - (v-ymin)/yr * ih).toFixed(1);

  const palette = (k: number) => `hsl(${(k/n_clusters)*300},70%,55%)`;

  // Dots coloured by cluster, opacity by dominant PCCA membership (or 1 for kmeans)
  const pcca: number[][] = result.pcca_membership ?? [];
  const dots = pts.map(([x,y], f) => {
    const k = assignments[f] as number;
    const opacity = pcca.length > 0
      ? (0.4 + 0.6 * (pcca[f]?.[k] ?? 0.5)).toFixed(2)
      : '0.7';
    return `<circle cx="${sx(x)}" cy="${sy(y)}" r="3.5" fill="${palette(k)}"
      opacity="${opacity}" data-f="${f}" style="cursor:${onFrameClick?'pointer':'default'}"/>`;
  }).join('');

  // Centroid markers (X)
  const centroids = centers.map((c: [number,number], k: number) => {
    const cx = sx(c[0]), cy = sy(c[1]);
    const col = palette(k);
    return `<line x1="${(+cx-6).toFixed(1)}" y1="${(+cy-6).toFixed(1)}" x2="${(+cx+6).toFixed(1)}" y2="${(+cy+6).toFixed(1)}" stroke="${col}" stroke-width="2.5"/>
            <line x1="${(+cx-6).toFixed(1)}" y1="${(+cy+6).toFixed(1)}" x2="${(+cx+6).toFixed(1)}" y2="${(+cy-6).toFixed(1)}" stroke="${col}" stroke-width="2.5"/>
            <text x="${(+cx+8).toFixed(1)}" y="${(+cy+4).toFixed(1)}" fill="${col}" font-size="9" font-family="monospace">C${k}</text>`;
  }).join('');

  // ITS legend
  const itsText = its && its.length > 0
    ? its.slice(0,3).map((t: number, i: number) => `ITS${i+2}=${t.toFixed(1)} fr`).join('  ')
    : '';

  // Population bars
  const bw   = (w-52-24) / n_clusters;
  const bars = populations.map((p: number, k: number) => {
    const bh = p * (BAR_H - 26);
    return `<rect x="${(52+k*bw+bw*0.1).toFixed(1)}" y="${(BAR_H-26-bh).toFixed(1)}"
      width="${(bw*0.8).toFixed(1)}" height="${bh.toFixed(1)}"
      fill="${palette(k)}" opacity="0.85" rx="1"/>
      <text x="${(52+(k+0.5)*bw).toFixed(1)}" y="${(BAR_H-6).toFixed(1)}" text-anchor="middle"
        fill="#7a7f85" font-size="9" font-family="monospace">${(p*100).toFixed(0)}%</text>`;
  }).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">${method === 'msm' ? 'MSM / PCCA' : 'K-means'} · ${n_clusters} clusters${itsText ? ' · ' + itsText : ''}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${svgH}" viewBox="0 0 ${w} ${svgH}" style="display:block">
      <g transform="translate(${pad.left},${pad.top})">
        ${dots}${centroids}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-38" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">PC2</text>
        <text x="${iw/2}" y="${(ih+36).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">PC1</text>
        ${onFrameClick ? `<text x="${iw}" y="-4" text-anchor="end" fill="#3d4245" font-size="9">click frame → 3D view</text>` : ''}
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
      dot.addEventListener('mouseenter', () => dot.setAttribute('opacity', '1'));
      dot.addEventListener('mouseleave', () => {
        const k = assignments[parseInt(dot.dataset.f!, 10)];
        const base = pcca.length > 0 ? (0.4 + 0.6*(pcca[parseInt(dot.dataset.f!,10)]?.[k]??0.5)).toFixed(2) : '0.7';
        dot.setAttribute('opacity', base);
      });
    });
  }
}

// ─── Geometry chart ───────────────────────────────────────────────────────────
//
// Multi-line chart for distance/angle time series.
// For umbrella source: x-axis = window index (CV order), overlaid with
// an optional "show on PMF" button.
// For trajectory source: x-axis = frame number with click-to-3D.

function renderGeometryChart(
  result: any,
  onFrameClick?: (frame: number) => void,
) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 620;
  const svgH    = (area.clientHeight || 420) - TITLE_H;
  const pad     = { top: 20, right: 24, bottom: 48, left: 62 };
  const iw      = w    - pad.left - pad.right;
  const ih      = svgH - pad.top  - pad.bottom;
  const { series, n_frames, source } = result;
  if (!series || series.length === 0) {
    showChartPlaceholder('Geometry Monitor');
    return;
  }

  // Collect all values to get global y range
  const allVals: number[] = (series as any[]).flatMap((s: any) => s.values.filter(isFinite));
  const ymin = Math.min(...allVals), ymax = Math.max(...allVals);
  const yr   = (ymax - ymin) || 1;
  const sy   = (v: number) => ih - ((v - ymin) / yr * ih);
  const sx   = (i: number) => (i / (n_frames - 1 || 1)) * iw;

  const palette = ['#00c4a7','#5b8dee','#e09a2e','#e05c5c','#a78dee'];

  const lines = (series as any[]).map((s: any, si: number) => {
    const isComposite = s.kind === 'composite';
    const col  = isComposite ? '#ffffff' : palette[si % palette.length];
    const dash = isComposite ? 'stroke-dasharray="5,3"' : '';
    const width = isComposite ? '2' : '1.5';
    const pts = (s.values as number[])
      .map((v, i) => isFinite(v) ? `${sx(i).toFixed(1)},${sy(v).toFixed(1)}` : null)
      .filter(Boolean).join(' ');
    return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="${width}" ${dash}
      stroke-linejoin="round" opacity="0.9"/>`;
  }).join('');

  const yTicks = Array.from({length:5}, (_,i) => {
    const v = ymin + yr*i/4;
    const y = sy(v);
    return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-6" y="${(y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="10">${v.toFixed(2)}</text>`;
  }).join('');

  const nTicks = Math.min(8, n_frames);
// @ts-ignore
  const _xTicks = Array.from({length:nTicks}, (_,i) => {
    const idx = Math.round(i/(nTicks-1||1)*(n_frames-1));
    return `<text x="${sx(idx).toFixed(1)}" y="${(ih+16).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">${idx}</text>`;
  }).join('');

  // Legend — vertical stack in top-right corner, background box for readability
  const LEG_ROW_H = 14;
  const LEG_PAD   = 6;
  const LEG_W     = 160;
  const LEG_H     = (series as any[]).length * LEG_ROW_H + LEG_PAD * 2;
  const LEG_X     = iw - LEG_W - 4;
  const LEG_Y     = 4;
  const legendRows = (series as any[]).map((s: any, si: number) => {
    const isComposite = s.kind === 'composite';
    const col  = isComposite ? '#e2e4e6' : palette[si % palette.length];
    const dash = isComposite ? 'stroke-dasharray="4,2"' : '';
    const ry   = LEG_Y + LEG_PAD + si * LEG_ROW_H + LEG_ROW_H / 2;
    const lx1  = LEG_X + LEG_PAD;
    const lx2  = lx1 + 16;
    const tx   = lx2 + 5;
    return `<line x1="${lx1}" y1="${ry}" x2="${lx2}" y2="${ry}" stroke="${col}" stroke-width="1.5" ${dash}/>
            <text x="${tx}" y="${ry + 3.5}" fill="${col}" font-size="9" font-family="monospace">${s.label}</text>`;
  }).join('');
  const legend = `<rect x="${LEG_X}" y="${LEG_Y}" width="${LEG_W}" height="${LEG_H}"
    rx="3" fill="#141618" opacity="0.82" stroke="#2e3235" stroke-width="0.5"/>
    ${legendRows}`;

  // Hover line
  const hitRects = onFrameClick ? Array.from({length: n_frames}, (_,i) =>
    `<rect x="${(sx(i)-5).toFixed(1)}" y="0" width="10" height="${ih}" fill="transparent" data-f="${i}" style="cursor:pointer"/>`
  ).join('') : '';

  const xLabel = source === 'umbrella' ? 'Window index' : 'Frame';

  // PMF overlay button (umbrella only)
  const overlayBtn = source === 'umbrella'
    ? `<button class="chart-export-btn" id="pmf-overlay-btn" style="margin-right:8px">Overlay on PMF</button>`
    : '';

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
        ${yTicks}${lines}${hitRects}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-48" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="11">Distance (Å) / Angle (°)</text>
        <text x="${iw/2}" y="${(ih+40).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="11">${xLabel}</text>
        ${legend}
        ${onFrameClick ? `<line id="geo-hover" x1="0" y1="0" x2="0" y2="${ih}" stroke="#00c4a7" stroke-width="0.5" stroke-dasharray="3,3" opacity="0" pointer-events="none"/>` : ''}
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);

  // PMF overlay: re-render PMF with geometry panel appended
  document.getElementById('pmf-overlay-btn')?.addEventListener('click', () => {
    const mbarData = (window as any).__lastMbarResult;
    if (!mbarData) { alert('Run MBAR / PMF first.'); return; }
    renderPmfChart(mbarData, undefined, result);
  });

  if (onFrameClick) {
    const svg = area.querySelector('svg')!;
    const hoverLine = document.getElementById('geo-hover');
    svg.addEventListener('mousemove', (e) => {
      const rect = svg.getBoundingClientRect();
      const idx  = Math.round((e.clientX - rect.left - pad.left) / iw * (n_frames-1));
      const x    = sx(Math.max(0, Math.min(n_frames-1, idx))).toFixed(1);
      hoverLine?.setAttribute('x1', x); hoverLine?.setAttribute('x2', x);
      hoverLine?.setAttribute('opacity', '1');
    });
    svg.addEventListener('mouseleave', () => hoverLine?.setAttribute('opacity', '0'));
    area.querySelectorAll<SVGRectElement>('rect[data-f]').forEach(r =>
      r.addEventListener('click', () => onFrameClick(parseInt(r.dataset.f!, 10)))
    );
  }
}

// ─── Dihedral time-series chart ───────────────────────────────────────────────
//
// Two-panel chart (φ top, ψ bottom) for one or more residues.
// Points coloured cold→warm by time. Horizontal bands mark the main
// rotameric regions (α helix −60°, β sheet −120°/+120°).
// Clicking a frame opens the analysis viewer for that frame.

function renderDihedralTimeSeries(
  seriesData: any[],
  onFrameClick?: (frame: number) => void,
) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 620;
  const panelH  = Math.floor(((area.clientHeight || 500) - TITLE_H) / 2);
  const pad     = { top: 18, right: 24, bottom: 36, left: 52 };
  const iw      = w - pad.left - pad.right;
  const ih      = panelH - pad.top - pad.bottom;
  const nFrames = seriesData[0]?.phi?.length ?? 0;
  if (nFrames === 0) { showChartPlaceholder('Dihedral Time Series'); return; }

  const palette = ['#00c4a7','#5b8dee','#e09a2e','#e05c5c','#a78dee'];
  const sx = (i: number) => (i / (nFrames-1||1)) * iw;

  // Build one SVG panel for either phi or psi
  const makePanel = (angleKey: 'phi' | 'psi', yLabel: string, svgY: number) => {
    // Background bands for rotameric regions
    const toY  = (deg: number) => ih - ((deg + 180) / 360 * ih);
    const band = (lo: number, hi: number, col: string) => {
      const y1 = Math.min(toY(lo), toY(hi));
      const h  = Math.abs(toY(lo) - toY(hi));
      return `<rect x="0" y="${y1.toFixed(1)}" width="${iw}" height="${h.toFixed(1)}" fill="${col}" opacity="0.06"/>`;
    };

    const bands = [
      band(-145, -30, '#00c4a7'),   // α-helix range
      band(90,  180, '#5b8dee'),    // β-sheet range
      band(-180,-155,'#5b8dee'),    // β wrap-around
    ].join('');

    const lines = seriesData.map((s: any, si: number) => {
      const col = palette[si % palette.length];
      const vals: number[] = s[angleKey];
      const pts = vals.map((v, i) =>
        isFinite(v) ? `${sx(i).toFixed(1)},${toY(v).toFixed(1)}` : null
      ).filter(Boolean).join(' ');
      return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.2"
        stroke-linejoin="round" opacity="0.85"/>`;
    }).join('');

    const yTicks = [-180,-90,0,90,180].map(v => {
      const y = toY(v);
      return `<line x1="0" y1="${y.toFixed(1)}" x2="${iw}" y2="${y.toFixed(1)}" stroke="#2e3235" stroke-width="0.5"/>
              <text x="-6" y="${(y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="9">${v}</text>`;
    }).join('');

    const nxt = Math.min(6, nFrames);
    const xTicks = angleKey === 'psi' ? Array.from({length:nxt}, (_,i) => {
      const idx = Math.round(i/(nxt-1||1)*(nFrames-1));
      return `<text x="${sx(idx).toFixed(1)}" y="${(ih+14).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">${idx}</text>`;
    }).join('') : '';

    return `<g transform="translate(${pad.left},${svgY + pad.top})">
      ${bands}${yTicks}${lines}
      <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
      <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
      <text x="${-(ih/2)}" y="-38" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">${yLabel} (°)</text>
      ${xTicks}
    </g>`;
  };

// @ts-ignore
  const _legend = seriesData.map((s: any, si: number) =>
    `<tspan fill="${palette[si%palette.length]}">${s.res_name} ${s.res_seq}</tspan>  `
  ).join('');

  const hitRects = onFrameClick ? Array.from({length: nFrames}, (_,i) =>
    `<rect x="${(pad.left + sx(i) - 4).toFixed(1)}" y="0" width="8" height="${TITLE_H + panelH*2}"
       fill="transparent" data-f="${i}" style="cursor:pointer"/>`
  ).join('') : '';

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">Dihedral Time Series · ${seriesData.map((s:any) => `${s.res_name} ${s.res_seq}`).join(', ')}</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${panelH*2}" viewBox="0 0 ${w} ${panelH*2}" style="display:block">
      ${makePanel('phi', 'φ', 0)}
      ${makePanel('psi', 'ψ', panelH)}
      <text x="${pad.left + iw/2}" y="${panelH*2 - 4}" text-anchor="middle" fill="#7a7f85" font-size="10">Frame</text>
      ${hitRects}
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
  if (onFrameClick) {
    area.querySelectorAll<SVGRectElement>('rect[data-f]').forEach(r =>
      r.addEventListener('click', () => onFrameClick(parseInt(r.dataset.f!, 10)))
    );
  }
}

// ─── SASA chart ───────────────────────────────────────────────────────────────
//
// Two-panel layout:
//   Top: per-residue mean SASA bar chart with ±1σ error bars, coloured by
//        exposure level (blue=buried → red=exposed). Clicking a bar opens the
//        residue in the analysis viewer with SASA mapped to bfactor.
//   Bottom: total SASA vs frame line chart.

function renderSasaChart(result: any, onResidueClick?: (atomIdx: number) => void) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 620;
  const totalH  = (area.clientHeight || 520) - TITLE_H;
  const BAR_H   = Math.floor(totalH * 0.62);
  const LINE_H  = totalH - BAR_H;

  const { per_residue_mean, per_residue_std, total_per_frame, res_labels } = result;
  const n_res    = per_residue_mean.length;
  const n_frames = total_per_frame.length;

  // ── Per-residue bar chart ────────────────────────────────────────────────
  const PAD_L = 58, PAD_R = 16, PAD_T = 20, PAD_B = 36;
  const bw    = (w - PAD_L - PAD_R) / n_res;
// @ts-ignore
  const _iw   = w - PAD_L - PAD_R;
  const ih_b  = BAR_H - PAD_T - PAD_B;
  const maxV  = Math.max(...per_residue_mean, 1e-9);

  const bars = per_residue_mean.map((v: number, i: number) => {
    const sigma = per_residue_std[i] ?? 0;
    const frac  = v / maxV;
    // Color: blue (buried, low SASA) → teal → red (exposed, high SASA)
    const r = Math.round(frac * 220);
    const g = Math.round((1 - Math.abs(frac - 0.5) * 2) * 150);
    const b = Math.round((1 - frac) * 210);
    const col  = `rgb(${r},${g},${b})`;
    const bh   = frac * ih_b;
    const bx   = (PAD_L + i * bw + bw * 0.1).toFixed(1);
    const by   = (PAD_T + ih_b - bh).toFixed(1);
    const bwi  = (bw * 0.8).toFixed(1);
    const errH = Math.min(sigma / maxV * ih_b, bh * 0.5);
    const errY = (PAD_T + ih_b - bh - errH).toFixed(1);
    const errX = (PAD_L + (i + 0.5) * bw).toFixed(1);
    return `<rect x="${bx}" y="${by}" width="${bwi}" height="${bh.toFixed(1)}"
      fill="${col}" opacity="0.85" rx="1" data-i="${i}"
      style="cursor:${onResidueClick ? 'pointer' : 'default'}"/>
      <line x1="${errX}" y1="${errY}" x2="${errX}" y2="${(PAD_T + ih_b - bh + errH).toFixed(1)}"
        stroke="${col}" stroke-width="1" opacity="0.6"/>`;
  }).join('');

  // X-axis labels — show every Nth residue to avoid crowding
  const step   = Math.max(1, Math.ceil(n_res / 20));
  const xTicks = per_residue_mean.map((_: number, i: number) => {
    if (i % step !== 0) return '';
    const x   = (PAD_L + (i + 0.5) * bw).toFixed(1);
    const lbl = res_labels[i] ?? String(i);
    return `<text x="${x}" y="${(PAD_T + ih_b + 14).toFixed(1)}" text-anchor="middle"
      fill="#7a7f85" font-size="8" transform="rotate(-35,${x},${(PAD_T + ih_b + 14).toFixed(1)})">${lbl}</text>`;
  }).join('');

  const yTicks = Array.from({length: 5}, (_, i) => {
    const v = maxV * i / 4;
    const y = (PAD_T + ih_b - (v / maxV) * ih_b).toFixed(1);
    return `<line x1="${PAD_L}" y1="${y}" x2="${w - PAD_R}" y2="${y}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="${(PAD_L - 4).toFixed(1)}" y="${(+y + 4).toFixed(1)}" text-anchor="end"
              fill="#7a7f85" font-size="9">${v.toFixed(0)}</text>`;
  }).join('');

  // ── Total SASA line chart ────────────────────────────────────────────────
  const L2 = { t: 12, b: 28, l: PAD_L, r: PAD_R };
  const iw2 = w - L2.l - L2.r;
  const ih2 = LINE_H - L2.t - L2.b;
  const tMin = Math.min(...total_per_frame), tMax = Math.max(...total_per_frame);
  const tRange = (tMax - tMin) || 1;
  const tsx = (i: number) => (i / (n_frames - 1 || 1)) * iw2;
  const tsy = (v: number) => ih2 - ((v - tMin) / tRange * ih2);
  const linePts = total_per_frame.map((v: number, i: number) =>
    `${tsx(i).toFixed(1)},${tsy(v).toFixed(1)}`).join(' ');
  const tyTick = [tMin, (tMin + tMax) / 2, tMax].map(v => {
    const y = tsy(v);
    return `<text x="${(L2.l - 4).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="end"
      fill="#7a7f85" font-size="9">${v.toFixed(0)}</text>`;
  }).join('');

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">SASA · ${n_res} residues · ${n_frames} frames</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    <svg width="${w}" height="${BAR_H}" viewBox="0 0 ${w} ${BAR_H}" style="display:block">
      ${yTicks}${bars}${xTicks}
      <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${PAD_T + ih_b}" stroke="#555" stroke-width="1"/>
      <line x1="${PAD_L}" y1="${PAD_T + ih_b}" x2="${w - PAD_R}" y2="${PAD_T + ih_b}" stroke="#555" stroke-width="1"/>
      <text x="${(PAD_L - 40).toFixed(1)}" y="${(PAD_T + ih_b / 2).toFixed(1)}"
        transform="rotate(-90,${(PAD_L - 40).toFixed(1)},${(PAD_T + ih_b / 2).toFixed(1)})"
        text-anchor="middle" fill="#7a7f85" font-size="10">SASA (Å²)</text>
      ${onResidueClick ? `<text x="${w - PAD_R}" y="${PAD_T - 4}" text-anchor="end" fill="#3d4245" font-size="9">click → 3D view</text>` : ''}
    </svg>
    <svg width="${w}" height="${LINE_H}" viewBox="0 0 ${w} ${LINE_H}"
         style="display:block;border-top:1px solid #2e3235">
      <g transform="translate(${L2.l},${L2.t})">
        ${tyTick}
        <polyline points="${linePts}" fill="none" stroke="#00c4a7" stroke-width="1.5"/>
        <line x1="0" y1="0" x2="0" y2="${ih2}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih2}" x2="${iw2}" y2="${ih2}" stroke="#555" stroke-width="1"/>
        <text x="${iw2/2}" y="${(ih2 + 20).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="10">Frame</text>
        <text x="${(-ih2/2).toFixed(1)}" y="-40" transform="rotate(-90)"
          text-anchor="middle" fill="#7a7f85" font-size="10">Total SASA (Å²)</text>
      </g>
    </svg>`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);

  if (onResidueClick) {
    area.querySelectorAll<SVGRectElement>('rect[data-i]').forEach(bar => {
      bar.addEventListener('click', () => onResidueClick(parseInt(bar.dataset.i!, 10)));
      bar.addEventListener('mouseenter', () => bar.setAttribute('opacity', '1'));
      bar.addEventListener('mouseleave', () => bar.setAttribute('opacity', '0.85'));
    });
  }
}

// ─── Membrane chart ───────────────────────────────────────────────────────────
//
// Four-panel SVG layout:
//   Panel A  Thickness vs frame (teal line)
//   Panel B  Area per lipid vs frame (amber line) — hidden if has_apl=false
//   Panel C  Upper/lower leaflet z-density (overlaid step histograms)
//   Panel D  |SCD| order parameter profile (horizontal bars, coloured by value)

function renderMembraneChart(result: any) {
  const area    = $('#chart-display')!;
  const TITLE_H = 36;
  const w       = area.clientWidth  || 700;
  const totalH  = (area.clientHeight || 560) - TITLE_H;

  const { thickness_per_frame: tpf, apl_per_frame: apf,
          z_density_upper: zdu, z_density_lower: zdl,
          order_params: op, order_labels: ol,
          mean_thickness, mean_apl, has_apl,
          n_upper, n_lower, n_frames } = result;

  // n_upper / n_lower are now lipid (residue) counts, not atom counts

  // Layout: split vertically into rows
  const hasOrder = op && op.length > 0;
  const nRows    = has_apl ? (hasOrder ? 4 : 3) : (hasOrder ? 3 : 2);
  const rowH     = Math.floor(totalH / nRows);

  // ── Shared helpers ──────────────────────────────────────────────────────
  const PL = 60, PR = 20, PT = 14, PB = 28;
  const iw = w - PL - PR;

  const makeLine = (
    values: number[], color: string, label: string, yUnit: string, panelH: number,
  ): string => {
    const ih  = panelH - PT - PB;
    const n   = values.length;
    const mn  = Math.min(...values), mx = Math.max(...values);
    const yr  = (mx - mn) || 1;
    const sx  = (i: number) => (i / (n - 1 || 1)) * iw;
    const sy  = (v: number) => ih - ((v - mn) / yr * ih);
    const pts = values.map((v, i) => `${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(' ');
    const yTicks = [mn, (mn+mx)/2, mx].map(v =>
      `<text x="-5" y="${(sy(v)+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="9">${v.toFixed(1)}</text>
       <line x1="0" y1="${sy(v).toFixed(1)}" x2="${iw}" y2="${sy(v).toFixed(1)}" stroke="#2e3235" stroke-width="0.4"/>`
    ).join('');
    const nxt   = Math.min(8, n);
    const xLbls = Array.from({length: nxt}, (_, k) => {
      const idx = Math.round(k / (nxt - 1 || 1) * (n - 1));
      return `<text x="${sx(idx).toFixed(1)}" y="${(ih + 16).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">${idx}</text>`;
    }).join('');
// @ts-ignore
    const _meanY = sy((mn + mx) / 2 + (mx - mn) * 0.05);
    return `<svg width="${w}" height="${panelH}" style="display:block;border-bottom:1px solid #1e2428">
      <g transform="translate(${PL},${PT})">
        ${yTicks}
        <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round"/>
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-44" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">${label} (${yUnit})</text>
        <text x="${iw}" y="${PT - 2}" text-anchor="end" fill="${color}" font-size="9" font-family="monospace" opacity="0.7">mean ${((mn+mx)/2).toFixed(2)} ${yUnit}</text>
        ${xLbls}
        <text x="${iw/2}" y="${(ih + 26).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">Frame</text>
      </g>
    </svg>`;
  };

  // ── Z-density panel ─────────────────────────────────────────────────────
  // Both leaflets share a single z-axis so the separation between bilayer
  // halves is visually meaningful.  The Rust backend returns independently-
  // normalised 50-bin histograms (each from its own zmin/zmax); we re-use
  // them as-is but map bin index → shared x-axis, draw lower on the left
  // half and upper on the right half, mirroring the physical bilayer layout
  // (lower leaflet = lower z = left side of plot).
  const makeZDensity = (panelH: number): string => {
    const ih = panelH - PT - PB;
    const nb = zdu.length;  // 50
    const bw = iw / nb;

    // Lower leaflet: bins 0–49 map to left half of plot
    const lowerBars = zdl.map((v: number, i: number) => {
      const x = (i * bw).toFixed(1);
      const h = (ih * v).toFixed(1);
      return `<rect x="${x}" y="${(ih * (1 - v)).toFixed(1)}" width="${(bw + 0.3).toFixed(1)}" height="${h}" fill="#534AB7" opacity="0.60"/>`;
    }).join('');

    // Upper leaflet: bins 0–49 map to right half (reversed so high z = right edge)
    const upperBars = zdu.map((v: number, i: number) => {
      const x = (iw - (i + 1) * bw).toFixed(1);
      const h = (ih * v).toFixed(1);
      return `<rect x="${x}" y="${(ih * (1 - v)).toFixed(1)}" width="${(bw + 0.3).toFixed(1)}" height="${h}" fill="#1D9E75" opacity="0.55"/>`;
    }).join('');

    // Centre divider
    const midX = (iw / 2).toFixed(1);

    return `<svg width="${w}" height="${panelH}" style="display:block;border-bottom:1px solid #1e2428">
      <g transform="translate(${PL},${PT})">
        ${lowerBars}${upperBars}
        <line x1="${midX}" y1="0" x2="${midX}" y2="${ih}" stroke="#444" stroke-width="0.8" stroke-dasharray="4,3"/>
        <line x1="0"       y1="0" x2="0"       y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0"       y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-44" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">Density (norm.)</text>
        <text x="${(+midX/2).toFixed(1)}"           y="${(ih+16).toFixed(1)}" text-anchor="middle" fill="#534AB7" font-size="9" opacity="0.8">← lower z</text>
        <text x="${(+midX + (iw-+midX)/2).toFixed(1)}" y="${(ih+16).toFixed(1)}" text-anchor="middle" fill="#1D9E75" font-size="9" opacity="0.8">upper z →</text>
        <rect x="${iw-124}" y="4" width="120" height="34" rx="3" fill="#141618" opacity="0.85"/>
        <rect x="${iw-116}" y="12" width="10" height="8" fill="#1D9E75" opacity="0.7"/>
        <text x="${iw-102}" y="20" fill="#7a7f85" font-size="9" font-family="monospace">Upper (${n_upper} lipids)</text>
        <rect x="${iw-116}" y="25" width="10" height="8" fill="#534AB7" opacity="0.7"/>
        <text x="${iw-102}" y="33" fill="#7a7f85" font-size="9" font-family="monospace">Lower (${n_lower} lipids)</text>
      </g>
    </svg>`;
  };

  // ── Order parameter panel ───────────────────────────────────────────────
  // When there are many carbons (acyl chains), draw as a polyline profile
  // rather than individual bars — much cleaner and matches what's published
  // in membrane MD papers.  For ≤40 carbons keep the bar chart.
  const makeOrderPanel = (panelH: number): string => {
    if (!op || op.length === 0) return '';
    const ih   = panelH - PT - PB;
    const n    = op.length;
    const maxV = Math.max(...op, 0.5);

    // How many y-axis labels to show — cap at one per 14px to avoid overlap
    const maxLbls = Math.max(2, Math.floor(ih / 14));
    const step    = Math.ceil(n / maxLbls);

    const xTick = [0, 0.25, 0.5, 0.75, 1.0].map(f => {
      const x = (f * maxV / maxV * iw).toFixed(1);
      const v = (f * maxV).toFixed(2);
      return `<line x1="${x}" y1="0" x2="${x}" y2="${ih}" stroke="#2e3235" stroke-width="0.4"/>
              <text x="${x}" y="${(ih + 14).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">${v}</text>`;
    }).join('');

    let content: string;
    if (n <= 40) {
      // Bar chart for short acyl chains
      const bh = ih / n;
      const bars = op.map((v: number, i: number) => {
        const t   = v / maxV;
        const col = `rgb(${Math.round(t*220)},${Math.round((1-Math.abs(t-0.5)*2)*140+40)},${Math.round((1-t)*200+30)})`;
        return `<rect x="0" y="${(i * bh).toFixed(1)}" width="${(v / maxV * iw).toFixed(1)}" height="${(bh * 0.78).toFixed(1)}" fill="${col}" opacity="0.85" rx="1"/>`;
      }).join('');
      const yLbls = op.map((_: number, i: number) => {
        if (i % step !== 0) return '';
        const y = (i * bh + bh * 0.5).toFixed(1);
        return `<text x="-4" y="${(+y + 3).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="8">${ol[i] ?? i}</text>`;
      }).join('');
      content = bars + yLbls;
    } else {
      // Polyline profile for long chains — draw as |SCD| vs carbon index
      const sy  = (i: number) => ((i / (n - 1)) * ih).toFixed(1);
      const sx2 = (v: number) => (v / maxV * iw).toFixed(1);
      const pts = op.map((v: number, i: number) => `${sx2(v)},${sy(i)}`).join(' ');
      // Filled area for visual weight
      const area2 = `M0,0 ` + op.map((v: number, i: number) => `L${sx2(v)},${sy(i)}`).join(' ') + ` L0,${ih} Z`;
      const yLbls = op.map((_: number, i: number) => {
        if (i % step !== 0) return '';
        const y = sy(i);
        return `<text x="-4" y="${(+y + 3).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="8">${ol[i] ?? i}</text>`;
      }).join('');
      content = `<path d="${area2}" fill="#1D9E75" opacity="0.18"/>
                 <polyline points="${pts}" fill="none" stroke="#1D9E75" stroke-width="1.5" stroke-linejoin="round" opacity="0.9"/>
                 ${yLbls}`;
    }

    return `<svg width="${w}" height="${panelH}" style="display:block">
      <g transform="translate(${PL},${PT})">
        ${xTick}
        ${content}
        <line x1="0" y1="0" x2="0" y2="${ih}" stroke="#555" stroke-width="1"/>
        <line x1="0" y1="${ih}" x2="${iw}" y2="${ih}" stroke="#555" stroke-width="1"/>
        <text x="${-(ih/2)}" y="-44" transform="rotate(-90)" text-anchor="middle" fill="#7a7f85" font-size="10">Carbon</text>
        <text x="${iw/2}" y="${(ih + 24).toFixed(1)}" text-anchor="middle" fill="#7a7f85" font-size="9">|SCD| order parameter</text>
      </g>
    </svg>`;
  };

  const aplLabel = has_apl
    ? `· APL ${mean_apl.toFixed(1)} Ų`
    : '· APL unavailable (no box)';

  area.innerHTML = `
    <div class="chart-title-bar">
      <span class="chart-title-text">Membrane · thickness ${mean_thickness.toFixed(2)} Å ${aplLabel} · ${n_frames} frames</span>
      <button class="chart-export-btn" id="chart-save-btn">Export CSV</button>
    </div>
    ${makeLine(tpf, '#1D9E75', 'Thickness', 'Å', rowH)}
    ${has_apl ? makeLine(apf, '#e09a2e', 'Area per lipid', 'Ų', rowH) : ''}
    ${makeZDensity(rowH)}
    ${hasOrder ? makeOrderPanel(rowH) : ''}`;

  $('#chart-save-btn')?.addEventListener('click', exportCsv);
}

// ─── Analysis 3D viewer ───────────────────────────────────────────────────────
async function openAnalysisViewer(params: Record<string, string | number>, title: string) {
  try {
    const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
    const qs    = new URLSearchParams(
      Object.fromEntries(Object.entries(params).map(([k, v]) => [k, String(v)]))
    ).toString();
    const label = `analysis-${params.mode}-${Date.now()}`;
    const viewer = new WebviewWindow(label, {
      url: `analysis_viewer.html?${qs}`, title: `3D · ${title}`,
      width: 860, height: 680, focus: true, visible: true,
    });
    viewer.once('tauri://created', () => log(`3D viewer: ${title}`, 'success'));
    viewer.once('tauri://error',   (e: any) => log(`3D viewer error: ${JSON.stringify(e)}`, 'error'));
  } catch (err) { log(`3D viewer failed: ${err}`, 'error'); }
}

// ─── PMF chart ────────────────────────────────────────────────────────────────
//
// Two-panel SVG: top = KDE overlap (window distributions), bottom = PMF with
// error band, ΔG‡ annotation, and clickable points → analysis viewer.

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

  // Rainbow palette
  const col = (i: number) => `hsl(${(i / n_windows) * 300},70%,55%)`;

  // ── Histogram panel ───────────────────────────────────────────────────────
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

  // ── PMF panel ─────────────────────────────────────────────────────────────
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

  // ΔG annotation near the maximum
  const dgIdx   = pmf.reduce((mi: number, v: number, i: number) =>
    isFinite(v) && v > (pmf[mi] ?? -Infinity) ? i : mi, 0);
  const dgAnnX  = sx(bc[dgIdx]).toFixed(1);
  const dgAnnY  = (sy(pmfMax) - 22).toFixed(1);

  // Y ticks
  const yTicks  = Array.from({ length: 5 }, (_, i) => {
    const v = pmfMin + pmfR * i / 4;
    const y = sy(v).toFixed(1);
    return `<line x1="0" y1="${y}" x2="${iw}" y2="${y}" stroke="#2e3235" stroke-width="0.5"/>
            <text x="-8" y="${(+y+4).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="9">${v.toFixed(1)}</text>`;
  }).join('');

  // X tick labels (window centres)
  const nXT = Math.min(10, wv.length);
  const xTicks = Array.from({ length: nXT }, (_, i) => {
    const v = wv[Math.round(i / (nXT - 1) * (wv.length - 1))];
    return `<text x="${sx(v).toFixed(1)}" y="${(pmfIH + 16).toFixed(1)}"
              text-anchor="middle" fill="#7a7f85" font-size="9">${v.toFixed(2)}</text>`;
  }).join('');

  // Clickable hit-dots
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

  // Store for geometry overlay
  (window as any).__lastMbarResult = result;

  // Geometry overlay panel (appended below PMF when geo data provided)
  if (geoOverlay && geoOverlay.series?.length > 0) {
    const geo     = geoOverlay;
    const GEO_H   = 120;
    const GEO_PL  = PL; 
// @ts-ignore
    const _GEO_PR = PR, GEO_PT = 10, GEO_PB = 30;
    const geoIH   = GEO_H - GEO_PT - GEO_PB;
    const geoIW   = iw;
    const allVals: number[] = geo.series.flatMap((s: any) => s.values.filter(isFinite));
    const gMin    = Math.min(...allVals), gMax = Math.max(...allVals);
    const gRange  = (gMax - gMin) || 1;
    const geoPalette = ['#00c4a7','#e09a2e','#e05c5c','#5b8dee'];
    const gsy = (v: number) => geoIH - ((v-gMin)/gRange * geoIH);

    const gLines = geo.series.map((s: any, si: number) => {
      const isComposite = s.kind === 'composite';
      const col  = isComposite ? '#e2e4e6' : geoPalette[si % geoPalette.length];
      const dash = isComposite ? 'stroke-dasharray="4,2"' : '';
      const nWin = geo.n_frames;
      const pts  = s.values.map((v: number, i: number) => {
        const x = (i / (nWin-1||1)) * geoIW;
        return isFinite(v) ? `${x.toFixed(1)},${gsy(v).toFixed(1)}` : null;
      }).filter(Boolean).join(' ');
      return `<polyline points="${pts}" fill="none" stroke="${col}" stroke-width="1.5" ${dash}/>`;
    }).join('');

    // Vertical legend box (same style as main geometry chart)
    const gLegRowH = 12;
    const gLegPad  = 5;
    const gLegW    = 150;
    const nSeries  = geo.series.length;
    const gLegH    = nSeries * gLegRowH + gLegPad * 2;
    const gLegX    = geoIW - gLegW - 2;
    const gLegY    = 2;
    const gLegRows = geo.series.map((s: any, si: number) => {
      const isComposite = s.kind === 'composite';
      const col  = isComposite ? '#e2e4e6' : geoPalette[si % geoPalette.length];
      const dash = isComposite ? 'stroke-dasharray="4,2"' : '';
      const ry   = gLegY + gLegPad + si * gLegRowH + gLegRowH / 2;
      const lx1  = gLegX + gLegPad;
      const lx2  = lx1 + 14;
      return `<line x1="${lx1}" y1="${ry}" x2="${lx2}" y2="${ry}" stroke="${col}" stroke-width="1.5" ${dash}/>
              <text x="${lx2+4}" y="${ry+3}" fill="${col}" font-size="8" font-family="monospace">${s.label}</text>`;
    }).join('');
    const gLegend = `<rect x="${gLegX}" y="${gLegY}" width="${gLegW}" height="${gLegH}"
      rx="2" fill="#141618" opacity="0.82" stroke="#2e3235" stroke-width="0.5"/>
      ${gLegRows}`;

    const gTicks = [gMin, (gMin+gMax)/2, gMax].map(v => {
      const y = gsy(v);
      return `<text x="-4" y="${(y+3).toFixed(1)}" text-anchor="end" fill="#7a7f85" font-size="8">${v.toFixed(1)}</text>`;
    }).join('');

    const geoSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
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

  const runBtn    = $('#run-btn')    as HTMLButtonElement;
  const cancelBtn = $('#cancel-btn') as HTMLButtonElement;

  // Tools that support server-side cancellation
  const CANCELLABLE: Record<string, string> = {
    sasa:     'cancel_sasa',
    membrane: 'cancel_membrane',
  };
  const cancelCmd = CANCELLABLE[tool.id];

  if (runBtn)    { runBtn.disabled = true; runBtn.innerHTML = '<span class="spinner"></span> Running…'; }
  if (cancelBtn && cancelCmd) {
    cancelBtn.classList.add('visible');
    cancelBtn.onclick = () => {
      invoke(cancelCmd).catch(() => {});
      log(`Cancelling ${tool.label}…`, 'warn');
      cancelBtn.disabled = true;
      cancelBtn.textContent = 'Cancelling…';
    };
  }

  // Build click callbacks — each opens a self-contained analysis viewer window
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
        // Parse atom indices from params
        const idxStr = (document.getElementById('param-atomIndices') as HTMLInputElement)?.value ?? '0';
        const atomIndices = idxStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
        try {
          const series = await invoke<any[]>('get_dihedral_time_series', { atomIndices });
          renderDihedralTimeSeries(series, frameClick);
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

    // Build invoke params — geometry tool needs special pre-processing
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
      const composites:     [number,number][] = [];
      const compWeights:    number[]          = [];
      const compLabels:     string[]          = [];

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
        log('No atom pairs to measure — load a cv.rst file or enter manual pairs below.', 'warn');
        showChartPlaceholder(tool.label);
        if (runBtn) { runBtn.disabled = false; runBtn.textContent = 'Run'; }
        return;
      }
      invokeParams = {
        pairs: allPairs, triplets: manualTriplets, source: sourceVal,
        labels: allLabels, composites, compositeWeights: compWeights, compositeLabels: compLabels,
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
        // Cache PCA points for the cluster chart
        (window as any).__lastPcaPts = pts;
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
        const sasaResClick = (atomIdx: number) =>
          openAnalysisViewer({ mode: 'residue', index: atomIdx }, `SASA · Residue ${atomIdx}`);
        renderSasaChart(result.data, sasaResClick);
        // Map mean SASA to bfactor for 3D coloring
        if (result.data.per_residue_mean) {
          try { await invoke('set_bfactors', { values: result.data.per_residue_mean }); } catch (_) {}
        }
        const map3dBtn = document.getElementById('map3d-btn') as HTMLButtonElement | null;
        if (map3dBtn) { map3dBtn.disabled = false; map3dBtn.classList.remove('disabled'); }
        break;
      }
      case 'membrane':
        renderMembraneChart(result.data);
        break;
      case 'cluster': {
        const clusterFrameClick = (frame: number) =>
          openAnalysisViewer({ mode: 'cluster', cluster_frame: frame }, `Cluster · Frame ${frame}`);
        renderClusterChart(result.data, clusterFrameClick);
        const pop = (result.data.populations as number[]).map((p: number, k: number) => `C${k}=${(p*100).toFixed(0)}%`).join(' ');
        log(`${result.data.n_clusters} clusters · ${pop}`, 'success');
        break;
      }
      case 'geometry': {
        renderGeometryChart(result.data, result.data.source === 'trajectory' ? frameClick : undefined);
        break;
      }
      case 'dihedral_ts':
        // handled below — no invoke call needed
        break;
      case 'ramachandran': {        const residues = result.data?.residues ?? [];
        if (residues.length === 0) {
          log('No residue dihedral data returned — check atom selection includes backbone atoms.', 'warn');
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
        // Store effectiveness as bfactor for Map to 3D
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
    const msg = String(e);
    if (msg === 'Cancelled') {
      log(`${tool.label} cancelled.`, 'warn');
    } else {
      log(`${tool.label} failed: ${e}`, 'error');
    }
    showChartPlaceholder(tool.label);
  } finally {
    if (runBtn)    { runBtn.disabled = false; runBtn.textContent = 'Run'; }
    if (cancelBtn) { cancelBtn.classList.remove('visible'); cancelBtn.disabled = false; cancelBtn.textContent = '✕ Cancel'; }
  }
}

// ─── File / load actions ──────────────────────────────────────────────────────

async function exportCsv() {
  const path = await save({
    defaultPath: 'analysis_results.csv',
    filters: [{ name: 'CSV', extensions: ['csv'] }]
  });
  if (!path) return;
  log(`Exporting to ${path}…`);
  try {
    const msg: string = await invoke('export_csv', { path });
    log(msg, 'success');
  } catch (e) {
    log(`Export failed: ${e}`, 'error');
  }
}

async function openUmbrellaViewer() {
  if (!umbrellaLoaded) {
    log('Scan & load umbrella windows first (QM/MM sidebar).', 'warn');
    return;
  }
  try {
    const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
    const win = new WebviewWindow(`umbrella-viewer-${Date.now()}`, {
      url:     'umbrella_viewer.html',
      title:   'Umbrella Window Trajectory',
      width:   920,
      height:  700,
      focus:   true,
      visible: true,
    });
    win.once('tauri://created',  () => log('Umbrella viewer opened.', 'success'));
    win.once('tauri://error', (e: any) => log(`Umbrella viewer error: ${JSON.stringify(e)}`, 'error'));
  } catch (err) {
    log(`Umbrella viewer failed: ${err}`, 'error');
  }
}

async function openVisualizer() {
  log('Opening 3D Visualizer…');
  try {
    const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
    const viewer = new WebviewWindow('visualizer', {
      url: `visualizer.html?frames=${totalFrames}`,
      title: 'MD Trajectory Visualizer',
      width: 900, height: 700,
      focus: true, visible: true,
    });
    viewer.once('tauri://created', () => log('Visualizer window opened.', 'success'));
    viewer.once('tauri://error',   (e: any) => log(`Visualizer error: ${JSON.stringify(e)}`, 'error'));
  } catch (err) {
    log(`Visualizer failed: ${err}`, 'error');
  }
}

async function openLigandBuilder() {
  log('Opening Ligand Builder…');
  try {
    const { WebviewWindow } = await import('@tauri-apps/api/webviewWindow');
    const win = new WebviewWindow('ligand-builder', {
      url:     'ligand_builder.html',
      title:   'Ligand Builder',
      width:   1280,
      height:  860,
      focus:   true,
      visible: true,
      resizable: true,
    });
    win.once('tauri://created', () => log('Ligand Builder opened.', 'success'));
    win.once('tauri://error',   (e: any) => log(`Ligand Builder error: ${JSON.stringify(e)}`, 'error'));
  } catch (err) {
    log(`Ligand Builder failed: ${err}`, 'error');
  }
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
    gap: 0;          /* category nav handles its own spacing */
    height: 44px;
  }
  .header-logo { font-family: var(--font-mono); font-size: 12px; font-weight: 500; color: var(--accent); letter-spacing: .05em; text-transform: uppercase; flex-shrink: 0; }
  .header-sep  { width: 1px; height: 18px; background: var(--border); margin: 0 14px; flex-shrink: 0; }
  .header-sub  { color: var(--muted); font-size: 10px; letter-spacing: .07em; text-transform: uppercase; flex-shrink: 0; }
  .header-cat { color: var(--muted); height: 15px; width: auto; display: block; flex-shrink: 0; margin-left: 10px; opacity: 0.65; transition: opacity .15s; cursor: pointer; }
  .header-cat:hover { opacity: 1; }

  /* ── About modal ── */
  .about-backdrop {
    position: fixed; inset: 0; background: rgba(0,0,0,0.65);
    display: flex; align-items: center; justify-content: center;
    z-index: 999; opacity: 0; pointer-events: none;
    transition: opacity .18s;
  }
  .about-backdrop.open { opacity: 1; pointer-events: all; }
  .about-modal {
    background: var(--bg1); border: 1px solid var(--border-hi);
    border-radius: 10px; padding: 32px 36px 28px;
    width: 420px; max-width: 92vw;
    box-shadow: 0 24px 64px rgba(0,0,0,0.6);
    transform: translateY(8px); transition: transform .18s;
    position: relative;
  }
  .about-backdrop.open .about-modal { transform: translateY(0); }
  .about-close {
    position: absolute; top: 14px; right: 16px;
    background: none; border: none; color: var(--muted);
    font-size: 18px; cursor: pointer; line-height: 1;
    padding: 2px 6px; border-radius: 4px;
    transition: color .12s, background .12s;
  }
  .about-close:hover { color: var(--text); background: var(--bg3); }
  .about-name {
    font-family: var(--font-mono); font-size: 22px; font-weight: 500;
    color: var(--accent); letter-spacing: .1em; margin-bottom: 4px;
  }
  .about-full { font-size: 11px; color: var(--muted); margin-bottom: 20px; letter-spacing: .02em; }
  .about-version {
    display: inline-block; background: var(--bg3); border: 1px solid var(--border);
    color: var(--muted); font-family: var(--font-mono); font-size: 10px;
    padding: 2px 8px; border-radius: 3px; margin-bottom: 20px;
  }
  .about-divider { border: none; border-top: 1px solid var(--border); margin: 16px 0; }
  .about-section { font-size: 10px; color: var(--muted); line-height: 1.9; }
  .about-section strong { color: var(--text); font-weight: 500; }
  .about-stack {
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px;
  }
  .about-badge {
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--muted); font-family: var(--font-mono); font-size: 9px;
    padding: 3px 8px; border-radius: 3px;
  }
  .about-cat {
    display: block; margin: 18px auto 0;
    color: var(--accent-dim); opacity: 0.45;
  }

  /* ── Dependency manager modal ── */
  .dep-modal-body { margin-top: 8px; }
  .dep-row {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid var(--border);
  }
  .dep-row:last-child { border-bottom: none; }
  .dep-info { flex: 1; min-width: 0; }
  .dep-name { font-size: 12px; font-weight: 500; color: var(--text); }
  .dep-meta { font-size: 10px; color: var(--muted); font-family: var(--font-mono); margin-top: 2px; }
  .dep-status-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
  }
  .dep-status-dot.installed { background: var(--success); }
  .dep-status-dot.missing   { background: var(--border-hi); }
  .dep-action-btn {
    font-family: var(--font-mono); font-size: 10px; font-weight: 500;
    border-radius: 3px; padding: 4px 10px; cursor: pointer;
    border: 1px solid; flex-shrink: 0; transition: all .15s;
  }
  .dep-action-btn.install {
    color: var(--accent); border-color: var(--accent-dim);
    background: transparent;
  }
  .dep-action-btn.install:hover { background: var(--accent-dim); color: #000; }
  .dep-action-btn.remove {
    color: var(--muted); border-color: var(--border);
    background: transparent;
  }
  .dep-action-btn.remove:hover { color: var(--danger); border-color: var(--danger); }
  .dep-action-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .dep-progress-bar {
    height: 3px; background: var(--bg3); border-radius: 2px;
    margin-top: 5px; overflow: hidden; display: none;
  }
  .dep-progress-bar.active { display: block; }
  .dep-progress-fill {
    height: 100%; background: var(--accent);
    border-radius: 2px; transition: width .1s linear; width: 0%;
  }
  .dep-no-platform {
    font-size: 10px; color: var(--muted); font-style: italic;
  }
  .dep-notice {
    margin-top: 14px; padding: 8px 10px; background: var(--bg3);
    border-radius: 4px; font-size: 10px; color: var(--muted); line-height: 1.6;
  }
  .dep-notice strong { color: var(--text); }

  /* ── First-launch dep banner ── */
  .dep-banner {
    position: fixed; bottom: 18px; right: 18px; z-index: 800;
    background: var(--bg2); border: 1px solid var(--border-hi);
    border-radius: 8px; padding: 12px 16px; max-width: 320px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    animation: slide-up .25s ease;
  }
  @keyframes slide-up {
    from { transform: translateY(12px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
  }
  .dep-banner-title { font-size: 11px; font-weight: 500; color: var(--text); margin-bottom: 4px; }
  .dep-banner-body  { font-size: 10px; color: var(--muted); line-height: 1.5; margin-bottom: 10px; }
  .dep-banner-btns  { display: flex; gap: 8px; }
  .dep-banner-btn-open {
    font-family: var(--font-mono); font-size: 10px; font-weight: 500;
    background: var(--accent); color: #000; border: none; border-radius: 3px;
    padding: 4px 10px; cursor: pointer;
  }
  .dep-banner-btn-dismiss {
    font-family: var(--font-mono); font-size: 10px;
    background: none; color: var(--muted); border: 1px solid var(--border);
    border-radius: 3px; padding: 4px 10px; cursor: pointer;
  }

  /* Category nav in the header — sits between the subtitle and the status pill */
  .cat-nav {
    display: flex;
    align-items: stretch;
    gap: 0;
    margin-left: 20px;
    height: 100%;
    flex: 1;          /* takes all available middle space */
    overflow-x: auto; /* graceful on narrow windows */
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

  .lb-btn {
    flex-shrink: 0;
    background: transparent;
    border: 1px solid var(--accent-dim);
    border-radius: 3px;
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    padding: 4px 12px;
    cursor: pointer;
    margin-right: 8px;
    transition: all .15s;
    letter-spacing: .04em;
  }
  .lb-btn:hover { background: var(--accent-dim); color: #000; }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--bg1);
    border-right: 1px solid var(--border);
    display: grid;
    /* files section fixed, tools section fills remaining, footer fixed */
    grid-template-rows: auto 1fr auto;
    overflow: hidden;
  }

  .sb-files {
    padding: 14px;
    display: flex; flex-direction: column; gap: 10px;
    border-bottom: 1px solid var(--border);
    overflow-y: auto;          /* scroll when QM/MM panel is taller than available space */
    max-height: 60vh;          /* never consume more than 60% of viewport height */
    flex-shrink: 0;
  }
  .sb-lbl   { font-family: var(--font-mono); font-size: 9px; font-weight: 500; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 6px; }

  .file-btn {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 3px;
    color: var(--text); font-family: var(--font-sans); font-size: 11px;
    padding: 6px 9px; text-align: left; cursor: pointer;
    display: flex; align-items: center; gap: 7px;
    transition: border-color .15s, background .15s;
  }
  .file-btn:hover    { border-color: var(--border-hi); background: var(--bg3); }
  .file-btn.selected { border-color: var(--accent-dim); }
  .file-lbl { font-size: 9px; color: var(--muted); font-family: var(--font-mono); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .file-lbl.set { color: var(--accent); }

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

  /* ── Tool browser (fills remaining sidebar height) ── */
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

  /* Tool list scrolls within its section — no category dropdown here */
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
  .sb-footer { padding: 10px 14px; border-top: 1px solid var(--border); font-size: 9px; color: var(--muted); line-height: 1.7; flex-shrink: 0; position: relative; overflow: hidden; }
  .sb-cat-bg { position: absolute; right: 12px; top: 0; height: 100%; width: auto; color: #1D9E75; opacity: 0.35; pointer-events: none; }

  /* ── Main panel: tool detail / chart / log ── */
  .main {
    background: var(--bg0);
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto 140px;
    overflow: hidden;
  }

  /* Tool detail bar (name, description, run button, params) */
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
  .tool-cancel-btn {
    flex-shrink: 0; display: none;
    background: transparent; color: var(--danger);
    border: 1px solid var(--danger); border-radius: 3px;
    font-family: var(--font-mono); font-size: 11px; font-weight: 500;
    padding: 6px 12px; cursor: pointer; transition: all .15s;
    opacity: 0.85;
  }
  .tool-cancel-btn.visible { display: inline-flex; align-items: center; gap: 5px; }
  .tool-cancel-btn:hover { background: var(--danger); color: #000; opacity: 1; }

  /* Parameter grid */
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

  /* ── Chart display area ── */
  #chart-display {
    background: var(--bg0);
    overflow: hidden;
    position: relative;
    display: flex;
    flex-direction: column;
  }

  /* Empty / loading state */
  .chart-placeholder {
    flex: 1;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 8px; color: var(--muted);
  }
  .placeholder-icon { font-size: 28px; opacity: .25; }
  .placeholder-name { font-family: var(--font-mono); font-size: 12px; color: var(--muted); }
  .placeholder-hint { font-size: 10px; color: var(--border-hi); }

  /* Welcome panel — shown on first launch */
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

  /* Chart title bar */
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

  /* ── Export bar ── */
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

  /* ── Log ── */
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

  /* No-tool selected state for detail bar */
  .detail-empty { font-size: 11px; color: var(--muted); padding: 4px 0; }
</style>

<div class="shell">

  <!-- Header with category navigation -->
  <header class="header">
    <span class="header-logo">ATMOS</span>
    <span class="header-sep"></span>
    <svg id="header-cat-btn" class="header-cat" viewBox="0 0 70 120" fill="none"
         stroke="currentColor" stroke-width="3"
         stroke-linecap="round" stroke-linejoin="round"
         aria-label="About Atmos" role="button" tabindex="0" title="About Atmos">
      <circle  cx="46" cy="34" r="16"/>
      <ellipse cx="46" cy="64" rx="21" ry="25"/>
      <path d="M33 23L26 4L40 17"/>
      <path d="M59 23L66 4L52 17"/>
      <path d="M25 83C4 99,7 117,30 112"/>
    </svg>
    <nav class="cat-nav" id="cat-nav">
      ${CATEGORIES.map((c, i) => `<button class="cat-tab${i === 0 ? ' active' : ''}" data-cat="${c.id}">${c.label}</button>`).join('')}
    </nav>
    <button class="lb-btn" id="lb-btn" title="Open Ligand Builder — build small molecules, run UFF minimization, generate QM input files">⬡ Ligand Builder</button>
    <div class="status-pill">
      <span class="status-dot idle" id="status-dot"></span>
      <span id="status-text">No files selected</span>
    </div>
  </header>

  <!-- About modal — opened by clicking the cat in the header -->
  <div class="about-backdrop" id="about-backdrop" role="dialog" aria-modal="true" aria-label="About Atmos">
    <div class="about-modal">
      <button class="about-close" id="about-close" aria-label="Close">✕</button>
      <div class="about-name">ATMOS</div>
      <div class="about-full">Analysis, Trajectory, Molecular &amp; Optimization Suite</div>
      <span class="about-version">v0.1.0</span>
      <hr class="about-divider"/>
      <div class="about-section">
        <strong>Built by</strong> Evan J<br>
        <strong>Licence</strong> MIT<br>
        <strong>Platform</strong> Tauri · macOS · Windows · Linux
      </div>
      <hr class="about-divider"/>
      <div class="about-section"><strong>Powered by</strong></div>
      <div class="about-stack">
        <span class="about-badge">rust</span>
        <span class="about-badge">chemfiles</span>
        <span class="about-badge">rayon</span>
        <span class="about-badge">ndarray</span>
        <span class="about-badge">ndarray-linalg</span>
        <span class="about-badge">tauri 2</span>
        <span class="about-badge">NGL viewer</span>
      </div>
      <svg class="about-cat" viewBox="0 0 70 120" fill="none"
           stroke="currentColor" stroke-width="2.5" width="32" height="55"
           stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle  cx="46" cy="34" r="16"/>
        <ellipse cx="46" cy="64" rx="21" ry="25"/>
        <path d="M33 23L26 4L40 17"/>
        <path d="M59 23L66 4L52 17"/>
        <path d="M25 83C4 99,7 117,30 112"/>
      </svg>
      <div style="margin-top:18px;text-align:center">
        <button id="open-dep-manager-btn" style="
          font-family:var(--font-mono);font-size:10px;color:var(--muted);
          background:none;border:1px solid var(--border);border-radius:3px;
          padding:4px 12px;cursor:pointer;">
          Manage Dependencies
        </button>
      </div>
    </div>
  </div>

  <!-- Dependency manager modal -->
  <div class="about-backdrop" id="dep-backdrop" role="dialog" aria-modal="true" aria-label="Manage Dependencies">
    <div class="about-modal" style="width:480px">
      <button class="about-close" id="dep-close" aria-label="Close">✕</button>
      <div class="about-name" style="font-size:16px">Dependencies</div>
      <div class="about-full">Optional external tools downloaded on demand</div>
      <hr class="about-divider"/>
      <div class="dep-modal-body" id="dep-modal-body">
        <div style="color:var(--muted);font-size:11px">Checking…</div>
      </div>
    </div>
  </div>

  <!-- Sidebar -->
  <aside class="sidebar">

    <!-- File / selection / load — MD / QM-MM mode switch -->
    <div class="sb-files">

      <!-- Project bar -->
      <div style="display:flex;gap:5px;margin-bottom:8px;align-items:center;">
        <button class="file-btn" id="load-project-btn" style="flex:1;font-size:9px;padding:5px 8px;"
                title="Open an .atmos project file">
          <span style="opacity:.5">📂</span> Load Project
        </button>
        <button class="file-btn" id="save-project-btn" style="flex:1;font-size:9px;padding:5px 8px;"
                title="Save current session as .atmos project" disabled>
          <span style="opacity:.5">💾</span> Save Project
        </button>
      </div>

      <div style="display:flex;margin-bottom:6px;">
        <button class="mode-tab active" data-mode="md">MD</button>
        <button class="mode-tab"        data-mode="qmm">QM/MM</button>
      </div>

      <!-- MD panel -->
      <div id="panel-md" style="display:flex;flex-direction:column;gap:8px;">
        <div>
          <div class="sb-lbl">Trajectory</div>
          <button class="file-btn" id="pick-traj"><span style="opacity:.5">▶</span> Select trajectory</button>
          <div class="file-lbl" id="traj-label" style="margin-top:4px">No file — .nc .ncrst .xtc .dcd .trr</div>
        </div>
        <div>
          <div class="sb-lbl">Topology <span style="color:#3d4245">(optional)</span></div>
          <button class="file-btn" id="pick-topo"><span style="opacity:.5">◈</span> Select topology</button>
          <div class="file-lbl" id="topo-label" style="margin-top:4px">No file — .pdb</div>
        </div>
        <div>
          <div class="sb-lbl">Atom Selection</div>
          <input class="sel-in" type="text" id="sel" value="name CA" spellcheck="false"/>
          <div style="margin-top:4px;font-size:9px;color:var(--muted)">
            e.g. <code style="color:#5b8dee">name CA</code> &nbsp;
                 <code style="color:#5b8dee">(resid >= 1) and (resid <= 50)</code>
          </div>
        </div>
        <div style="display:flex;gap:6px;align-items:stretch;">
          <button class="btn-primary disabled" id="load-btn" style="flex:1">Load &amp; Index</button>
          <div style="display:flex;flex-direction:column;gap:2px;flex-shrink:0;">
            <div class="sb-lbl" style="font-size:8px;margin-bottom:0">Stride</div>
            <input class="sel-in" type="number" id="stride-input" value="1" min="1" step="1"
                   style="width:52px;padding:4px 6px;font-size:11px;" title="Load every Nth frame (1 = all frames)"/>
          </div>
        </div>
        <button class="btn-primary disabled" id="visualize-btn">Visualize Trajectory</button>
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
          <button class="file-btn" id="pick-qmm-topo"><span style="opacity:.5">◈</span> Select topology</button>
          <div class="file-lbl" id="qmm-topo-label" style="margin-top:4px">No file — .pdb </div>
        </div>
        <div>
          <div class="sb-lbl">CV restraint file <span style="color:#3d4245">(optional — cv.rst)</span></div>
          <button class="file-btn" id="pick-cv-rst"><span style="opacity:.5">◈</span> Select cv.rst</button>
          <div class="file-lbl" id="cv-rst-label" style="margin-top:4px">No file — .rst .cv.rst NMR.def</div>
          <div id="cv-rst-status" style="margin-top:4px;font-size:9px;color:var(--muted);display:none"></div>
        </div>
        <button class="btn-primary" id="load-umbrella-btn">Scan &amp; Load Windows</button>
        <div class="file-lbl" id="umbrella-status">No windows loaded</div>
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
      <span style="color:#3d4245">load once · analyze in parallel</span>
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
        <div class="welcome-tagline">Analysis, Trajectory, Molecular & Optimization Suite · Rust + chemfiles + rayon</div>
        <div class="welcome-steps">
          <div class="welcome-step">
            <div class="ws-num">01 · FILES</div>
            <div class="ws-title">Load trajectory</div>
            <div class="ws-desc">Select a trajectory (.nc, .xtc, .dcd) and an optional topology (.pdb). The topology enables residue names, chains, and full-atom representations.</div>
          </div>
          <div class="welcome-step">
            <div class="ws-num">02 · SELECT</div>
            <div class="ws-title">Define atom selection</div>
            <div class="ws-desc">Enter a chemfiles selection string. The default <code style="color:#5b8dee">name CA</code> selects Cα atoms. All analyses run on this subset — load once, analyze many times.</div>
          </div>
          <div class="welcome-step">
            <div class="ws-num">03 · ANALYZE</div>
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
      <button class="export-btn" id="batch-export-btn" title="Export all cached analyses to a folder">Batch Export</button>
    </div>

    <!-- Log -->
    <div class="log-panel" id="log"></div>

  </main>
</div>`;

  // ── File pickers ──────────────────────────────────────────────────────────

  $('#pick-traj')!.addEventListener('click', async () => {
    const sel = await open({ multiple: false, filters: [
      { name: 'Trajectory / Project', extensions: ['nc', 'ncrst', 'xtc','dcd','trr','atmos'] },
    ]});
    if (sel) {
      const path = sel as string;
      if (path.endsWith('.atmos')) { await openProject(path); return; }
      trajectoryPath = path;
      const name = trajectoryPath.split('/').pop()!;
      $('#traj-label')!.textContent = name;
      $('#traj-label')!.className = 'file-lbl set';
      $('#pick-traj')!.classList.add('selected');
      log(`Trajectory: ${name}`);
      updateLoadState();
    }
  });

  $('#pick-topo')!.addEventListener('click', async () => {
    const sel = await open({ multiple: false, filters: [{ name: 'Topology', extensions: ['pdb'] }] });
    if (sel) {
      topologyPath = sel as string;
      const name = topologyPath.split('/').pop()!;
      $('#topo-label')!.textContent = name;
      $('#topo-label')!.className = 'file-lbl set';
      $('#pick-topo')!.classList.add('selected');
      log(`Topology: ${name}`);
      updateLoadState();
    }
  });

  // ── Load & Index ──────────────────────────────────────────────────────────

  $('#load-btn')!.addEventListener('click', async () => {
    if (!trajectoryPath) return;
    const btn = $('#load-btn') as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Loading…';
    log('Indexing trajectory into memory…');
    try {
      const msg: string = await invoke('load_trajectory', {
        path: trajectoryPath,
        topPath: topologyPath ?? null,
        selectionStr: ($('#sel') as HTMLInputElement).value,
        stride: parseInt(($('#stride-input') as HTMLInputElement).value ?? '1', 10) || 1,
      });
      const match = msg.match(/\d+/);
      if (match) totalFrames = parseInt(match[0]);
      isLoaded = true;
      saveSession();
      // Route the main message at success level; if a ⚠ memory warning is
      // embedded, split it out and emit it separately at warn level so it
      // appears in a distinct colour and isn't buried in the success line.
      const warnIdx = msg.indexOf(' ⚠');
      if (warnIdx !== -1) {
        log(msg.slice(0, warnIdx), 'success');
        log(msg.slice(warnIdx + 1), 'warn');   // strip leading space
      } else {
        log(msg, 'success');
      }
      updateLoadState();
    } catch (e) {
      log(`Load error: ${e}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Load & Index';
    }
  });

  $('#visualize-btn')!.addEventListener('click', () => {
        if (isLoaded) {
            openVisualizer(); // This calls your existing TS function that spawns the WebviewWindow
        } else {
            log('Please load and index the trajectory first.', 'warn');
        }
    log('Opening 3D Visualization…');
    });
 
   $('#lb-btn')!.addEventListener('click', openLigandBuilder);

  // ── Category nav tabs → rebuild tool list ────────────────────────────────

  function switchCategory(catId: string) {
    activeToolId = null;

    // Update tab active state
    $$('.cat-tab').forEach(tab => {
      tab.classList.toggle('active', (tab as HTMLElement).dataset.cat === catId);
    });

    // Update sidebar label
    const cat = CATEGORIES.find(c => c.id === catId);
    const tools = TOOLS.filter(t => t.category === catId);
    $('#tools-section-label')!.textContent = cat?.label ?? '';
    $('#tools-count')!.textContent = `${tools.length} tool${tools.length !== 1 ? 's' : ''}`;

    renderToolList(catId);

    // Reset detail bar (keep chart as-is — don't clobber a visible result)
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

  // ── QM/MM topology picker ────────────────────────────────────────────────
  $('#pick-qmm-topo')?.addEventListener('click', async () => {
    const sel = await open({
      multiple: false,
      filters: [{ name: 'Topology', extensions: ['pdb'] }]
    });
    if (sel) {
      qmmTopoPath = sel as string;
      const name  = qmmTopoPath.split('/').pop()!;
      $('#qmm-topo-label')!.textContent  = name;
      ($('#qmm-topo-label')! as HTMLElement).style.color = 'var(--accent)';
      ($('#pick-qmm-topo')! as HTMLElement).classList.add('selected');
      log(`QM/MM topology: ${name}`);
      try {
        const msg: string = await invoke('set_qmm_topology', { path: qmmTopoPath });
        log(msg, 'success');
      } catch (e) {
        log(`Topology error: ${e}`, 'error');
      }
    }
  });

  // ── cv.rst restraint file picker ──────────────────────────────────────────
  $('#pick-cv-rst')?.addEventListener('click', async () => {
    const sel = await open({
      multiple: false,
      filters: [{ name: 'AMBER restraint', extensions: ['rst', 'def', 'txt', 'cv'] }]
    });
    if (!sel) return;
    cvRstPath = sel as string;
    const name = cvRstPath.split('/').pop()!;
    $('#cv-rst-label')!.textContent = name;
    ($('#cv-rst-label')! as HTMLElement).style.color = 'var(--accent)';
    ($('#pick-cv-rst')! as HTMLElement).classList.add('selected');
    log(`Parsing cv.rst: ${name}`);
    try {
      cvRstBlocks = await invoke<any[]>('parse_cv_rst', { path: cvRstPath });
      const statusEl = $('#cv-rst-status') as HTMLElement;
      statusEl.style.display = 'block';
      statusEl.style.color   = 'var(--success)';
      statusEl.textContent   = `${cvRstBlocks.length} CV block${cvRstBlocks.length !== 1 ? 's' : ''} parsed: ${cvRstBlocks.map(b => b.cv_label).join(', ')}`;
      log(`cv.rst: ${cvRstBlocks.length} blocks — ${cvRstBlocks.map((b: any) => b.cv_label).join(', ')}`, 'success');
    } catch (e) {
      log(`cv.rst parse failed: ${e}`, 'error');
      cvRstBlocks = [];
    }
  });
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
        cvPattern, nWindows, valMin, valMax, cvCol, rstPattern: rstPat,
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

  // ── Batch Export ───────────────────────────────────────────────────────────
  // Opens a folder picker, exports every cached analyses to CSV files there,
  // then also triggers SVG downloads for the currently-visible chart.
  $('#batch-export-btn')!.addEventListener('click', async () => {
    const { open: openDir } = await import('@tauri-apps/plugin-dialog');
    const dir = await openDir({ directory: true, title: 'Choose export folder' });
    if (!dir) return;

    const btn = $('#batch-export-btn') as HTMLButtonElement;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Exporting…';
    log(`Batch exporting to ${dir}…`);

    try {
      const written: string[] = await invoke('batch_export', { dir });

      // Also export the currently-visible SVG chart(s)
      const svgs = document.querySelectorAll<SVGSVGElement>('#chart-display svg');
      if (svgs.length > 0) {
        const { writeTextFile } = await import('@tauri-apps/plugin-fs');
        const svgBlobs: string[] = [];
        svgs.forEach((svg) => {
          svgBlobs.push(new XMLSerializer().serializeToString(svg));
        });
        const combined = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg">\n`
          + svgBlobs.join('\n') + `\n</svg>`;
        const svgPath = `${dir}/chart.svg`;
        await writeTextFile(svgPath, combined);
        written.push('chart.svg');
      }

      log(`Batch export complete: ${written.length} files — ${written.join(', ')}`, 'success');
    } catch (e) {
      log(`Batch export failed: ${e}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Batch Export';
    }
  });

  // ── Dependency manager (conda) ────────────────────────────────────────────

  type ToolAvailability = {
    id: string; name: string; bin: string;
    installed: boolean; path: string | null;
  };
  type CondaEnvStatus = {
    conda_available: boolean;
    conda_path: string | null;
    env_exists: boolean;
    tools: ToolAvailability[];
  };

  async function renderDepModal(): Promise<void> {
    const body = document.getElementById('dep-modal-body')!;
    body.innerHTML = '<div style="color:var(--muted);font-size:11px">Checking environment…</div>';

    let status: CondaEnvStatus;
    try {
      status = await invoke<CondaEnvStatus>('check_conda_env');
    } catch (e) {
      body.innerHTML = `<div style="color:var(--danger);font-size:11px">Check failed: ${e}</div>`;
      return;
    }

    if (!status.conda_available) {
      body.innerHTML = `
        <div style="color:var(--warn);font-size:11px;line-height:1.7;margin-bottom:14px">
          <strong style="color:var(--text)">conda / mamba not found.</strong><br>
          Atmos uses conda to install xtb, Open Babel, and AutoDock Vina.<br>
          Install <strong>Miniforge</strong> (recommended) then restart Atmos.
        </div>
        <a href="#" id="open-miniforge-link" style="
          font-family:var(--font-mono);font-size:10px;color:var(--accent);
          text-decoration:none;border-bottom:1px solid var(--accent-dim);
          padding-bottom:1px;">
          miniforge.github.io →
        </a>
        <div style="margin-top:14px;font-size:10px;color:var(--muted)">
          Already have conda? Make sure it is on your PATH and restart Atmos.<br>
          All features work without these tools — built-in UFF minimisation is always available.
        </div>`;
      document.getElementById('open-miniforge-link')?.addEventListener('click', async (e) => {
        e.preventDefault();
        const { openUrl } = await import('@tauri-apps/plugin-opener');
        await openUrl('https://github.com/conda-forge/miniforge#miniforge');
      });
      return;
    }

    // Conda available — show tool status table + install/remove buttons
    const allInstalled = status.tools.every(t => t.installed);
    const tableRows = status.tools.map(t => `
      <div class="dep-row" data-tool="${t.id}">
        <span class="dep-status-dot ${t.installed ? 'installed' : 'missing'}"></span>
        <div class="dep-info">
          <div class="dep-name">${t.name}</div>
          <div class="dep-meta">${t.installed
            ? `<span style="color:var(--success)">installed</span>${t.path ? '  ·  ' + t.path : ''}`
            : '<span style="color:var(--muted)">not installed</span>'}</div>
        </div>
      </div>`).join('');

    body.innerHTML = `
      <div style="font-size:10px;color:var(--muted);margin-bottom:10px;font-family:var(--font-mono)">
        conda: ${status.conda_path ?? 'found'}
        &nbsp;·&nbsp; env: <span style="color:${status.env_exists ? 'var(--success)' : 'var(--muted)'}">
          ${status.env_exists ? 'atmos-env ✓' : 'not created'}
        </span>
      </div>
      ${tableRows}
      <div style="margin-top:14px;display:flex;gap:8px;flex-wrap:wrap">
        ${allInstalled
          ? `<button class="dep-action-btn remove" id="dep-remove-env-btn">Remove atmos-env</button>`
          : `<button class="dep-action-btn install" id="dep-install-env-btn">
               ${status.env_exists ? 'Update' : 'Install'} all tools via conda
             </button>`
        }
      </div>
      <div class="dep-progress-bar" id="dep-install-bar">
        <div class="dep-progress-fill" id="dep-install-fill"></div>
      </div>
      <pre id="dep-log" style="
        margin-top:10px;background:var(--bg0);border:1px solid var(--border);
        border-radius:4px;padding:8px 10px;font-size:9px;font-family:var(--font-mono);
        color:var(--muted);max-height:160px;overflow-y:auto;display:none;
        white-space:pre-wrap;word-break:break-all;"></pre>`;

    // Install button
    document.getElementById('dep-install-env-btn')?.addEventListener('click', async () => {
      const btn    = document.getElementById('dep-install-env-btn') as HTMLButtonElement;
      const bar    = document.getElementById('dep-install-bar')!;
      const fill   = document.getElementById('dep-install-fill')!;
      const logEl  = document.getElementById('dep-log')!;

      btn.disabled = true;
      btn.textContent = 'Installing…';
      bar.classList.add('active');
      logEl.style.display = 'block';
      logEl.textContent   = '';

      // Listen for conda output lines
      const { listen } = await import('@tauri-apps/api/event');
      let lineCount = 0;
      const unlisten = await listen<string>('dep:output', (ev) => {
        logEl.textContent += ev.payload + '\n';
        logEl.scrollTop = logEl.scrollHeight;
        lineCount++;
        // Fake progress: conda doesn't report %; use line count as proxy
        fill.style.width = `${Math.min(95, lineCount * 2)}%`;
      });

      try {
        await invoke('install_conda_env');
        fill.style.width = '100%';
        log('atmos-env installed — xtb, Open Babel, and Vina are ready.', 'success');
        setTimeout(() => renderDepModal(), 600);
      } catch (e) {
        btn.disabled   = false;
        btn.textContent = 'Retry';
        bar.classList.remove('active');
        log(`conda install failed: ${e}`, 'error');
      } finally {
        unlisten();
      }
    });

    // Remove button
    document.getElementById('dep-remove-env-btn')?.addEventListener('click', async () => {
      const btn = document.getElementById('dep-remove-env-btn') as HTMLButtonElement;
      btn.disabled = true;
      btn.textContent = 'Removing…';
      try {
        await invoke('remove_conda_env');
        log('atmos-env removed.', 'warn');
        renderDepModal();
      } catch (e) {
        btn.disabled = false;
        btn.textContent = 'Remove atmos-env';
        log(`Remove failed: ${e}`, 'error');
      }
    });
  }

  const depBackdrop = document.getElementById('dep-backdrop')!;
  const openDepManager = async () => {
    depBackdrop.classList.add('open');
    await renderDepModal();
  };
  const closeDepManager = () => depBackdrop.classList.remove('open');

  document.getElementById('open-dep-manager-btn')!.addEventListener('click', openDepManager);
  document.getElementById('dep-close')!.addEventListener('click', closeDepManager);
  depBackdrop.addEventListener('click', (e) => { if (e.target === depBackdrop) closeDepManager(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && depBackdrop.classList.contains('open')) closeDepManager();
  });

  // First-launch banner: shown once if conda is available but atmos-env is missing
  const BANNER_KEY = 'atmos-dep-banner-dismissed-v2';
  if (!localStorage.getItem(BANNER_KEY)) {
    try {
      const s = await invoke<CondaEnvStatus>('check_conda_env');
      if (s.conda_available && !s.env_exists) {
        const banner = document.createElement('div');
        banner.className = 'dep-banner';
        banner.innerHTML = `
          <div class="dep-banner-title">Optional tools available</div>
          <div class="dep-banner-body">
            conda found. Install xtb, Open Babel, and AutoDock Vina into
            a managed environment for QM calculations and docking.
          </div>
          <div class="dep-banner-btns">
            <button class="dep-banner-btn-open">Set up</button>
            <button class="dep-banner-btn-dismiss">Later</button>
          </div>`;
        document.body.appendChild(banner);
        banner.querySelector('.dep-banner-btn-open')!.addEventListener('click', () => {
          banner.remove(); localStorage.setItem(BANNER_KEY, '1'); openDepManager();
        });
        banner.querySelector('.dep-banner-btn-dismiss')!.addEventListener('click', () => {
          banner.remove(); localStorage.setItem(BANNER_KEY, '1');
        });
      }
    } catch (_) {}
  }

  // ── About modal ───────────────────────────────────────────────────────────
  const aboutBackdrop = document.getElementById('about-backdrop')!;
  const openAbout  = () => aboutBackdrop.classList.add('open');
  const closeAbout = () => aboutBackdrop.classList.remove('open');

  document.getElementById('header-cat-btn')!.addEventListener('click', openAbout);
  document.getElementById('header-cat-btn')!.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openAbout(); }
  });
  document.getElementById('about-close')!.addEventListener('click', closeAbout);
  aboutBackdrop.addEventListener('click', (e) => {
    if (e.target === aboutBackdrop) closeAbout();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && aboutBackdrop.classList.contains('open')) closeAbout();
  });

  // ── Session persistence ───────────────────────────────────────────────────
  const SESSION_KEY = 'atmos-session-v1';

  function saveSession() {
    try {
      localStorage.setItem(SESSION_KEY, JSON.stringify({
        trajectoryPath,
        topologyPath,
        selection: ($('#sel') as HTMLInputElement)?.value ?? 'name CA',
        stride:    ($('#stride-input') as HTMLInputElement)?.value ?? '1',
      }));
    } catch (_) {}
  }

  function restoreSession() {
    try {
      const raw = localStorage.getItem(SESSION_KEY);
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.trajectoryPath) {
        trajectoryPath = s.trajectoryPath;
        const name = trajectoryPath!.split(/[\\/]/).pop()!;
        $('#traj-label')!.textContent = name;
        $('#traj-label')!.className = 'file-lbl set';
        $('#pick-traj')!.classList.add('selected');
      }
      if (s.topologyPath) {
        topologyPath = s.topologyPath;
        const name = topologyPath!.split(/[\\/]/).pop()!;
        $('#topo-label')!.textContent = name;
        $('#topo-label')!.className = 'file-lbl set';
        $('#pick-topo')!.classList.add('selected');
      }
      if (s.selection) ($('#sel') as HTMLInputElement).value = s.selection;
      if (s.stride)    ($('#stride-input') as HTMLInputElement).value = s.stride;
      if (s.trajectoryPath) updateLoadState();
    } catch (_) {}
  }

  // Save on every meaningful change
  $('#pick-traj')!.addEventListener('click', () => setTimeout(saveSession, 200));
  $('#pick-topo')!.addEventListener('click', () => setTimeout(saveSession, 200));
  $('#sel')!.addEventListener('change', saveSession);
  $('#stride-input')!.addEventListener('change', saveSession);

  restoreSession();

  // ── Project file (.atmos) ──────────────────────────────────────────────────

  type ProjectMeta = {
    traj_path:      string | null;
    topo_path:      string | null;
    selection:      string;
    stride:         number;
    n_frames:       number;
    atmos_version:  string;
    traj_embedded:  boolean;
    topo_embedded:  boolean;
    analyses_saved: string[];
  };

  async function applyProjectMeta(meta: ProjectMeta) {
    if (meta.traj_path) {
      trajectoryPath = meta.traj_path;
      const name = trajectoryPath.split(/[\\/]/).pop()!;
      $('#traj-label')!.textContent = meta.traj_embedded ? `${name} (embedded)` : name;
      $('#traj-label')!.className = 'file-lbl set';
      $('#pick-traj')!.classList.add('selected');
    }
    if (meta.topo_path) {
      topologyPath = meta.topo_path;
      const name = topologyPath.split(/[\\/]/).pop()!;
      $('#topo-label')!.textContent = meta.topo_embedded ? `${name} (embedded)` : name;
      $('#topo-label')!.className = 'file-lbl set';
      $('#pick-topo')!.classList.add('selected');
    }
    ($('#sel') as HTMLInputElement).value = meta.selection;
    ($('#stride-input') as HTMLInputElement).value = String(meta.stride);

    if (meta.traj_path) {
      const btn = $('#load-btn') as HTMLButtonElement;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Loading…';
      try {
        const msg: string = await invoke('load_trajectory', {
          path:            meta.traj_path,
          topPath:         meta.topo_path ?? null,
          selectionStr:    meta.selection,
          stride:          meta.stride,
          preserveCaches:  true,   // keep caches restored by load_project
        });
        totalFrames = meta.n_frames || parseInt(msg.match(/\d+/)?.[0] ?? '0');
        isLoaded = true;
        log(`Project loaded — ${meta.analyses_saved.length} cached analyses restored`, 'success');
        if (meta.analyses_saved.length > 0)
          log(`Cached: ${meta.analyses_saved.join(', ')}`, 'info' as any);
        updateLoadState();
        saveSession();
      } catch (e) {
        log(`Project load error: ${e}`, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Load & Index';
      }
    }
  }

  async function openProject(path: string) {
    log(`Opening project: ${path.split(/[\\/]/).pop()}…`);
    try {
      const meta: ProjectMeta = await invoke('load_project', { srcPath: path });
      await applyProjectMeta(meta);
    } catch (e) {
      log(`Failed to open project: ${e}`, 'error');
    }
  }

  async function saveProjectDialog() {
    const cached: string[] = await invoke('get_cached_analyses');
    const hasTraj = !!trajectoryPath;
    const hasTopo = !!topologyPath;

    const result = await new Promise<{ embed_traj: boolean; embed_topo: boolean; cancelled: boolean }>((resolve) => {
      const overlay = document.createElement('div');
      overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:1000;display:flex;align-items:center;justify-content:center;';
      overlay.innerHTML = `
        <div style="background:var(--bg1);border:1px solid var(--border-hi);border-radius:10px;padding:28px 32px;width:380px;max-width:92vw;box-shadow:0 24px 64px rgba(0,0,0,.6);">
          <div style="font-family:var(--font-mono);font-size:14px;font-weight:500;color:var(--accent);margin-bottom:8px;">Save Project</div>
          <div style="font-size:10px;color:var(--muted);margin-bottom:16px;">
            ${cached.length} cached ${cached.length === 1 ? 'analysis' : 'analyses'} will be saved.
            Embedding files makes the project fully portable.
          </div>
          ${hasTraj ? `
          <label style="display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text);margin-bottom:10px;cursor:pointer;">
            <input type="checkbox" id="embed-traj-cb" style="accent-color:var(--accent)"/>
            Embed trajectory <span style="color:var(--muted);font-size:10px;">(increases file size)</span>
          </label>` : ''}
          ${hasTopo ? `
          <label style="display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text);margin-bottom:16px;cursor:pointer;">
            <input type="checkbox" id="embed-topo-cb" checked style="accent-color:var(--accent)"/>
            Embed topology
          </label>` : '<div style="margin-bottom:16px;"></div>'}
          <div style="display:flex;gap:8px;justify-content:flex-end;">
            <button id="proj-cancel" style="font-family:var(--font-mono);font-size:10px;background:none;color:var(--muted);border:1px solid var(--border);border-radius:3px;padding:5px 12px;cursor:pointer;">Cancel</button>
            <button id="proj-save"   style="font-family:var(--font-mono);font-size:10px;background:var(--accent);color:#000;border:none;border-radius:3px;padding:5px 12px;cursor:pointer;font-weight:500;">Choose location…</button>
          </div>
        </div>`;
      document.body.appendChild(overlay);
      overlay.querySelector('#proj-cancel')!.addEventListener('click', () => {
        overlay.remove(); resolve({ embed_traj: false, embed_topo: false, cancelled: true });
      });
      overlay.querySelector('#proj-save')!.addEventListener('click', () => {
        const et = (overlay.querySelector('#embed-traj-cb') as HTMLInputElement)?.checked ?? false;
        const ep = (overlay.querySelector('#embed-topo-cb') as HTMLInputElement)?.checked ?? false;
        overlay.remove(); resolve({ embed_traj: et, embed_topo: ep, cancelled: false });
      });
    });

    if (result.cancelled) return;

    const destPath = await save({
      defaultPath: 'project.atmos',
      filters: [{ name: 'Atmos Project', extensions: ['atmos'] }],
    });
    if (!destPath) return;

    try {
      await invoke('save_project', {
        destPath,
        embedTraj: result.embed_traj,
        embedTopo: result.embed_topo,
        trajPath:  trajectoryPath ?? null,
        topoPath:  topologyPath  ?? null,
        selection: ($('#sel') as HTMLInputElement).value,
        stride:    parseInt(($('#stride-input') as HTMLInputElement).value ?? '1', 10) || 1,
      });
      log(`Project saved → ${(destPath as string).split(/[\\/]/).pop()}`, 'success');
    } catch (e) {
      log(`Save failed: ${e}`, 'error');
    }
  }

  $('#load-project-btn')!.addEventListener('click', async () => {
    const sel = await open({ multiple: false, filters: [{ name: 'Atmos Project', extensions: ['atmos'] }] });
    if (sel) await openProject(sel as string);
  });

  $('#save-project-btn')!.addEventListener('click', saveProjectDialog);

  // ── File association / drag-drop handler ──────────────────────────────────
  // Receives paths from: OS file association (via Rust setup()/RunEvent::Opened),
  // and from window drag-drop.
  const { listen } = await import('@tauri-apps/api/event');

  const handleOpenPath = (path: string) => {
    const ext = path.split('.').pop()?.toLowerCase() ?? '';
    if (ext === 'atmos') { openProject(path); return; }
    const trajExts = ['nc', 'ncrst', 'xtc', 'trr', 'dcd'];
    const topoExts = ['pdb'];
    if (trajExts.includes(ext)) {
      trajectoryPath = path;
      const name = path.split(/[\\/]/).pop()!;
      $('#traj-label')!.textContent = name;
      $('#traj-label')!.className = 'file-lbl set';
      $('#pick-traj')!.classList.add('selected');
      log(`Trajectory: ${name}`);
      updateLoadState();
    } else if (topoExts.includes(ext)) {
      topologyPath = path;
      const name = path.split(/[\\/]/).pop()!;
      $('#topo-label')!.textContent = name;
      $('#topo-label')!.className = 'file-lbl set';
      $('#pick-topo')!.classList.add('selected');
      log(`Topology: ${name}`);
      updateLoadState();
    } else {
      log(`Unrecognised file type: ${path.split(/[\\/]/).pop()}`, 'warn');
    }
  };

  // From Rust setup() / RunEvent::Opened
  await listen('atmos://open-file', (event) => {
    handleOpenPath(event.payload as string);
  });

  // From window drag-drop (Tauri v2 event name)
  await listen('tauri://drag-drop', (event: any) => {
    const paths: string[] = event.payload?.paths ?? event.payload ?? [];
    paths.forEach(handleOpenPath);
  });

  // ── Initial render ────────────────────────────────────────────────────────
  switchCategory(CATEGORIES[0].id);
  log('ATMOS ready.', 'success');
  updateLoadState();
}

window.addEventListener('DOMContentLoaded', initUI);