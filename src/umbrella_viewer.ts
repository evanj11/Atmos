import { invoke } from '@tauri-apps/api/core';

// ─── NGL stubs ────────────────────────────────────────────────────────────────
declare const NGL: {
  Stage: new (id: string, params?: Record<string, unknown>) => NGLStage;
};
interface NGLStage {
  loadFile(blob: Blob, params: Record<string, unknown>): Promise<NGLComponent>;
  removeComponent(comp: NGLComponent): void;
  autoView(): void;
  handleResize(): void;
  viewer: { requestRender(): void };
  signals: { clicked: { add(fn: (pd: any) => void): void } };
}
interface NGLComponent {
  removeAllRepresentations(): void;
  addRepresentation(type: string, params?: Record<string, unknown>): void;
  updateRepresentations(params: Record<string, unknown>): void;
  autoView(sel?: string, ms?: number): void;
  structure: { updatePosition(data: Float32Array): void };
}

// ─── State ────────────────────────────────────────────────────────────────────
let stage:         NGLStage | null = null;
let comp:          NGLComponent | null = null;
let currentMode:   string  = 'cartoon';
let currentColor:  string  = 'resname';
let totalWindows:  number  = 0;
let currentWindow: number  = 0;
let isPlaying:     boolean = false;
let playInterval:  number | null = null;

// ─── DOM helpers ─────────────────────────────────────────────────────────────
const el = (id: string) => document.getElementById(id)!;

function setStatus(msg: string, hide = false) {
  const ov = el('status-overlay');
  ov.textContent = msg;
  ov.classList.toggle('hidden', hide);
}

function setProgress(pct: number) {
  const bar = el('progress-bar') as HTMLElement;
  bar.style.width = `${pct.toFixed(0)}%`;
  bar.style.display = pct >= 100 ? 'none' : 'block';
}

function updateWindowLabel() {
  el('window-label').textContent = `Window ${currentWindow} / ${totalWindows - 1}`;
  (el('window-slider') as HTMLInputElement).value = String(currentWindow);
}

// ─── Representation applier ───────────────────────────────────────────────────
function applyReps(): void {
  if (!comp) return;
  comp.removeAllRepresentations();
  comp.addRepresentation(currentMode, {
    sele: 'protein', colorScheme: currentColor, opacity: 0.85,
  });
  comp.addRepresentation('licorice', {
    sele: 'not (protein or water or ion)', colorScheme: 'element',
    radius: 0.15, multipleBond: 'symmetric',
  });
  stage?.viewer.requestRender();
}

// ─── NGL ready helper — used only on the initial load ────────────────────────
function applyAfterNglReady(applyFn: () => void): void {
  setTimeout(() => {
    if (!comp) return;
    comp.removeAllRepresentations();
    setTimeout(() => {
      if (!comp) return;
      comp.removeAllRepresentations();
      applyFn();
      stage?.viewer.requestRender();
    }, 150);
  }, 0);
}

// ─── Frame scrubbing — MD-style coordinate swap ───────────────────────────────
// This is the key to smooth playback. Instead of re-loading the PDB file for
// each window (which triggers NGL's bond inference pipeline every time), we:
//   1. Load window 0 as a PDB once to establish the NGL component + topology
//   2. For every subsequent window, just swap the flat coordinate array using
//      structure.updatePosition() and call updateRepresentations({position:true})
//
// This is exactly how the MD trajectory visualizer works. The result is
// essentially zero per-frame overhead — just a Float32Array copy and a render.

async function goToWindow(idx: number): Promise<void> {
  if (!comp || idx < 0 || idx >= totalWindows) return;
  currentWindow = idx;
  updateWindowLabel();

  try {
    const flat = await invoke<number[]>('get_umbrella_window_coords', { windowIdx: idx });
    comp.structure.updatePosition(new Float32Array(flat));
    comp.updateRepresentations({ position: true });
    stage?.viewer.requestRender();
  } catch (err) {
    console.error('get_umbrella_window_coords failed:', err);
  }
}

// ─── Playback ─────────────────────────────────────────────────────────────────
function startPlay(): void {
  if (isPlaying) return;
  isPlaying = true;
  el('play-btn').textContent = '⏹ Stop';
  el('play-btn').classList.add('playing');
  const fps = Math.max(0.5, Math.min(60,
    parseFloat((el('fps-input') as HTMLInputElement).value || '5')));
  playInterval = window.setInterval(() => {
    goToWindow((currentWindow + 1) % totalWindows);
  }, 1000 / fps);
}

function stopPlay(): void {
  if (!isPlaying) return;
  isPlaying = false;
  if (playInterval !== null) { clearInterval(playInterval); playInterval = null; }
  el('play-btn').textContent = '▶ Play';
  el('play-btn').classList.remove('playing');
}

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  // @ts-ignore
  stage = new NGL.Stage('viewport', { backgroundColor: '#060708' });

  try {
    totalWindows = await invoke<number>('get_umbrella_window_count');
  } catch (e) {
    setStatus(`Failed to get window count: ${e}`); return;
  }
  if (totalWindows === 0) {
    setStatus('No umbrella windows loaded. Use the QM/MM sidebar to scan & load windows first.');
    return;
  }

  const slider = el('window-slider') as HTMLInputElement;
  slider.max   = String(totalWindows - 1);
  slider.value = '0';

  // ── Step 1: Pre-load all window coordinates into Rust cache ──────────────
  // Reads each ncrst sequentially on a Rust blocking thread, caches flat f32
  // coords. Subsequent goToWindow calls are zero-cost cache lookups.
  setStatus(`Pre-loading ${totalWindows} window snapshots…`);
  setProgress(0);
  try {
    const msg = await invoke<string>('preload_umbrella_coords');
    setStatus(msg + ' Loading structure…');
    setProgress(100);
  } catch (e) {
    setStatus(`Pre-load failed: ${e}`);
    return;
  }

  // ── Step 2: Load window 0 as PDB to establish the NGL component ───────────
  // This is the only full PDB load — topology, bonds, and atom names are
  // established here. All subsequent windows reuse this component.
  let pdb0: string;
  try {
    pdb0 = await invoke<string>('get_umbrella_snapshot_pdb', { windowIdx: 0 });
  } catch (e) {
    setStatus(`Failed to load window 0: ${e}`); return;
  }

  const blob = new Blob([pdb0], { type: 'text/plain' });
  comp = await stage.loadFile(blob, { ext: 'pdb', firstModelOnly: true });

  applyAfterNglReady(() => {
    applyReps();
    comp!.autoView('protein', 0);
    setStatus('', true);
    updateWindowLabel();

    // Wire controls only after structure is loaded
    wireControls(slider);
  });
}

function wireControls(slider: HTMLInputElement): void {
  // Slider
  slider.addEventListener('input', () => {
    stopPlay();
    goToWindow(parseInt(slider.value, 10));
  });

  // Play / stop
  el('play-btn').addEventListener('click', () => {
    isPlaying ? stopPlay() : startPlay();
  });

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if      (e.key === 'ArrowRight' || e.key === 'ArrowUp')  { stopPlay(); goToWindow(currentWindow + 1); }
    else if (e.key === 'ArrowLeft'  || e.key === 'ArrowDown') { stopPlay(); goToWindow(currentWindow - 1); }
    else if (e.key === ' ') { e.preventDefault(); isPlaying ? stopPlay() : startPlay(); }
  });

  // Style buttons
  el('style-group').addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('[data-mode]') as HTMLButtonElement | null;
    if (!btn) return;
    currentMode = btn.dataset.mode!;
    document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyReps();
  });

  // Color dropdown
  el('color-scheme').addEventListener('change', (e) => {
    currentColor = (e.target as HTMLSelectElement).value;
    applyReps();
  });

  // NGL atom click
  stage!.signals.clicked.add((pd: any) => {
    if (!pd?.atom) { el('atom-info').textContent = 'Click a residue for details'; return; }
    const a = pd.atom;
    el('atom-info').textContent = `${a.resname} ${a.resno}  ·  ${a.atomname}  ·  chain ${a.chainname}`;
  });

  window.addEventListener('resize', () => stage?.handleResize());
}

window.addEventListener('DOMContentLoaded', init);
