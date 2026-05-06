import { invoke } from './api';
import { listen } from './events';

// ─── NGL type stubs ───────────────────────────────────────────────────────────

declare const NGL: {
  Stage: new (elementId: string, params?: Record<string, unknown>) => NGLStage;
};
interface NGLStage {
  loadFile(blob: Blob, params: Record<string, unknown>): Promise<NGLComponent>;
  autoView(): void;
  handleResize(): void;
  viewer: { requestRender(): void };
}
interface NGLComponent {
  removeAllRepresentations(): void;
  addRepresentation(type: string, params?: Record<string, unknown>): void;
  updateRepresentations(params: Record<string, unknown>): void;
  structure: { updatePosition(data: Float32Array): void };
  autoView(selection?: string, duration?: number): void;
}

// ─── State ────────────────────────────────────────────────────────────────────

let stage:        NGLStage | null     = null;
let comp:         NGLComponent | null = null;
let currentMode:  string              = 'cartoon';
let currentColor: string              = 'resname';
let totalFrames:  number              = 0;
let currentFrame: number              = 0;
let isPlaying:    boolean             = false;
let playInterval: number | null       = null;

// ─── DOM helpers ─────────────────────────────────────────────────────────────

function setStatus(msg: string, hide = false) {
  const el = document.getElementById('status-overlay')!;
  el.textContent = msg;
  el.classList.toggle('hidden', hide);
}

function setHighlightLabel(text: string) {
  const el = document.getElementById('highlight-label')!;
  el.textContent = text;
  el.style.opacity = text ? '1' : '0';
}

// ─── Representation ───────────────────────────────────────────────────────────

function applyRepresentation() {
  if (!comp) return;
  comp.removeAllRepresentations();

  // Protein / nucleic polymer
  comp.addRepresentation(currentMode, {
    sele:        'polymer',
    colorScheme: currentColor,
  });

  // Ligand / heteroatoms — licorice uses only the bond table (CONECT records
  // or NGL's residue dictionary), never distance inference. This prevents
  // the spurious cross-residue bonds you get with ball+stick on dense systems.
  comp.addRepresentation('licorice', {
    sele:        'not (polymer or water or ion)',
    colorScheme: 'element',
    radius:      0.15,
    multipleBond: 'symmetric',
  });

  // Ions — simple spheres, no bonds
  comp.addRepresentation('spacefill', {
    sele:        'ion',
    colorScheme: 'element',
    radius:      0.5,
  });

  stage?.viewer.requestRender();
}

// ─── Highlight helpers ────────────────────────────────────────────────────────
//
// NGL resno is 1-based and matches the residue sequence number from topology.
// Our analysis arrays are 0-based atom indices, so resno = atomIndex + 1.

function highlightResidue(atomIndex: number) {
  if (!comp) return;
  // Remove old highlight by reapplying the main representation cleanly
  comp.removeAllRepresentations();
  comp.addRepresentation(currentMode, {
    sele: 'polymer', colorScheme: currentColor,
  });
  comp.addRepresentation('licorice', {
    sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15,
  });
  // Add highlight on top
  const resno = atomIndex + 1;
  comp.addRepresentation('ball+stick', {
    sele:        `resno ${resno}`,
    colorScheme: 'uniform',
    color:       '#ffdd00',
    radius:      0.6,
  });
  comp.autoView(`resno ${resno}`, 600);
  stage?.viewer.requestRender();
  setHighlightLabel(`Residue ${resno} (index ${atomIndex})`);
}

function highlightPair(i: number, j: number) {
  if (!comp) return;
  comp.removeAllRepresentations();
  comp.addRepresentation(currentMode, {
    sele: 'polymer', colorScheme: currentColor,
  });
  comp.addRepresentation('licorice', {
    sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15,
  });
  comp.addRepresentation('ball+stick', {
    sele: `resno ${i + 1}`, colorScheme: 'uniform', color: '#ffdd00', radius: 0.6,
  });
  comp.addRepresentation('ball+stick', {
    sele: `resno ${j + 1}`, colorScheme: 'uniform', color: '#00c4a7', radius: 0.6,
  });
  comp.autoView(`resno ${i + 1} or resno ${j + 1}`, 600);
  stage?.viewer.requestRender();
  setHighlightLabel(`Residues ${i + 1} & ${j + 1}`);
}

function clearHighlight() {
  applyRepresentation();
  setHighlightLabel('');
}

// ─── B-factor reload ──────────────────────────────────────────────────────────
//
// Fetches a fresh PDB snapshot from Rust (which now has the bfactor column
// filled in) and replaces the current NGL component. The color scheme is
// automatically switched to 'bfactor' so the mapping is immediately visible.

async function reloadWithBfactors() {
  if (!stage) return;
  setStatus('Updating B-factor mapping…');
  try {
    const pdbString = await invoke<string>('get_snapshot_pdb', { frame_idx: currentFrame });
    const blob = new Blob([pdbString], { type: 'text/plain' });

    // Dispose the old component properly before loading the new one
    if (comp) { comp.removeAllRepresentations(); }
    comp = await stage.loadFile(blob, { ext: 'pdb', firstModelOnly: true });

    currentColor = 'bfactor';
    comp.addRepresentation(currentMode, {
      sele: 'polymer', colorScheme: 'bfactor',
    });
    comp.autoView();

    // Sync the color dropdown
    const sel = document.getElementById('color-scheme') as HTMLSelectElement | null;
    if (sel) sel.value = 'bfactor';

    const indicator = document.getElementById('bfactor-indicator');
    if (indicator) indicator.style.display = 'flex';

    setStatus('', true);
    stage.viewer.requestRender();
  } catch (err) {
    setStatus(`B-factor reload failed: ${err}`);
    console.error(err);
  }
}

// ─── Frame scrubbing ──────────────────────────────────────────────────────────

async function goToFrame(idx: number) {
  if (!comp || !stage) return;
  currentFrame = Math.max(0, Math.min(idx, totalFrames - 1));

  const label  = document.getElementById('frame-label')!;
  const slider = document.getElementById('frame-slider') as HTMLInputElement;
  label.textContent = `Frame ${currentFrame} / ${totalFrames - 1}`;
  slider.value      = String(currentFrame);

  try {
    const flat = await invoke<number[]>('get_frame_coords', { frame_idx: currentFrame });
    comp.structure.updatePosition(new Float32Array(flat));
    comp.updateRepresentations({ position: true });
    stage.viewer.requestRender();
  } catch (err) {
    console.error('get_frame_coords failed:', err);
  }
}

// ─── Playback ─────────────────────────────────────────────────────────────────

function startPlayback() {
  if (isPlaying) return;
  isPlaying = true;
  const playBtn  = document.getElementById('play-btn')!;
  const fpsInput = document.getElementById('fps-input') as HTMLInputElement;
  playBtn.textContent = '⏹ Stop';
  playBtn.classList.add('playing');
  const fps = Math.max(1, Math.min(120, parseInt(fpsInput.value || '25', 10)));
  playInterval = window.setInterval(
    () => goToFrame((currentFrame + 1) % totalFrames),
    1000 / fps
  );
}

function stopPlayback() {
  if (!isPlaying) return;
  isPlaying = false;
  if (playInterval !== null) { clearInterval(playInterval); playInterval = null; }
  const playBtn = document.getElementById('play-btn')!;
  playBtn.textContent = '▶ Play';
  playBtn.classList.remove('playing');
}

// ─── Main ─────────────────────────────────────────────────────────────────────
//
// CRITICAL ORDER:
//   1. Register all listen() calls FIRST — before any async structure loading.
//      Events emitted while the structure is loading would otherwise be dropped.
//   2. Then load the structure.
//   3. Then wire up UI controls.
//
// listen() returns a Promise<UnlistenFn> but it resolves almost instantly
// (it just registers a callback in the WebSocket handler). We don't need to
// await it in sequence with the structure load — fire all registrations
// immediately and let them resolve in parallel.

async function initVisualizer() {
  const urlParams = new URLSearchParams(window.location.search);
  totalFrames     = parseInt(urlParams.get('frames') || '0', 10);
  if (totalFrames === 0) {
    setStatus('No trajectory loaded — open a trajectory first.');
    return;
  }

  const slider = document.getElementById('frame-slider') as HTMLInputElement;
  slider.max   = String(totalFrames - 1);

  // ── Step 1: Register event listeners IMMEDIATELY ──────────────────────────
  // Do this before the await on stage.loadFile so no events are missed
  // during the ~1 second it takes NGL to parse and render the PDB.
  listen<{ frame: number }>('viz:frame', (e) => {
    stopPlayback();
    goToFrame(e.payload.frame);
  });

  listen<{ index: number }>('viz:residue', (e) => {
    highlightResidue(e.payload.index);
  });

  listen<{ i: number; j: number }>('viz:pair', (e) => {
    highlightPair(e.payload.i, e.payload.j);
  });

  listen('viz:bfactor', () => {
    reloadWithBfactors();
  });

  // ── Step 2: Load structure ────────────────────────────────────────────────
  // @ts-ignore — NGL loaded via CDN script tag
  stage = new NGL.Stage('viewport', { backgroundColor: '#060708' });
  setStatus('Loading structure…');

  try {
    const pdbString = await invoke<string>('get_snapshot_pdb', { frame_idx: 0 });
    const blob = new Blob([pdbString], { type: 'text/plain' });
    comp = await stage!.loadFile(blob, { ext: 'pdb', firstModelOnly: true });
    applyRepresentation();
    comp.autoView();
    setStatus('', true);
  } catch (err) {
    setStatus(`Failed to load structure: ${err}`);
    console.error(err);
    return;
  }

  // ── Step 3: Wire up UI controls ───────────────────────────────────────────
  document.getElementById('style-group')!.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('[data-mode]') as HTMLButtonElement | null;
    if (!btn) return;
    currentMode = btn.dataset.mode!;
    document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyRepresentation();
  });

  document.getElementById('color-scheme')!.addEventListener('change', (e) => {
    currentColor = (e.target as HTMLSelectElement).value;
    const indicator = document.getElementById('bfactor-indicator');
    if (indicator && currentColor !== 'bfactor') indicator.style.display = 'none';
    applyRepresentation();
  });

  slider.addEventListener('input', (e) => {
    stopPlayback();
    goToFrame(parseInt((e.target as HTMLInputElement).value, 10));
  });

  document.getElementById('play-btn')!.addEventListener('click', () => {
    if (isPlaying) stopPlayback(); else startPlayback();
  });

  document.getElementById('fps-input')!.addEventListener('change', () => {
    if (isPlaying) { stopPlayback(); startPlayback(); }
  });

  document.getElementById('clear-highlight-btn')?.addEventListener('click', () => {
    clearHighlight();
  });

  window.addEventListener('resize', () => stage?.handleResize());
}

window.addEventListener('DOMContentLoaded', initVisualizer);
