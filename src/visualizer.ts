import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// ─── NGL type stubs ───────────────────────────────────────────────────────────

declare const NGL: {
  Stage: new (elementId: string, params?: Record<string, unknown>) => NGLStage;
  Shape: new (name: string) => NGLShape;
};
interface NGLStage {
  loadFile(blob: Blob, params: Record<string, unknown>): Promise<NGLComponent>;
  addComponentFromObject(obj: unknown, params?: Record<string, unknown>): NGLComponent;
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
interface NGLShape {
  addCylinder(
    start:  [number,number,number],
    end:    [number,number,number],
    color:  [number,number,number],
    radius: number,
    name?:  string,
  ): void;
  addSphere(
    center: [number,number,number],
    color:  [number,number,number],
    radius: number,
    name?:  string,
  ): void;
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
  el.textContent  = text;
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
    sele:         'not (polymer or water or ion)',
    colorScheme:  'element',
    radius:       0.15,
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
  comp.removeAllRepresentations();
  comp.addRepresentation(currentMode, { sele: 'polymer', colorScheme: currentColor });
  comp.addRepresentation('licorice',  { sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15 });
  const resno = atomIndex + 1;
  comp.addRepresentation('ball+stick', {
    sele: `resno ${resno}`, colorScheme: 'uniform', color: '#ffdd00', radius: 0.6,
  });
  comp.autoView(`resno ${resno}`, 600);
  stage?.viewer.requestRender();
  setHighlightLabel(`Residue ${resno} (index ${atomIndex})`);
}

function highlightPair(i: number, j: number) {
  if (!comp) return;
  comp.removeAllRepresentations();
  comp.addRepresentation(currentMode, { sele: 'polymer', colorScheme: currentColor });
  comp.addRepresentation('licorice',  { sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15 });
  comp.addRepresentation('ball+stick', { sele: `resno ${i+1}`, colorScheme: 'uniform', color: '#ffdd00', radius: 0.6 });
  comp.addRepresentation('ball+stick', { sele: `resno ${j+1}`, colorScheme: 'uniform', color: '#00c4a7', radius: 0.6 });
  comp.autoView(`resno ${i+1} or resno ${j+1}`, 600);
  stage?.viewer.requestRender();
  setHighlightLabel(`Residues ${i+1} & ${j+1}`);
}

function clearHighlight() {
  applyRepresentation();
  setHighlightLabel('');
}

// ─── B-factor reload ──────────────────────────────────────────────────────────

async function reloadWithBfactors() {
  if (!stage) return;
  setStatus('Updating B-factor mapping…');
  try {
    const pdbString = await invoke<string>('get_snapshot_pdb', { frameIdx: currentFrame });
    const blob = new Blob([pdbString], { type: 'text/plain' });
    if (comp) { comp.removeAllRepresentations(); }
    comp = await stage.loadFile(blob, { ext: 'pdb', firstModelOnly: true });
    currentColor = 'bfactor';
    comp.addRepresentation(currentMode, { sele: 'polymer', colorScheme: 'bfactor' });
    comp.autoView();
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
    const flat = await invoke<number[]>('get_frame_coords', { frameIdx: currentFrame });
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

// ─── NGL dock box ─────────────────────────────────────────────────────────────
//
// Renders the 12 edges of the docking search box as thin cylinders using
// NGL.Shape, plus a centre sphere.  Renders in the same stage as the receptor
// so it rotates and scales together.

function drawDockBox(
  cx: number, cy: number, cz: number,
  sx: number, sy: number, sz: number,
): void {
  if (!stage) return;

  const hx = sx / 2, hy = sy / 2, hz = sz / 2;
  // Accent teal  #00c4a7 → [0, 0.77, 0.65]
  const col: [number,number,number] = [0.0, 0.77, 0.65];
  const r = 0.25;   // cylinder radius in Å

  const C: [number,number,number][] = [
    [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
    [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
    [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
    [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
  ];

  const edges: [number,number][] = [
    [0,1],[1,2],[2,3],[3,0],   // bottom face
    [4,5],[5,6],[6,7],[7,4],   // top face
    [0,4],[1,5],[2,6],[3,7],   // verticals
  ];

  // @ts-ignore — NGL loaded via CDN
  const shape = new NGL.Shape('dock-box');
  for (const [a, b] of edges) shape.addCylinder(C[a], C[b], col, r);
  shape.addSphere([cx, cy, cz], col, 0.5);   // centre marker

  const boxComp = stage.addComponentFromObject(shape);
  boxComp.addRepresentation('buffer');
  stage.viewer.requestRender();
}

// ─── Dock preview mode ────────────────────────────────────────────────────────
//
// URL params:  mode=dock_preview  receptor=…  cx cy cz sx sy sz
//
// Loads the receptor PDB/PDBQT via lb_get_receptor_pdb (which converts PDBQT
// to clean PDB automatically), then overlays the search box.  No trajectory
// controls are shown.

async function initDockPreview(params: URLSearchParams): Promise<void> {
  // Remove the playback controls bar and collapse the shell grid
  const controls = document.querySelector('.controls') as HTMLElement | null;
  if (controls) controls.style.display = 'none';
  const shell = document.querySelector('.shell') as HTMLElement | null;
  if (shell)    shell.style.gridTemplateRows = '44px 1fr';

  const receptorPath = params.get('receptor') ?? '';
  const cx = parseFloat(params.get('cx') ?? '0');
  const cy = parseFloat(params.get('cy') ?? '0');
  const cz = parseFloat(params.get('cz') ?? '0');
  const sx = parseFloat(params.get('sx') ?? '20');
  const sy = parseFloat(params.get('sy') ?? '20');
  const sz = parseFloat(params.get('sz') ?? '20');

  // @ts-ignore
  stage = new NGL.Stage('viewport', { backgroundColor: '#060708' });
  setStatus('Loading receptor…');

  try {
    const pdbString = await invoke<string>('lb_get_receptor_pdb', { receptorPath });
    const blob = new Blob([pdbString], { type: 'text/plain' });
    comp = await stage!.loadFile(blob, { ext: 'pdb', firstModelOnly: true });

    comp.addRepresentation('cartoon',   { sele: 'polymer', colorScheme: 'resname' });
    comp.addRepresentation('licorice',  { sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15 });
    comp.addRepresentation('spacefill', { sele: 'ion',     colorScheme: 'element', radius: 0.5 });
    comp.autoView();

    drawDockBox(cx, cy, cz, sx, sy, sz);
    setStatus('', true);
  } catch (err) {
    setStatus(`Failed to load receptor: ${err}`);
    console.error(err);
    return;
  }

  // Style/colour controls still drive the receptor component
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
  window.addEventListener('resize', () => stage?.handleResize());
}

// ─── Docked complex mode ──────────────────────────────────────────────────────
//
// URL params:  mode=docked_complex  receptor=…  pdbqt=…  pose=1  total_poses=9
//
// Loads the receptor as one NGL component and the selected docked pose as a
// second component.  Arrow keys navigate between poses; the ligand builder can
// also push viz:dock-pose events to switch poses remotely.

async function initDockedComplex(params: URLSearchParams): Promise<void> {
  const controls = document.querySelector('.controls') as HTMLElement | null;
  if (controls) controls.style.display = 'none';
  const shell = document.querySelector('.shell') as HTMLElement | null;
  if (shell)    shell.style.gridTemplateRows = '44px 1fr';

  const receptorPath = params.get('receptor')    ?? '';
  const pdbqtPath    = params.get('pdbqt')       ?? '';
  const totalPoses   = parseInt(params.get('total_poses') ?? '1', 10);
  let   activePose   = parseInt(params.get('pose')        ?? '1', 10);

  // @ts-ignore
  stage = new NGL.Stage('viewport', { backgroundColor: '#060708' });
  setStatus('Loading docked complex…');

  // ── Load receptor ──────────────────────────────────────────────────────────
  let receptorComp: NGLComponent | null = null;
  try {
    const pdbString = await invoke<string>('lb_get_receptor_pdb', { receptorPath });
    const blob = new Blob([pdbString], { type: 'text/plain' });
    receptorComp = await stage!.loadFile(blob, { ext: 'pdb', firstModelOnly: true });
    receptorComp.addRepresentation('cartoon',   { sele: 'polymer', colorScheme: 'resname' });
    receptorComp.addRepresentation('licorice',  { sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15 });
    receptorComp.addRepresentation('spacefill', { sele: 'ion', colorScheme: 'element', radius: 0.5 });
  } catch (err) {
    setStatus(`Failed to load receptor: ${err}`);
    console.error(err);
    return;
  }

  // ── Ligand pose loader ────────────────────────────────────────────────────
  let ligandComp: NGLComponent | null = null;

  async function loadPose(rank: number): Promise<void> {
    try {
      setStatus(`Loading pose ${rank} / ${totalPoses}…`);
      const posePdb = await invoke<string>('lb_get_pose_pdb', {
        outputPdbqtPath: pdbqtPath,
        poseRank:        rank,
      });
      const blob = new Blob([posePdb], { type: 'text/plain' });

      // Remove old ligand component before adding new one
      if (ligandComp) { ligandComp.removeAllRepresentations(); }
      ligandComp = await stage!.loadFile(blob, { ext: 'pdb', firstModelOnly: true });
      ligandComp.addRepresentation('ball+stick', {
        colorScheme:  'element',
        multipleBond: 'symmetric',
        radius:       0.2,
      });

      setHighlightLabel(`Pose ${rank} / ${totalPoses}  (← → to navigate)`);
      setStatus('', true);
      stage!.viewer.requestRender();
    } catch (err) {
      setStatus(`Failed to load pose ${rank}: ${err}`);
      console.error(err);
    }
  }

  await loadPose(activePose);
  stage!.autoView();

  // ── Keyboard pose navigation ──────────────────────────────────────────────
  document.addEventListener('keydown', async (e) => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      activePose = Math.min(totalPoses, activePose + 1);
      await loadPose(activePose);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      activePose = Math.max(1, activePose - 1);
      await loadPose(activePose);
    }
  });

  // ── Remote pose switch from ligand builder (clicking a row) ───────────────
  listen<{ pose: number }>('viz:dock-pose', async (e) => {
    activePose = e.payload.pose;
    await loadPose(activePose);
  });

  // ── Style / colour controls apply to receptor ─────────────────────────────
  document.getElementById('style-group')!.addEventListener('click', (e) => {
    const btn = (e.target as HTMLElement).closest('[data-mode]') as HTMLButtonElement | null;
    if (!btn || !receptorComp) return;
    currentMode = btn.dataset.mode!;
    document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    receptorComp.removeAllRepresentations();
    receptorComp.addRepresentation(currentMode, { sele: 'polymer', colorScheme: currentColor });
    receptorComp.addRepresentation('licorice',  { sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15 });
    stage?.viewer.requestRender();
  });
  document.getElementById('color-scheme')!.addEventListener('change', (e) => {
    if (!receptorComp) return;
    currentColor = (e.target as HTMLSelectElement).value;
    receptorComp.removeAllRepresentations();
    receptorComp.addRepresentation(currentMode, { sele: 'polymer', colorScheme: currentColor });
    receptorComp.addRepresentation('licorice',  { sele: 'not (polymer or water or ion)', colorScheme: 'element', radius: 0.15 });
    const indicator = document.getElementById('bfactor-indicator');
    if (indicator && currentColor !== 'bfactor') indicator.style.display = 'none';
    stage?.viewer.requestRender();
  });
  window.addEventListener('resize', () => stage?.handleResize());
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
// (it just registers a callback in the IPC channel). We don't need to await
// it in sequence with the structure load — we can fire all registrations
// immediately and let them resolve in parallel.

async function initVisualizer() {
  const urlParams = new URLSearchParams(window.location.search);
  const mode      = urlParams.get('mode') ?? 'trajectory';

  // Branch into specialised init functions for non-trajectory modes.
  // Each handles its own stage setup and control wiring, then returns.
  if (mode === 'dock_preview')   { await initDockPreview(urlParams);   return; }
  if (mode === 'docked_complex') { await initDockedComplex(urlParams); return; }

  // ── Trajectory mode (default) ─────────────────────────────────────────────
  totalFrames = parseInt(urlParams.get('frames') || '0', 10);
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
    const pdbString = await invoke<string>('get_snapshot_pdb', { frameIdx: 0 });
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
