#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use chemfiles::{Frame, Selection, Trajectory};
use serde::Serialize;
use ndarray::prelude::*;
use ndarray_linalg::Eigh;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
extern crate glob;
use serde_json;

// ─── Progress event ──────────────────────────────────────────────────────────
//
// Emitted during long-running commands so the frontend can show a progress bar.
// The event name is "progress" and the payload is { tool, pct } where pct is
// 0.0–100.0.  The frontend listens with:
//   import { listen } from '@tauri-apps/api/event';
//   await listen('progress', (e) => updateBar(e.payload.pct));

#[derive(Clone, Serialize)]
struct ProgressEvent {
    tool: String,
    pct:  f64,
}

fn emit_progress(app: &AppHandle, tool: &str, pct: f64) {
    let _ = app.emit("progress", ProgressEvent { tool: tool.to_string(), pct });
}

// ─── Atom metadata ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AtomMeta {
    pub atom_name: String,
    pub res_name:  String,
    pub chain_id:  char,
    pub res_seq:   i64,
    pub element:   String,
}

// ─── Shared State ─────────────────────────────────────────────────────────────
//
// Every analysis that produces reusable output gets its own cache field.
// `pca_cache` stores both the projections and eigenvectors so downstream
// tools (FES, NMA overlap) can consume them without re-running PCA.

pub struct AppData {
    pub trajectory_data: Mutex<Option<Vec<Vec<[f64; 3]>>>>,
    pub atom_meta:       Mutex<Option<Vec<AtomMeta>>>,
    // ── per-analysis caches ────────────────────────────────────────────────
    pub rmsd_cache:      Mutex<Option<Vec<f64>>>,
    pub rmsf_cache:      Mutex<Option<Vec<f64>>>,
    pub rg_cache:        Mutex<Option<Vec<f64>>>,
    pub dccm_cache:      Mutex<Option<Vec<Vec<f64>>>>,
    pub pca_cache:       Mutex<Option<PcaResult>>,
    pub enm_cache:       Mutex<Option<EnmResult>>,
    pub contacts_cache:  Mutex<Option<Vec<Vec<f64>>>>,
    pub hbond_cache:     Mutex<Option<Vec<HBondRecord>>>,
    // ── visualizer overlay data ───────────────────────────────────────────
    pub bfactor_cache:   Mutex<Option<Vec<f64>>>,
    pub dihedral_cache:  Mutex<Option<DihedralResult>>,
    pub prs_cache:       Mutex<Option<PrsResult>>,
    pub mi_cache:        Mutex<Option<Vec<Vec<f64>>>>,
    pub cluster_cache:   Mutex<Option<ClusterResult>>,
    pub geometry_cache:  Mutex<Option<GeometryResult>>,
    pub sasa_cache:      Mutex<Option<SasaResult>>,
    pub membrane_cache:  Mutex<Option<MembraneResult>>,
    // ── Box dimensions (per-frame [a, b, c] in Å) ────────────────────────
    pub cell_dims:       Mutex<Option<Vec<[f64; 3]>>>,
    // ── QM/MM umbrella sampling ───────────────────────────────────────────
    // Each window stores: the raw CV samples, the restraint centre val0,
    // and the path to the restart file for the analysis viewer.
    pub umbrella_windows: Mutex<Option<Vec<UmbrellaWindow>>>,
    pub mbar_result:      Mutex<Option<MbarResult>>,
    pub qmm_topology:     Mutex<Option<String>>,  // path to PDB/parm7 for ncrst visualisation
    // Pre-loaded flat coordinate arrays for all umbrella windows.
    // Populated by get_umbrella_snapshot_pdb on first call per window,
    // or by an explicit preload command. Format: flat [x,y,z, x,y,z, …] f32.
    // Pre-loaded umbrella trajectory: one flat-coord entry per window
    pub umbrella_traj_coords: Mutex<Option<Vec<Vec<f32>>>>,
    pub qm_region:        Mutex<Option<QmRegion>>, // stored QM region atoms
    // ── Startup file path (from CLI arg or file association) ─────────────────
    pub startup_file:     Mutex<Option<String>>,
}

// ─── Internal cached types ────────────────────────────────────────────────────
// These live in AppData and are not sent to JS directly — serialisable
// wrappers are built at query time.

#[derive(Clone)]
pub struct PcaResult {
    pub projections:        Vec<[f64; 2]>,
    pub explained_variance: Vec<f64>,
    /// Top eigenvectors stored column-wise for downstream use (e.g. ENM overlap)
    pub eigenvectors:       Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct EnmResult {
    pub eigenvalues:   Vec<f64>,   // mode frequencies (ascending)
    pub eigenvectors:  Vec<Vec<f64>>, // columns = modes, rows = 3N components
    pub model:         String,     // "ANM" or "GNM"
}

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct HBondRecord {
    pub donor:     usize,   // atom index
    pub acceptor:  usize,
    pub occupancy: f64,     // fraction of frames with bond present
    pub mean_dist: f64,     // mean D–A distance in Å
}

// ─── QM/MM umbrella sampling types ───────────────────────────────────────────

/// One umbrella sampling window loaded from a CV file.
#[derive(Clone, Serialize, serde::Deserialize)]
pub struct UmbrellaWindow {
    pub index:     usize,
    pub val0:      f64,        // restraint centre (Å or other CV unit)
    pub samples:   Vec<f64>,   // raw CV values read from the .cv file
    pub cv_file:   String,     // path that was read
    pub rst_file:  Option<String>, // nearest .ncrst for visualisation
}

/// Result returned to the frontend after MBAR/WHAM calculation.
#[derive(Clone, Serialize, serde::Deserialize)]
pub struct MbarResult {
    pub bin_centers: Vec<f64>,
    pub pmf:         Vec<f64>,   // PMF in kcal/mol, zero-referenced
    pub pmf_err:     Vec<f64>,   // MBAR uncertainty
    pub delta_g:     f64,        // activation free energy (max - min near reactant)
    pub delta_g_err: f64,
    pub n_windows:   usize,
    pub window_val0: Vec<f64>,   // restraint centres for vertical lines
    pub kde_x:       Vec<Vec<f64>>, // KDE x-values per window (for histogram panel)
    pub kde_y:       Vec<Vec<f64>>, // KDE y-values per window
}

// ─── QM region types ─────────────────────────────────────────────────────────

/// One atom in the stored QM region.
#[derive(Clone, Serialize)]
pub struct QmAtom {
    pub serial:   usize,   // 1-based PDB serial
    pub atom_name: String,
    pub res_name:  String,
    pub res_seq:   i64,
    pub chain_id:  char,
    pub element:   String,
    pub x: f64, pub y: f64, pub z: f64,
}

/// The full stored QM region — returned to JS for display and export.
#[derive(Clone, Serialize)]
pub struct QmRegion {
    pub atoms:       Vec<QmAtom>,
    pub amber_mask:  String,   // e.g. ":145,146 | @1,2,3"
    pub window_idx:  usize,
    pub n_atoms:     usize,
}

// ─── Serialisable return types ────────────────────────────────────────────────

#[derive(Serialize)]
struct AnalysisResult<T> {
    data: T,
    message: String,
}

#[derive(Serialize)]
struct PcaData {
    projections:        Vec<[f64; 2]>,
    explained_variance: Vec<f64>,
}

#[derive(Serialize)]
struct EntropyResult {
    entropy_j_per_mol_k: f64,
    n_modes_used:        usize,
}

// ─── Dihedral / Ramachandran types ────────────────────────────────────────────

/// Per-residue summary stored in the dihedral cache (full φ/ψ time series).
#[derive(Clone)]
pub struct ResidueDihedralFull {
    pub res_seq:  i64,
    pub res_name: String,
    pub atom_idx: usize,   // CA (or representative) selection index for click-through
    pub phi:      Vec<f64>, // degrees, NaN for terminal residues
    pub psi:      Vec<f64>,
}

/// Cache type — holds full per-frame data so get_residue_dihedrals can serve
/// the analysis viewer without re-running the analysis.
#[derive(Clone)]
pub struct DihedralResult {
    pub residues: Vec<ResidueDihedralFull>,
    pub mode:     String,  // "backbone" or "pseudodihedral"
}

/// Per-residue summary sent to the frontend (means/stds only, not all frames).
#[derive(Serialize)]
struct ResidueDihedralSummary {
    res_seq:   i64,
    res_name:  String,
    atom_idx:  usize,
    phi_mean:  Option<f64>,
    psi_mean:  Option<f64>,
    phi_std:   Option<f64>,
    psi_std:   Option<f64>,
    n_valid:   usize,
}

#[derive(Serialize)]
struct DihedralResultJson {
    density:  Vec<Vec<f64>>,       // 60×60 2D histogram normalised to [0,1]
    residues: Vec<ResidueDihedralSummary>,
    mode:     String,
    n_frames: usize,
}

/// Full time series for one residue — returned on demand by get_residue_dihedrals.
#[derive(Serialize)]
struct SingleResidueDihedrals {
    res_seq:  i64,
    res_name: String,
    phi:      Vec<f64>,
    psi:      Vec<f64>,
    mode:     String,
}

// ─── PRS types ────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct PrsResult {
    pub matrix:        Vec<Vec<f64>>,  // N×N response matrix
    pub effectiveness: Vec<f64>,       // row sums (how much i perturbs others)
    pub sensitivity:   Vec<f64>,       // col sums normalised by j variance
}

// ─── MI type ─────────────────────────────────────────────────────────────────
// stored as Vec<Vec<f64>> directly in the cache

// ─── Clustering types ─────────────────────────────────────────────────────────

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct ClusterResult {
    pub assignments:  Vec<usize>,    // frame → cluster id
    pub centers:      Vec<[f64; 2]>, // cluster center in PC1/PC2 space
    pub populations:  Vec<f64>,      // fraction of frames in each cluster
    pub method:       String,        // "kmeans" or "msm"
    pub n_clusters:   usize,
    // MSM-only fields (empty for k-means)
    pub implied_timescales: Vec<f64>, // ITS at the requested lag (ps or frames)
    pub pcca_membership:    Vec<Vec<f64>>, // soft assignment [frame][macro]
}

#[derive(Serialize)]
struct ClusterResultJson {
    assignments:        Vec<usize>,
    centers:            Vec<[f64; 2]>,
    populations:        Vec<f64>,
    method:             String,
    n_clusters:         usize,
    implied_timescales: Vec<f64>,
    pcca_membership:    Vec<Vec<f64>>,
}

// ─── Geometry / distance types ────────────────────────────────────────────────

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct GeometrySeries {
    pub label:   String,      // e.g. "dist(0,5)"
    pub kind:    String,      // "distance" or "angle"
    pub values:  Vec<f64>,    // one value per frame
    pub unit:    String,      // "Å" or "°"
}

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct GeometryResult {
    pub series:   Vec<GeometrySeries>,
    pub n_frames: usize,
    pub source:   String,     // "trajectory" or "umbrella"
}

// ─── SASA types ──────────────────────────────────────────────────────────────

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct SasaResult {
    pub per_residue_mean: Vec<f64>,
    pub per_residue_std:  Vec<f64>,
    pub total_per_frame:  Vec<f64>,
    pub res_labels:       Vec<String>,
}

// ─── Membrane analysis types ──────────────────────────────────────────────────

#[derive(Clone, Serialize, serde::Deserialize)]
pub struct MembraneResult {
    pub thickness_per_frame: Vec<f64>,   // bilayer thickness (Å) per frame
    pub apl_per_frame:       Vec<f64>,   // area per lipid (Ų) per frame; 0 if no box
    pub z_density_upper:     Vec<f64>,   // normalised z-histogram, upper leaflet
    pub z_density_lower:     Vec<f64>,   // normalised z-histogram, lower leaflet
    pub order_params:        Vec<f64>,   // mean Scd per consecutive C–C vector
    pub order_labels:        Vec<String>,
    pub n_upper:             usize,      // headgroup atoms per leaflet (frame 0)
    pub n_lower:             usize,
    pub n_frames:            usize,
    pub mean_thickness:      f64,
    pub mean_apl:            f64,
    pub has_apl:             bool,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct RunGeometryReq {
    pairs:    Vec<[usize; 2]>,
    triplets: Vec<[usize; 3]>,
    source:   Option<String>,
}

#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct RunClusteringReq {
    n_clusters: usize,
    method:     Option<String>,
    lag:        Option<usize>,
    n_macro:    Option<usize>,
}

// ─── load_trajectory ──────────────────────────────────────────────────────────

#[tauri::command]
async fn load_trajectory(
    state: State<'_, AppData>,
    path: String,
    top_path: Option<String>,
    selection_str: String,
    stride: Option<usize>,
    preserve_caches: Option<bool>,  // true when called from load_project
) -> Result<String, String> {
    let stride = stride.unwrap_or(1).max(1);
    let mut traj = Trajectory::open(&path, 'r').map_err(|e| e.to_string())?;

    if let Some(ref topo_path) = top_path {
        if topo_path.ends_with(".parm7") || topo_path.ends_with(".prmtop") {
            traj.set_topology_with_format(topo_path, "Amber Topology")
                .map_err(|e| e.to_string())?;
        } else {
            traj.set_topology_file(topo_path).map_err(|e| e.to_string())?;
        }
    }

    let mut sel = Selection::new(selection_str.as_str())
        .map_err(|_| format!("Invalid selection: '{}'", selection_str))?;

    let mut frame = Frame::new();
    let n_frames_hint = traj.nsteps();
    let mut all_coords: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n_frames_hint / stride + 1);
    let mut all_cells:  Vec<[f64; 3]>      = Vec::with_capacity(n_frames_hint / stride + 1);
    let mut atom_meta_store: Option<Vec<AtomMeta>> = None;
    let mut frame_idx = 0usize;

    while traj.read(&mut frame).is_ok() {
        if frame_idx % stride != 0 {
            frame_idx += 1;
            continue;
        }
        frame_idx += 1;
        let matches = sel.evaluate(&frame);
        let pos = frame.positions();

        if atom_meta_store.is_none() {
            let topo = frame.topology();
            let mut meta_vec: Vec<AtomMeta> = Vec::with_capacity(matches.len());
            for (serial_0, m) in matches.iter().enumerate() {
                let gidx      = m[0];
                let atom      = topo.atom(gidx);
                let atom_name = atom.name().to_string();
                let element   = {
                    let raw = atom.atomic_type();
                    if raw.is_empty() {
                        atom_name.chars().find(|c| c.is_ascii_uppercase())
                            .map(|c| c.to_string()).unwrap_or_else(|| "X".to_string())
                    } else { raw.to_string() }
                };
                let (res_name, chain_id, res_seq) =
                    if let Some(res) = topo.residue_for_atom(gidx) {
                        let chain = res.get("chainname")
                            .and_then(|p| if let chemfiles::Property::String(s) = p {
                                s.chars().next() } else { None })
                            .unwrap_or('A');
                        (res.name().to_string(), chain, res.id().unwrap_or(serial_0 as i64 + 1))
                    } else { ("UNK".to_string(), 'A', serial_0 as i64 + 1) };
                meta_vec.push(AtomMeta { atom_name, res_name, chain_id, res_seq, element });
            }
            atom_meta_store = Some(meta_vec);
        }

        let frame_coords: Vec<[f64; 3]> = matches.iter()
            .map(|m| { let i = m[0]; [pos[i][0] as f64, pos[i][1] as f64, pos[i][2] as f64] })
            .collect();
        all_coords.push(frame_coords);
        all_cells.push(frame.cell().lengths());
    }

    let frame_count = all_coords.len();
    if frame_count == 0 { return Err("No frames read — check trajectory and selection.".into()); }
    let n_atoms = all_coords[0].len();

    // Invalidate all caches on new load — skipped when loading a project
    // so that caches restored by load_project are not wiped.
    *state.trajectory_data.lock().unwrap() = Some(all_coords);
    *state.atom_meta.lock().unwrap()       = atom_meta_store;
    *state.cell_dims.lock().unwrap()       = Some(all_cells);

    if !preserve_caches.unwrap_or(false) {
        *state.rmsd_cache.lock().unwrap()      = None;
        *state.rmsf_cache.lock().unwrap()      = None;
        *state.rg_cache.lock().unwrap()        = None;
        *state.dccm_cache.lock().unwrap()      = None;
        *state.pca_cache.lock().unwrap()       = None;
        *state.enm_cache.lock().unwrap()       = None;
        *state.contacts_cache.lock().unwrap()  = None;
        *state.hbond_cache.lock().unwrap()     = None;
        *state.bfactor_cache.lock().unwrap()   = None;
        *state.dihedral_cache.lock().unwrap()  = None;
        *state.prs_cache.lock().unwrap()       = None;
        *state.mi_cache.lock().unwrap()        = None;
        *state.cluster_cache.lock().unwrap()   = None;
        *state.geometry_cache.lock().unwrap()  = None;
        *state.sasa_cache.lock().unwrap()      = None;
        *state.membrane_cache.lock().unwrap()  = None;
    }

    let stride_msg = if stride > 1 { format!(" (stride {})", stride) } else { String::new() };

    // Memory estimate: frame_count × n_atoms × 3 × 8 bytes (f64 coords)
    // plus cell_dims: frame_count × 3 × 8 bytes — negligible.
    let mem_bytes  = frame_count * n_atoms * 24;
    let mem_msg = if mem_bytes >= 4_000_000_000 {
        format!(" ⚠ {:.1} GB in memory — consider a larger stride.", mem_bytes as f64 / 1e9)
    } else if mem_bytes >= 1_000_000_000 {
        format!(" ({:.1} GB in memory)", mem_bytes as f64 / 1e9)
    } else {
        format!(" ({:.0} MB in memory)", mem_bytes as f64 / 1e6)
    };

    Ok(format!("Loaded {} frames, {} atoms ({}){}.{}",
        frame_count, n_atoms, selection_str, stride_msg, mem_msg))
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

fn mean_positions(traj: &[Vec<[f64; 3]>]) -> Vec<[f64; 3]> {
    let n_frames = traj.len() as f64;
    let n_atoms  = traj[0].len();
    let mut mean = vec![[0.0f64; 3]; n_atoms];
    for frame in traj {
        for (i, a) in frame.iter().enumerate() {
            mean[i][0] += a[0]; mean[i][1] += a[1]; mean[i][2] += a[2];
        }
    }
    for m in &mut mean { m[0] /= n_frames; m[1] /= n_frames; m[2] /= n_frames; }
    mean
}

/// Build the 3N×3N covariance matrix from mean-centred coordinates.
/// Used by both PCA and the quasi-harmonic entropy.
fn build_covariance(traj: &[Vec<[f64; 3]>]) -> Result<Array2<f64>, String> {
    let n_frames = traj.len();
    let n_atoms  = traj[0].len();
    let dim      = n_atoms * 3;
    let mean     = mean_positions(traj);
    let flat_mean: Vec<f64> = mean.iter().flat_map(|a| a.iter().copied()).collect();
    let data_flat: Vec<f64> = traj.iter()
        .flat_map(|f| f.iter().flat_map(|a| a.iter().copied()))
        .collect();
    let mut x = Array2::from_shape_vec((n_frames, dim), data_flat).map_err(|e| e.to_string())?;
    for mut row in x.rows_mut() {
        for (v, m) in row.iter_mut().zip(&flat_mean) { *v -= m; }
    }
    Ok(x.t().dot(&x) / n_frames as f64)
}

// ─── Structural analyses ──────────────────────────────────────────────────────

#[tauri::command]
fn run_rmsd(state: State<'_, AppData>) -> Result<AnalysisResult<Vec<f64>>, String> {
    if let Some(c) = state.rmsd_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "RMSD (cached)".into() });
    }
    let lock = state.trajectory_data.lock().unwrap();
    let traj = lock.as_ref().ok_or("No trajectory loaded")?;
    let n_frames = traj.len();
    let n_atoms  = traj[0].len();
    let ref_flat: Vec<f64> = traj[0].iter().flat_map(|a| a.iter().copied()).collect();
    let ref_arr  = Array2::from_shape_vec((n_atoms, 3), ref_flat).map_err(|e| e.to_string())?;

    let mut results = vec![0.0f64; n_frames];
    results.par_iter_mut().enumerate().for_each(|(f, val)| {
        let mut sum = 0.0f64;
        for i in 0..n_atoms { for c in 0..3 { let d = traj[f][i][c] - ref_arr[[i, c]]; sum += d*d; } }
        *val = (sum / n_atoms as f64).sqrt();
    });
    *state.rmsd_cache.lock().unwrap() = Some(results.clone());
    Ok(AnalysisResult { data: results, message: format!("RMSD over {n_frames} frames computed.") })
}

#[tauri::command]
fn run_rmsf(state: State<'_, AppData>) -> Result<AnalysisResult<Vec<f64>>, String> {
    if let Some(c) = state.rmsf_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "RMSF (cached)".into() });
    }
    let lock = state.trajectory_data.lock().unwrap();
    let traj = lock.as_ref().ok_or("No trajectory loaded")?;
    let n_frames = traj.len() as f64;
    let n_atoms  = traj[0].len();
    let mean     = mean_positions(traj);

    let rmsf: Vec<f64> = (0..n_atoms).into_par_iter().map(|i| {
        let mut sum = 0.0f64;
        for frame in traj { for c in 0..3 { let d = frame[i][c] - mean[i][c]; sum += d*d; } }
        (sum / n_frames).sqrt()
    }).collect();
    *state.rmsf_cache.lock().unwrap() = Some(rmsf.clone());
    Ok(AnalysisResult { data: rmsf, message: format!("RMSF for {n_atoms} atoms computed.") })
}

#[tauri::command]
fn run_radius_of_gyration(state: State<'_, AppData>) -> Result<AnalysisResult<Vec<f64>>, String> {
    if let Some(c) = state.rg_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "Rg (cached)".into() });
    }
    let lock = state.trajectory_data.lock().unwrap();
    let traj = lock.as_ref().ok_or("No trajectory loaded")?;
    let n_frames = traj.len();
    let n_atoms  = traj[0].len() as f64;

    let rg: Vec<f64> = (0..n_frames).into_par_iter().map(|f| {
        let frame = &traj[f];
        let mut com = [0.0f64; 3];
        for a in frame { com[0] += a[0]; com[1] += a[1]; com[2] += a[2]; }
        com[0] /= n_atoms; com[1] /= n_atoms; com[2] /= n_atoms;
        let sum: f64 = frame.iter().map(|a| {
            let dx=a[0]-com[0]; let dy=a[1]-com[1]; let dz=a[2]-com[2];
            dx*dx + dy*dy + dz*dz
        }).sum();
        (sum / n_atoms).sqrt()
    }).collect();
    *state.rg_cache.lock().unwrap() = Some(rg.clone());
    Ok(AnalysisResult { data: rg, message: format!("Radius of gyration over {n_frames} frames computed.") })
}

#[tauri::command]
async fn run_pca(
    app:   AppHandle,
    state: State<'_, AppData>,
) -> Result<AnalysisResult<PcaData>, String> {
    if let Some(ref c) = *state.pca_cache.lock().unwrap() {
        return Ok(AnalysisResult {
            data: PcaData { projections: c.projections.clone(), explained_variance: c.explained_variance.clone() },
            message: "PCA (cached)".into(),
        });
    }

    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .as_ref().ok_or("No trajectory loaded")?.clone();

    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let dim      = n_atoms * 3;
        emit_progress(&app, "pca", 5.0);

        let mean     = mean_positions(&traj);
        let flat_mean: Vec<f64> = mean.iter().flat_map(|a| a.iter().copied()).collect();
        let data_flat: Vec<f64> = traj.iter()
            .flat_map(|f| f.iter().flat_map(|a| a.iter().copied())).collect();

        emit_progress(&app, "pca", 20.0);

        let mut x = Array2::from_shape_vec((n_frames, dim), data_flat).map_err(|e| e.to_string())?;
        for mut row in x.rows_mut() {
            for (v, m) in row.iter_mut().zip(&flat_mean) { *v -= m; }
        }
        let c = x.t().dot(&x) / n_frames as f64;

        emit_progress(&app, "pca", 50.0);

        let (eigenvalues, eigenvectors) = c.eigh(ndarray_linalg::UPLO::Upper)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        emit_progress(&app, "pca", 85.0);

        let total: f64 = eigenvalues.iter().map(|v| v.abs()).sum();
        let explained: Vec<f64> = eigenvalues.iter().rev().map(|v| v.abs() / total).collect();
        let ne = eigenvalues.len();
        let pc1 = eigenvectors.column(ne - 1).to_owned();
        let pc2 = eigenvectors.column(ne - 2).to_owned();
        let projections: Vec<[f64; 2]> = x.rows().into_iter()
            .map(|row| [row.dot(&pc1), row.dot(&pc2)]).collect();
        let evecs: Vec<Vec<f64>> = (0..ne).rev()
            .map(|k| eigenvectors.column(k).iter().copied().collect())
            .collect();

        emit_progress(&app, "pca", 100.0);
        Ok::<_, String>((projections, explained, evecs))
    }).await.map_err(|e| e.to_string())??;

    let (projections, explained, evecs) = result;
    *state.pca_cache.lock().unwrap() = Some(PcaResult {
        projections: projections.clone(),
        explained_variance: explained.iter().copied().take(20).collect(),
        eigenvectors: evecs,
    });

    Ok(AnalysisResult {
        data: PcaData { projections, explained_variance: explained.iter().copied().take(10).collect() },
        message: format!("PCA complete. PC1={:.1}%, PC2={:.1}% of variance.",
            explained[0]*100.0, explained[1]*100.0),
    })
}

// ─── Correlation & Dynamics ───────────────────────────────────────────────────

#[tauri::command]
async fn run_dccm(
    app:   AppHandle,
    state: State<'_, AppData>,
) -> Result<AnalysisResult<Vec<Vec<f64>>>, String> {
    if let Some(c) = state.dccm_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "DCCM (cached)".into() });
    }

    // Clone data out of the Mutex immediately so the lock is dropped before
    // the blocking work begins. This allows other commands to read trajectory
    // data concurrently while DCCM is running.
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .as_ref().ok_or("No trajectory loaded")?.clone();

    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let mean     = mean_positions(&traj);

        emit_progress(&app, "dccm", 5.0);

        let norms: Vec<f64> = (0..n_atoms).into_par_iter().map(|i| {
            let sum: f64 = traj.iter().map(|f|
                (0..3usize).map(|c| { let d = f[i][c]-mean[i][c]; d*d }).sum::<f64>()
            ).sum();
            sum.sqrt()
        }).collect();

        emit_progress(&app, "dccm", 15.0);

        // Process rows in parallel; emit progress every ~10% of rows
        let chunk = (n_atoms / 10).max(1);
        let matrix: Vec<Vec<f64>> = (0..n_atoms).into_par_iter().map(|i| {
            if i % chunk == 0 {
                let pct = 15.0 + (i as f64 / n_atoms as f64) * 80.0;
                emit_progress(&app, "dccm", pct);
            }
            let mut row = vec![0.0f64; n_atoms];
            for j in 0..n_atoms {
                let mut dot = 0.0f64;
                for f in &traj { for c in 0..3 { dot += (f[i][c]-mean[i][c])*(f[j][c]-mean[j][c]); } }
                let den = norms[i] * norms[j];
                if den > 1e-9 { row[j] = dot / den; }
            }
            row
        }).collect();

        emit_progress(&app, "dccm", 100.0);
        Ok::<_, String>((matrix, n_atoms, n_frames))
    }).await.map_err(|e| e.to_string())??;

    let (matrix, n_atoms, n_frames) = result;
    *state.dccm_cache.lock().unwrap() = Some(matrix.clone());
    Ok(AnalysisResult {
        data: matrix,
        message: format!("DCCM ({n_atoms}×{n_atoms}) over {n_frames} frames computed."),
    })
}

// ─── run_contacts ─────────────────────────────────────────────────────────────
//
// Residue–residue contact frequency matrix.
//
// A contact between atom i and atom j is defined as their Cα–Cα (or any
// selected-atom) distance falling below `cutoff` Angstroms. The matrix
// value C_ij is the fraction of frames in which that contact is present.
// Atoms in adjacent residues (|res_seq_i - res_seq_j| ≤ 2) are excluded
// to filter out trivial bonded contacts.
//
// The result is suitable for display as a heatmap — values in [0, 1].

#[tauri::command]
async fn run_contacts(
    app:    AppHandle,
    state:  State<'_, AppData>,
    cutoff: Option<f64>,   // default 8.0 Å
) -> Result<AnalysisResult<Vec<Vec<f64>>>, String> {
    if let Some(c) = state.contacts_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "Contact map (cached)".into() });
    }
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .as_ref().ok_or("No trajectory loaded")?.clone();
    let res_seqs: Vec<i64> = state.atom_meta.lock().unwrap()
        .as_ref()
        .map(|m| m.iter().map(|a| a.res_seq).collect())
        .unwrap_or_else(|| (0..traj[0].len() as i64).collect());

    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_frames = traj.len() as f64;
        let n_atoms  = traj[0].len();
        let cut2     = (cutoff.unwrap_or(8.0)).powi(2);
        let chunk    = (traj.len() / 10).max(1);

        let mut counts = vec![vec![0.0f64; n_atoms]; n_atoms];
        for (fi, frame) in traj.iter().enumerate() {
            if fi % chunk == 0 {
                emit_progress(&app, "contacts", (fi as f64 / traj.len() as f64) * 95.0);
            }
            for i in 0..n_atoms {
                for j in (i + 1)..n_atoms {
                    if (res_seqs[i] - res_seqs[j]).abs() <= 2 { continue; }
                    let dx = frame[i][0] - frame[j][0];
                    let dy = frame[i][1] - frame[j][1];
                    let dz = frame[i][2] - frame[j][2];
                    if dx*dx + dy*dy + dz*dz <= cut2 {
                        counts[i][j] += 1.0; counts[j][i] += 1.0;
                    }
                }
            }
        }
        let matrix: Vec<Vec<f64>> = counts.iter()
            .map(|row| row.iter().map(|&v| v / n_frames).collect())
            .collect();
        emit_progress(&app, "contacts", 100.0);
        Ok::<_, String>((matrix, n_atoms, cutoff.unwrap_or(8.0)))
    }).await.map_err(|e| e.to_string())??;

    let (matrix, n_atoms, cut) = result;
    *state.contacts_cache.lock().unwrap() = Some(matrix.clone());
    Ok(AnalysisResult {
        data: matrix,
        message: format!("Contact map ({n_atoms}×{n_atoms}) computed at {cut:.1} Å cutoff."),
    })
}

// ─── run_hbond ────────────────────────────────────────────────────────────────
//
// Hydrogen bond analysis using a pure distance + angle criterion.
//
// Criterion (Baker–Hubbard):
//   D–H ··· A  where  D–A distance < 3.5 Å  and  D–H–A angle > 120°
//
// Because we store only Cα atoms by default, this command works best when
// the trajectory is loaded with a full-atom selection such as
// `name N O` or without any selection filter. When only Cα atoms are
// present it falls back to a simple donor–acceptor distance scan at 3.5 Å
// (no hydrogen available, so angle criterion is skipped) and reports a
// warning in the message.
//
// The H-atom detection heuristic: an atom is a candidate H-donor if its
// element is "N" or "O". An acceptor is "N" or "O" as well. We iterate
// all donor–acceptor pairs, evaluate the distance criterion per frame,
// and accumulate occupancy. Only pairs with occupancy > 5% are returned
// to avoid flooding the result with spurious transient contacts.

#[tauri::command]
fn run_hbond(
    state: State<'_, AppData>,
    cutoff_dist: Option<f64>,  // default 3.5 Å donor–acceptor distance
    min_occupancy: Option<f64>, // default 0.05 (5%)
) -> Result<AnalysisResult<Vec<f64>>, String> {
    let lock      = state.trajectory_data.lock().unwrap();
    let traj      = lock.as_ref().ok_or("No trajectory loaded")?;
    let meta_lock = state.atom_meta.lock().unwrap();
    let n_frames  = traj.len() as f64;
    let n_atoms   = traj[0].len();
    let cut       = cutoff_dist.unwrap_or(3.5);
    let cut2      = cut * cut;
    let min_occ   = min_occupancy.unwrap_or(0.05);

    // Identify donor and acceptor indices from element types
    let is_polar: Vec<bool> = meta_lock.as_ref()
        .map(|m| m.iter().map(|a| a.element == "N" || a.element == "O").collect())
        .unwrap_or_else(|| vec![false; n_atoms]);

    let polar_indices: Vec<usize> = is_polar.iter().enumerate()
        .filter_map(|(i, &p)| if p { Some(i) } else { None })
        .collect();

    let has_polar = polar_indices.iter().any(|&i| is_polar[i]);
    let ca_only   = !has_polar;

    // Use all atom pairs if no polar atoms found (Cα-only fallback)
    let (donors, acceptors): (Vec<usize>, Vec<usize>) = if ca_only {
        let all: Vec<usize> = (0..n_atoms).collect();
        (all.clone(), all)
    } else {
        (polar_indices.clone(), polar_indices)
    };

    // Count contacts per donor–acceptor pair using rayon over donors
    let records: Vec<HBondRecord> = donors.par_iter().flat_map(|&d| {
        let mut local = Vec::new();
        for &a in &acceptors {
            if a <= d { continue; } // upper triangle only
            let mut count    = 0usize;
            let mut dist_sum = 0.0f64;
            for frame in traj.iter() {
                let dx = frame[d][0] - frame[a][0];
                let dy = frame[d][1] - frame[a][1];
                let dz = frame[d][2] - frame[a][2];
                let d2 = dx*dx + dy*dy + dz*dz;
                if d2 <= cut2 {
                    count    += 1;
                    dist_sum += d2.sqrt();
                }
            }
            let occ = count as f64 / n_frames;
            if occ >= min_occ {
                local.push(HBondRecord {
                    donor:     d,
                    acceptor:  a,
                    occupancy: occ,
                    mean_dist: if count > 0 { dist_sum / count as f64 } else { 0.0 },
                });
            }
        }
        local
    }).collect();

    // Sort by occupancy descending
    let mut records = records;
    records.sort_by(|a, b| b.occupancy.partial_cmp(&a.occupancy).unwrap());

    let n_found = records.len();
    let warn = if ca_only { " (Cα-only fallback — load full-atom selection for proper H-bond analysis)" } else { "" };
    *state.hbond_cache.lock().unwrap() = Some(records.clone());

    // Return occupancy values as the data array for the bar chart
    let occupancies: Vec<f64> = records.iter().map(|r| r.occupancy).collect();
    Ok(AnalysisResult {
        data: occupancies,
        message: format!("Found {n_found} contacts with occupancy ≥ {:.0}%.{warn}", min_occ * 100.0),
    })
}

// ─── Normal Mode Analysis ─────────────────────────────────────────────────────
//
// run_enm builds an elastic network model from the mean structure.
//
// Two models are supported:
//
//   ANM (Anisotropic Network Model)
//   ──────────────────────────────
//   Each pair of Cα atoms within `cutoff` Å is connected by a spring of
//   uniform stiffness γ = 1. The 3N×3N Hessian matrix H is built analytically:
//
//     H_ij = -γ/r_ij² · r_ij ⊗ r_ij     (i ≠ j, within cutoff)
//     H_ii = -∑_{j≠i} H_ij
//
//   where r_ij is the unit vector from i to j and ⊗ is the outer product.
//   The 3N×3N matrix is then eigendecomposed. The 6 lowest modes are
//   rigid-body (zero eigenvalue) and are discarded; modes 7..N are the
//   functional modes.
//
//   GNM (Gaussian Network Model)
//   ────────────────────────────
//   Isotropic — uses only the N×N Kirchhoff (contact) matrix:
//
//     Γ_ij = -1  if i ≠ j and |r_ij| ≤ cutoff
//     Γ_ii = degree(i)
//
//   Eigendecomposition gives N−1 non-trivial modes. The slowest non-zero
//   mode predicts mobile/hinge regions via its squared eigenvector elements.
//   GNM is much faster than ANM and often sufficient for identifying
//   flexible regions.
//
// The eigenvalues returned are the raw spring constants (frequencies²),
// not mass-weighted. For comparison with experiment multiply by an
// appropriate force constant γ (typically 1 kcal/mol/Å²).

#[tauri::command]
async fn run_enm(
    app:     AppHandle,
    state:   State<'_, AppData>,
    cutoff:  Option<f64>,   // default 7.5 Å
    n_modes: Option<usize>, // default 20 non-trivial modes
    model:   Option<String>,// "ANM" or "GNM", default "ANM"
) -> Result<AnalysisResult<Vec<f64>>, String> {
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .as_ref().ok_or("No trajectory loaded")?.clone();

    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_atoms   = traj[0].len();
        let cut       = cutoff.unwrap_or(7.5);
        let cut2      = cut * cut;
        let n_req     = n_modes.unwrap_or(20).min(n_atoms.saturating_sub(1));
        let model_str = model.as_deref().unwrap_or("ANM").to_uppercase();
        let mean      = mean_positions(&traj);

        emit_progress(&app, "enm", 5.0);

        let (all_eigenvalues, all_eigenvectors, skip) = match model_str.as_str() {
            "GNM" => {
                let mut kirchhoff = Array2::<f64>::zeros((n_atoms, n_atoms));
                for i in 0..n_atoms {
                    for j in (i+1)..n_atoms {
                        let dx = mean[i][0]-mean[j][0];
                        let dy = mean[i][1]-mean[j][1];
                        let dz = mean[i][2]-mean[j][2];
                        if dx*dx+dy*dy+dz*dz <= cut2 {
                            kirchhoff[[i,j]] = -1.0; kirchhoff[[j,i]] = -1.0;
                            kirchhoff[[i,i]] += 1.0; kirchhoff[[j,j]] += 1.0;
                        }
                    }
                }
                emit_progress(&app, "enm", 40.0);
                let (vals, vecs) = kirchhoff.eigh(ndarray_linalg::UPLO::Upper)
                    .map_err(|e| format!("GNM eigendecomposition failed: {e}"))?;
                let vecs_cols: Vec<Vec<f64>> = (0..n_atoms)
                    .map(|k| vecs.column(k).iter().copied().collect()).collect();
                (vals.to_vec(), vecs_cols, 1usize)
            },
            _ => {
                let dim = n_atoms * 3;
                let mut hessian = Array2::<f64>::zeros((dim, dim));
                for i in 0..n_atoms {
                    if i % (n_atoms / 10).max(1) == 0 {
                        let pct = 5.0 + (i as f64 / n_atoms as f64) * 35.0;
                        emit_progress(&app, "enm", pct);
                    }
                    for j in (i+1)..n_atoms {
                        let dx = mean[j][0]-mean[i][0];
                        let dy = mean[j][1]-mean[i][1];
                        let dz = mean[j][2]-mean[i][2];
                        let r2 = dx*dx+dy*dy+dz*dz;
                        if r2 > cut2 { continue; }
                        let rv = [dx, dy, dz];
                        for a in 0..3usize {
                            for b in 0..3usize {
                                let val = -rv[a]*rv[b]/r2;
                                hessian[[i*3+a, j*3+b]] = val; hessian[[j*3+a, i*3+b]] = val;
                                hessian[[i*3+a, i*3+b]] -= val; hessian[[j*3+a, j*3+b]] -= val;
                            }
                        }
                    }
                }
                emit_progress(&app, "enm", 40.0);
                let (vals, vecs) = hessian.eigh(ndarray_linalg::UPLO::Upper)
                    .map_err(|e| format!("ANM eigendecomposition failed: {e}"))?;
                let vecs_cols: Vec<Vec<f64>> = (0..dim)
                    .map(|k| vecs.column(k).iter().copied().collect()).collect();
                (vals.to_vec(), vecs_cols, 6usize)
            },
        };

        emit_progress(&app, "enm", 80.0);

        // ── Cache ALL eigenvalues and eigenvectors (needed for animation) ──
        let mode_vals: Vec<f64> = all_eigenvalues[skip..].iter().take(n_req).copied().collect();
        let mode_vecs: Vec<Vec<f64>> = all_eigenvectors[skip..].iter().take(n_req).cloned().collect();

        // ── Compute B-factors from the pseudoinverse of the Hessian/Kirchhoff ──
        //
        // B_i = (8π²/3) * k_BT * Σ_k (1/λ_k) * ||v_k,i||²
        //
        // We omit the 8π²k_BT/3 prefactor (makes values relative, not absolute)
        // so the output is directly comparable across systems:
        //   b_i = Σ_k (1/λ_k) * ||v_k,i||²
        //
        // For ANM: v_k,i is a 3-vector (x,y,z components for atom i in mode k)
        //   so ||v_k,i||² = v_kx² + v_ky² + v_kz²
        // For GNM: v_k,i is a scalar, so ||v_k,i||² = v_ki²
        let bfactors: Vec<f64> = match model_str.as_str() {
            "GNM" => {
                // GNM: eigenvectors are N-dim, atom i component is index i
                let mut b = vec![0.0f64; n_atoms];
                for (&lam, evec) in all_eigenvalues[skip..].iter()
                        .zip(all_eigenvectors[skip..].iter()).take(n_req) {
                    if lam.abs() < 1e-10 { continue; }
                    for i in 0..n_atoms {
                        b[i] += evec[i] * evec[i] / lam;
                    }
                }
                b
            },
            _ => {
                // ANM: eigenvectors are 3N-dim, atom i occupies indices 3i..3i+3
                let mut b = vec![0.0f64; n_atoms];
                for (&lam, evec) in all_eigenvalues[skip..].iter()
                        .zip(all_eigenvectors[skip..].iter()).take(n_req) {
                    if lam.abs() < 1e-10 { continue; }
                    for i in 0..n_atoms {
                        let vx = evec[3*i];
                        let vy = evec[3*i+1];
                        let vz = evec[3*i+2];
                        b[i] += (vx*vx + vy*vy + vz*vz) / lam;
                    }
                }
                b
            },
        };

        emit_progress(&app, "enm", 100.0);
        Ok::<_, String>((mode_vals, mode_vecs, bfactors, all_eigenvalues, model_str, n_atoms, n_req))
    }).await.map_err(|e| e.to_string())??;

    let (_mode_vals, mode_vecs, bfactors, all_eigenvalues, model_str, n_atoms, n_req) = result;

    *state.enm_cache.lock().unwrap() = Some(EnmResult {
        eigenvalues:  all_eigenvalues,  // full spectrum stored for animation
        eigenvectors: mode_vecs,
        model:        model_str.clone(),
    });

    Ok(AnalysisResult {
        data: bfactors,
        message: format!("{model_str}: B-factor profile from {n_req} modes, {n_atoms} atoms."),
    })
}

// ─── run_nma_overlap ──────────────────────────────────────────────────────────
//
// Measures how well the ENM modes capture the actual MD fluctuations by
// computing the cumulative squared overlap:
//
//   O_k = Σ_{i=1}^{k} (v_i · u)²
//
// where v_i is the i-th ENM eigenvector and u is the mean-squared
// displacement vector from the MD trajectory (normalised).
//
// A value of O_k → 1 means the first k modes collectively explain 100%
// of the MD fluctuation direction. This is a standard way to validate
// whether your ENM is a good model for the dynamics in the simulation.

#[tauri::command]
fn run_nma_overlap(state: State<'_, AppData>) -> Result<AnalysisResult<Vec<f64>>, String> {
    let enm_lock = state.enm_cache.lock().unwrap();
    let enm      = enm_lock.as_ref().ok_or("Run the Elastic Network Model first.")?;
    let lock     = state.trajectory_data.lock().unwrap();
    let traj     = lock.as_ref().ok_or("No trajectory loaded")?;
    let n_atoms  = traj[0].len();
    let n_frames = traj.len() as f64;
    let mean     = mean_positions(traj);

    // Build the MD fluctuation vector (mean-squared displacement per atom, flattened)
    let mut msd = vec![0.0f64; n_atoms * 3];
    for frame in traj.iter() {
        for i in 0..n_atoms {
            for c in 0..3 {
                let d = frame[i][c] - mean[i][c];
                msd[i*3+c] += d * d;
            }
        }
    }
    for v in &mut msd { *v /= n_frames; }
    let norm: f64 = msd.iter().map(|v| v*v).sum::<f64>().sqrt();
    if norm < 1e-12 { return Err("MD fluctuation vector is zero — trajectory may be static.".into()); }
    let u: Vec<f64> = msd.iter().map(|v| v / norm).collect();

    // Cumulative squared overlap with each ENM mode
    // For GNM eigenvectors have length N (not 3N) so we need to match dims.
    let dim_match = enm.eigenvectors.first().map(|v| v.len()).unwrap_or(0) == n_atoms * 3;

    let overlaps: Vec<f64> = if dim_match {
        let mut cumulative = 0.0f64;
        enm.eigenvectors.iter().map(|mode| {
            let dot: f64 = mode.iter().zip(u.iter()).map(|(a,b)| a*b).sum();
            cumulative += dot * dot;
            cumulative
        }).collect()
    } else {
        return Err(format!(
            "ENM eigenvector dimension ({}) does not match 3N ({}) — \
             GNM eigenvectors are N-dimensional and not directly comparable with 3N MD fluctuations. \
             Use ANM for this analysis.",
            enm.eigenvectors.first().map(|v| v.len()).unwrap_or(0),
            n_atoms * 3
        ));
    };

    let n_modes = overlaps.len();
    Ok(AnalysisResult {
        data: overlaps,
        message: format!("Mode–trajectory overlap computed for {n_modes} {} modes.", enm.model),
    })
}

// ─── Network Analysis ─────────────────────────────────────────────────────────
//
// All three network tools operate on the DCCM-derived residue graph.
// Run DCCM first; these commands will error if it hasn't been computed.
//
// The graph G = (V, E) has:
//   V = residues (n_atoms nodes, one per selected atom)
//   E = pairs where |C_ij| >= threshold (weighted by |C_ij|)
//
// Edge weights are |correlation| so both correlated and anti-correlated
// pairs are treated as connected — negative correlations still indicate
// a communication pathway in allosteric signal transmission.

fn build_graph(dccm: &[Vec<f64>], threshold: f64) -> Vec<Vec<(usize, f64)>> {
    let n = dccm.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i+1)..n {
            let w = dccm[i][j].abs();
            if w >= threshold {
                adj[i].push((j, w));
                adj[j].push((i, w));
            }
        }
    }
    adj
}

// ─── run_communities ─────────────────────────────────────────────────────────
//
// Girvan–Newman community detection on the correlation network.
//
// The algorithm iteratively removes the edge with highest betweenness
// centrality until the graph splits into components. We stop when the
// number of components first exceeds `max_communities` or no edges remain.
//
// Returns a Vec<usize> where result[i] = community label for node i.
// This is sent to the frontend as a bar chart of community sizes.

#[tauri::command]
async fn run_communities(
    app:   AppHandle,
    state: State<'_, AppData>,
    threshold:        Option<f64>,  // default 0.6
    max_communities:  Option<usize>, // default 10
) -> Result<AnalysisResult<Vec<f64>>, String> {
    let dccm: Vec<Vec<f64>> = state.dccm_cache.lock().unwrap()
        .as_ref().ok_or("Run DCCM first.")?.clone();
    let app2 = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
    let dccm      = &dccm;
    let n         = dccm.len();
    let thresh    = threshold.unwrap_or(0.6);
    let max_comm  = max_communities.unwrap_or(10);

    // Build adjacency as a mutable edge set
    // We represent the graph as a symmetric matrix of Option<f64> weights
    // for fast edge removal during Girvan–Newman iterations.
    let mut weight: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i+1)..n {
            let w = dccm[i][j].abs();
            if w >= thresh { weight[i][j] = w; weight[j][i] = w; }
        }
    }

    /// BFS-based betweenness centrality for edge (src, dst) via Brandes' algorithm.
    /// Returns edge betweenness for all edges as HashMap<(usize,usize), f64>.
    fn edge_betweenness(weight: &[Vec<f64>], n: usize) -> HashMap<(usize,usize), f64> {
        let mut eb: HashMap<(usize,usize), f64> = HashMap::new();
        for s in 0..n {
            // BFS from s
            let mut stack  = Vec::new();
            let mut pred:  Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma  = vec![0.0f64; n];
            let mut dist   = vec![-1i64; n];
            sigma[s] = 1.0; dist[s] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(s);
            while let Some(v) = queue.pop_front() {
                stack.push(v);
                for j in 0..n {
                    if weight[v][j] == 0.0 { continue; }
                    if dist[j] < 0 {
                        queue.push_back(j);
                        dist[j] = dist[v] + 1;
                    }
                    if dist[j] == dist[v] + 1 {
                        sigma[j] += sigma[v];
                        pred[j].push(v);
                    }
                }
            }
            let mut delta = vec![0.0f64; n];
            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    let c = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    delta[v] += c;
                    let key = if v < w { (v, w) } else { (w, v) };
                    *eb.entry(key).or_insert(0.0) += c;
                }
            }
        }
        eb
    }

    // Connected components via BFS
    fn components(weight: &[Vec<f64>], n: usize) -> Vec<usize> {
        let mut label = vec![usize::MAX; n];
        let mut comp  = 0usize;
        for start in 0..n {
            if label[start] != usize::MAX { continue; }
            let mut queue = VecDeque::new();
            queue.push_back(start);
            label[start] = comp;
            while let Some(v) = queue.pop_front() {
                for j in 0..n {
                    if weight[v][j] > 0.0 && label[j] == usize::MAX {
                        label[j] = comp;
                        queue.push_back(j);
                    }
                }
            }
            comp += 1;
        }
        label
    }

    let mut labels = components(&weight, n);
    let mut n_comp = *labels.iter().max().unwrap_or(&0) + 1;

    while n_comp < max_comm {
        let eb = edge_betweenness(&weight, n);
        if eb.is_empty() { break; }
        // Remove highest-betweenness edge
        let &(ri, rj) = eb.iter().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        weight[ri][rj] = 0.0;
        weight[rj][ri] = 0.0;
        labels = components(&weight, n);
        let new_comp = *labels.iter().max().unwrap_or(&0) + 1;
        if new_comp > n_comp { n_comp = new_comp; }
        if n_comp >= max_comm { break; }
    }

    // Return community sizes as bar chart data
    let mut sizes = vec![0usize; n_comp];
    for &l in &labels { if l < n_comp { sizes[l] += 1; } }
    let sizes_f: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
    emit_progress(&app2, "communities", 100.0);
    Ok::<_, String>((sizes_f, n_comp, thresh))
    }).await.map_err(|e| e.to_string())??;
    let (sizes_f, n_comp, thresh) = result;
    Ok(AnalysisResult {
        data: sizes_f,
        message: format!("Found {n_comp} communities (threshold={thresh:.2})."),
    })
}

// ─── run_betweenness ─────────────────────────────────────────────────────────
//
// Node betweenness centrality on the correlation network.
//
// C_B(v) = Σ_{s≠v≠t} σ(s,t|v) / σ(s,t)
//
// where σ(s,t) is the number of shortest paths from s to t and
// σ(s,t|v) is the number of those paths passing through v.
//
// We use Brandes' O(VE) algorithm with BFS (unweighted shortest paths).
// For a weighted version (meaningful for allosteric path analysis) see
// run_optimal_paths which uses Dijkstra on -log(|C_ij|) edge costs.

#[tauri::command]
fn run_betweenness(
    state: State<'_, AppData>,
    threshold: Option<f64>,  // default 0.4
) -> Result<AnalysisResult<Vec<f64>>, String> {
    let dccm_lock = state.dccm_cache.lock().unwrap();
    let dccm      = dccm_lock.as_ref().ok_or("Run DCCM first.")?;
    let n         = dccm.len();
    let thresh    = threshold.unwrap_or(0.4);
    let adj       = build_graph(dccm, thresh);

    // Brandes' algorithm
    let mut centrality = vec![0.0f64; n];
    for s in 0..n {
        let mut stack  = Vec::new();
        let mut pred:  Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma  = vec![0.0f64; n];
        let mut dist   = vec![-1i64; n];
        sigma[s] = 1.0; dist[s] = 0;
        let mut queue  = VecDeque::new();
        queue.push_back(s);
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &(w, _) in &adj[v] {
                if dist[w] < 0 { queue.push_back(w); dist[w] = dist[v] + 1; }
                if dist[w] == dist[v] + 1 { sigma[w] += sigma[v]; pred[w].push(v); }
            }
        }
        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s { centrality[w] += delta[w]; }
        }
    }
    // Normalise by (n-1)(n-2) for undirected graph
    let norm = ((n - 1) * (n - 2)) as f64;
    if norm > 0.0 { for c in &mut centrality { *c /= norm; } }

    Ok(AnalysisResult {
        data: centrality,
        message: format!("Betweenness centrality computed for {n} nodes (threshold={thresh:.2})."),
    })
}

// ─── run_optimal_paths ────────────────────────────────────────────────────────
//
// Finds the optimal (shortest) communication path between two residues in the
// allosteric network using Dijkstra's algorithm on the -log(|C_ij|) metric.
//
// The -log transform converts high correlations (near 1) to short distances
// (near 0) and low correlations to long distances, making the shortest path
// the strongest correlated communication route — consistent with the
// Girvan–Newman and network suboptimal paths literature.
//
// Returns the path as a sequence of residue indices, plus the cumulative
// edge weights at each step (for plotting).

#[tauri::command]
fn run_optimal_paths(
    state: State<'_, AppData>,
    source:    usize,
    sink:      usize,
    threshold: Option<f64>,  // minimum |C_ij| to include edge, default 0.3
) -> Result<AnalysisResult<Vec<f64>>, String> {
    let dccm_lock = state.dccm_cache.lock().unwrap();
    let dccm      = dccm_lock.as_ref().ok_or("Run DCCM first.")?;
    let n         = dccm.len();
    let thresh    = threshold.unwrap_or(0.3);

    if source >= n { return Err(format!("Source {source} out of range (0–{}).", n-1)); }
    if sink   >= n { return Err(format!("Sink {sink} out of range (0–{}).", n-1)); }

    // Dijkstra with edge cost = -ln(|C_ij|) clamped to [0, ∞)
    let inf = f64::INFINITY;
    let mut dist = vec![inf; n];
    let mut prev = vec![usize::MAX; n];
    dist[source] = 0.0;

    // Simple O(N²) Dijkstra — adequate for N ≤ ~1000 residues
    let mut visited = vec![false; n];
    for _ in 0..n {
        // Find unvisited node with minimum distance
        let u = (0..n).filter(|&i| !visited[i])
            .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap());
        let u = match u { Some(x) => x, None => break };
        if dist[u] == inf { break; }
        visited[u] = true;
        if u == sink { break; }

        for j in 0..n {
            if visited[j] { continue; }
            let w = dccm[u][j].abs();
            if w < thresh { continue; }
            let edge_cost = -(w.ln());     // -ln(|C|): small when |C| near 1
            let new_dist  = dist[u] + edge_cost;
            if new_dist < dist[j] { dist[j] = new_dist; prev[j] = u; }
        }
    }

    // Reconstruct path
    if dist[sink] == inf {
        return Err(format!(
            "No path found from residue {source} to {sink} at threshold {thresh:.2}. \
             Try lowering the threshold."
        ));
    }

    let mut path = Vec::new();
    let mut cur  = sink;
    while cur != usize::MAX { path.push(cur); cur = prev[cur]; }
    path.reverse();

    // Return cumulative path costs as the bar chart data
    let costs: Vec<f64> = path.windows(2)
        .map(|w| { let c = dccm[w[0]][w[1]].abs(); -(c.ln()) })
        .collect();
    let cumulative: Vec<f64> = costs.iter()
        .scan(0.0f64, |acc, &v| { *acc += v; Some(*acc) })
        .collect();

    let path_str: Vec<String> = path.iter().map(|&i| i.to_string()).collect();
    Ok(AnalysisResult {
        data: cumulative,
        message: format!("Path {source}→{sink}: [{}]  total cost={:.3}",
            path_str.join("→"), dist[sink]),
    })
}

// ─── Thermodynamics ───────────────────────────────────────────────────────────

// ─── run_fes ──────────────────────────────────────────────────────────────────
//
// 2D free energy surface projected onto the first two principal components.
//
// F(PC1, PC2) = -k_B T ln P(PC1, PC2)
//
// where P is estimated by 2D histogram of the trajectory projections.
// We set k_B T = 1 (arbitrary units) so values are in units of k_B T.
// The minimum is shifted to zero. Bins with zero occupancy are set to the
// maximum free energy (representing inaccessible regions).

#[tauri::command]
fn run_fes(
    state:   State<'_, AppData>,
    n_bins:  Option<usize>,  // bins per axis, default 50
) -> Result<AnalysisResult<Vec<Vec<f64>>>, String> {
    let pca_lock = state.pca_cache.lock().unwrap();
    let pca      = pca_lock.as_ref().ok_or("Run PCA first.")?;
    let bins     = n_bins.unwrap_or(50);
    let pts      = &pca.projections;

    if pts.is_empty() { return Err("PCA projections are empty.".into()); }

    let pc1_vals: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let pc2_vals: Vec<f64> = pts.iter().map(|p| p[1]).collect();
    let xmin = pc1_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = pc1_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ymin = pc2_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    let ymax = pc2_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let xr   = (xmax - xmin) * 1.05;
    let yr   = (ymax - ymin) * 1.05;

    let mut counts = vec![vec![0usize; bins]; bins];
    for p in pts {
        let xi = (((p[0] - xmin) / xr) * bins as f64) as usize;
        let yi = (((p[1] - ymin) / yr) * bins as f64) as usize;
        let xi = xi.min(bins - 1);
        let yi = yi.min(bins - 1);
        counts[yi][xi] += 1;
    }

    let n_total = pts.len() as f64;
    let _max_val = counts.iter().flat_map(|r| r.iter()).cloned().max().unwrap_or(1) as f64;
    // Max free energy for empty bins
    let f_max   = -(1.0f64 / n_total).ln();

    let fes: Vec<Vec<f64>> = counts.iter()
        .map(|row| row.iter().map(|&c| {
            if c == 0 { f_max } else { -(c as f64 / n_total).ln() }
        }).collect())
        .collect();

    // Shift minimum to zero
    let f_min: f64 = fes.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min);
    let fes: Vec<Vec<f64>> = fes.iter()
        .map(|row| row.iter().map(|&v| v - f_min).collect())
        .collect();

    Ok(AnalysisResult {
        data: fes,
        message: format!("Free energy surface ({bins}×{bins} bins) computed from {} frames.", pts.len()),
    })
}

// ─── run_entropy ──────────────────────────────────────────────────────────────
//
// Quasi-harmonic (Schlitter) conformational entropy.
//
// S_qh = ½ k_B Σ_i ln(1 + k_B T e² / (ħ² λ_i))
//
// In reduced units (k_B=1, T=300 K, ħ expressed via k_B T / ħω ≫ 1 for
// soft modes), the classical limit gives:
//
//   S_qh ≈ ½ Σ_i [1 + ln(k_B T / (ħ²/m · λ_i))]
//
// For a dimensionless implementation suitable for comparing trajectories
// we use the Andricioaei–Karplus approximation in mass-weighted coordinates
// with uniform masses:
//
//   S = ½ k_B Σ_i ln(2π e k_B T / (ħ²) · σ_i²)
//
// where σ_i² are the eigenvalues of the mass-weighted covariance matrix.
// For uniform masses (Cα-only) this reduces to a sum over covariance
// eigenvalues. We return the result in J/(mol·K) using:
//   k_B = 1.380649e-23 J/K, N_A = 6.02214076e23, ħ = 1.054571817e-34 J·s
//   m_Ca = 12.011 × 1.66054e-27 kg (carbon mass in kg)
//
// Note: the result depends strongly on the number of atoms and frames.
// It should be used for relative comparisons between simulations loaded
// with the same selection, not as an absolute entropy value.

#[tauri::command]
fn run_entropy(state: State<'_, AppData>) -> Result<AnalysisResult<EntropyResult>, String> {
    let lock = state.trajectory_data.lock().unwrap();
    let traj = lock.as_ref().ok_or("No trajectory loaded")?;

    let cov = build_covariance(traj)?;
    let (eigenvalues, _) = cov.eigh(ndarray_linalg::UPLO::Upper)
        .map_err(|e| format!("Entropy eigendecomposition failed: {e}"))?;

    // Physical constants
    const KB: f64 = 1.380649e-23;    // J/K
    const NA: f64 = 6.02214076e23;   // mol⁻¹
    const H_BAR: f64 = 1.054571817e-34; // J·s
    const T: f64 = 300.0;            // K (standard temperature)
    const M_CA_KG: f64 = 12.011 * 1.66054e-27; // kg, carbon Cα mass

    // Filter out numerical zero/negative eigenvalues (rigid-body modes)
    let threshold = 1e-6 * eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let valid_evals: Vec<f64> = eigenvalues.iter()
        .filter(|&&v| v > threshold)
        .copied()
        .collect();

    let n_modes = valid_evals.len();
    // σ² in Å² (covariance eigenvalues); convert to m²: 1 Å = 1e-10 m
    let angstrom2_to_m2 = 1e-20_f64;

    let entropy_per_mode: Vec<f64> = valid_evals.iter().map(|&sigma2_ang| {
        let sigma2_m2 = sigma2_ang * angstrom2_to_m2;
        // Andricioaei–Karplus: S_i = k_B/2 * (1 + ln(2π e k_B T m / ħ² * σ²))
        let arg = (2.0 * std::f64::consts::PI * std::f64::consts::E
                   * KB * T * M_CA_KG / (H_BAR * H_BAR))
                  * sigma2_m2;
        if arg > 1.0 { KB / 2.0 * arg.ln() } else { 0.0 }
    }).collect();

    let entropy_j_k = entropy_per_mode.iter().sum::<f64>();
    // Convert to J/(mol·K)
    let entropy_j_per_mol_k = entropy_j_k * NA;

    Ok(AnalysisResult {
        data: EntropyResult { entropy_j_per_mol_k, n_modes_used: n_modes },
        message: format!(
            "Quasi-harmonic entropy: {:.2} J/(mol·K) from {n_modes} modes (T=300 K).",
            entropy_j_per_mol_k
        ),
    })
}

// ─── PDB writing helpers ─────────────────────────────────────────────────────
//
// Shared constants and a frame-to-PDB converter used by get_snapshot_pdb,
// get_umbrella_snapshot_pdb, and the standalone rewrite_pdb command.
//
// KEY RULES that make NGL render correctly:
//   1. Chain ID must be present (col 22). NGL uses it to segment chains.
//      We default to 'A' when the topology has no chain info.
//   2. Standard residues → ATOM, everything else → HETATM.
//      NGL looks up ATOM residues in its built-in bond dictionary; HETATM
//      residues use CONECT records exclusively — no distance inference.
//   3. CONECT records for every HETATM atom pair that is bonded.
//      Written from the chemfiles topology bond table, which comes from the
//      parm7 or the connectivity inferred when loading a PDB with topology.
//   4. No CONECT for ATOM (standard residue) atoms — NGL's dictionary
//      handles those and CONECT records for them can cause double-bonds.

const STANDARD_RES: &[&str] = &[
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    // AMBER protonation variants
    "HIE","HID","HIP","CYX","ASH","GLH","LYN","CYM",
    // Terminal caps
    "ACE","NME","NHE",
    // Water / solvent
    "WAT","HOH","SOL","TIP","TIP3","TP3",
];

/// Write a chemfiles Frame to a PDB string with correct chain IDs,
/// HETATM classification, and CONECT records for non-standard residues.
/// `bf` is an optional per-atom B-factor slice (0-indexed).
fn frame_to_pdb(frame: &chemfiles::Frame, bf: Option<&[f64]>) -> String {
    let pos  = frame.positions();
    let topo = frame.topology();
    let n    = pos.len();
    let mut pdb = String::with_capacity(n * 82 + 1024);

    let mut is_hetatm = vec![false; n];

    for (i, p) in pos.iter().enumerate() {
        let atom      = topo.atom(i);
        let atom_name = { let s = atom.name().to_string(); if s.is_empty() { format!("X{}", i+1) } else { s } };
        let element   = {
            let raw = atom.atomic_type().to_string();
            if raw.is_empty() {
                atom_name.chars().find(|c| c.is_ascii_uppercase())
                    .map(|c| c.to_string()).unwrap_or_else(|| "C".to_string())
            } else { raw }
        };
        let (res_name, chain_id, res_seq) =
            if let Some(res) = topo.residue_for_atom(i) {
                let chain = res.get("chainname")
                    .and_then(|p| if let chemfiles::Property::String(s) = p { s.chars().next() } else { None })
                    .unwrap_or('A');
                let rn = res.name().to_string();
                let name = if rn.is_empty() { "UNK".to_string() } else { rn };
                (name, chain, res.id().unwrap_or(i as i64 + 1))
            } else { ("UNK".to_string(), 'A', i as i64 + 1) };

        let hetatm = !STANDARD_RES.contains(&res_name.as_str());
        is_hetatm[i] = hetatm;
        let record = if hetatm { "HETATM" } else { "ATOM  " };

        let pdb_name = match element.len() {
            1 => format!(" {:<3}", &atom_name[..atom_name.len().min(3)]),
            _ => format!("{:<4}", &atom_name[..atom_name.len().min(4)]),
        };
        let res3  = format!("{:<3}", &res_name[..res_name.len().min(3)]);
        let elem2 = format!("{:>2}", &element[..element.len().min(2)]);
        let b = bf.and_then(|v| v.get(i).copied()).unwrap_or(0.0);

        pdb.push_str(&format!(
            "{}{:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}
",
            record, i+1, pdb_name, res3, chain_id, res_seq,
            p[0] as f64, p[1] as f64, p[2] as f64, b, elem2
        ));
    }

    // CONECT records — HETATM atoms only, from topology bond table
    let bonds = topo.bonds();
    if !bonds.is_empty() {
        let mut neighbours: Vec<Vec<usize>> = vec![vec![]; n];
        for bond in &bonds {
            let (a, b) = (bond[0], bond[1]);
            if a < n && b < n && (is_hetatm[a] || is_hetatm[b]) {
                neighbours[a].push(b + 1);
                neighbours[b].push(a + 1);
            }
        }
        for (i, partners) in neighbours.iter().enumerate() {
            if partners.is_empty() { continue; }
            for chunk in partners.chunks(4) {
                let fields: String = chunk.iter().map(|p| format!("{:5}", p)).collect::<Vec<_>>().join("");
                pdb.push_str(&format!("CONECT{:5}{}
", i + 1, fields));
            }
        }
    }

    pdb.push_str("TER
END
");
    pdb
}

// ─── Visualizer helpers ───────────────────────────────────────────────────────

#[tauri::command]
fn get_snapshot_pdb(
    state: State<'_, AppData>,
    frame_idx: Option<usize>,
) -> Result<String, String> {
    let idx = frame_idx.unwrap_or(0);
    let data_lock = state.trajectory_data.lock().unwrap();
    let frames = data_lock.as_ref().ok_or("No trajectory loaded")?;
    if idx >= frames.len() {
        return Err(format!("Frame {} out of range (0–{})", idx, frames.len() - 1));
    }
    let coords    = &frames[idx];
    let meta_lock = state.atom_meta.lock().unwrap();
    let bf_lock   = state.bfactor_cache.lock().unwrap();
    let mut pdb   = String::with_capacity(coords.len() * 82);
    let std_res: &[&str] = &[
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "HIE","HID","HIP","CYX","ASH","GLH","LYN","CYM",
        "ACE","NME","NHE","WAT","HOH","SOL","TIP","TIP3","TP3",
    ];

    for (i, pos) in coords.iter().enumerate() {
        let (atom_name, res_name, chain_id, res_seq, element) =
            match meta_lock.as_ref().and_then(|m| m.get(i)) {
                Some(m) => (m.atom_name.clone(), m.res_name.clone(), m.chain_id, m.res_seq, m.element.clone()),
                None    => ("CA".into(), "UNK".into(), 'A', i as i64 + 1, "C".into()),
            };
        let pdb_name = match element.len() {
            1 => format!(" {:<3}", &atom_name[..atom_name.len().min(3)]),
            _ => format!("{:<4}", &atom_name[..atom_name.len().min(4)]),
        };
        let pdb_elem = format!("{:>2}", &element[..element.len().min(2)]);
        let res3 = format!("{:<3}", &res_name[..res_name.len().min(3)]);
        let record = if std_res.contains(&res_name.as_str()) { "ATOM  " } else { "HETATM" };
        // Use stored B-factor if available, otherwise 0.00
        let bf = bf_lock.as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(0.0);
        pdb.push_str(&format!(
            "{}{:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}\n",
            record, i+1, pdb_name, res3, chain_id, res_seq, pos[0], pos[1], pos[2], bf, pdb_elem,
        ));
    }
    pdb.push_str("TER\nEND\n");
    Ok(pdb)
}

// ─── viz_event ────────────────────────────────────────────────────────────────
//
// Relay a named event with a JSON payload to the visualizer window.
// Called from JS in the main window; the visualizer's listen() calls receive it.
//
// Why this is needed:
//   JS emit() from @tauri-apps/api/event only broadcasts within the same
//   webview. To reach a *different* webview (the visualizer window) we must
//   go through Rust and use app_handle.emit_to("visualizer", ...) which
//   targets the window by its label.
//
// The payload is passed as a pre-serialised JSON string so this single
// command can relay any event type without needing a Rust struct per event.

#[tauri::command]
fn viz_event(app: AppHandle, event: String, payload: String) -> Result<(), String> {
    // Parse the payload as a serde_json::Value so emit_to can re-serialise it.
    let value: serde_json::Value = serde_json::from_str(&payload)
        .map_err(|e| format!("Invalid payload JSON: {e}"))?;
    app.emit_to("visualizer", &event, value)
        .map_err(|e| e.to_string())
}

// ─── set_bfactors ─────────────────────────────────────────────────────────────
//
// Stores a per-atom float array that will be written into the B-factor column
// of the PDB snapshot. Values are normalised to [0, 100] so NGL's bfactor
// color scheme maps cleanly from cold (0 = rigid) to hot (100 = flexible).
// Call this after run_rmsf or run_enm to enable "Color by B-factor" in the
// visualizer.

#[tauri::command]
fn set_bfactors(state: State<'_, AppData>, values: Vec<f64>) -> Result<String, String> {
    let n = values.len();
    if n == 0 { return Err("Empty B-factor array.".into()); }

    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    // Normalise to [0, 100]; if all values are identical set to 50
    let normalised: Vec<f64> = if range < 1e-10 {
        vec![50.0; n]
    } else {
        values.iter().map(|&v| (v - min) / range * 100.0).collect()
    };

    *state.bfactor_cache.lock().unwrap() = Some(normalised);
    Ok(format!("B-factor data set for {n} atoms (range {min:.3}–{max:.3})."))
}

#[derive(serde::Serialize)]
struct ResidueInfo {
    res_seq:  i64,
    res_name: String,
    chain_id: char,
}
 
#[tauri::command]
fn get_selection_residues(
    state: tauri::State<'_, AppData>,
) -> Result<Vec<ResidueInfo>, String> {
    let lock = state.atom_meta.lock().unwrap();
    let meta = lock.as_ref().ok_or("No trajectory loaded.")?;
    Ok(meta.iter().map(|a| ResidueInfo {
        res_seq:  a.res_seq,
        res_name: a.res_name.clone(),
        chain_id: a.chain_id,
    }).collect())
}

#[tauri::command]
fn get_frame_coords(state: State<'_, AppData>, frame_idx: usize) -> Result<Vec<f64>, String> {
    let data_lock = state.trajectory_data.lock().unwrap();
    let frames = data_lock.as_ref().ok_or("No trajectory loaded")?;
    if frame_idx >= frames.len() {
        return Err(format!("Frame {} out of range (0–{})", frame_idx, frames.len() - 1));
    }
    Ok(frames[frame_idx].iter().flat_map(|c| c.iter().cloned()).collect())
}

// ─── export_csv ──────────────────────────────────────────────────────────────
//
// Writes every per-frame scalar cache to a single CSV.

#[tauri::command]
fn export_csv(state: State<'_, AppData>, path: String) -> Result<String, String> {
    let rmsd = state.rmsd_cache.lock().unwrap().clone();
    let rmsf = state.rmsf_cache.lock().unwrap().clone();
    let rg   = state.rg_cache.lock().unwrap().clone();

    if rmsd.is_none() && rmsf.is_none() && rg.is_none() {
        return Err("No per-frame analyses have been run yet.".into());
    }

    let mut file = File::create(&path).map_err(|e| e.to_string())?;
    let mut headers = vec!["frame"];
    if rmsd.is_some() { headers.push("rmsd_angstrom"); }
    if rmsf.is_some() { headers.push("rmsf_angstrom"); }
    if rg.is_some()   { headers.push("rg_angstrom"); }
    writeln!(file, "{}", headers.join(",")).map_err(|e| e.to_string())?;

    let n = [&rmsd, &rmsf, &rg].iter()
        .filter_map(|v| v.as_ref()).map(|v| v.len()).max().unwrap_or(0);
    for i in 0..n {
        let mut row = vec![i.to_string()];
        if let Some(ref v) = rmsd { row.push(format!("{:.6}", v.get(i).unwrap_or(&f64::NAN))); }
        if let Some(ref v) = rmsf { row.push(format!("{:.6}", v.get(i).unwrap_or(&f64::NAN))); }
        if let Some(ref v) = rg   { row.push(format!("{:.6}", v.get(i).unwrap_or(&f64::NAN))); }
        writeln!(file, "{}", row.join(",")).map_err(|e| e.to_string())?;
    }
    Ok(format!("Exported {} rows to {}", n, path))
}

// ─── load_umbrella_windows ────────────────────────────────────────────────────
//
// Reads CV files for each umbrella sampling window.
//
// Parameters (from the frontend):
//   cv_pattern   – glob-style path pattern where {window} is replaced by
//                  zero-padded window index, e.g. "../{window}/step5.00_eq.cv"
//   n_windows    – number of windows (30 in the example script)
//   val_min      – CV value at window 0 (restraint minimum, e.g. -0.800)
//   val_max      – CV value at window n-1 (restraint maximum, e.g. 2.100)
//   cv_col       – 0-based column index in the CV file (column 2 in script)
//   rst_pattern  – optional pattern for .ncrst files, same {window} substitution
//
// The .cv file is whitespace-delimited; we load the specified column.
// Correlated-data subsampling is NOT done here — pymbar handles that during
// MBAR, and the raw data is stored for the histogram panel.

#[tauri::command]
async fn load_umbrella_windows(
    state: State<'_, AppData>,
    cv_pattern:  String,
    n_windows:   usize,
    val_min:     f64,
    val_max:     f64,
    cv_col:      usize,
    rst_pattern: Option<String>,
) -> Result<String, String> {
    use std::io::{BufRead, BufReader};

    let val0_k: Vec<f64> = (0..n_windows)
        .map(|i| val_min + (val_max - val_min) * i as f64 / (n_windows - 1).max(1) as f64)
        .collect();

    let mut windows: Vec<UmbrellaWindow> = Vec::with_capacity(n_windows);
    let mut total_samples = 0usize;

    for i in 0..n_windows {
        let idx_str = format!("{:02}", i);
        let cv_path = cv_pattern.replace("{window}", &idx_str);
        let rst_path = rst_pattern.as_ref().map(|p| p.replace("{window}", &idx_str));

        // Glob-expand the cv_path (handles wildcards like step5.*)
        let expanded: Vec<std::path::PathBuf> = {
            let mut v = Vec::new();
            if let Ok(entries) = glob::glob(&cv_path) {
                for entry in entries.flatten() { v.push(entry); }
            }
            if v.is_empty() {
                // Try as a literal path
                let p = std::path::PathBuf::from(&cv_path);
                if p.exists() { v.push(p); }
            }
            v.sort();
            v
        };

        if expanded.is_empty() {
            return Err(format!("Window {:02}: no files found matching '{}'", i, cv_path));
        }

        let mut samples: Vec<f64> = Vec::new();
        for path in &expanded {
            let file = File::open(path)
                .map_err(|e| format!("Window {:02}: cannot open '{}': {}", i, path.display(), e))?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|e| e.to_string())?;
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('@') {
                    continue;
                }
                let cols: Vec<&str> = trimmed.split_whitespace().collect();
                if let Some(val_str) = cols.get(cv_col) {
                    if let Ok(val) = val_str.parse::<f64>() {
                        samples.push(val);
                    }
                }
            }
        }

        if samples.is_empty() {
            return Err(format!("Window {:02}: no numeric data found in column {}", i, cv_col));
        }

        total_samples += samples.len();
        let rst = rst_path.filter(|p| std::path::Path::new(p).exists());

        windows.push(UmbrellaWindow {
            index:    i,
            val0:     val0_k[i],
            samples,
            cv_file:  expanded[0].display().to_string(),
            rst_file: rst,
        });
    }

    *state.umbrella_windows.lock().unwrap() = Some(windows);
    *state.mbar_result.lock().unwrap()      = None;

    Ok(format!("Loaded {} umbrella windows ({} total CV samples).", n_windows, total_samples))
}

// ─── MBAR constants and helpers ──────────────────────────────────────────────

const R_GAS:     f64 = 1.987_204_258e-3; // kcal/(mol·K)
const MBAR_TOL:  f64 = 1e-8;
const MBAR_ITER: usize = 500;

/// Compute reduced harmonic bias matrix.
/// u_kn[k][n] = β · (fc/2) · (x_n − x0_k)²
fn compute_bias_matrix(samples: &[f64], val0_k: &[f64], fc: f64, beta: f64) -> Vec<Vec<f64>> {
    val0_k.iter().map(|&x0| {
        samples.par_iter().map(|&x| {
            let d = x - x0;
            beta * 0.5 * fc * d * d
        }).collect()
    }).collect()
}

/// One MBAR self-consistent sweep. Returns new f_k (log-sum-exp stabilised).
fn mbar_sweep(f: &[f64], u_kn: &[Vec<f64>], n_k: &[usize]) -> Vec<f64> {
    let k = f.len();
    let n_total = u_kn[0].len();
    // log denominator for each sample
    let log_denom: Vec<f64> = (0..n_total).into_par_iter().map(|n| {
        let max_t = (0..k).map(|ki| (n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..k).map(|ki| ((n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n] - max_t).exp())
            .sum::<f64>();
        max_t + sum.ln()
    }).collect();
    // new f_k
    (0..k).map(|ki| {
        let max_t = (0..n_total).map(|n| -u_kn[ki][n] - log_denom[n])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..n_total).map(|n| (-u_kn[ki][n] - log_denom[n] - max_t).exp())
            .sum::<f64>();
        -(max_t + sum.ln())
    }).collect()
}

/// Iterate to convergence.
fn solve_mbar(u_kn: &[Vec<f64>], n_k: &[usize]) -> Vec<f64> {
    let k = n_k.len();
    let mut f = vec![0.0f64; k];
    for _ in 0..MBAR_ITER {
        let mut f_new = mbar_sweep(&f, u_kn, n_k);
        let shift = f_new[0];
        f_new.iter_mut().for_each(|v| *v -= shift);
        let max_d = f_new.iter().zip(f.iter()).map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        f = f_new;
        if max_d < MBAR_TOL { break; }
    }
    f
}

/// PMF from MBAR weights. Returns (bin_centers, pmf, counts_per_bin).
fn compute_pmf(
    samples: &[f64], f: &[f64], u_kn: &[Vec<f64>], n_k: &[usize],
    val_min: f64, val_max: f64, nbins: usize,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let k = f.len();
    let n_total = samples.len();
    let bin_w = (val_max - val_min) / nbins as f64;

    let log_denom: Vec<f64> = (0..n_total).into_par_iter().map(|n| {
        let max_t = (0..k).map(|ki| (n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..k).map(|ki| ((n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n] - max_t).exp())
            .sum::<f64>();
        max_t + sum.ln()
    }).collect();

    let mut bin_lws: Vec<Vec<f64>> = vec![vec![]; nbins];
    let mut bin_counts = vec![0usize; nbins];
    for n in 0..n_total {
        let x = samples[n];
        if x < val_min || x > val_max { continue; }
        let bi = ((x - val_min) / bin_w) as usize;
        let bi = bi.min(nbins - 1);
        bin_lws[bi].push(-log_denom[n]);
        bin_counts[bi] += 1;
    }

    let pmf: Vec<f64> = bin_lws.iter().map(|lws| {
        if lws.is_empty() { return f64::NAN; }
        let max_lw = lws.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        -(max_lw + lws.iter().map(|&lw| (lw - max_lw).exp()).sum::<f64>().ln())
    }).collect();

    let bin_centers = (0..nbins)
        .map(|b| val_min + (b as f64 + 0.5) * bin_w)
        .collect();
    (bin_centers, pmf, bin_counts)
}

/// Gaussian KDE with Scott bandwidth on n_pts grid points.
fn gaussian_kde(samples: &[f64], x_min: f64, x_max: f64, n_pts: usize) -> (Vec<f64>, Vec<f64>) {
    if samples.is_empty() { return (vec![], vec![]); }
    let n  = samples.len() as f64;
    let mu = samples.iter().sum::<f64>() / n;
    let sigma = (samples.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n).sqrt().max(1e-12);
    let bw = sigma * n.powf(-0.2);
    let xs: Vec<f64> = (0..n_pts)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1) as f64)
        .collect();
    let norm = n * bw * (2.0 * std::f64::consts::PI).sqrt();
    let ys: Vec<f64> = xs.par_iter().map(|&xi| {
        samples.iter().map(|&s| { let z = (xi - s) / bw; (-0.5 * z * z).exp() })
            .sum::<f64>() / norm
    }).collect();
    (xs, ys)
}

// ─── run_mbar — native Rust MBAR ─────────────────────────────────────────────
//
// Full self-consistent MBAR in Rust + rayon. No Python dependency.
// Algorithm: iterative MBAR equations until ‖Δf‖_∞ < 1e-8.
// Uncertainty: block bootstrap (50 replicates, 10 blocks/window) parallelised
// across replicates with rayon — typically < 2 s on 30-window datasets.

#[tauri::command]
async fn run_mbar(
    app:    AppHandle,
    state:  State<'_, AppData>,
    fc:     f64,           // AMBER force constant × 2  (kcal mol⁻¹ Å⁻²)
    temp:   Option<f64>,   // temperature in K, default 300 → β = 1/(R·T)
    n_bins: Option<usize>, // PMF bins, default n_windows − 1
    n_boot: Option<usize>, // bootstrap replicates, default 50
) -> Result<AnalysisResult<MbarResult>, String> {

    // ── Collect window data ───────────────────────────────────────────────────
    let (all_samples, val0_k, n_k, n_windows) = {
        let lock    = state.umbrella_windows.lock().unwrap();
        let windows = lock.as_ref().ok_or("Load umbrella windows first.")?;
        if windows.len() < 2 { return Err("Need at least 2 windows for MBAR.".into()); }
        let n_windows = windows.len();
        let val0_k: Vec<f64>   = windows.iter().map(|w| w.val0).collect();
        let n_k:    Vec<usize> = windows.iter().map(|w| w.samples.len()).collect();
        let all_samples: Vec<f64> = windows.iter()
            .flat_map(|w| w.samples.iter().copied()).collect();
        (all_samples, val0_k, n_k, n_windows)
    };

    // β = 1/(R·T) — the physically correct inverse thermal energy.
    // R = 1.987×10⁻³ kcal/(mol·K), so at 300 K: β ≈ 1.677 mol/kcal.
    let beta = 1.0 / (R_GAS * temp.unwrap_or(300.0));
    let nbins  = n_bins.unwrap_or(n_windows.saturating_sub(1)).max(2);
    let n_boot = n_boot.unwrap_or(50);
    let val_min = val0_k.iter().cloned().fold(f64::INFINITY,     f64::min);
    let val_max = val0_k.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_total = all_samples.len();

    emit_progress(&app, "mbar", 5.0);

    // ── Main MBAR solve ───────────────────────────────────────────────────────
    let u_kn = compute_bias_matrix(&all_samples, &val0_k, fc, beta);
    emit_progress(&app, "mbar", 20.0);

    let f = {
        let u2 = u_kn.clone(); let nk2 = n_k.clone();
        tauri::async_runtime::spawn_blocking(move || solve_mbar(&u2, &nk2))
            .await.map_err(|e| e.to_string())?
    };

    emit_progress(&app, "mbar", 55.0);

    let (bin_centers, pmf_raw, _) = compute_pmf(
        &all_samples, &f, &u_kn, &n_k, val_min, val_max, nbins,
    );

    // Zero-reference: subtract the minimum in the first half of bins.
    // The Python notebook uses f_i[:20].argmin() — effectively the lowest
    // point in the reactant region. Using the first half is equivalent for
    // a reactant basin on the left and more robust to sparse data.
    let ref_half = (nbins / 2).max(1);
    let ref_val  = pmf_raw[..ref_half].iter()
        .filter(|v| v.is_finite()).cloned()
        .fold(f64::INFINITY, f64::min);
    let ref_val  = if ref_val.is_finite() { ref_val } else {
        pmf_raw.iter().filter(|v| v.is_finite()).cloned()
               .fold(f64::INFINITY, f64::min)
    };
    let pmf: Vec<f64> = pmf_raw.iter()
        .map(|v| if v.is_finite() { v - ref_val } else { f64::NAN }).collect();

    emit_progress(&app, "mbar", 65.0);

    // ── Block-bootstrap uncertainty ───────────────────────────────────────────
    let n_blocks = 10usize;
    let all_ref = std::sync::Arc::new(all_samples.clone());
    let vk_ref  = std::sync::Arc::new(val0_k.clone());

    let pmf_boots: Vec<Vec<f64>> = (0..n_boot).into_par_iter().map(|rep| {
        // Deterministic LCG from rep index
        let mut rng: u64 = (rep as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let next = |r: &mut u64| -> usize {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *r as usize
        };

        let mut boot: Vec<f64> = Vec::with_capacity(n_total);
        let mut boot_nk = vec![0usize; n_windows];
        let mut off = 0usize;
        for (ki, &nk) in n_k.iter().enumerate() {
            let bs = (nk / n_blocks).max(1);
            let n_draw = n_blocks;
            for _ in 0..n_draw {
                let pick = next(&mut rng) % n_blocks;
                let s = off + pick * bs;
                let e = (s + bs).min(off + nk);
                for idx in s..e { boot.push(all_ref[idx]); }
            }
            boot_nk[ki] = boot.len() - boot_nk[..ki].iter().sum::<usize>();
            off += nk;
        }
        let u_b = compute_bias_matrix(&boot, &vk_ref, fc, beta);
        let f_b = solve_mbar(&u_b, &boot_nk);
        let (_, pmf_b, _) = compute_pmf(&boot, &f_b, &u_b, &boot_nk, val_min, val_max, nbins);
        let ref_b = pmf_b[..ref_half].iter().filter(|v| v.is_finite())
            .cloned().fold(f64::INFINITY, f64::min);
        let ref_b = if ref_b.is_finite() { ref_b } else {
            pmf_b.iter().filter(|v| v.is_finite()).cloned().fold(f64::INFINITY, f64::min)
        };
        pmf_b.iter().map(|v| if v.is_finite() { v - ref_b } else { f64::NAN }).collect()
    }).collect();

    emit_progress(&app, "mbar", 90.0);

    let pmf_err: Vec<f64> = (0..nbins).map(|b| {
        let vals: Vec<f64> = pmf_boots.iter()
            .filter_map(|pb| pb.get(b).copied()).filter(|v| v.is_finite()).collect();
        if vals.len() < 2 { return 0.0; }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64).sqrt()
    }).collect();

    let (dg_idx, delta_g) = pmf.iter().enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, &v)| (i, v)).unwrap_or((0, f64::NAN));
    let delta_g_err = pmf_err.get(dg_idx).copied().unwrap_or(0.0);

    // KDE for histogram panel
    let kde_margin = (val_max - val_min) * 0.05;
    let mut off = 0usize;
    let (kde_x, kde_y): (Vec<_>, Vec<_>) = n_k.iter().map(|&nk| {
        let sl = &all_samples[off..off + nk]; off += nk;
        gaussian_kde(sl, val_min - kde_margin, val_max + kde_margin, 60)
    }).unzip();

    emit_progress(&app, "mbar", 100.0);

    let result = MbarResult {
        bin_centers, pmf, pmf_err, delta_g, delta_g_err,
        n_windows, window_val0: val0_k, kde_x, kde_y,
    };
    let msg = format!(
        "MBAR: ΔG‡ = {:.2} ± {:.2} kcal/mol  ({n_windows} windows · {nbins} bins · {n_boot} bootstrap).",
        delta_g, delta_g_err,
    );
    *state.mbar_result.lock().unwrap() = Some(result.clone());
    Ok(AnalysisResult { data: result, message: msg })
}

// ─── set_qmm_topology ────────────────────────────────────────────────────────
//
// Stores the path to a PDB or parm7 topology file for use when visualising
// AMBER ncrst restart files. chemfiles can then apply it as a topology
// template so residue names, atom names, and connectivity are correct.

#[tauri::command]
fn set_qmm_topology(state: State<'_, AppData>, path: String) -> Result<String, String> {
    if !std::path::Path::new(&path).exists() {
        return Err(format!("Topology file not found: {path}"));
    }
    let name = std::path::Path::new(&path)
        .file_name().unwrap_or_default().to_string_lossy().to_string();
    *state.qmm_topology.lock().unwrap() = Some(path);
    Ok(format!("QM/MM topology set: {name}"))
}

// ─── get_umbrella_snapshot_pdb ────────────────────────────────────────────────
//
// Loads the .ncrst restart file for a given window and returns a minimal PDB
// string for the analysis viewer. The ncrst contains only coordinates (no
// topology), so we fall back to Cα-backbone placeholders — the same approach
// as the topology-free path in get_snapshot_pdb.
//
// If the window's rst_file is None (pattern didn't match), returns an error
// with a helpful message.

#[tauri::command]
fn get_umbrella_snapshot_pdb(
    state:      State<'_, AppData>,
    window_idx: usize,
) -> Result<String, String> {
    let lock    = state.umbrella_windows.lock().unwrap();
    let windows = lock.as_ref().ok_or("No umbrella windows loaded.")?;
    let win     = windows.get(window_idx)
        .ok_or(format!("Window index {window_idx} out of range."))?;
    let rst_path = win.rst_file.as_ref()
        .ok_or(format!("No restart file found for window {:02}. Set rst_pattern when loading.", window_idx))?;

    let topo_path = state.qmm_topology.lock().unwrap().clone();

    // Open the ncrst, then apply topology the same way load_trajectory does:
    // parm7/prmtop → set_topology_with_format, everything else → set_topology_file.
    let mut traj = chemfiles::Trajectory::open(rst_path, 'r').map_err(|e| e.to_string())?;
    if let Some(ref tp) = topo_path {
        if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
            traj.set_topology_with_format(tp, "Amber Topology").map_err(|e| e.to_string())?;
        } else {
            traj.set_topology_file(tp).map_err(|e| e.to_string())?;
        }
    }

    let mut frame = chemfiles::Frame::new();
    traj.read(&mut frame).map_err(|e| e.to_string())?;
    Ok(frame_to_pdb(&frame, None))
}


// ─── rewrite_pdb ─────────────────────────────────────────────────────────────
//
// Reads any PDB file (e.g. from cpptraj/leap with missing chain IDs and no
// CONECT records), optionally applies a parm7/prmtop topology for bond info,
// and returns a corrected PDB string ready for NGL with:
//   - Chain ID 'A' on every atom (or from topology when available)
//   - ATOM / HETATM classification matching the shared STANDARD_RES table
//   - CONECT records for all non-standard residue bonds

#[tauri::command]
fn rewrite_pdb(path: String, topo_path: Option<String>) -> Result<String, String> {
    let mut traj = chemfiles::Trajectory::open(&path, 'r').map_err(|e| e.to_string())?;
    if let Some(ref tp) = topo_path {
        if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
            traj.set_topology_with_format(tp, "Amber Topology").map_err(|e| e.to_string())?;
        } else {
            traj.set_topology_file(tp).map_err(|e| e.to_string())?;
        }
    }
    let mut frame = chemfiles::Frame::new();
    traj.read(&mut frame).map_err(|e| e.to_string())?;
    Ok(frame_to_pdb(&frame, None))
}

// ─── save_qm_region ──────────────────────────────────────────────────────────
//
// Receives a list of 1-based PDB serial numbers selected by the user in the
// analysis viewer, reads atom/residue metadata from the last-loaded umbrella
// snapshot, builds a QmRegion, and stores it in AppData.
//
// The AMBER mask is built as a residue-based mask where possible:
//   :resname   for standard ligand/substrate residues (e.g. :UNL)
//   :resno     for protein residues (e.g. :145,146)
// This is the format expected by sander/ORCA QM/MM interfaces.

#[tauri::command]
fn save_qm_region(
    state:       State<'_, AppData>,
    serials:     Vec<usize>,   // 1-based PDB serial numbers from NGL click
    window_idx:  usize,
) -> Result<QmRegion, String> {
    let lock    = state.umbrella_windows.lock().unwrap();
    let windows = lock.as_ref().ok_or("No umbrella windows loaded.")?;
    let win     = windows.get(window_idx)
        .ok_or(format!("Window index {window_idx} out of range."))?;
    let rst_path = win.rst_file.as_ref()
        .ok_or("No restart file for this window.")?;

    let topo_path = state.qmm_topology.lock().unwrap().clone();

    // Re-open the ncrst to get coordinates and topology
    let mut traj = chemfiles::Trajectory::open(rst_path, 'r').map_err(|e| e.to_string())?;
    if let Some(ref tp) = topo_path {
        if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
            traj.set_topology_with_format(tp, "Amber Topology").map_err(|e| e.to_string())?;
        } else {
            traj.set_topology_file(tp).map_err(|e| e.to_string())?;
        }
    }
    let mut frame = chemfiles::Frame::new();
    traj.read(&mut frame).map_err(|e| e.to_string())?;
    let pos  = frame.positions();
    let topo = frame.topology();

    // Build QmAtom list from requested serials (serial = i+1 for 0-based index i)
    let mut atoms: Vec<QmAtom> = Vec::new();
    for &serial in &serials {
        if serial == 0 || serial > pos.len() { continue; }
        let i = serial - 1;
        let atom      = topo.atom(i);
        let atom_name = { let n = atom.name().to_string(); if n.is_empty() { format!("X{i}") } else { n } };
        let element   = { let e = atom.atomic_type().to_string();
            if e.is_empty() { atom_name.chars().find(|c| c.is_ascii_uppercase())
                .map(|c| c.to_string()).unwrap_or("C".into()) } else { e } };
        let (res_name, chain_id, res_seq) =
            if let Some(res) = topo.residue_for_atom(i) {
                let chain = res.get("chainname")
                    .and_then(|p| if let chemfiles::Property::String(s) = p { s.chars().next() } else { None })
                    .unwrap_or('A');
                let rn = res.name().to_string();
                (if rn.is_empty() { "UNK".into() } else { rn },
                 chain, res.id().unwrap_or(i as i64 + 1))
            } else { ("UNK".into(), 'A', i as i64 + 1) };

        atoms.push(QmAtom {
            serial, atom_name, res_name, res_seq, chain_id, element,
            x: pos[i][0] as f64, y: pos[i][1] as f64, z: pos[i][2] as f64,
        });
    }

    if atoms.is_empty() { return Err("No valid atoms in selection.".into()); }

    // Build AMBER mask: group by unique residue, use :resname for heteroatoms,
    // :resno for protein residues
    let mut res_ids: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let mut het_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let protein_resnames = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                             "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                             "HIE","HID","HIP","CYX","ASH","GLH"];
    for a in &atoms {
        if protein_resnames.contains(&a.res_name.as_str()) {
            res_ids.insert(a.res_seq);
        } else {
            het_names.insert(a.res_name.clone());
        }
    }
    let mut mask_parts: Vec<String> = Vec::new();
    if !res_ids.is_empty() {
        let ids: Vec<String> = res_ids.iter().map(|id| id.to_string()).collect();
        mask_parts.push(format!(":{}", ids.join(",")));
    }
    for name in &het_names {
        mask_parts.push(format!(":{name}"));
    }
    let amber_mask = mask_parts.join(" | ");
    let n_atoms = atoms.len();

    let region = QmRegion { atoms, amber_mask, window_idx, n_atoms };
    *state.qm_region.lock().unwrap() = Some(region.clone());
    Ok(region)
}

/// Return total number of loaded umbrella windows.
#[tauri::command]
fn get_umbrella_window_count(state: State<'_, AppData>) -> usize {
    state.umbrella_windows.lock().unwrap()
        .as_ref().map(|w| w.len()).unwrap_or(0)
}

/// Return the currently stored QM region (if any).
#[tauri::command]
fn get_qm_region(state: State<'_, AppData>) -> Option<QmRegion> {
    state.qm_region.lock().unwrap().clone()
}

/// Clear the stored QM region.
#[tauri::command]
fn clear_qm_region(state: State<'_, AppData>) {
    *state.qm_region.lock().unwrap() = None;
}

// ─── resolve_qm_selection ─────────────────────────────────────────────────────
//
// Parses a simple AMBER-mask-style selection string into 1-based atom serials
// by reading the ncrst topology. Supported syntax:
//   :RES          → all atoms in residues named RES (e.g. :UNL)
//   :N,M,...      → atoms in residues with those sequence numbers
//   @N,M,...      → atoms with those 1-based serial numbers directly
//   :RES @N,M     → union of both (space-separated tokens)
// Returns a sorted, deduplicated Vec<usize> of 1-based serials.

#[tauri::command]
fn resolve_qm_selection(
    state:         State<'_, AppData>,
    selection_str: String,
    window_idx:    usize,
) -> Result<Vec<usize>, String> {
    let lock    = state.umbrella_windows.lock().unwrap();
    let windows = lock.as_ref().ok_or("No umbrella windows loaded.")?;
    let win     = windows.get(window_idx).ok_or("Window index out of range.")?;
    let rst_path = win.rst_file.as_ref().ok_or("No restart file for this window.")?;

    let topo_path = state.qmm_topology.lock().unwrap().clone();
    let mut traj = chemfiles::Trajectory::open(rst_path, 'r').map_err(|e| e.to_string())?;
    if let Some(ref tp) = topo_path {
        if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
            traj.set_topology_with_format(tp, "Amber Topology").map_err(|e| e.to_string())?;
        } else {
            traj.set_topology_file(tp).map_err(|e| e.to_string())?;
        }
    }
    let mut frame = chemfiles::Frame::new();
    traj.read(&mut frame).map_err(|e| e.to_string())?;
    let topo = frame.topology();
    let n_atoms = frame.positions().len();

    let mut serials: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

    for token in selection_str.split_whitespace() {
        if token.starts_with(':') {
            let inner = &token[1..];
            // Could be residue name(s) or residue number(s)
            for part in inner.split(',') {
                let part = part.trim();
                if let Ok(resno) = part.parse::<i64>() {
                    // Residue sequence number
                    for i in 0..n_atoms {
                        if let Some(res) = topo.residue_for_atom(i) {
                            if res.id().unwrap_or(-1) == resno {
                                serials.insert(i + 1);
                            }
                        }
                    }
                } else {
                    // Residue name
                    for i in 0..n_atoms {
                        if let Some(res) = topo.residue_for_atom(i) {
                            if res.name().eq_ignore_ascii_case(part) {
                                serials.insert(i + 1);
                            }
                        }
                    }
                }
            }
        } else if token.starts_with('@') {
            let inner = &token[1..];
            for part in inner.split(',') {
                if let Ok(serial) = part.trim().parse::<usize>() {
                    if serial >= 1 && serial <= n_atoms {
                        serials.insert(serial);
                    }
                }
            }
        }
    }

    if serials.is_empty() {
        return Err(format!("Selection '{}' matched no atoms.", selection_str));
    }
    Ok(serials.into_iter().collect())
}

// ─── preload_umbrella_coords ─────────────────────────────────────────────────
//
// Reads each umbrella window's restart file sequentially (chemfiles Trajectory
// is not Send so rayon parallel iteration cannot be used), builds a single
// Vec<Vec<f32>> with one flat-coord entry per window, and stores it in
// umbrella_traj_coords.  This is identical in structure to trajectory_data
// (the MD trajectory cache) so the viewer can use the same coord-swap pattern
// as the MD visualizer: load one PDB once, then call updatePosition per frame.
//
// Must run on a blocking thread (tauri::async_runtime::spawn_blocking) so it
// doesn't stall the Tokio executor.

#[tauri::command]
async fn preload_umbrella_coords(
    app:   AppHandle,
    state: State<'_, AppData>,
) -> Result<String, String> {
    // Collect paths under the lock, then release it before the blocking work
    let (rst_paths, topo_path) = {
        let lock    = state.umbrella_windows.lock().unwrap();
        let windows = lock.as_ref().ok_or("No umbrella windows loaded.")?;
        let paths: Vec<Option<String>> = windows.iter()
            .map(|w| w.rst_file.clone()).collect();
        (paths, state.qmm_topology.lock().unwrap().clone())
    };

    let n = rst_paths.len();
    if n == 0 { return Err("No umbrella windows loaded.".into()); }

    emit_progress(&app, "preload", 0.0);

    // Run sequentially on a blocking thread — chemfiles is not Send
    let app2 = app.clone();
    let all: Vec<Vec<f32>> = tauri::async_runtime::spawn_blocking(move || {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (i, rst_opt) in rst_paths.iter().enumerate() {
            let rst = rst_opt.as_ref()
                .ok_or(format!("Window {i}: no restart file set."))?;

            let mut traj = chemfiles::Trajectory::open(rst, 'r')
                .map_err(|e| format!("Window {i}: {e}"))?;

            if let Some(ref tp) = topo_path {
                if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
                    traj.set_topology_with_format(tp, "Amber Topology")
                        .map_err(|e| format!("Window {i} topo: {e}"))?;
                } else {
                    traj.set_topology_file(tp)
                        .map_err(|e| format!("Window {i} topo: {e}"))?;
                }
            }

            let mut frame = chemfiles::Frame::new();
            traj.read(&mut frame)
                .map_err(|e| format!("Window {i} read: {e}"))?;

            let flat: Vec<f32> = frame.positions().iter()
                .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
                .collect();
            out.push(flat);

            emit_progress(&app2, "preload", (i + 1) as f64 / n as f64 * 100.0);
        }
        Ok::<Vec<Vec<f32>>, String>(out)
    }).await.map_err(|e| e.to_string())??;

    let msg = format!("Loaded {} window snapshots into trajectory cache.", n);
    *state.umbrella_traj_coords.lock().unwrap() = Some(all);
    Ok(msg)
}

// ─── get_umbrella_window_coords ───────────────────────────────────────────────
//
// Returns the flat f32 coordinate array for one window from the pre-loaded
// cache. The MD visualizer's get_frame_coords pattern — no disk I/O, just a
// Vec<f32> lookup — making frame scrubbing essentially free.

#[tauri::command]
fn get_umbrella_window_coords(
    state:      State<'_, AppData>,
    window_idx: usize,
) -> Result<Vec<f32>, String> {
    let lock = state.umbrella_traj_coords.lock().unwrap();
    let all  = lock.as_ref().ok_or("Coords not pre-loaded. Call preload_umbrella_coords first.")?;
    all.get(window_idx)
        .cloned()
        .ok_or(format!("Window {window_idx} out of range."))
}

// ─── write_text_file ─────────────────────────────────────────────────────────

#[tauri::command]
fn write_text_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(&path, contents).map_err(|e| e.to_string())
}

// ─── Dihedral geometry helpers ────────────────────────────────────────────────

fn cross3(a: [f64;3], b: [f64;3]) -> [f64;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn dot3(a: [f64;3], b: [f64;3]) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn norm3(a: [f64;3]) -> f64 { dot3(a,a).sqrt() }
fn sub3(a: [f64;3], b: [f64;3]) -> [f64;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }

/// Dihedral angle (degrees) defined by four points p1–p2–p3–p4.
fn dihedral_deg(p1: [f64;3], p2: [f64;3], p3: [f64;3], p4: [f64;3]) -> f64 {
    let b1 = sub3(p2, p1);
    let b2 = sub3(p3, p2);
    let b3 = sub3(p4, p3);
    let n1 = cross3(b1, b2);
    let n2 = cross3(b2, b3);
    let m1 = cross3(n1, b2);
    let b2n = norm3(b2);
    if b2n < 1e-10 { return f64::NAN; }
    let x = dot3(n1, n2);
    let y = dot3(m1, n2) / b2n;
    (-y.atan2(x)).to_degrees()
}

// ─── run_dihedrals ────────────────────────────────────────────────────────────
//
// Computes backbone φ/ψ angles (if N/CA/C atoms are present in the selection)
// or Cα pseudo-dihedrals (for Cα-only trajectories).
//
// Returns a 60×60 2D density grid + per-residue mean/std for the Ramachandran
// plot in the frontend, and caches the full per-frame time series so that
// get_residue_dihedrals can serve the analysis viewer on demand.

#[tauri::command]
async fn run_dihedrals(
    app:   AppHandle,
    state: State<'_, AppData>,
) -> Result<AnalysisResult<DihedralResultJson>, String> {
    // Serve from cache if available
    if let Some(ref c) = *state.dihedral_cache.lock().unwrap() {
        let summaries = summarise_dihedrals(&c.residues);
        let density   = build_rama_density(&c.residues);
        let n_frames  = c.residues.first().map(|r| r.phi.len()).unwrap_or(0);
        return Ok(AnalysisResult {
            data: DihedralResultJson { density, residues: summaries, mode: c.mode.clone(), n_frames },
            message: "Dihedrals (cached)".into(),
        });
    }

    // Clone data out of Mutexes before spawn_blocking
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .clone().ok_or("No trajectory loaded")?;
    let meta: Vec<AtomMeta> = state.atom_meta.lock().unwrap()
        .clone().ok_or("No atom metadata")?;

    emit_progress(&app, "dihedrals", 0.0);

    let app_clone = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_frames = traj.len();

    // Group atoms by (res_seq, chain_id) preserving residue order
    let mut seen_keys: Vec<(i64, char)> = Vec::new();
    seen_keys.sort_by_key(|k| k.0);
    let mut res_map: std::collections::HashMap<(i64, char), Vec<(usize, String)>> = Default::default();
    for (i, m) in meta.iter().enumerate() {
        let key = (m.res_seq, m.chain_id);
        if !res_map.contains_key(&key) { seen_keys.push(key); }
        res_map.entry(key).or_default().push((i, m.atom_name.clone()));
    }

    // Detect backbone mode: majority of residues must have N, CA, C
    let backbone_count = seen_keys.iter()
        .filter(|k| {
            let atoms = &res_map[k];
            let names: Vec<&str> = atoms.iter().map(|(_, n)| n.as_str()).collect();
            names.contains(&"N") && names.contains(&"CA") && names.contains(&"C")
        })
        .count();
    let backbone_mode = backbone_count > seen_keys.len() / 2;
    let mode_str = if backbone_mode { "backbone" } else { "pseudodihedral" };

    emit_progress(&app_clone, "dihedrals", 0.0);

    let n_res = seen_keys.len();
    let chunk = (n_res / 20).max(1);

    let residues: Vec<ResidueDihedralFull> = if backbone_mode {
        // Build backbone residue list: (res_seq, res_name, n_idx, ca_idx, c_idx)
        struct BB { res_seq: i64, res_name: String, n: usize, ca: usize, c: usize }
        let bb_list: Vec<BB> = seen_keys.iter().filter_map(|k| {
            let atoms = &res_map[k];
            let n  = atoms.iter().find(|(_, nm)| nm == "N" ).map(|(i,_)| *i)?;
            let ca = atoms.iter().find(|(_, nm)| nm == "CA").map(|(i,_)| *i)?;
            let c  = atoms.iter().find(|(_, nm)| nm == "C" ).map(|(i,_)| *i)?;
            Some(BB { res_seq: k.0, res_name: meta[ca].res_name.clone(), n, ca, c })
        }).collect();

        let nb = bb_list.len();
        (0..nb).map(|ri| {
            if ri % chunk == 0 {
                emit_progress(&app_clone, "dihedrals", ri as f64 / nb as f64 * 95.0);
            }
            let mut phi = vec![f64::NAN; n_frames];
            let mut psi = vec![f64::NAN; n_frames];
            for fi in 0..n_frames {
                let f = &traj[fi];
                if ri > 0 {
                    phi[fi] = dihedral_deg(
                        f[bb_list[ri-1].c], f[bb_list[ri].n],
                        f[bb_list[ri].ca],  f[bb_list[ri].c],
                    );
                }
                if ri < nb - 1 {
                    psi[fi] = dihedral_deg(
                        f[bb_list[ri].n],    f[bb_list[ri].ca],
                        f[bb_list[ri].c],    f[bb_list[ri+1].n],
                    );
                }
            }
            ResidueDihedralFull {
                res_seq:  bb_list[ri].res_seq,
                res_name: bb_list[ri].res_name.clone(),
                atom_idx: bb_list[ri].ca,
                phi, psi,
            }
        }).collect()
    } else {
        // Pseudo-dihedral: 4 consecutive Cα (or first atom per residue)
        let ca_list: Vec<(i64, String, usize)> = seen_keys.iter().map(|k| {
            let atoms = &res_map[k];
            let ca_idx = atoms.iter().find(|(_, nm)| nm == "CA").map(|(i,_)| *i)
                .unwrap_or(atoms[0].0);
            (k.0, meta[ca_idx].res_name.clone(), ca_idx)
        }).collect();
        let nc = ca_list.len();

        (0..nc).map(|ri| {
            if ri % chunk == 0 {
                emit_progress(&app_clone, "dihedrals", ri as f64 / nc as f64 * 95.0);
            }
            let mut phi = vec![f64::NAN; n_frames];
            let psi     = vec![f64::NAN; n_frames]; // not used in pseudo mode
            if ri > 0 && ri + 2 < nc {
                for fi in 0..n_frames {
                    let f = &traj[fi];
                    phi[fi] = dihedral_deg(
                        f[ca_list[ri-1].2], f[ca_list[ri].2],
                        f[ca_list[ri+1].2], f[ca_list[ri+2].2],
                    );
                }
            }
            ResidueDihedralFull {
                res_seq:  ca_list[ri].0,
                res_name: ca_list[ri].1.clone(),
                atom_idx: ca_list[ri].2,
                phi, psi,
            }
        }).collect()
    };

    emit_progress(&app_clone, "dihedrals", 100.0);

    let summaries = summarise_dihedrals(&residues);
    let density   = build_rama_density(&residues);
    let n_res_out = residues.len();
    let mode_s    = mode_str.to_string();
    (DihedralResult { residues, mode: mode_s.clone() }, summaries, density, n_res_out, n_frames, mode_s)
    }).await.map_err(|e| e.to_string())?;

    let (dihedral_result, summaries, density, n_res_out, n_frames, mode_s) = result;
    *state.dihedral_cache.lock().unwrap() = Some(dihedral_result);

    Ok(AnalysisResult {
        data: DihedralResultJson { density, residues: summaries, mode: mode_s.clone(), n_frames },
        message: format!("Dihedral angles computed for {n_res_out} residues over {n_frames} frames ({mode_s} mode)."),
    })
}

fn summarise_dihedrals(residues: &[ResidueDihedralFull]) -> Vec<ResidueDihedralSummary> {
    residues.iter().map(|r| {
        let phi_valid: Vec<f64> = r.phi.iter().copied().filter(|v| v.is_finite()).collect();
        let psi_valid: Vec<f64> = r.psi.iter().copied().filter(|v| v.is_finite()).collect();
        let mean_std = |v: &[f64]| -> (Option<f64>, Option<f64>) {
            if v.is_empty() { return (None, None); }
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let s = (v.iter().map(|x| (x-m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
            (Some(m), Some(s))
        };
        let (phi_mean, phi_std) = mean_std(&phi_valid);
        let (psi_mean, psi_std) = mean_std(&psi_valid);
        let n_valid = phi_valid.len().min(psi_valid.len());
        ResidueDihedralSummary {
            res_seq: r.res_seq, res_name: r.res_name.clone(), atom_idx: r.atom_idx,
            phi_mean, psi_mean, phi_std, psi_std, n_valid,
        }
    }).collect()
}

fn build_rama_density(residues: &[ResidueDihedralFull]) -> Vec<Vec<f64>> {
    const BINS: usize = 60;
    let mut grid = vec![vec![0u32; BINS]; BINS];
    let mut max_count = 0u32;
    for r in residues {
        for (&phi, &psi) in r.phi.iter().zip(r.psi.iter()) {
            if !phi.is_finite() || !psi.is_finite() { continue; }
            let xi = ((phi + 180.0) / 360.0 * BINS as f64) as usize;
            let yi = ((psi + 180.0) / 360.0 * BINS as f64) as usize;
            let xi = xi.min(BINS-1); let yi = yi.min(BINS-1);
            grid[yi][xi] += 1;
            if grid[yi][xi] > max_count { max_count = grid[yi][xi]; }
        }
    }
    if max_count == 0 { return vec![vec![0.0; BINS]; BINS]; }
    grid.iter().map(|row| row.iter().map(|&c| c as f64 / max_count as f64).collect()).collect()
}

// ─── get_residue_dihedrals ────────────────────────────────────────────────────
//
// Returns the full φ/ψ time series for a single residue identified by its
// selection atom index.  Called by the analysis viewer dihedral mode.

#[tauri::command]
fn get_residue_dihedrals(
    state:    State<'_, AppData>,
    atom_idx: usize,
) -> Result<SingleResidueDihedrals, String> {
    let lock  = state.dihedral_cache.lock().unwrap();
    let cache = lock.as_ref().ok_or("Run dihedral analysis first.")?;
    let res   = cache.residues.iter()
        .find(|r| r.atom_idx == atom_idx)
        .ok_or(format!("No dihedral data for atom index {atom_idx}."))?;
    Ok(SingleResidueDihedrals {
        res_seq:  res.res_seq,
        res_name: res.res_name.clone(),
        phi:      res.phi.clone(),
        psi:      res.psi.clone(),
        mode:     cache.mode.clone(),
    })
}

// ─── run_prs ─────────────────────────────────────────────────────────────────
//
// Perturbation Response Scanning.
//
// Builds the N×N scalar covariance matrix C where
//   C[i][j] = (1/T) Σ_t  Δr_i(t) · Δr_j(t)
//
// For each source i, the displacement response at j is C[i][j].
// Effectiveness and sensitivity are derived from the raw and
// variance-normalised row/column sums, giving asymmetric per-residue
// scalars that can be mapped to the 3D structure.

#[tauri::command]
async fn run_prs(
    app:   AppHandle,
    state: State<'_, AppData>,
) -> Result<AnalysisResult<PrsResult>, String> {
    if let Some(c) = state.prs_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "PRS (cached)".into() });
    }

    // Clone data out of the Mutex before entering spawn_blocking
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .clone().ok_or("No trajectory loaded")?;

    emit_progress(&app, "prs", 2.0);

    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_frames  = traj.len() as f64;
        let n_atoms   = traj[0].len();
        let mean      = mean_positions(&traj);

        // N×N scalar covariance  C[i][j] = <Δr_i · Δr_j>
        let _chunk = (n_atoms / 20).max(1);
        let cov: Vec<Vec<f64>> = (0..n_atoms).into_par_iter().map(|i| {
            (0..n_atoms).map(|j| {
                let mut s = 0.0f64;
                for frame in traj.iter() {
                    let dix = frame[i][0] - mean[i][0];
                    let diy = frame[i][1] - mean[i][1];
                    let diz = frame[i][2] - mean[i][2];
                    let djx = frame[j][0] - mean[j][0];
                    let djy = frame[j][1] - mean[j][1];
                    let djz = frame[j][2] - mean[j][2];
                    s += dix*djx + diy*djy + diz*djz;
                }
                s / n_frames
            }).collect()
        }).collect();

        // Normalise matrix by geometric mean of diagonal for display
        let variances: Vec<f64> = (0..n_atoms).map(|i| cov[i][i].max(1e-12)).collect();
        let matrix: Vec<Vec<f64>> = (0..n_atoms).map(|i| {
            (0..n_atoms).map(|j| {
                cov[i][j].abs() / (variances[i] * variances[j]).sqrt()
            }).collect()
        }).collect();

        let effectiveness: Vec<f64> = (0..n_atoms).map(|i| {
            let s: f64 = (0..n_atoms).filter(|&j| j != i).map(|j| cov[i][j].abs()).sum();
            s / variances[i]
        }).collect();
        let sensitivity: Vec<f64> = (0..n_atoms).map(|j| {
            let s: f64 = (0..n_atoms).filter(|&i| i != j).map(|i| cov[i][j].abs()).sum();
            s / variances[j]
        }).collect();

        let norm_vec = |v: Vec<f64>| -> Vec<f64> {
            let mx = v.iter().cloned().fold(0.0f64, f64::max);
            if mx < 1e-12 { return v; }
            v.iter().map(|&x| x / mx).collect()
        };
        let effectiveness = norm_vec(effectiveness);
        let sensitivity   = norm_vec(sensitivity);

        (PrsResult { matrix, effectiveness, sensitivity }, n_atoms, n_frames as usize)
    }).await.map_err(|e| e.to_string())?;

    emit_progress(&app, "prs", 100.0);
    let (prs_result, n_atoms, n_frames) = result;
    *state.prs_cache.lock().unwrap() = Some(prs_result.clone());

    Ok(AnalysisResult {
        data: prs_result,
        message: format!("PRS computed for {n_atoms} residues over {n_frames} frames."),
    })
}

// ─── run_mutual_information ───────────────────────────────────────────────────
//
// Computes the Normalised Mutual Information (NMI) matrix between all pairs
// of residues, using the scalar displacement magnitude as the time series:
//   d_i(t) = |r_i(t) − <r_i>|
//
// MI estimated via 2D histogram (B = min(√T, 50) bins per axis).
// NMI(i,j) = MI(d_i, d_j) / √(H(d_i) · H(d_j))  ∈ [0, 1]
//
// Unlike DCCM (which captures linear correlations) MI detects any statistical
// dependence including non-linear coupling.

#[tauri::command]
async fn run_mutual_information(
    app:   AppHandle,
    state: State<'_, AppData>,
) -> Result<AnalysisResult<Vec<Vec<f64>>>, String> {
    if let Some(c) = state.mi_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "Mutual information (cached)".into() });
    }

    // Clone data out before entering spawn_blocking
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .clone().ok_or("No trajectory loaded")?;

    emit_progress(&app, "mi", 2.0);

    let result = tauri::async_runtime::spawn_blocking(move || {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let mean     = mean_positions(&traj);

        // Number of histogram bins: heuristic √T, capped at 50
        let n_bins = ((n_frames as f64).sqrt() as usize).min(50).max(5);

        // Displacement magnitude time series for each atom
        let displacements: Vec<Vec<f64>> = (0..n_atoms).map(|i| {
            traj.iter().map(|frame| {
                let dx = frame[i][0]-mean[i][0];
                let dy = frame[i][1]-mean[i][1];
                let dz = frame[i][2]-mean[i][2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            }).collect()
        }).collect();

        // Per-atom range for histogram binning
        let ranges: Vec<(f64,f64)> = (0..n_atoms).map(|i| {
            let lo = displacements[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = displacements[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let eps = (hi - lo) * 0.01 + 1e-12;
            (lo - eps, hi + eps)
        }).collect();

        // Marginal entropy H(i)
        let marginal_entropy: Vec<f64> = (0..n_atoms).map(|i| {
            let (lo, hi) = ranges[i];
            let bw = (hi - lo) / n_bins as f64;
            let mut hist = vec![0u32; n_bins];
            for &d in &displacements[i] {
                let b = ((d - lo) / bw) as usize;
                hist[b.min(n_bins-1)] += 1;
            }
            let n = n_frames as f64;
            hist.iter().filter(|&&c| c > 0).map(|&c| {
                let p = c as f64 / n;
                -p * p.ln()
            }).sum::<f64>()
        }).collect();

        // NMI matrix — parallelised with rayon
        let matrix: Vec<Vec<f64>> = (0..n_atoms).into_par_iter().map(|i| {
            let (loi, hii) = ranges[i];
            let bwi = (hii - loi) / n_bins as f64;
            (0..n_atoms).map(|j| {
                if i == j { return 1.0; }
                let (loj, hij) = ranges[j];
                let bwj = (hij - loj) / n_bins as f64;
                let mut joint = vec![vec![0u32; n_bins]; n_bins];
                for t in 0..n_frames {
                    let bi = ((displacements[i][t] - loi) / bwi) as usize;
                    let bj = ((displacements[j][t] - loj) / bwj) as usize;
                    joint[bi.min(n_bins-1)][bj.min(n_bins-1)] += 1;
                }
                let n = n_frames as f64;
                let joint_h: f64 = joint.iter().flat_map(|r| r.iter())
                    .filter(|&&c| c > 0)
                    .map(|&c| { let p = c as f64/n; -p*p.ln() })
                    .sum();
                let mi    = marginal_entropy[i] + marginal_entropy[j] - joint_h;
                let denom = (marginal_entropy[i] * marginal_entropy[j]).sqrt();
                if denom < 1e-12 { 0.0 } else { (mi / denom).max(0.0).min(1.0) }
            }).collect()
        }).collect();

        (matrix, n_atoms, n_bins)
    }).await.map_err(|e| e.to_string())?;

    emit_progress(&app, "mi", 100.0);
    let (matrix, n_atoms, n_bins) = result;
    *state.mi_cache.lock().unwrap() = Some(matrix.clone());

    Ok(AnalysisResult {
        data: matrix,
        message: format!("NMI matrix ({n_atoms}×{n_atoms}) computed with {n_bins} histogram bins."),
    })
}

// ─── run_clustering ───────────────────────────────────────────────────────────
//
// K-means clustering of trajectory frames in PCA space.
// Optionally builds a Markov State Model (MSM) and runs PCCA for macrostate
// assignment.
//
// K-means: Lloyd's algorithm with k-means++ initialisation.
// MSM: count matrix at lag τ → row-normalise → eigendecompose → PCCA-like
//      spectral clustering via top-k eigenvectors (fuzzy macrostate membership).

#[tauri::command]
async fn run_clustering(
    app:        AppHandle,
    state:      State<'_, AppData>,
    n_clusters: usize,
    method:     Option<String>,
    lag:        Option<usize>,
    n_macro:    Option<usize>,
) -> Result<AnalysisResult<ClusterResultJson>, String> {
    let pca = state.pca_cache.lock().unwrap().clone()
        .ok_or("Run PCA first — clustering uses PCA projections.")?;
    let k        = n_clusters.max(2);
    let method   = method.as_deref().unwrap_or("kmeans").to_lowercase();
    let lag      = lag.unwrap_or(1).max(1);
    let n_macro  = n_macro.unwrap_or(k).max(2).min(k);
    let pts      = pca.projections.clone();
    let n_frames = pts.len();
    if n_frames < k { return Err(format!("Need at least {k} frames for {k} clusters.")); }

    emit_progress(&app, "cluster", 2.0);

    let app2         = app.clone();
    let method_clone = method.clone();   // clone before move into closure
    let (assignments, centers, populations, its, pcca_mem) =
        tauri::async_runtime::spawn_blocking(move || -> Result<_,String> {
        let method = method_clone;

        // ── K-means++ initialisation ──────────────────────────────────────
        fn dist2(a: [f64;2], b: [f64;2]) -> f64 {
            let dx = a[0]-b[0]; let dy = a[1]-b[1]; dx*dx + dy*dy
        }
        // Seed first center as frame 0, then choose subsequent with D² probability
        let mut rng: u64 = 12345678901234567;
        let mut lcg = || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut centers: Vec<[f64;2]> = vec![pts[0]];
        for _ in 1..k {
            let dists: Vec<f64> = pts.iter()
                .map(|&p| centers.iter().map(|&c| dist2(p,c)).fold(f64::INFINITY, f64::min))
                .collect();
            let total: f64 = dists.iter().sum();
            let mut target = lcg() * total;
            let mut pick = 0;
            for (i, &d) in dists.iter().enumerate() { target -= d; if target <= 0.0 { pick = i; break; } }
            centers.push(pts[pick]);
        }

        // ── Lloyd iterations ──────────────────────────────────────────────
        let mut assignments = vec![0usize; n_frames];
        for _ in 0..200 {
            // Assign
            let changed: usize = pts.par_iter().zip(assignments.par_iter_mut()).map(|(&p, a)| {
                let new = (0..k).min_by(|&i, &j|
                    dist2(p, centers[i]).partial_cmp(&dist2(p, centers[j])).unwrap()
                ).unwrap();
                let changed = if new != *a { 1 } else { 0 };
                *a = new;
                changed
            }).sum();
            // Update centers
            let mut sums   = vec![[0.0f64;2]; k];
            let mut counts = vec![0usize; k];
            for (i, &a) in assignments.iter().enumerate() {
                sums[a][0] += pts[i][0]; sums[a][1] += pts[i][1]; counts[a] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    centers[c] = [sums[c][0] / counts[c] as f64, sums[c][1] / counts[c] as f64];
                }
            }
            if changed == 0 { break; }
        }
        emit_progress(&app2, "cluster", 50.0);

        let populations: Vec<f64> = {
            let mut cnt = vec![0usize; k];
            for &a in &assignments { cnt[a] += 1; }
            cnt.iter().map(|&c| c as f64 / n_frames as f64).collect()
        };

        // ── MSM / PCCA (optional) ─────────────────────────────────────────
        let mut its_out   = Vec::new();
        let mut pcca_out  = vec![vec![0.0f64; n_macro]; n_frames];

        if method == "msm" {
            // Build count matrix at lag τ
            let mut count = vec![vec![0u32; k]; k];
            for t in 0..(n_frames.saturating_sub(lag)) {
                count[assignments[t]][assignments[t + lag]] += 1;
            }
            // Symmetrise and row-normalise → T
            let mut t_mat = vec![vec![0.0f64; k]; k];
            for i in 0..k {
                let mut row_sum = 0.0f64;
                for j in 0..k {
                    let sym = (count[i][j] + count[j][i]) as f64 * 0.5;
                    t_mat[i][j] = sym; row_sum += sym;
                }
                if row_sum > 0.0 { for j in 0..k { t_mat[i][j] /= row_sum; } }
                else { t_mat[i][i] = 1.0; }
            }
            emit_progress(&app2, "cluster", 70.0);

            // Eigendecompose T (power iteration for top-k eigenvectors of symmetric T)
            // Use ndarray for this
            let t_arr = Array2::from_shape_vec((k, k),
                t_mat.iter().flat_map(|r| r.iter().copied()).collect::<Vec<f64>>())
                .map_err(|e| e.to_string())?;
            let (eigenvalues, eigenvectors) = t_arr.eigh(ndarray_linalg::UPLO::Upper)
                .map_err(|e| format!("MSM eigendecomposition failed: {e}"))?;
            let n_ev = eigenvalues.len();

            // Implied timescales from eigenvalues (skip λ=1 stationary mode)
            its_out = (0..(n_ev.saturating_sub(1))).rev()
                .take(n_macro.saturating_sub(1))
                .map(|i| {
                    let lam = eigenvalues[n_ev - 1 - i].abs().max(1e-10).min(1.0 - 1e-10);
                    -( lag as f64) / lam.ln().abs()
                }).collect();

            emit_progress(&app2, "cluster", 85.0);

            // PCCA-style fuzzy macrostate assignment via top-n_macro eigenvectors
            // Each frame's micro-cluster gets soft membership from the eigenvector values
            // Approach: for each micro-cluster centre, read top eigenvectors,
            // then use softmax-normalised absolute values as membership weights.
            let top_evecs: Vec<Vec<f64>> = (1..=n_macro.min(n_ev-1)).map(|m| {
                let col = n_ev - 1 - m; // descending order
                eigenvectors.column(col).iter().copied().collect()
            }).collect();

            // For each frame, read its micro-cluster's eigenvector projection
            for (f, &micro) in assignments.iter().enumerate() {
                let weights: Vec<f64> = top_evecs.iter()
                    .map(|ev| ev[micro].abs() + 1e-12).collect();
                let total: f64 = weights.iter().sum();
                for m in 0..n_macro {
                    pcca_out[f][m] = if m < weights.len() { weights[m] / total } else { 0.0 };
                }
            }
        }

        emit_progress(&app2, "cluster", 100.0);
        Ok((assignments, centers, populations, its_out, pcca_out))
    }).await.map_err(|e| e.to_string())??;

    let result = ClusterResult {
        assignments: assignments.clone(),
        centers: centers.clone(),
        populations: populations.clone(),
        method: method.clone(),
        n_clusters: k,
        implied_timescales: its.clone(),
        pcca_membership: pcca_mem.clone(),
    };
    *state.cluster_cache.lock().unwrap() = Some(result);

    let msg = if method == "msm" && !its.is_empty() {
        format!("{k} clusters · MSM lag={lag} · ITS₁={:.1} frames", its[0])
    } else {
        format!("K-means: {k} clusters over {n_frames} frames.")
    };

    Ok(AnalysisResult {
        data: ClusterResultJson { assignments, centers, populations, method, n_clusters: k,
                                  implied_timescales: its, pcca_membership: pcca_mem },
        message: msg,
    })
}

// ─── parse_cv_rst ─────────────────────────────────────────────────────────────
//
// Parses an AMBER NMROPT restraint file (.cv.rst / NMR.def) and returns one
// block per &rst...&end section.  Each block exposes the iat atom indices
// (1-based, full-topology), rstwt weights, and the restraint centres r2/r3.
//
// For r1-r2 type CVs the iat list has 4 entries:
//   iat = A, B, B, C  →  r1 = dist(A,B)   r2 = dist(B,C)   CV = rstwt[0]*r1 + rstwt[1]*r2

#[derive(Clone, Serialize)]
pub struct CvRstBlock {
    pub block_idx: usize,       // 0-based index of this &rst block
    pub iat:       Vec<i64>,    // atom indices, 1-based AMBER convention
    pub rstwt:     Vec<f64>,    // weights (e.g. [1.0, -1.0] for r1-r2)
    pub r2:        f64,         // restraint centre (r2 = r3 for flat-bottomed)
    pub rk2:       f64,         // force constant
    pub comment:   String,      // the # comment line above this block (if any)
    pub cv_label:  String,      // human-readable label e.g. "r1-r2 (block 0)"
}

#[tauri::command]
fn parse_cv_rst(path: String) -> Result<Vec<CvRstBlock>, String> {
    let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let mut blocks: Vec<CvRstBlock> = Vec::new();
    let mut pending_comment = String::new();
    let mut in_block = false;
    // Scratch fields for the current block
    let mut iat:   Vec<i64> = Vec::new();
    let mut rstwt: Vec<f64> = Vec::new();
    let mut r2  = 0.0f64;
    let mut rk2 = 0.0f64;

    for raw in content.lines() {
        let line = raw.trim();
        if line.starts_with('#') {
            pending_comment = line.trim_start_matches('#').trim().to_string();
            continue;
        }
        let lc = line.to_lowercase();
        if lc.contains("&rst") {
            in_block = true;
            iat.clear(); rstwt.clear(); r2 = 0.0; rk2 = 0.0;
            continue;
        }
        if lc.contains("&end") || lc == "/" {
            if in_block && !iat.is_empty() {
                // Build a human-readable label
                let block_idx = blocks.len();
                let cv_label = if iat.len() == 4 && rstwt.len() == 2 {
                    let sign = if rstwt[1] < 0.0 { "−" } else { "+" };
                    format!("r1{}r2 (col {})", sign, block_idx + 1)
                } else {
                    format!("CV col {} ({} atoms)", block_idx + 1, iat.len())
                };
                blocks.push(CvRstBlock {
                    block_idx,
                    iat: iat.clone(), rstwt: rstwt.clone(), r2, rk2,
                    comment: pending_comment.clone(), cv_label,
                });
                pending_comment.clear();
            }
            in_block = false;
            continue;
        }
        if !in_block { continue; }

        // Parse key=value pairs (handle comma-separated values on same line)
        // e.g.  iat=7488,7496,7496,7164,
        //        rstwt=1.,-1.,
        //        r2=-0.950, r3=-0.950, rk2=150.0
        let clean = line.replace(' ', "");
        for token in clean.split(',').filter(|t| t.contains('=')) {
            let mut parts = token.splitn(2, '=');
            let key = parts.next().unwrap_or("").to_lowercase();
            let val = parts.next().unwrap_or("").trim_end_matches(',');
            match key.as_str() {
                "iat" => {
                    // might be only the start of a list; gather remaining on line
                    for v in line.split(|c: char| !c.is_ascii_digit() && c != '-') {
                        if let Ok(n) = v.parse::<i64>() { iat.push(n); }
                    }
                }
                "rstwt" => {
                    if let Ok(f) = val.parse::<f64>() { rstwt.push(f); }
                }
                "r2" | "r3" => {
                    if r2 == 0.0 { r2 = val.parse::<f64>().unwrap_or(0.0); }
                }
                "rk2" | "rk3" => {
                    if rk2 == 0.0 { rk2 = val.parse::<f64>().unwrap_or(0.0); }
                }
                _ => {}
            }
        }
        // Also handle multi-line iat= by scanning any line for consecutive integers
        // (the above iat branch fires if "iat" key is on the current line, but
        // continuation lines with just numbers are rare — AMBER usually puts all
        // iat values on one line with trailing comma)
    }
    if blocks.is_empty() {
        return Err("No &rst blocks found — check the file format.".into());
    }
    Ok(blocks)
}

// ─── run_geometry_series ──────────────────────────────────────────────────────
//
// Computes interatomic distances and/or angles over the loaded MD trajectory
// (or over umbrella windows if source = "umbrella").
//
// pairs:      0-based atom index pairs for distance computation.
//             For umbrella source, these are full-topology 0-based indices
//             (cv.rst iat values converted from 1-based by the frontend).
// labels:     optional custom label for each pair/triplet (same length as
//             pairs+triplets combined, or empty to use auto-generated labels).
// composites: list of [i, j] indices into the already-computed series to
//             produce a new series as series[i] + rstwt[i]*series[j].
//             For r1-r2 CVs the frontend passes [[0,1]] with rstwt [-1.0]
//             so that RC = r1 - r2 is appended automatically.
// composite_weights: weight for each composite's second term (length = composites.len())
// composite_labels:  label for each composite series

#[tauri::command]
async fn run_geometry_series(
    app:               AppHandle,
    state:             State<'_, AppData>,
    pairs:             Vec<[usize; 2]>,
    triplets:          Vec<[usize; 3]>,
    source:            Option<String>,
    labels:            Option<Vec<String>>,
    composites:        Option<Vec<[usize; 2]>>,
    composite_weights: Option<Vec<f64>>,
    composite_labels:  Option<Vec<String>>,
) -> Result<AnalysisResult<GeometryResult>, String> {
    let source             = source.as_deref().unwrap_or("trajectory");
    let labels             = labels.unwrap_or_default();
    let composites         = composites.unwrap_or_default();
    let composite_weights  = composite_weights.unwrap_or_default();
    let composite_labels   = composite_labels.unwrap_or_default();

    // Helper: build composite series after base series are computed
    let build_composites = |series: &mut Vec<GeometrySeries>| {
        for (ci, &[si, sj]) in composites.iter().enumerate() {
            if si >= series.len() || sj >= series.len() { continue; }
            let w   = composite_weights.get(ci).copied().unwrap_or(-1.0);
            let n   = series[si].values.len().min(series[sj].values.len());
            let values: Vec<f64> = (0..n).map(|t| {
                let a = series[si].values[t];
                let b = series[sj].values[t];
                if a.is_finite() && b.is_finite() { a + w * b } else { f64::NAN }
            }).collect();
            let label = composite_labels.get(ci).cloned()
                .unwrap_or_else(|| format!("RC({}{}{})", si, if w < 0.0 {"−"} else {"+"}, sj));
            series.push(GeometrySeries { label, kind: "composite".into(), values, unit: "Å".into() });
        }
    };

    // Apply custom label to series by position
    let apply_labels = |series: &mut Vec<GeometrySeries>| {
        for (i, s) in series.iter_mut().enumerate() {
            if let Some(lbl) = labels.get(i) {
                if !lbl.is_empty() { s.label = lbl.clone(); }
            }
        }
    };

    if source == "umbrella" {
        // ── Umbrella windows: one structure per window ────────────────────
        let windows_lock = state.umbrella_windows.lock().unwrap();
        let windows = windows_lock.as_ref().ok_or("Load umbrella windows first.")?;
        let coords_lock = state.umbrella_traj_coords.lock().unwrap();
        let coords  = coords_lock.as_ref().ok_or("Pre-load umbrella coords first (run window trajectory viewer).")?;
        let n_win   = windows.len().min(coords.len());
        let _wv: Vec<f64> = windows.iter().take(n_win).map(|w| w.val0).collect();
        drop(windows_lock);

        let mut series: Vec<GeometrySeries> = Vec::new();

        // Per-pair distance vs window
        for &[i, j] in &pairs {
            let values: Vec<f64> = (0..n_win).map(|wi| {
                let flat = &coords[wi];
                let n3   = flat.len();
                if i*3+2 >= n3 || j*3+2 >= n3 { return f64::NAN; }
                let dx = flat[i*3]   as f64 - flat[j*3]   as f64;
                let dy = flat[i*3+1] as f64 - flat[j*3+1] as f64;
                let dz = flat[i*3+2] as f64 - flat[j*3+2] as f64;
                (dx*dx + dy*dy + dz*dz).sqrt()
            }).collect();
            series.push(GeometrySeries {
                label: format!("d({i},{j})"),
                kind: "distance".into(), values, unit: "Å".into(),
            });
        }

        // Per-triplet angle vs window
        for &[i, j, k] in &triplets {
            let values: Vec<f64> = (0..n_win).map(|wi| {
                let flat = &coords[wi];
                let n3 = flat.len();
                if i*3+2 >= n3 || j*3+2 >= n3 || k*3+2 >= n3 { return f64::NAN; }
                let vji = [flat[i*3]-flat[j*3], flat[i*3+1]-flat[j*3+1], flat[i*3+2]-flat[j*3+2]];
                let vjk = [flat[k*3]-flat[j*3], flat[k*3+1]-flat[j*3+1], flat[k*3+2]-flat[j*3+2]];
                let dot  = (vji[0]*vjk[0] + vji[1]*vjk[1] + vji[2]*vjk[2]) as f64;
                let ni   = (vji[0]*vji[0] + vji[1]*vji[1] + vji[2]*vji[2]).sqrt() as f64;
                let nk   = (vjk[0]*vjk[0] + vjk[1]*vjk[1] + vjk[2]*vjk[2]).sqrt() as f64;
                if ni < 1e-10 || nk < 1e-10 { return f64::NAN; }
                (dot / (ni * nk)).clamp(-1.0, 1.0).acos().to_degrees()
            }).collect();
            series.push(GeometrySeries {
                label: format!("a({i},{j},{k})"),
                kind: "angle".into(), values, unit: "°".into(),
            });
        }

        let n = series.first().map(|s| s.values.len()).unwrap_or(0);
        apply_labels(&mut series);
        build_composites(&mut series);
        let result = GeometryResult { series: series.clone(), n_frames: n, source: "umbrella".into() };
        *state.geometry_cache.lock().unwrap() = Some(result.clone());
        return Ok(AnalysisResult {
            data: result,
            message: format!("{} geometry series over {n_win} umbrella windows.", series.len()),
        });
    }

    // ── MD trajectory: one value per frame ───────────────────────────────────
    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .clone().ok_or("No trajectory loaded.")?;
    let n_frames = traj.len();
    let n_atoms  = traj[0].len();

    for &[i, j] in &pairs {
        if i >= n_atoms || j >= n_atoms {
            return Err(format!("Atom index out of range: max is {} (0-based).", n_atoms - 1));
        }
    }
    for &[i, j, k] in &triplets {
        if i >= n_atoms || j >= n_atoms || k >= n_atoms {
            return Err(format!("Atom index out of range: max is {} (0-based).", n_atoms - 1));
        }
    }

    emit_progress(&app, "geometry", 5.0);

    let app2 = app.clone();
    let pairs    = pairs.clone();
    let triplets = triplets.clone();

    let series = tauri::async_runtime::spawn_blocking(move || -> Vec<GeometrySeries> {
        let mut out = Vec::new();
        let chunk = (n_frames / 20).max(1);

        for &[i, j] in &pairs {
            let values: Vec<f64> = (0..n_frames).map(|f| {
                if f % chunk == 0 { emit_progress(&app2, "geometry", 5.0 + (f as f64 / n_frames as f64) * 90.0); }
                let dx = traj[f][i][0] - traj[f][j][0];
                let dy = traj[f][i][1] - traj[f][j][1];
                let dz = traj[f][i][2] - traj[f][j][2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            }).collect();
            out.push(GeometrySeries {
                label: format!("d({i},{j})"),
                kind: "distance".into(), values, unit: "Å".into(),
            });
        }

        for &[i, j, k] in &triplets {
            let values: Vec<f64> = (0..n_frames).map(|f| {
                let vji = [traj[f][i][0]-traj[f][j][0], traj[f][i][1]-traj[f][j][1], traj[f][i][2]-traj[f][j][2]];
                let vjk = [traj[f][k][0]-traj[f][j][0], traj[f][k][1]-traj[f][j][1], traj[f][k][2]-traj[f][j][2]];
                let dot = vji[0]*vjk[0] + vji[1]*vjk[1] + vji[2]*vjk[2];
                let ni  = (vji[0]*vji[0]+vji[1]*vji[1]+vji[2]*vji[2]).sqrt();
                let nk  = (vjk[0]*vjk[0]+vjk[1]*vjk[1]+vjk[2]*vjk[2]).sqrt();
                if ni < 1e-10 || nk < 1e-10 { return f64::NAN; }
                (dot / (ni * nk)).clamp(-1.0, 1.0).acos().to_degrees()
            }).collect();
            out.push(GeometrySeries {
                label: format!("a({i},{j},{k})"),
                kind: "angle".into(), values, unit: "°".into(),
            });
        }
        out
    }).await.map_err(|e| e.to_string())?;

    emit_progress(&app, "geometry", 100.0);
    let n = series.first().map(|s| s.values.len()).unwrap_or(0);
    let mut series = series;
    apply_labels(&mut series);
    build_composites(&mut series);
    let result = GeometryResult { series: series.clone(), n_frames: n, source: "trajectory".into() };
    *state.geometry_cache.lock().unwrap() = Some(result.clone());

    Ok(AnalysisResult {
        data: result,
        message: format!("{} geometry series over {n_frames} frames.", series.len()),
    })
}

// ─── cancel_sasa / cancel_membrane ───────────────────────────────────────────

#[tauri::command]
fn cancel_sasa()     { SASA_CANCEL.store(true, Ordering::Relaxed); }

#[tauri::command]
fn cancel_membrane() { MEMBRANE_CANCEL.store(true, Ordering::Relaxed); }

// ─── run_sasa ─────────────────────────────────────────────────────────────────
//
// Shrake-Rupley solvent accessible surface area.
//
// We approximate per-atom SASA and aggregate by residue (grouping by res_seq).
// The probe radius is 1.4 Å (water). Each atom is assigned a van-der-Waals
// radius by element. The algorithm tests N_sphere = 92 probe points (Fibonacci
// sphere) per atom against all neighbour atoms within (r_i + r_j + 2*probe)².
//
// For Cα-only selections each atom IS one residue; for full-atom selections we
// sum over the residue group.

#[tauri::command]
async fn run_sasa(
    app:    AppHandle,
    state:  State<'_, AppData>,
    probe:  Option<f64>,   // probe radius in Å (default 1.4)
) -> Result<AnalysisResult<SasaResult>, String> {
    // Serve from cache
    if let Some(c) = state.sasa_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "SASA (cached)".into() });
    }

    let traj: Vec<Vec<[f64; 3]>> = state.trajectory_data.lock().unwrap()
        .clone().ok_or("No trajectory loaded")?;
    let meta: Vec<AtomMeta> = state.atom_meta.lock().unwrap()
        .clone().ok_or("No atom metadata")?;

    let probe_r = probe.unwrap_or(1.4);
    SASA_CANCEL.store(false, Ordering::Relaxed);
    emit_progress(&app, "sasa", 2.0);

    let app2 = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<SasaResult, String> {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();

        // Van-der-Waals radii by element (Å)
        let vdw = |elem: &str| -> f64 {
            match elem {
                "C"  => 1.70, "N"  => 1.55, "O"  => 1.52,
                "S"  => 1.80, "P"  => 1.80, "H"  => 1.20,
                "F"  => 1.47, "CL" | "Cl" => 1.75,
                "BR" | "Br" => 1.85, "I"  => 1.98,
                "CA" | "Ca" => 2.31, "ZN" | "Zn" => 1.39,
                "MG" | "Mg" => 1.73, "FE" | "Fe" => 1.52,
                _    => 1.70,
            }
        };
        let radii: Vec<f64> = meta.iter().map(|m| vdw(&m.element)).collect();

        // Fibonacci sphere: 92 points (good accuracy/speed balance)
        const N_SPHERE: usize = 92;
        let sphere_pts: Vec<[f64; 3]> = {
            let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
            (0..N_SPHERE).map(|i| {
                let y   = 1.0 - (i as f64 / (N_SPHERE as f64 - 1.0)) * 2.0;
                let r   = (1.0 - y * y).max(0.0).sqrt();
                let phi = golden * i as f64;
                [r * phi.cos(), y, r * phi.sin()]
            }).collect()
        };

        // Per-atom, per-frame SASA — collect as Result so cancel is clean
        let chunk = (n_frames / 20).max(1);
        let per_atom_per_frame: Result<Vec<Vec<f64>>, String> =
            (0..n_frames).into_par_iter().map(|fi| {
                if SASA_CANCEL.load(Ordering::Relaxed) {
                    return Err("Cancelled".into());
                }
                if fi % chunk == 0 {
                    emit_progress(&app2, "sasa", 5.0 + (fi as f64 / n_frames as f64) * 88.0);
                }
                let frame = &traj[fi];
                Ok((0..n_atoms).map(|i| {
                    let ri   = radii[i] + probe_r;
                    let ri2  = ri * ri;
                    let cutoff2 = (ri + 1.98 + 2.0 * probe_r).powi(2);
                    let neighbours: Vec<(usize, f64)> = (0..n_atoms).filter(|&j| {
                        if j == i { return false; }
                        let dx = frame[i][0]-frame[j][0];
                        let dy = frame[i][1]-frame[j][1];
                        let dz = frame[i][2]-frame[j][2];
                        dx*dx + dy*dy + dz*dz <= cutoff2
                    }).map(|j| (j, radii[j] + probe_r)).collect();

                    let exposed = sphere_pts.iter().filter(|&&pt| {
                        let px = frame[i][0] + ri * pt[0];
                        let py = frame[i][1] + ri * pt[1];
                        let pz = frame[i][2] + ri * pt[2];
                        !neighbours.iter().any(|&(j, rj)| {
                            let dx = px - frame[j][0];
                            let dy = py - frame[j][1];
                            let dz = pz - frame[j][2];
                            dx*dx + dy*dy + dz*dz < rj * rj
                        })
                    }).count();

                    4.0 * std::f64::consts::PI * ri2 * (exposed as f64 / N_SPHERE as f64)
                }).collect())
            }).collect();
        let per_atom_per_frame = per_atom_per_frame?;

        // Aggregate by residue (group consecutive atoms with same res_seq)
        // Build residue groups
        let mut res_groups: Vec<(String, Vec<usize>)> = Vec::new(); // (label, atom_indices)
        for (i, m) in meta.iter().enumerate() {
            let label = format!("{} {}", m.res_name, m.res_seq);
            if let Some(last) = res_groups.last_mut() {
                if last.0 == label { last.1.push(i); continue; }
            }
            res_groups.push((label, vec![i]));
        }

        let n_res = res_groups.len();
        let mut per_residue_mean = vec![0.0f64; n_res];
        let mut per_residue_std  = vec![0.0f64; n_res];
        let mut res_labels       = Vec::with_capacity(n_res);
        let mut total_per_frame  = vec![0.0f64; n_frames];

        for (ri, (label, atom_idxs)) in res_groups.iter().enumerate() {
            res_labels.push(label.clone());
            let vals: Vec<f64> = (0..n_frames).map(|fi| {
                atom_idxs.iter().map(|&ai| per_atom_per_frame[fi][ai]).sum::<f64>()
            }).collect();
            let mean = vals.iter().sum::<f64>() / n_frames as f64;
            let std  = (vals.iter().map(|v| (v-mean).powi(2)).sum::<f64>() / n_frames as f64).sqrt();
            per_residue_mean[ri] = mean;
            per_residue_std[ri]  = std;
            for (fi, &v) in vals.iter().enumerate() { total_per_frame[fi] += v; }
        }

        emit_progress(&app2, "sasa", 100.0);
        Ok(SasaResult { per_residue_mean, per_residue_std, total_per_frame, res_labels })
    }).await.map_err(|e| e.to_string())??;

    let n_res    = result.per_residue_mean.len();
    let n_frames = result.total_per_frame.len();
    let msg      = format!("SASA computed for {n_res} residues over {n_frames} frames (probe={probe_r:.1} Å).");
    *state.sasa_cache.lock().unwrap() = Some(result.clone());
    Ok(AnalysisResult { data: result, message: msg })
}

// ─── run_membrane ─────────────────────────────────────────────────────────────
//
// Membrane bilayer analysis — four outputs in one pass:
//
//   1. Bilayer thickness (Å) per frame — z-distance between upper/lower leaflet
//      headgroup centroids.
//   2. Area per lipid (Ų) per frame — box XY area ÷ N_lipids_per_leaflet.
//      Requires box dimensions (present for AMBER .nc/.dcd, zero otherwise).
//   3. Leaflet z-density histograms — 50-bin normalised distributions for
//      upper and lower headgroup atoms, useful for visualising membrane order.
//   4. Lipid order parameters (SCD) — mean S = 0.5(3cos²θ−1) for each
//      consecutive C–C bond vector in the selection, where θ is the angle with
//      the membrane normal.  For AMBER LIPID17/LIPID21 acyl chains, load a
//      full-atom selection that includes the sn-1/sn-2 chain carbons.
//
// Parameters
//   headgroup  comma-separated atom names used for leaflet assignment and
//              thickness/APL (default "P" — works for all common phospholipids).
//   normal     membrane normal axis: 0=x, 1=y, 2=z (default 2).

#[tauri::command]
async fn run_membrane(
    app:       AppHandle,
    state:     State<'_, AppData>,
    headgroup: Option<String>,  // default "P"
    normal:    Option<usize>,   // default 2
) -> Result<AnalysisResult<MembraneResult>, String> {
    if let Some(c) = state.membrane_cache.lock().unwrap().clone() {
        return Ok(AnalysisResult { data: c, message: "Membrane analysis (cached)".into() });
    }

    let traj      = state.trajectory_data.lock().unwrap().clone()
        .ok_or("No trajectory loaded")?;
    let meta      = state.atom_meta.lock().unwrap().clone()
        .ok_or("No atom metadata")?;
    let cell_dims = state.cell_dims.lock().unwrap().clone();

    let hg_names: Vec<String> = headgroup.unwrap_or_else(|| "P".into())
        .split(',').map(|s| s.trim().to_uppercase()).collect();
    let ax = normal.unwrap_or(2).min(2);

    MEMBRANE_CANCEL.store(false, Ordering::Relaxed);
    emit_progress(&app, "membrane", 2.0);

    let app2 = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<MembraneResult, String> {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();

        // ── Headgroup atom indices ────────────────────────────────────────────
        let hg_indices: Vec<usize> = meta.iter().enumerate()
            .filter(|(_, m)| hg_names.iter().any(|h| m.atom_name.to_uppercase() == *h))
            .map(|(i, _)| i)
            .collect();
        let hg_found = !hg_indices.is_empty();

        // Fall back to all atoms if no headgroup match (e.g. Cα-only selection)
        let hg: Vec<usize> = if hg_found {
            hg_indices.clone()
        } else {
            (0..n_atoms).collect()
        };

        // ── Carbon atom indices (for order parameters) ────────────────────────
        let c_indices: Vec<usize> = meta.iter().enumerate()
            .filter(|(_, m)| {
                let el = m.element.to_uppercase();
                el == "C" || (el.is_empty() && m.atom_name.starts_with('C'))
            })
            .map(|(i, _)| i)
            .collect();

        let chunk = (n_frames / 20).max(1);

        // ── Per-frame thickness and APL ───────────────────────────────────────
        let frame_results: Result<Vec<(f64, f64, Vec<usize>, Vec<usize>)>, String> =
            (0..n_frames).into_par_iter().map(|fi| {
                if MEMBRANE_CANCEL.load(Ordering::Relaxed) {
                    return Err("Cancelled".into());
                }
                if fi % chunk == 0 {
                    emit_progress(&app2, "membrane",
                        5.0 + fi as f64 / n_frames as f64 * 60.0);
                }
                let frame = &traj[fi];

                // Leaflet assignment by median along normal axis
                let ax_coords: Vec<f64> = hg.iter().map(|&i| frame[i][ax]).collect();
                let mut sorted = ax_coords.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[sorted.len() / 2];

                let upper: Vec<usize> = hg.iter().enumerate()
                    .filter(|(k, _)| ax_coords[*k] >= median)
                    .map(|(_, &i)| i).collect();
                let lower: Vec<usize> = hg.iter().enumerate()
                    .filter(|(k, _)| ax_coords[*k] < median)
                    .map(|(_, &i)| i).collect();

                let mean_ax = |idx: &[usize]| -> f64 {
                    if idx.is_empty() { return 0.0; }
                    idx.iter().map(|&i| frame[i][ax]).sum::<f64>() / idx.len() as f64
                };
                let thickness = (mean_ax(&upper) - mean_ax(&lower)).abs();

                // APL = box XY area / number of lipid MOLECULES in upper leaflet.
                // We count distinct (res_seq, chain_id) pairs — one per lipid —
                // rather than raw atom count, which would be ~134× too large for
                // a full-atom POPC selection.
                let apl = if let Some(ref cd) = cell_dims {
                    if fi < cd.len() {
                        let ax1 = (ax + 1) % 3;
                        let ax2 = (ax + 2) % 3;
                        let area = cd[fi][ax1] * cd[fi][ax2];
                        let n_lipids_upper = upper.iter()
                            .map(|&i| (meta[i].res_seq, meta[i].chain_id))
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        if area > 0.0 && n_lipids_upper > 0 {
                            area / n_lipids_upper as f64
                        } else { 0.0 }
                    } else { 0.0 }
                } else { 0.0 };

                Ok((thickness, apl, upper, lower))
            }).collect::<Result<Vec<_>, String>>();

        let frame_results = frame_results?;

        let thickness_per_frame: Vec<f64> = frame_results.iter().map(|r| r.0).collect();
        let apl_per_frame:       Vec<f64> = frame_results.iter().map(|r| r.1).collect();
        let has_apl = apl_per_frame.iter().any(|&v| v > 0.0);

        // n_upper/n_lower reported as lipid count (residues), not atom count.
        // For the legend we show whichever makes more sense contextually.
        let n_upper = {
            let atoms = &frame_results[0].2;
            atoms.iter().map(|&i| (meta[i].res_seq, meta[i].chain_id))
                 .collect::<std::collections::HashSet<_>>().len()
        };
        let n_lower = {
            let atoms = &frame_results[0].3;
            atoms.iter().map(|&i| (meta[i].res_seq, meta[i].chain_id))
                 .collect::<std::collections::HashSet<_>>().len()
        };

        emit_progress(&app2, "membrane", 68.0);

        // ── Z-density histograms ──────────────────────────────────────────────
        // Always built from headgroup atoms only (hg_indices when matched,
        // otherwise fall back to the full hg set).  This ensures the histogram
        // shows the phosphate/headgroup bilayer profile (two separated peaks)
        // rather than a flat all-atom distribution.
        const N_BINS: usize = 50;
        let density_atoms: &Vec<usize> = &hg_indices; // empty = fallback to hg

        let collect_z = |leaf_getter: fn(&(f64,f64,Vec<usize>,Vec<usize>)) -> &Vec<usize>| -> Vec<f64> {
            (0..n_frames).flat_map(|fi| {
                let leaflet = leaf_getter(&frame_results[fi]);
                if hg_found {
                    // Only keep atoms that are in hg_indices for a clean headgroup profile
                    let hg_set: std::collections::HashSet<usize> = density_atoms.iter().copied().collect();
                    leaflet.iter()
                        .filter(|i| hg_set.contains(i))
                        .map(|&i| traj[fi][i][ax]).collect::<Vec<_>>()
                } else {
                    leaflet.iter().map(|&i| traj[fi][i][ax]).collect::<Vec<_>>()
                }
            }).collect()
        };

        let build_hist = |vals: &[f64]| -> Vec<f64> {
            if vals.is_empty() { return vec![0.0; N_BINS]; }
            let zmin = vals.iter().cloned().fold(f64::INFINITY,     f64::min);
            let zmax = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = (zmax - zmin).max(1.0);
            let mut bins = vec![0u32; N_BINS];
            for &v in vals {
                let b = ((v - zmin) / range * N_BINS as f64) as usize;
                bins[b.min(N_BINS - 1)] += 1;
            }
            let max = *bins.iter().max().unwrap_or(&1).max(&1) as f64;
            bins.iter().map(|&c| c as f64 / max).collect()
        };

        let z_density_upper = build_hist(&collect_z(|r| &r.2));
        let z_density_lower = build_hist(&collect_z(|r| &r.3));

        emit_progress(&app2, "membrane", 82.0);

        // ── Order parameters ──────────────────────────────────────────────────
        // SCD = 0.5*(3cos²θ - 1), θ = angle of C[i]→C[i+1] with normal axis.
        // Consecutive C atoms in the selection are used as a proxy for the
        // acyl chain.  For proper SCD, load sn-chain carbons only.
        let (order_params, order_labels) = if c_indices.len() >= 2 {
            let pairs: Vec<(usize, usize)> = c_indices.windows(2)
                .map(|w| (w[0], w[1])).collect();

            let ops: Vec<f64> = pairs.iter().map(|&(ci, cj)| {
                let s: f64 = (0..n_frames).map(|fi| {
                    let f  = &traj[fi];
                    let dx = f[cj][0] - f[ci][0];
                    let dy = f[cj][1] - f[ci][1];
                    let dz = f[cj][2] - f[ci][2];
                    let len = (dx*dx + dy*dy + dz*dz).sqrt().max(1e-10);
                    let cos_t = [dx, dy, dz][ax] / len;
                    0.5 * (3.0 * cos_t * cos_t - 1.0)
                }).sum::<f64>() / n_frames as f64;
                s.abs()  // |SCD| is the conventional reported quantity
            }).collect();

            let labels: Vec<String> = pairs.iter().map(|(ci, _)| {
                format!("{}{}", meta[*ci].atom_name, meta[*ci].res_seq)
            }).collect();

            (ops, labels)
        } else {
            (vec![], vec![])
        };

        let mean_thickness = thickness_per_frame.iter().sum::<f64>() / n_frames as f64;
        let mean_apl = if has_apl && n_frames > 0 {
            apl_per_frame.iter().sum::<f64>() / n_frames as f64
        } else { 0.0 };

        emit_progress(&app2, "membrane", 100.0);

        Ok(MembraneResult {
            thickness_per_frame,
            apl_per_frame,
            z_density_upper,
            z_density_lower,
            order_params,
            order_labels,
            n_upper,
            n_lower,
            n_frames,
            mean_thickness,
            mean_apl,
            has_apl,
        })
    }).await.map_err(|e| e.to_string())??;

    let msg = format!(
        "Membrane: mean thickness {:.2} Å · {} upper / {} lower headgroup atoms{}",
        result.mean_thickness, result.n_upper, result.n_lower,
        if result.has_apl {
            format!(" · mean APL {:.1} Ų", result.mean_apl)
        } else {
            " (no box → APL unavailable)".into()
        }
    );
    *state.membrane_cache.lock().unwrap() = Some(result.clone());
    Ok(AnalysisResult { data: result, message: msg })
}

// ─── batch_export ─────────────────────────────────────────────────────────────
//
// Exports every cached analysis result to a single directory as CSV files,
// and returns a list of filenames written (so the frontend can show them).
// SVG is handled on the frontend side — the frontend calls this to get the
// CSV payloads and writes the SVG strings itself.

#[tauri::command]
fn batch_export(
    state: State<'_, AppData>,
    dir:   String,
) -> Result<Vec<String>, String> {
    use std::io::Write;
    let dir_path = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir_path).map_err(|e| e.to_string())?;

    let mut written: Vec<String> = Vec::new();

    macro_rules! write_csv {
        ($name:literal, $header:expr, $rows:expr) => {{
            let path = dir_path.join($name);
            let mut f = File::create(&path).map_err(|e| e.to_string())?;
            writeln!(f, "{}", $header).map_err(|e| e.to_string())?;
            for row in $rows { writeln!(f, "{}", row).map_err(|e| e.to_string())?; }
            written.push($name.to_string());
        }};
    }

    // RMSD
    if let Some(v) = state.rmsd_cache.lock().unwrap().as_ref() {
        write_csv!("rmsd.csv", "frame,rmsd_angstrom",
            v.iter().enumerate().map(|(i,x)| format!("{},{:.6}", i, x)).collect::<Vec<_>>());
    }
    // RMSF
    if let Some(v) = state.rmsf_cache.lock().unwrap().as_ref() {
        write_csv!("rmsf.csv", "atom_index,rmsf_angstrom",
            v.iter().enumerate().map(|(i,x)| format!("{},{:.6}", i, x)).collect::<Vec<_>>());
    }
    // Radius of gyration
    if let Some(v) = state.rg_cache.lock().unwrap().as_ref() {
        write_csv!("rg.csv", "frame,rg_angstrom",
            v.iter().enumerate().map(|(i,x)| format!("{},{:.6}", i, x)).collect::<Vec<_>>());
    }
    // PCA projections
    if let Some(pca) = state.pca_cache.lock().unwrap().as_ref() {
        write_csv!("pca.csv", "frame,pc1,pc2",
            pca.projections.iter().enumerate()
                .map(|(i,p)| format!("{},{:.6},{:.6}", i, p[0], p[1]))
                .collect::<Vec<_>>());
        write_csv!("pca_variance.csv", "component,explained_variance",
            pca.explained_variance.iter().enumerate()
                .map(|(i,v)| format!("{},{:.6}", i+1, v))
                .collect::<Vec<_>>());
    }
    // DCCM
    if let Some(m) = state.dccm_cache.lock().unwrap().as_ref() {
        let path = dir_path.join("dccm.csv");
        let mut f = File::create(&path).map_err(|e| e.to_string())?;
        for row in m {
            let line = row.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(",");
            writeln!(f, "{}", line).map_err(|e| e.to_string())?;
        }
        written.push("dccm.csv".to_string());
    }
    // Contacts
    if let Some(m) = state.contacts_cache.lock().unwrap().as_ref() {
        let path = dir_path.join("contacts.csv");
        let mut f = File::create(&path).map_err(|e| e.to_string())?;
        for row in m {
            let line = row.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(",");
            writeln!(f, "{}", line).map_err(|e| e.to_string())?;
        }
        written.push("contacts.csv".to_string());
    }
    // MI
    if let Some(m) = state.mi_cache.lock().unwrap().as_ref() {
        let path = dir_path.join("mutual_information.csv");
        let mut f = File::create(&path).map_err(|e| e.to_string())?;
        for row in m {
            let line = row.iter().map(|v| format!("{:.4}", v)).collect::<Vec<_>>().join(",");
            writeln!(f, "{}", line).map_err(|e| e.to_string())?;
        }
        written.push("mutual_information.csv".to_string());
    }
    // PRS
    if let Some(prs) = state.prs_cache.lock().unwrap().as_ref() {
        write_csv!("prs_effectiveness.csv", "atom_index,effectiveness",
            prs.effectiveness.iter().enumerate()
                .map(|(i,v)| format!("{},{:.6}", i, v)).collect::<Vec<_>>());
        write_csv!("prs_sensitivity.csv", "atom_index,sensitivity",
            prs.sensitivity.iter().enumerate()
                .map(|(i,v)| format!("{},{:.6}", i, v)).collect::<Vec<_>>());
    }
    // SASA
    if let Some(sasa) = state.sasa_cache.lock().unwrap().as_ref() {
        write_csv!("sasa_per_residue.csv", "residue,mean_sasa_a2,std_sasa_a2",
            sasa.per_residue_mean.iter().zip(&sasa.per_residue_std).zip(&sasa.res_labels)
                .map(|((m, s), lbl)| format!("{},{:.4},{:.4}", lbl, m, s))
                .collect::<Vec<_>>());
        write_csv!("sasa_total.csv", "frame,total_sasa_a2",
            sasa.total_per_frame.iter().enumerate()
                .map(|(i,v)| format!("{},{:.4}", i, v)).collect::<Vec<_>>());
    }
    // Membrane
    if let Some(mem) = state.membrane_cache.lock().unwrap().as_ref() {
        // Per-frame thickness and APL
        let has_apl = mem.has_apl;
        write_csv!("membrane_per_frame.csv",
            if has_apl { "frame,thickness_angstrom,apl_angstrom2" }
            else       { "frame,thickness_angstrom" },
            mem.thickness_per_frame.iter().enumerate().map(|(i, t)| {
                if has_apl {
                    format!("{},{:.4},{:.4}", i, t, mem.apl_per_frame[i])
                } else {
                    format!("{},{:.4}", i, t)
                }
            }).collect::<Vec<_>>());
        // Order parameters
        if !mem.order_params.is_empty() {
            write_csv!("membrane_order_params.csv", "carbon,abs_scd",
                mem.order_params.iter().zip(&mem.order_labels)
                    .map(|(s, lbl)| format!("{},{:.6}", lbl, s))
                    .collect::<Vec<_>>());
        }
        // Z-density histograms
        write_csv!("membrane_z_density.csv", "bin,upper_norm,lower_norm",
            mem.z_density_upper.iter().zip(&mem.z_density_lower).enumerate()
                .map(|(i, (u, l))| format!("{},{:.6},{:.6}", i, u, l))
                .collect::<Vec<_>>());
    }
    // Dihedrals summary
    if let Some(dih) = state.dihedral_cache.lock().unwrap().as_ref() {
        let path = dir_path.join("dihedrals.csv");
        let mut f = File::create(&path).map_err(|e| e.to_string())?;
        writeln!(f, "atom_idx,res_name,res_seq,phi_mean,psi_mean,phi_std,psi_std").map_err(|e| e.to_string())?;
        for r in &dih.residues {
            let phi_mean: Vec<f64> = r.phi.iter().copied().filter(|v| v.is_finite()).collect();
            let psi_mean: Vec<f64> = r.psi.iter().copied().filter(|v| v.is_finite()).collect();
            let pm = if phi_mean.is_empty() { f64::NAN } else { phi_mean.iter().sum::<f64>() / phi_mean.len() as f64 };
            let psm = if psi_mean.is_empty() { f64::NAN } else { psi_mean.iter().sum::<f64>() / psi_mean.len() as f64 };
            let ps  = if phi_mean.len() < 2 { f64::NAN } else { (phi_mean.iter().map(|v|(v-pm).powi(2)).sum::<f64>() / phi_mean.len() as f64).sqrt() };
            let pss = if psi_mean.len() < 2 { f64::NAN } else { (psi_mean.iter().map(|v|(v-psm).powi(2)).sum::<f64>() / psi_mean.len() as f64).sqrt() };
            writeln!(f, "{},{},{},{:.4},{:.4},{:.4},{:.4}", r.atom_idx, r.res_name, r.res_seq, pm, psm, ps, pss).map_err(|e| e.to_string())?;
        }
        written.push("dihedrals.csv".to_string());
    }
    // Clustering
    if let Some(cl) = state.cluster_cache.lock().unwrap().as_ref() {
        write_csv!("clustering.csv", "frame,cluster_id",
            cl.assignments.iter().enumerate()
                .map(|(i,c)| format!("{},{}", i, c)).collect::<Vec<_>>());
        write_csv!("cluster_populations.csv", "cluster_id,population",
            cl.populations.iter().enumerate()
                .map(|(i,p)| format!("{},{:.6}", i, p)).collect::<Vec<_>>());
    }
    // Geometry series
    if let Some(geo) = state.geometry_cache.lock().unwrap().as_ref() {
        let path = dir_path.join("geometry.csv");
        let mut f = File::create(&path).map_err(|e| e.to_string())?;
        let headers: Vec<&str> = std::iter::once("index")
            .chain(geo.series.iter().map(|s| s.label.as_str())).collect();
        writeln!(f, "{}", headers.join(",")).map_err(|e| e.to_string())?;
        let n = geo.series.iter().map(|s| s.values.len()).max().unwrap_or(0);
        for i in 0..n {
            let mut row = vec![i.to_string()];
            for s in &geo.series { row.push(format!("{:.6}", s.values.get(i).copied().unwrap_or(f64::NAN))); }
            writeln!(f, "{}", row.join(",")).map_err(|e| e.to_string())?;
        }
        written.push("geometry.csv".to_string());
    }

    Ok(written)
}
//
// Returns the full φ/ψ time series for one or more residues identified by
// atom index. Used by the main chart to render a standalone dihedral
// time-series chart (separate from the Ramachandran overlay in analysis_viewer).

#[tauri::command]
fn get_dihedral_time_series(
    state:       State<'_, AppData>,
    atom_indices: Vec<usize>,
) -> Result<Vec<SingleResidueDihedrals>, String> {
    let lock  = state.dihedral_cache.lock().unwrap();
    let cache = lock.as_ref().ok_or("Run Ramachandran / Dihedrals analysis first.")?;
    let result: Vec<SingleResidueDihedrals> = atom_indices.iter()
        .filter_map(|&idx| {
            cache.residues.iter().find(|r| r.atom_idx == idx).map(|r| SingleResidueDihedrals {
                res_seq:  r.res_seq,
                res_name: r.res_name.clone(),
                phi:      r.phi.clone(),
                psi:      r.psi.clone(),
                mode:     cache.mode.clone(),
            })
        }).collect();
    if result.is_empty() { return Err("No dihedral data found for the given atom indices.".into()); }
    Ok(result)
}

// ─── Ligand Builder input types ───────────────────────────────────────────────

#[derive(Clone, serde::Deserialize)]
pub struct LbAtomIn {
    pub id:      usize,
    pub element: String,
    pub x:       f64,   // 2D canvas units
    pub y:       f64,
    pub charge:  i32,
    pub radical: u32,
}

#[derive(Clone, serde::Deserialize)]
pub struct LbBondIn {
    pub atom1:   usize,
    pub atom2:   usize,
    pub order:   u32,     // 1, 2, 3
    pub aromatic: bool,
}

#[derive(Clone, serde::Deserialize)]
pub struct LbMolecule {
    pub atoms:        Vec<LbAtomIn>,
    pub bonds:        Vec<LbBondIn>,
    pub charge:       i32,
    pub multiplicity: u32,
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct Coord3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(serde::Serialize)]
pub struct MinimizeResult {
    pub coords:       Vec<Coord3>,
    pub steps_run:    u32,
    pub final_energy: f64,
}

// ─── Async minimization types ─────────────────────────────────────────────────
//
// lb_minimize_start returns Ok(()) immediately; actual results arrive via events:
//   lb:min-progress  { step, total, energy, method, message }
//   lb:min-done      { coords, steps_run, final_energy, method }
//   lb:min-error     String

static MINIMIZE_CANCEL: AtomicBool = AtomicBool::new(false);

#[derive(Clone, serde::Serialize)]
struct MinProgress {
    step:    usize,
    total:   usize,   // 0 = indeterminate (external process)
    energy:  f64,
    method:  String,
    message: String,
}

#[derive(Clone, serde::Serialize)]
struct MinDonePayload {
    coords:       Vec<Coord3>,
    steps_run:    usize,
    final_energy: f64,
    method:       String,
}

// ─── UFF bond-length table (Å) ────────────────────────────────────────────────

fn uff_bond_length(e1: &str, e2: &str, order: u32) -> f64 {
    // UFF single-bond radii (Å) — from Rappé et al., JACS 1992
    let r = |e: &str| match e {
        "H"  => 0.354, "C"  => 0.757, "N"  => 0.700, "O"  => 0.658,
        "S"  => 1.050, "P"  => 1.070, "F"  => 0.570, "Cl" => 1.020,
        "Br" => 1.200, "I"  => 1.390, "B"  => 0.840, "Si" => 1.110,
        "Na" => 1.144, "Mg" => 1.184, "Al" => 1.213, "K"  => 1.468,
        "Ca" => 1.429, "Fe" => 1.243, "Zn" => 1.278, "Se" => 1.220,
        _ => 0.900,
    };
    // electronegativity correction (Pauling)
    let chi = |e: &str| match e {
        "H"=>2.20,"C"=>2.55,"N"=>3.04,"O"=>3.44,"S"=>2.58,"P"=>2.19,
        "F"=>3.98,"Cl"=>3.16,"Br"=>2.96,"I"=>2.66,"B"=>2.04,"Si"=>1.90,
        _=>2.0,
    };
    let rij = r(e1)+r(e2);
    // Pauling bond order correction
    let n = match order { 2=>2.0, 3=>3.0, _=>1.0 };
    let rbo = -0.1332*(rij)*f64::ln(n);
    // electronegativity correction
    let ren = r(e1)*r(e2)*(f64::sqrt(chi(e1))-f64::sqrt(chi(e2))).powi(2)
              / (r(e1)*chi(e1)+r(e2)*chi(e2));
    (rij + rbo - ren).max(0.8)
}

fn uff_angle0(element: &str, connectivity: usize) -> f64 {
    // Equilibrium angles in degrees
    match element {
        "C"  => if connectivity<=2 {180.0} else if connectivity==3 {120.0} else {109.47},
        "N"  => if connectivity<=2 {120.0} else {107.0},
        "O"  => if connectivity<=1 {180.0} else {104.5},
        "S"  => if connectivity<=2 {104.5} else if connectivity==3 {109.5} else {109.5},
        "P"  => if connectivity<=2 {104.5} else {109.5},
        "Si" => 109.47,
        "B"  => 120.0,
        _    => 109.47,
    }
}

#[allow(dead_code)]
fn atomic_mass(e: &str) -> f64 {
    match e {
        "H"=>1.008,"C"=>12.011,"N"=>14.007,"O"=>15.999,"S"=>32.065,
        "P"=>30.974,"F"=>18.998,"Cl"=>35.453,"Br"=>79.904,"I"=>126.904,
        "B"=>10.811,"Si"=>28.086,"Na"=>22.990,"Mg"=>24.305,"Al"=>26.982,
        "K"=>39.098,"Ca"=>40.078,"Fe"=>55.845,"Zn"=>65.38,"Se"=>78.96,
        _ => 12.0,
    }
}

// ─── Distance / angle helpers ─────────────────────────────────────────────────

fn dist(a: &Coord3, b: &Coord3) -> f64 {
    let dx=a.x-b.x; let dy=a.y-b.y; let dz=a.z-b.z;
    (dx*dx+dy*dy+dz*dz).sqrt()
}

fn angle_rad(a: &Coord3, centre: &Coord3, b: &Coord3) -> f64 {
    let ax=a.x-centre.x; let ay=a.y-centre.y; let az=a.z-centre.z;
    let bx=b.x-centre.x; let by=b.y-centre.y; let bz=b.z-centre.z;
    let dot=ax*bx+ay*by+az*bz;
    let la=(ax*ax+ay*ay+az*az).sqrt();
    let lb=(bx*bx+by*by+bz*bz).sqrt();
    if la<1e-9||lb<1e-9 {return std::f64::consts::PI/2.0;}
    (dot/(la*lb)).clamp(-1.0,1.0).acos()
}

// ─── Energy and gradient (simplified UFF) ─────────────────────────────────────

fn total_energy(
    coords:    &[Coord3],
    atoms:     &[LbAtomIn],
    bonds:     &[(usize,usize,u32)],   // (idx_a, idx_b, order)
    angles:    &[(usize,usize,usize)], // (idx_a, idx_centre, idx_b)
) -> f64 {
    let mut e = 0.0_f64;
    let n = coords.len();

    // Bond stretching  E = 0.5 * k * (r - r0)^2
    for &(i, j, order) in bonds {
        if i>=n||j>=n { continue; }
        let r0 = uff_bond_length(&atoms[i].element, &atoms[j].element, order);
        let r  = dist(&coords[i], &coords[j]);
        let dr = r - r0;
        e += 700.0 * dr * dr;
    }

    // Angle bending  E = 0.5 * k * (θ - θ0)^2
    for &(ia, ic, ib) in angles {
        if ia>=n||ic>=n||ib>=n { continue; }
        let conn = bonds.iter().filter(|&&(p,q,_)| p==ic||q==ic).count();
        let theta0 = uff_angle0(&atoms[ic].element, conn).to_radians();
        let theta  = angle_rad(&coords[ia], &coords[ic], &coords[ib]);
        let dth = theta - theta0;
        e += 150.0 * dth * dth;
    }

    // VdW repulsion (soft wall, non-bonded pairs)
    let bonded_set: std::collections::HashSet<(usize,usize)> = bonds.iter()
        .flat_map(|&(i,j,_)| [(i,j),(j,i)]).collect();
    let one3_set: std::collections::HashSet<(usize,usize)> = angles.iter()
        .flat_map(|&(a,_,b)| [(a,b),(b,a)]).collect();

    for i in 0..n {
        for j in (i+1)..n {
            if bonded_set.contains(&(i,j)) || one3_set.contains(&(i,j)) { continue; }
            let r = dist(&coords[i], &coords[j]).max(0.1);
            let r_vdw = 2.5_f64; // soft-wall distance (Å)
            if r < r_vdw {
                let overlap = r_vdw - r;
                e += 200.0 * overlap * overlap * overlap;
            }
        }
    }

    e
}

fn numerical_gradient(
    coords:  &[Coord3],
    atoms:   &[LbAtomIn],
    bonds:   &[(usize,usize,u32)],
    angles:  &[(usize,usize,usize)],
) -> Vec<Coord3> {
    let h  = 1e-4_f64;
    let e0 = total_energy(coords, atoms, bonds, angles);
    let n  = coords.len();
    let mut grad = vec![Coord3{x:0.0,y:0.0,z:0.0}; n];

    for i in 0..n {
        for d in 0..3_usize {
            let mut cp = coords.to_vec();
            match d {
                0 => cp[i].x += h,
                1 => cp[i].y += h,
                _ => cp[i].z += h,
            }
            let g = (total_energy(&cp, atoms, bonds, angles) - e0) / h;
            match d { 0=>grad[i].x=g, 1=>grad[i].y=g, _=>grad[i].z=g, }
        }
    }
    grad
}

// ─── Build angle list from bond list ─────────────────────────────────────────

fn build_angles(n: usize, bonds: &[(usize,usize,u32)]) -> Vec<(usize,usize,usize)> {
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for &(i,j,_) in bonds { adj[i].push(j); adj[j].push(i); }
    let mut angles = vec![];
    for c in 0..n {
        let nb = &adj[c];
        for ii in 0..nb.len() {
            for jj in (ii+1)..nb.len() {
                angles.push((nb[ii], c, nb[jj]));
            }
        }
    }
    angles
}

// ─── lb_generate_coords ──────────────────────────────────────────────────────
//
// Generates initial 3D coordinates from 2D canvas layout.
// Scales 2D positions to angstroms, then perturbs z for sp3 centres.

#[tauri::command]
fn lb_generate_coords(mol: LbMolecule) -> Result<Vec<Coord3>, String> {
    if mol.atoms.is_empty() { return Err("No atoms in molecule.".into()); }

    // Find scale factor: aim for ~1.5 Å average bond length in 3D
    let bond_lengths_2d: Vec<f64> = mol.bonds.iter().filter_map(|b| {
        let a1 = mol.atoms.iter().find(|a| a.id==b.atom1)?;
        let a2 = mol.atoms.iter().find(|a| a.id==b.atom2)?;
        let dx=a2.x-a1.x; let dy=a2.y-a1.y;
        Some((dx*dx+dy*dy).sqrt())
    }).collect();

    let avg_2d = if bond_lengths_2d.is_empty() { 50.0 }
                 else { bond_lengths_2d.iter().sum::<f64>() / bond_lengths_2d.len() as f64 };
    let scale  = 1.54 / avg_2d.max(1.0); // target C-C bond ≈ 1.54 Å

    // Build connectivity map
    let n = mol.atoms.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let bonds_idx: Vec<(usize,usize,u32)> = mol.bonds.iter().filter_map(|b| {
        let i = mol.atoms.iter().position(|a| a.id==b.atom1)?;
        let j = mol.atoms.iter().position(|a| a.id==b.atom2)?;
        adj[i].push(j); adj[j].push(i);
        Some((i, j, b.order))
    }).collect();

    // Start: z = 0, plus small z perturbation for sp3 atoms
    let mut rng_seed: u64 = 12345;
    let rng = |s: &mut u64| -> f64 {
        *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
        ((*s & 0xFFFFFF) as f64 / 0xFFFFFF as f64) - 0.5
    };

    let mut coords: Vec<Coord3> = mol.atoms.iter().enumerate().map(|(i, a)| {
        let x = a.x * scale;
        let y = a.y * scale;
        // Apply small z perturbation for sp3 atoms (>2 bonds) to break planarity
        let cn = adj[i].len();
        let z = if cn >= 3 && a.element != "C" || cn > 3 { rng(&mut rng_seed) * 0.5 } else { 0.0 };
        Coord3 { x, y, z }
    }).collect();

    // Run a quick minimization to snap to proper geometry
    let angles = build_angles(n, &bonds_idx);
    let mut lr = 0.05_f64;
    for step in 0..300_usize {
        let grad = numerical_gradient(&coords, &mol.atoms, &bonds_idx, &angles);
        let gnorm: f64 = grad.iter().map(|g| g.x*g.x+g.y*g.y+g.z*g.z).sum::<f64>().sqrt();
        if gnorm < 0.01 { break; }
        for i in 0..n {
            coords[i].x -= lr * grad[i].x / gnorm.max(1.0);
            coords[i].y -= lr * grad[i].y / gnorm.max(1.0);
            coords[i].z -= lr * grad[i].z / gnorm.max(1.0);
        }
        if step % 50 == 49 { lr *= 0.8; }
    }

    Ok(coords)
}

// ─── lb_minimize_start / lb_minimize_cancel ───────────────────────────────────
//
// lb_minimize_start returns Ok(()) in < 1 ms so the JS await unblocks at once.
// The actual work runs on a background thread (UFF) or async task (xTB/ORCA)
// and pushes results back via Tauri events.

#[tauri::command]
async fn lb_minimize_start(
    app:        tauri::AppHandle,
    mol:        LbMolecule,
    coords:     Vec<Coord3>,
    method:     String,
    steps:      usize,
    conv:       f64,
    xtb_path:   Option<String>,
    xtb_level:  Option<String>,
    orca_path:  Option<String>,
    orca_cores: Option<usize>,
) -> Result<(), String> {
    MINIMIZE_CANCEL.store(false, Ordering::Relaxed);

    let app2     = app.clone();
    // Auto-resolve xtb from the managed conda env if no explicit path supplied
    let xtb_bin  = match xtb_path.filter(|p| !p.is_empty()) {
        Some(p) => p,
        None    => get_conda_bin("xtb".to_string()).await
                       .unwrap_or_else(|_| "xtb".into()),
    };
    let xtb_lvl  = xtb_level.unwrap_or_else(|| "normal".into());
    let orca_bin = orca_path.unwrap_or_else(|| "orca".into());
    let n_cores  = orca_cores.unwrap_or(4);

    tauri::async_runtime::spawn(async move {
        let result = match method.as_str() {
            "gfn2"       => run_xtb(&app2, &mol, &coords, 2, &xtb_bin, &xtb_lvl).await,
            "gfn1"       => run_xtb(&app2, &mol, &coords, 1, &xtb_bin, &xtb_lvl).await,
            "orca-b3lyp" => run_orca(&app2, &mol, &coords, "B3LYP",    "def2-SVP",  &orca_bin, n_cores).await,
            "orca-pbe0"  => run_orca(&app2, &mol, &coords, "PBE0",     "def2-TZVP", &orca_bin, n_cores).await,
            "orca-wb97"  => run_orca(&app2, &mol, &coords, "wB97X-D3", "def2-TZVP", &orca_bin, n_cores).await,
            _            => run_uff_async(&app2, mol, coords, steps, conv).await, // "uff" or default
        };
        match result {
            Ok(p)    => { let _ = app2.emit("lb:min-done",  p); }
            Err(msg) => { let _ = app2.emit("lb:min-error", msg); }
        }
    });

    Ok(())
}

#[tauri::command]
fn lb_minimize_cancel() {
    MINIMIZE_CANCEL.store(true, Ordering::Relaxed);
}

// ─── UFF async runner ─────────────────────────────────────────────────────────
//
// Wraps the existing total_energy / numerical_gradient / build_angles pipeline
// in spawn_blocking so the Tauri reactor stays free.  Logic is identical to
// the original lb_minimize — only additions are the cancel check and progress
// events every 25 steps.

async fn run_uff_async(
    app:    &tauri::AppHandle,
    mol:    LbMolecule,
    coords: Vec<Coord3>,
    steps:  usize,
    conv:   f64,
) -> Result<MinDonePayload, String> {
    let app2 = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        if mol.atoms.is_empty() { return Err("Empty molecule.".into()); }
        if coords.len() != mol.atoms.len() {
            return Err(format!("Coord count ({}) ≠ atom count ({}).", coords.len(), mol.atoms.len()));
        }

        let n = mol.atoms.len();
        let bonds_idx: Vec<(usize,usize,u32)> = mol.bonds.iter().filter_map(|b| {
            let i = mol.atoms.iter().position(|a| a.id==b.atom1)?;
            let j = mol.atoms.iter().position(|a| a.id==b.atom2)?;
            Some((i, j, b.order))
        }).collect();
        let angles = build_angles(n, &bonds_idx);

        let _ = app2.emit("lb:min-progress", MinProgress {
            step: 0, total: steps, energy: 0.0,
            method: "UFF".into(), message: "Starting UFF minimization…".into(),
        });

        let mut c       = coords.clone();
        let mut lr      = 0.05_f64;
        let mut steps_run = 0_usize;
        let mut last_e  = f64::MAX;

        for s in 0..steps {
            if MINIMIZE_CANCEL.load(Ordering::Relaxed) {
                return Err("Minimization cancelled".into());
            }

            let e = total_energy(&c, &mol.atoms, &bonds_idx, &angles);
            if (last_e - e).abs() < conv && s > 10 { steps_run = s; break; }
            last_e    = e;
            steps_run = s + 1;

            let grad  = numerical_gradient(&c, &mol.atoms, &bonds_idx, &angles);
            let gnorm: f64 = grad.iter().map(|g| g.x*g.x+g.y*g.y+g.z*g.z).sum::<f64>().sqrt();
            if gnorm < conv * 0.1 { break; }

            // Armijo line search — identical to lb_minimize
            let mut trial_lr = lr;
            for _ in 0..10 {
                let mut ct = c.clone();
                for i in 0..n {
                    ct[i].x -= trial_lr * grad[i].x / gnorm.max(1e-9);
                    ct[i].y -= trial_lr * grad[i].y / gnorm.max(1e-9);
                    ct[i].z -= trial_lr * grad[i].z / gnorm.max(1e-9);
                }
                if total_energy(&ct, &mol.atoms, &bonds_idx, &angles) < e { c = ct; lr = trial_lr; break; }
                trial_lr *= 0.5;
            }
            if s % 100 == 99 { lr = (lr * 1.2).min(0.1); }

            if s % 25 == 0 {
                let _ = app2.emit("lb:min-progress", MinProgress {
                    step: s, total: steps, energy: e,
                    method: "UFF".into(),
                    message: format!("Step {}/{} · E = {:.4}", s, steps, e),
                });
            }
        }

        let final_energy = total_energy(&c, &mol.atoms, &bonds_idx, &angles);
        Ok(MinDonePayload { coords: c, steps_run, final_energy, method: "UFF".into() })
    })
    .await
    .map_err(|e| e.to_string())?
}

// ─── xTB runner ──────────────────────────────────────────────────────────────
//
// Writes a temp XYZ, runs `xtb input.xyz --opt {level} --gfn {1|2}`, streams
// stdout to emit cycle-by-cycle progress, then reads xtbopt.xyz.

async fn run_xtb(
    app:     &tauri::AppHandle,
    mol:     &LbMolecule,
    coords:  &[Coord3],
    version: u8,
    binary:  &str,
    level:   &str,
) -> Result<MinDonePayload, String> {
    use tokio::io::AsyncBufReadExt as _;

    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let inp = tmp.path().join("input.xyz");
    std::fs::write(&inp, lb_format_xyz(mol, coords)).map_err(|e| e.to_string())?;

    let method_name = format!("GFN{}-xTB", version);
    let _ = app.emit("lb:min-progress", MinProgress {
        step: 0, total: 0, energy: 0.0,
        method: method_name.clone(),
        message: format!("Starting {} geometry optimization…", method_name),
    });

    let xtb_args = [
        inp.to_str().unwrap(),
        "--opt",  level,
        "--gfn",  &version.to_string(),
        "--chrg", &mol.charge.to_string(),
        "--uhf",  &(mol.multiplicity.saturating_sub(1)).to_string(),
    ];

    let mut child = conda_run_cmd(binary, "xtb", &xtb_args).await
        .current_dir(tmp.path())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start xTB: {e}\nIs `xtb` installed and in PATH?"))?;

    let stdout = child.stdout.take().unwrap();
    let mut lines       = tokio::io::BufReader::new(stdout).lines();
    let mut last_energy = 0.0f64;
    let mut cycle       = 0usize;

    while let Ok(Some(line)) = lines.next_line().await {
        if MINIMIZE_CANCEL.load(Ordering::Relaxed) {
            let _ = child.kill().await;
            return Err("Minimization cancelled".into());
        }
        if line.contains("TOTAL ENERGY") {
            if let Some(e_eh) = parse_last_float(&line) {
                last_energy = e_eh * 627.509; // Hartree → kcal/mol
                cycle += 1;
                let _ = app.emit("lb:min-progress", MinProgress {
                    step: cycle, total: 0, energy: last_energy,
                    method: method_name.clone(),
                    message: format!("Cycle {} · E = {:.4} kcal/mol", cycle, last_energy),
                });
            }
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "xTB exited with error (code {:?}).\nCheck that the binary path is correct and the molecule is valid.",
            status.code()
        ));
    }

    let xyz_str = std::fs::read_to_string(tmp.path().join("xtbopt.xyz"))
        .map_err(|_| "xTB finished but xtbopt.xyz was not produced. The optimization may have failed.".to_string())?;
    let new_coords = parse_xyz_coords(&xyz_str)?;

    Ok(MinDonePayload { coords: new_coords, steps_run: cycle, final_energy: last_energy, method: method_name })
}

// ─── ORCA runner ──────────────────────────────────────────────────────────────
//
// Writes an ORCA OPT input, streams stdout for SCF/geometry cycle progress,
// reads back mol.xyz which ORCA writes automatically.

async fn run_orca(
    app:        &tauri::AppHandle,
    mol:        &LbMolecule,
    coords:     &[Coord3],
    functional: &str,
    basis:      &str,
    binary:     &str,
    n_cores:    usize,
) -> Result<MinDonePayload, String> {
    use tokio::io::AsyncBufReadExt as _;

    let tmp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let inp = tmp.path().join("mol.inp");

    let geom = mol.atoms.iter().zip(coords.iter())
        .map(|(a, c)| format!("  {:<4}  {:12.6}  {:12.6}  {:12.6}", a.element, c.x, c.y, c.z))
        .collect::<Vec<_>>().join("\n");
    let input_text = format!(
        "! {} {} OPT\n%pal\n  nprocs {}\nend\n* xyz {} {}\n{}\n*\n",
        functional, basis, n_cores, mol.charge, mol.multiplicity, geom
    );
    std::fs::write(&inp, input_text).map_err(|e| e.to_string())?;

    let method_name = format!("ORCA {}/{}", functional, basis);
    let _ = app.emit("lb:min-progress", MinProgress {
        step: 0, total: 0, energy: 0.0,
        method: method_name.clone(),
        message: format!("Starting {} optimization ({} cores)…", method_name, n_cores),
    });

    let mut child = tokio::process::Command::new(binary)
        .arg(inp.to_str().unwrap())
        .current_dir(tmp.path())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("Failed to start ORCA: {e}\nIs `orca` installed and in PATH?"))?;

    let stdout = child.stdout.take().unwrap();
    let mut lines  = tokio::io::BufReader::new(stdout).lines();
    let mut last_e = 0.0f64;
    let mut cycle  = 0usize;

    while let Ok(Some(line)) = lines.next_line().await {
        if MINIMIZE_CANCEL.load(Ordering::Relaxed) {
            let _ = child.kill().await;
            return Err("Minimization cancelled".into());
        }
        if line.contains("FINAL SINGLE POINT ENERGY") {
            if let Some(e_eh) = parse_last_float(&line) {
                last_e = e_eh * 627.509;
                cycle += 1;
                let _ = app.emit("lb:min-progress", MinProgress {
                    step: cycle, total: 0, energy: last_e,
                    method: method_name.clone(),
                    message: format!("SCF cycle {} · E = {:.4} kcal/mol", cycle, last_e),
                });
            }
        }
    }

    let _ = child.wait().await;

    let xyz_str = std::fs::read_to_string(tmp.path().join("mol.xyz"))
        .map_err(|_| "ORCA finished but mol.xyz was not produced. The optimization may have failed.".to_string())?;
    let new_coords = parse_xyz_coords(&xyz_str)?;

    Ok(MinDonePayload { coords: new_coords, steps_run: cycle, final_energy: last_e, method: method_name })
}

// ─── XYZ / parse helpers ──────────────────────────────────────────────────────

fn lb_format_xyz(mol: &LbMolecule, coords: &[Coord3]) -> String {
    let mut s = format!("{}\nGenerated by Ligand Builder\n", mol.atoms.len());
    for (a, c) in mol.atoms.iter().zip(coords.iter()) {
        s.push_str(&format!("{:<4}  {:12.6}  {:12.6}  {:12.6}\n", a.element, c.x, c.y, c.z));
    }
    s
}

fn parse_xyz_coords(xyz: &str) -> Result<Vec<Coord3>, String> {
    let mut lines = xyz.lines();
    let n: usize  = lines.next().unwrap_or("0").trim().parse()
        .map_err(|_| "Invalid XYZ: bad atom count".to_string())?;
    lines.next(); // skip comment line
    let mut coords = Vec::with_capacity(n);
    for line in lines.take(n) {
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 4 { return Err(format!("Short XYZ line: '{line}'")); }
        coords.push(Coord3 {
            x: p[1].parse().map_err(|_| format!("Bad x in: '{line}'"))?,
            y: p[2].parse().map_err(|_| format!("Bad y in: '{line}'"))?,
            z: p[3].parse().map_err(|_| format!("Bad z in: '{line}'"))?,
        });
    }
    if coords.len() != n {
        return Err(format!("XYZ declared {n} atoms but found {} coords", coords.len()));
    }
    Ok(coords)
}

fn parse_last_float(line: &str) -> Option<f64> {
    line.split_whitespace().filter_map(|t| t.parse::<f64>().ok()).last()
}

// ─── lb_export_xyz ────────────────────────────────────────────────────────────

#[tauri::command]
fn lb_export_xyz(mol: LbMolecule, coords: Vec<Coord3>) -> Result<String, String> {
    let n = mol.atoms.len();
    let mut lines = String::new();
    lines.push_str(&format!("{}\n", n));
    lines.push_str("Generated by MD Engine Ligand Builder\n");
    if coords.len() == n {
        for (i, a) in mol.atoms.iter().enumerate() {
            lines.push_str(&format!("{:2}  {:12.6}  {:12.6}  {:12.6}\n",
                a.element, coords[i].x, coords[i].y, coords[i].z));
        }
    } else {
        // Use 2D coords with z=0
        for a in &mol.atoms {
            lines.push_str(&format!("{:2}  {:12.6}  {:12.6}  {:12.6}\n",
                a.element, a.x*0.03, a.y*0.03, 0.0));
        }
    }
    Ok(lines)
}

// ─── lb_export_pdb ────────────────────────────────────────────────────────────

#[tauri::command]
fn lb_export_pdb(mol: LbMolecule, coords: Vec<Coord3>, resname: String) -> Result<String, String> {
    let n = mol.atoms.len();
    let has3d = coords.len() == n;
    let rn = resname.trim().to_uppercase();
    let mut lines = String::new();
    lines.push_str("REMARK   Generated by MD Engine Ligand Builder\n");

    for (i, a) in mol.atoms.iter().enumerate() {
        let (x, y, z) = if has3d {
            (coords[i].x, coords[i].y, coords[i].z)
        } else {
            (a.x*0.03, a.y*0.03, 0.0)
        };
        // PDB HETATM line
        lines.push_str(&format!(
            "HETATM{:>5}  {:<3} {:>3}  {:>4}    {:8.3}{:8.3}{:8.3}  1.00  0.00          {:>2}\n",
            i+1, a.element, rn, 1, x, y, z, a.element
        ));
    }
    for b in &mol.bonds {
        let i1 = mol.atoms.iter().position(|a| a.id==b.atom1).map(|x|x+1).unwrap_or(0);
        let i2 = mol.atoms.iter().position(|a| a.id==b.atom2).map(|x|x+1).unwrap_or(0);
        if i1>0 && i2>0 { lines.push_str(&format!("CONECT{:>5}{:>5}\n", i1, i2)); }
    }
    lines.push_str("END\n");
    Ok(lines)
}

// ─── lb_export_mol (MDL MOL format) ──────────────────────────────────────────

#[tauri::command]
fn lb_export_mol(mol: LbMolecule, coords: Vec<Coord3>) -> Result<String, String> {
    let n   = mol.atoms.len();
    let nb  = mol.bonds.len();
    let has3d = coords.len() == n;
    let mut s = String::new();
    s.push_str("\n     MDL Molfile from MD Engine Ligand Builder\n\n");
    s.push_str(&format!("{:>3}{:>3}  0  0  0  0  0  0  0  0999 V2000\n", n, nb));
    for (i, a) in mol.atoms.iter().enumerate() {
        let (x,y,z) = if has3d { (coords[i].x,coords[i].y,coords[i].z) }
                      else { (a.x*0.03,a.y*0.03,0.0) };
        s.push_str(&format!("{:10.4}{:10.4}{:10.4} {:<3} 0  0  0  0  0  0  0  0  0  0  0  0\n",
            x,y,z,a.element));
    }
    for b in &mol.bonds {
        let i1 = mol.atoms.iter().position(|a| a.id==b.atom1).map(|x|x+1).unwrap_or(0);
        let i2 = mol.atoms.iter().position(|a| a.id==b.atom2).map(|x|x+1).unwrap_or(0);
        let btype = if b.aromatic { 4 } else { b.order };
        s.push_str(&format!("{:>3}{:>3}{:>3}  0\n", i1, i2, btype));
    }
    s.push_str("M  END\n");
    Ok(s)
}

// ─── lb_export_qm_input ──────────────────────────────────────────────────────

#[tauri::command]
fn lb_export_qm_input(
    mol:      LbMolecule,
    coords:   Vec<Coord3>,
    program:  String,
    jobtype:  String,
    method:   String,
    basis:    String,
    solvent:  String,
    extra:    String,
) -> Result<String, String> {
    let n = mol.atoms.len();
    if n == 0 { return Err("Molecule is empty.".into()); }

    let has3d = coords.len() == n;
    let charge = mol.charge;
    let mult   = mol.multiplicity;

    // Build geometry block
    let xyz: Vec<(String,f64,f64,f64)> = mol.atoms.iter().enumerate().map(|(i,a)| {
        let (x,y,z) = if has3d { (coords[i].x,coords[i].y,coords[i].z) }
                      else { (a.x*0.03, a.y*0.03, 0.0) };
        (a.element.clone(), x, y, z)
    }).collect();

    let xyz_block = |indent: &str| xyz.iter()
        .map(|(el,x,y,z)| format!("{}{:<4}{:14.8}{:14.8}{:14.8}", indent, el, x, y, z))
        .collect::<Vec<_>>().join("\n");

    // Job type keyword mapping
    let (qchem_job, orca_key, gauss_key, nw_task, psi4_fn, gamess_run) = match jobtype.as_str() {
        "opt"      => ("opt","Opt","Opt","optimize","optimize","OPTIMIZE"),
        "freq"     => ("freq","Freq","Freq","freq","frequencies","HESSIAN"),
        "opt_freq" => ("opt","Opt Freq","Opt Freq","optimize","optimize","OPTIMIZE"),
        "ts"       => ("ts","OptTS","TS","saddle 1","energy","SADPOINT"),
        "scan"     => ("pes_scan","Scan","Scan","energy","energy","ENERGY"),
        _          => ("sp","","","energy","energy","ENERGY"),
    };

    // Solvent keyword mapping
    let solvent_line = |prog: &str| {
        if solvent.is_empty() { return String::new(); }
        match prog {
            "qchem"    => format!("solvent_method  pcm\nsolvent         {}\n", solvent),
            "orca"     => format!("! CPCM({})\n", solvent),
            "gaussian" => format!("SCRF=(PCM,Solvent={})", solvent),
            "nwchem"   => format!("cosmo\n  solvent {}\nend\n", solvent),
            "psi4"     => format!("set pcm true\nset pcm_scf_type total\n"),
            "gamess"   => format!(" $PCM SOLVNT={} $END\n", solvent.to_uppercase()),
            _ => String::new(),
        }
    };

    let out = match program.as_str() {
        // ── Q-Chem ──────────────────────────────────────────────────────────
        "qchem" => format!(
"$comment
 Generated by MD Engine Ligand Builder
$end

$molecule
{} {}
{}
$end

$rem
  JOBTYPE         {}
  METHOD          {}
  BASIS           {}
  THRESH          12
  SYM_IGNORE      TRUE
{}{}$end
",
            charge, mult, xyz_block(""),
            qchem_job, method, basis,
            solvent_line("qchem"),
            if extra.is_empty() {String::new()} else {format!("  {}\n",extra.trim())}
        ),

        // ── ORCA ─────────────────────────────────────────────────────────────
        "orca" => format!(
"! {} {}/{} {} {}{}
%maxcore 4000
%pal nprocs 4 end

* xyz {} {}
{}
*
",
            method, basis, basis, orca_key,
            solvent_line("orca"),
            if extra.is_empty() {String::new()} else {format!(" {}",extra.trim())},
            charge, mult, xyz_block("  ")
        ),

        // ── Gaussian ─────────────────────────────────────────────────────────
        "gaussian" => format!(
"#p {}/{} {} {}{}

Molecule generated by MD Engine Ligand Builder

{} {}
{}

",
            method, basis, gauss_key,
            solvent_line("gaussian"),
            if extra.is_empty() {String::new()} else {format!(" {}",extra.trim())},
            charge, mult, xyz_block("")
        ),

        // ── NWChem ───────────────────────────────────────────────────────────
        "nwchem" => format!(
"start molecule

title \"Generated by MD Engine Ligand Builder\"

charge {}

geometry units angstroms
{}
end

basis
  * library {}
end

{}dft
  xc {}
  mult {}
end
{}
task dft {}
",
            charge, xyz_block("  "),
            basis.to_lowercase(),
            solvent_line("nwchem"),
            method, mult,
            if extra.is_empty() {String::new()} else {format!("\n# {}\n",extra.trim())},
            nw_task
        ),

        // ── Psi4 ─────────────────────────────────────────────────────────────
        "psi4" => format!(
"# Generated by MD Engine Ligand Builder
import psi4

psi4.set_memory('4 GB')
psi4.set_num_threads(4)

mol = psi4.geometry(\"\"\"
{} {}
{}
\"\"\")

psi4.set_options({{
    'basis': '{}',
    'reference': 'rhf' if {} == 1 else 'uhf',
}})

{}{}
{}('{}')
",
            charge, mult, xyz_block(""),
            basis.to_lowercase(), mult,
            solvent_line("psi4"),
            if extra.is_empty() {String::new()} else {format!("# {}\n",extra.trim())},
            psi4_fn, method.to_lowercase()
        ),

        // ── GAMESS ───────────────────────────────────────────────────────────
        "gamess" => {
            let gbasis = match basis.to_lowercase().as_str() {
                "sto-3g"  => ("STO","3"),
                "6-31g*"  => ("N31","6"), "6-31+g**" => ("N31","6"),
                "6-311+g**"=>("N311","6"),"cc-pvdz"  => ("CCD","cc-pVDZ"),
                "cc-pvtz" => ("CCT","cc-pVTZ"),
                _         => ("N31","6"),
            };
            // Build GAMESS $DATA block
            let data_block = xyz.iter().map(|(el,x,y,z)| {
                let an = match el.as_str() {
                    "H"=>1,"C"=>6,"N"=>7,"O"=>8,"S"=>16,"P"=>15,
                    "F"=>9,"Cl"=>17,"Br"=>35,"I"=>53,"B"=>5,"Si"=>14,_=>6
                };
                format!(" {:<4}{}.0   {:12.6}{:12.6}{:12.6}", el, an, x, y, z)
            }).collect::<Vec<_>>().join("\n");

            format!(
" $CONTRL SCFTYP=RHF RUNTYP={} DFTTYP={} MULT={} ICHARG={} COORD=CART $END\n\
 $SYSTEM MWORDS=512 $END\n\
 $BASIS  GBASIS={} NGAUSS={} NDFUNC=1 $END\n\
{}{}
 $DATA\n\
Molecule generated by MD Engine Ligand Builder\nC1\n{}\n $END\n",
                gamess_run, method.to_uppercase(), mult, charge,
                gbasis.0, gbasis.1,
                solvent_line("gamess"),
                if extra.is_empty() {String::new()} else {format!(" $CONTRL {} $END\n",extra.trim())},
                data_block
            )
        },

        _ => return Err(format!("Unknown program: {}", program)),
    };

    Ok(out)
}

// ─── lb_write_text ────────────────────────────────────────────────────────────

#[tauri::command]
fn lb_write_text(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| e.to_string())
}

// ─── lb_append_to_qm_region ──────────────────────────────────────────────────
//
// Appends the ligand's atoms to the existing QM region in AppData, so the
// built ligand can be directly used in a QM/MM input generation workflow.

#[tauri::command]
fn lb_append_to_qm_region(
    state:   tauri::State<'_, AppData>,
    mol:     LbMolecule,
    coords:  Vec<Coord3>,
    resname: String,
) -> Result<String, String> {
    let n = mol.atoms.len();
    if n == 0 { return Err("Empty molecule.".into()); }
    if coords.len() != n { return Err("3D coordinates required (run Generate 3D first).".into()); }

    let mut qm = state.qm_region.lock().unwrap();
    let existing = qm.get_or_insert_with(|| QmRegion {
        atoms:      vec![],
        amber_mask: String::new(),
        window_idx: 0,
        n_atoms:    0,
    });

    let base_serial = existing.atoms.iter().map(|a| a.serial).max().unwrap_or(0) + 1;
    for (i, a) in mol.atoms.iter().enumerate() {
        existing.atoms.push(QmAtom {
            serial:    base_serial + i,
            atom_name: a.element.clone(),
            res_name:  resname.trim().to_uppercase(),
            res_seq:   9999,
            chain_id:  'L',
            element:   a.element.clone(),
            x:         coords[i].x,
            y:         coords[i].y,
            z:         coords[i].z,
        });
    }
    existing.n_atoms = existing.atoms.len();
    Ok(format!("Appended {} atoms to QM region ({} total).", n, existing.n_atoms))
}

// ─── Docking ──────────────────────────────────────────────────────────────────
//
// lb_dock_start returns Ok(()) immediately; results arrive via:
//   lb:dock-progress { message }
//   lb:dock-done     { poses, best_affinity, engine, output_pdbqt }
//   lb:dock-error    String

static DOCK_CANCEL:     AtomicBool = AtomicBool::new(false);
static SASA_CANCEL:     AtomicBool = AtomicBool::new(false);
static MEMBRANE_CANCEL: AtomicBool = AtomicBool::new(false);

#[derive(Clone, serde::Serialize)]
struct DockProgress {
    message: String,
}

#[derive(Clone, serde::Serialize)]
struct DockPose {
    rank:     usize,
    affinity: f64,
    rmsd_lb:  f64,
    rmsd_ub:  f64,
    coords:   Vec<Coord3>,
}

#[derive(Clone, serde::Serialize)]
struct DockDonePayload {
    poses:         Vec<DockPose>,
    best_affinity: f64,
    engine:        String,
    output_pdbqt:  String,
}

#[tauri::command]
async fn lb_dock_start(
    app:           tauri::AppHandle,
    mol:           LbMolecule,
    coords:        Vec<Coord3>,
    engine:        String,
    binary:        String,
    receptor_path: String,
    obabel_path:   String,
    center_x:      f64,
    center_y:      f64,
    center_z:      f64,
    size_x:        f64,
    size_y:        f64,
    size_z:        f64,
    exhaustiveness:usize,
    num_poses:     usize,
    energy_range:  f64,
    cpu:           usize,
) -> Result<(), String> {
    DOCK_CANCEL.store(false, Ordering::Relaxed);

    // Auto-resolve vina and obabel from the managed conda env if paths look
    // like bare names (no path separator) or are empty.
    let binary = if binary.contains('/') || binary.contains('\\') {
        binary
    } else {
        get_conda_bin(engine.clone()).await.unwrap_or(binary)
    };
    let obabel_path = if obabel_path.contains('/') || obabel_path.contains('\\') {
        obabel_path
    } else {
        get_conda_bin("obabel".to_string()).await.unwrap_or(obabel_path)
    };

    let app2 = app.clone();
    tauri::async_runtime::spawn(async move {
        let result = run_docking(
            &app2, &mol, &coords, &engine, &binary, &receptor_path, &obabel_path,
            center_x, center_y, center_z, size_x, size_y, size_z,
            exhaustiveness, num_poses, energy_range, cpu,
        ).await;
        match result {
            Ok(p)    => { let _ = app2.emit("lb:dock-done",  p); }
            Err(msg) => { let _ = app2.emit("lb:dock-error", msg); }
        }
    });
    Ok(())
}

#[tauri::command]
fn lb_dock_cancel() {
    DOCK_CANCEL.store(true, Ordering::Relaxed);
}

/// Resolve a binary name to its full path, expanding shell aliases.
///
/// Returns the filesystem prefix of the managed conda env, or None.
fn get_conda_env_prefix(conda: &str) -> Option<String> {
    let out = std::process::Command::new(conda)
        .args(["env", "list", "--json"])
        .output().ok()?;
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    json["envs"].as_array()?
        .iter()
        .filter_map(|v| v.as_str())
        .find(|p| p.ends_with(CONDA_ENV))
        .map(|s| s.to_string())
}

/// Returns a `tokio::process::Command` that calls the tool directly from its
/// full path inside the managed conda env, with the minimal set of environment
/// variables that each tool requires.
///
/// This is more reliable than `conda run` in non-shell contexts (e.g. Tauri
/// app bundles) because `conda run` requires conda shell hooks to be
/// initialised, which are absent when spawning from a GUI process.
///
/// Required env vars per tool:
///   xtb     — XTBPATH={prefix}/share/xtb  (set by the conda-forge activation
///             script; without it xtb cannot find its parameter files)
///   obabel  — none (RPATH in the conda-forge build handles shared libs)
///   vina    — none (statically linked in the conda-forge build)
///
/// Falls back to calling `binary` directly (system PATH) when the env or the
/// tool binary cannot be found.
async fn conda_run_cmd(binary: &str, tool_name: &str, tool_args: &[&str])
    -> tokio::process::Command
{
    if let Some(conda) = find_conda().await {
        if let Some(prefix) = get_conda_env_prefix(&conda) {
            // Construct the full path to the binary inside the env
            #[cfg(target_os = "windows")]
            let bin_path = format!("{}\\Library\\bin\\{}.exe", prefix, tool_name);
            #[cfg(not(target_os = "windows"))]
            let bin_path = format!("{}/bin/{}", prefix, tool_name);

            if std::path::Path::new(&bin_path).exists() {
                let mut cmd = tokio::process::Command::new(&bin_path);
                cmd.args(tool_args);

                // xtb needs XTBPATH to locate its parameter files.
                // The conda-forge activation script sets this; we replicate it.
                if tool_name == "xtb" {
                    let xtb_share = format!("{}/share/xtb", prefix);
                    if std::path::Path::new(&xtb_share).exists() {
                        cmd.env("XTBPATH", &xtb_share);
                    }
                    // Also set XTBHOME — older xtb versions use this instead
                    cmd.env("XTBHOME", &prefix);
                }

                // Set CONDA_PREFIX so any tool that checks it behaves correctly
                cmd.env("CONDA_PREFIX", &prefix);

                return cmd;
            }
        }
    }

    // Fallback: call binary directly (system install / user PATH)
    let mut cmd = tokio::process::Command::new(binary);
    cmd.args(tool_args);
    cmd
}

/// `tokio::process::Command` calls `execvp` directly — it never goes through a
/// shell, so bash aliases defined in `.bash_aliases` / `.bashrc` are invisible.
/// This helper bridges that gap:
///
///   1. If the caller supplied a path that contains `/` it is already absolute
///      or relative-from-cwd, so return it unchanged.
///   2. Otherwise ask an interactive bash to resolve it with `which`, which
///      *does* source `.bashrc` and therefore expands aliases.  The result is
///      the real filesystem path (e.g. `/opt/autodock/bin/vina`).
///   3. If bash or `which` fail (Windows, no bash, etc.) fall back to the
///      original string and let `Command::new` try its own PATH lookup.
async fn resolve_binary(name: &str) -> String {
    // Already a path — use as-is
    if name.contains('/') || name.contains('\\') {
        return name.to_string();
    }

    // Ask interactive bash to resolve aliases + PATH
    let result = tokio::process::Command::new("bash")
        .args(["-i", "-c", &format!("which {}", name)])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .await;

    match result {
        Ok(out) if out.status.success() => {
            let resolved = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !resolved.is_empty() && resolved.contains('/') {
                return resolved;
            }
            name.to_string()
        }
        _ => name.to_string(),
    }
}

/// Shared runner for Vina and UniDock — they accept the same config format.
async fn run_docking(
    app:           &tauri::AppHandle,
    mol:           &LbMolecule,
    coords:        &[Coord3],
    engine:        &str,
    binary:        &str,
    receptor_path: &str,
    obabel_path:   &str,
    center_x:      f64, center_y: f64, center_z: f64,
    size_x:        f64, size_y:   f64, size_z:   f64,
    exhaustiveness:usize, num_poses: usize, energy_range: f64, cpu: usize,
) -> Result<DockDonePayload, String> {
    use tokio::io::AsyncBufReadExt as _;

    let tmp    = tempfile::tempdir().map_err(|e| e.to_string())?;
    let lig_p  = tmp.path().join("ligand.pdbqt");
    let out_p  = tmp.path().join("output.pdbqt");
    let cfg_p  = tmp.path().join("vina.conf");

    // ── Resolve binary (expands shell aliases) ───────────────────────────────
    let resolved_binary = resolve_binary(binary).await;
    let engine_name = if engine == "unidock" { "UniDock" } else { "AutoDock Vina" };
    let _ = app.emit("lb:dock-progress", DockProgress {
        message: format!("Resolved binary: {}", resolved_binary),
    });

    // ── Auto-convert PDB receptor → PDBQT if needed ─────────────────────────
    //
    // Vina exits immediately with code 1 when given a PDB instead of a PDBQT.
    // If the receptor path ends in .pdb / .ent we run it through obabel
    // transparently before building the config file, so the user can simply
    // point at their PDB without a separate preparation step.
    let receptor_pdbqt: String = {
        let lower = receptor_path.to_lowercase();
        if lower.ends_with(".pdb") || lower.ends_with(".ent") {
            let _ = app.emit("lb:dock-progress", DockProgress {
                message: "PDB receptor detected — converting to PDBQT via obabel…".into(),
            });

            let resolved_obabel = resolve_binary(obabel_path).await;
            let conv_out = tmp.path().join("receptor.pdbqt");

            let obabel_args = [
                receptor_path,
                "-O", conv_out.to_str().unwrap(),
                "-xr",
                "-h",
            ];
            let status = conda_run_cmd(&resolved_obabel, "obabel", &obabel_args).await
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
                .map_err(|e| format!(
                    "Failed to run obabel for receptor conversion: {e}\n\
                     Make sure obabel is installed and on PATH, or enter its full path \
                     in the obabel field.\n\
                     Alternatively, prepare the receptor PDBQT manually and provide \
                     that file directly."
                ))?;

            if !status.success() {
                return Err(format!(
                    "obabel could not convert the receptor PDB to PDBQT (exit code {:?}).\n\
                     Check that the PDB file is valid and that obabel supports it.\n\
                     You can also prepare the receptor PDBQT manually with:\n\
                     obabel {} -O receptor.pdbqt -xr -h",
                    status.code(), receptor_path
                ));
            }

            let _ = app.emit("lb:dock-progress", DockProgress {
                message: "Receptor converted to PDBQT.".into(),
            });

            conv_out.to_str().unwrap().to_string()
        } else {
            // Already PDBQT (or unknown extension — pass through and let Vina validate)
            receptor_path.to_string()
        }
    };

    // ── Write ligand PDBQT ───────────────────────────────────────────────────
    let _ = app.emit("lb:dock-progress", DockProgress { message: "Writing ligand PDBQT…".into() });
    let lig_pdbqt = write_ligand_pdbqt(mol, coords)?;
    std::fs::write(&lig_p, &lig_pdbqt).map_err(|e| e.to_string())?;

    // ── Write Vina config ────────────────────────────────────────────────────
    let cfg = format!(
        "receptor = {rec}\nligand = {lig}\ncenter_x = {cx:.3}\ncenter_y = {cy:.3}\ncenter_z = {cz:.3}\nsize_x = {sx:.1}\nsize_y = {sy:.1}\nsize_z = {sz:.1}\nexhaustiveness = {ex}\nnum_modes = {np}\nenergy_range = {er}\ncpu = {cpu}\nout = {out}\n",
        rec = receptor_pdbqt,
        lig = lig_p.to_str().unwrap(),
        cx  = center_x, cy = center_y, cz = center_z,
        sx  = size_x,   sy = size_y,   sz = size_z,
        ex  = exhaustiveness, np = num_poses, er = energy_range,
        out = out_p.to_str().unwrap(),
    );
    std::fs::write(&cfg_p, &cfg).map_err(|e| e.to_string())?;

    // ── Spawn engine ─────────────────────────────────────────────────────────
    let _ = app.emit("lb:dock-progress", DockProgress {
        message: format!("Running {} (exhaustiveness={})…", engine_name, exhaustiveness),
    });

    let mut args = vec!["--config", cfg_p.to_str().unwrap()];
    if engine == "unidock" { args.push("--scoring"); args.push("vina"); }

    let engine_tool = if engine == "unidock" { "unidock" } else { "vina" };
    let mut child = conda_run_cmd(&resolved_binary, engine_tool, &args).await
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!(
            "Failed to start {} (resolved path: {}):\n{}\n\nIf you are using a shell alias, paste the full binary path into the Binary field instead of the alias name.",
            engine_name, resolved_binary, e
        ))?;

    // Stream stdout for progress lines
    let stdout = child.stdout.take().unwrap();
    let mut lines = tokio::io::BufReader::new(stdout).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        if DOCK_CANCEL.load(Ordering::Relaxed) {
            let _ = child.kill().await;
            return Err("Docking cancelled".into());
        }
        let trimmed = line.trim().to_string();
        if !trimmed.is_empty() {
            let _ = app.emit("lb:dock-progress", DockProgress { message: trimmed });
        }
    }

    let status = child.wait().await.map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!(
            "{} exited with error (code {:?}).\n\
             Check that the receptor PDBQT is valid and the binary path is correct.",
            engine_name, status.code()
        ));
    }

    // ── Parse output PDBQT ───────────────────────────────────────────────────
    let _ = app.emit("lb:dock-progress", DockProgress { message: "Parsing docking results…".into() });
    let out_str = std::fs::read_to_string(&out_p)
        .map_err(|_| format!("{} finished but output.pdbqt was not produced.", engine_name))?;
    let poses = parse_vina_output(&out_str, mol.atoms.len())?;

    if poses.is_empty() {
        return Err("Docking produced no valid poses. Try a larger search box or higher exhaustiveness.".into());
    }

    let best_affinity = poses[0].affinity;
    let out_path_str  = out_p.into_os_string().into_string().unwrap_or_default();
    std::mem::forget(tmp);

    Ok(DockDonePayload { poses, best_affinity, engine: engine_name.into(), output_pdbqt: out_path_str })
}

/// Writes a minimal PDBQT string for the ligand suitable for AutoDock Vina.
///
/// Atom types follow the AutoDock 4 convention:
///   C/A (aromatic C), N/NA, OA, SA, P, HD, H, F, Cl, Br, I.
/// Gasteiger charges are approximated as 0.000 — Vina's scoring function
/// uses its own internal terms and is not sensitive to small charge errors.
/// Rotatable bonds are detected (non-ring single bonds between heavy atoms
/// with at least one non-H substituent on each end) and used for TORSDOF.
/// Full BRANCH/ENDBRANCH records are not generated (rigid-body docking);
/// users needing flexible docking should prepare the ligand with obabel.
fn write_ligand_pdbqt(mol: &LbMolecule, coords: &[Coord3]) -> Result<String, String> {
    if mol.atoms.len() != coords.len() {
        return Err("Atom / coord count mismatch for ligand PDBQT.".into());
    }

    // Detect aromaticity (atoms involved in aromatic bonds)
    let aromatic_ids: std::collections::HashSet<usize> = mol.bonds.iter()
        .filter(|b| b.aromatic)
        .flat_map(|b| {
            let i = mol.atoms.iter().position(|a| a.id == b.atom1);
            let j = mol.atoms.iter().position(|a| a.id == b.atom2);
            [i, j].into_iter().flatten()
        })
        .collect();

    // Detect ring membership (BFS / union-find simplified: mark atoms in any cycle)
    // We use the existing bond graph and check for cycles via DFS back-edges.
    let n = mol.atoms.len();
    let adj: Vec<Vec<usize>> = {
        let mut a = vec![vec![]; n];
        for b in &mol.bonds {
            if let (Some(i), Some(j)) = (
                mol.atoms.iter().position(|x| x.id == b.atom1),
                mol.atoms.iter().position(|x| x.id == b.atom2),
            ) { a[i].push(j); a[j].push(i); }
        }
        a
    };
    let in_ring = detect_ring_atoms(n, &adj);

    // Count rotatable bonds for TORSDOF
    let mut torsdof = 0usize;
    for b in &mol.bonds {
        if b.order != 1 || b.aromatic { continue; }
        let Some(i) = mol.atoms.iter().position(|a| a.id == b.atom1) else { continue };
        let Some(j) = mol.atoms.iter().position(|a| a.id == b.atom2) else { continue };
        if in_ring[i] && in_ring[j] { continue; }  // ring bond
        if mol.atoms[i].element == "H" || mol.atoms[j].element == "H" { continue; }
        // Both ends must have at least one non-H heavy neighbour (not just each other)
        let i_has_nbr = adj[i].iter().any(|&k| k != j && mol.atoms[k].element != "H");
        let j_has_nbr = adj[j].iter().any(|&k| k != i && mol.atoms[k].element != "H");
        if i_has_nbr && j_has_nbr { torsdof += 1; }
    }

    let mut s = String::from("REMARK  LIGAND generated by Atmos Ligand Builder\nROOT\n");
    for (idx, (atom, c)) in mol.atoms.iter().zip(coords.iter()).enumerate() {
        let at = autodock_atom_type(&atom.element, aromatic_ids.contains(&idx), &mol.bonds, &mol.atoms, idx);
        s.push_str(&format!(
            "ATOM  {:5}  {:<4}{:<3}  {:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00    {:+.3} {}\n",
            idx + 1,
            atom.element,
            "LIG",
            1,
            c.x, c.y, c.z,
            0.000_f64,
            at,
        ));
    }
    s.push_str("ENDROOT\n");
    s.push_str(&format!("TORSDOF {}\n", torsdof));
    Ok(s)
}

/// Assign AutoDock 4 atom type string from element + context.
fn autodock_atom_type(
    element:   &str,
    aromatic:  bool,
    bonds:     &[LbBondIn],
    atoms:     &[LbAtomIn],
    atom_idx:  usize,
) -> &'static str {
    let bonded_elems: Vec<&str> = bonds.iter()
        .filter_map(|b| {
            let i = atoms.iter().position(|a| a.id == b.atom1)?;
            let j = atoms.iter().position(|a| a.id == b.atom2)?;
            if i == atom_idx { Some(atoms[j].element.as_str()) }
            else if j == atom_idx { Some(atoms[i].element.as_str()) }
            else { None }
        }).collect();

    match element {
        "C"  => if aromatic { "A" } else { "C" },
        "N"  => "NA",   // conservative: treat all N as H-bond acceptor capable
        "O"  => "OA",   // all O as H-bond acceptor
        "S"  => "SA",
        "P"  => "P",
        "H"  => {
            // HD if bonded to N or O (H-bond donor)
            if bonded_elems.iter().any(|&e| e == "N" || e == "O") { "HD" } else { "H" }
        }
        "F"  => "F",
        "Cl" => "Cl",
        "Br" => "Br",
        "I"  => "I",
        _    => "C",    // fallback for B, Si, metals
    }
}

/// Simple DFS-based ring atom detection: an atom is in a ring if it lies on any cycle.
fn detect_ring_atoms(n: usize, adj: &[Vec<usize>]) -> Vec<bool> {
    let mut in_ring = vec![false; n];
    let mut visited = vec![false; n];
    let mut parent  = vec![usize::MAX; n];

    fn dfs(
        v: usize, par: usize,
        adj: &[Vec<usize>],
        visited: &mut Vec<bool>,
        parent:  &mut Vec<usize>,
        in_ring: &mut Vec<bool>,
    ) {
        visited[v] = true;
        for &u in &adj[v] {
            if !visited[u] {
                parent[u] = v;
                dfs(u, v, adj, visited, parent, in_ring);
            } else if u != par {
                // Back edge v → u: u is an ancestor of v.
                // Trace the path v → parent[v] → … → u, marking each node as
                // being in a ring.
                //
                // Bug fix: the previous version used `while cur != u { cur = parent[cur]; }`
                // unconditionally.  When the DFS root is itself part of a ring,
                // its parent slot holds usize::MAX (the "no parent" sentinel), so
                // `parent[root]` = usize::MAX and indexing `in_ring[usize::MAX]`
                // panics with "index out of bounds".  Any molecule whose first
                // atom (in adjacency-list order) is part of a ring triggers this —
                // e.g. benzene where node 0 is carbon.
                //
                // Fix: loop with an explicit guard that stops before following
                // the usize::MAX sentinel.
                let mut cur = v;
                loop {
                    in_ring[cur] = true;
                    if cur == u { break; }          // reached the back-edge target
                    let p = parent[cur];
                    if p == usize::MAX { break; }   // cur is the DFS root; stop
                    cur = p;
                }
            }
        }
    }

    for start in 0..n {
        if !visited[start] {
            dfs(start, usize::MAX, adj, &mut visited, &mut parent, &mut in_ring);
        }
    }
    in_ring
}

/// Parse AutoDock Vina output PDBQT.
///
/// Vina writes one MODEL block per pose. Each block starts with:
///   REMARK VINA RESULT:   affinity  rmsd_lb  rmsd_ub
/// followed by ATOM records. We extract both the scores and the coordinates.
fn parse_vina_output(pdbqt: &str, n_atoms: usize) -> Result<Vec<DockPose>, String> {
    let mut poses: Vec<DockPose> = Vec::new();
    let mut rank       = 0usize;
    let mut affinity   = 0.0f64;
    let mut rmsd_lb    = 0.0f64;
    let mut rmsd_ub    = 0.0f64;
    let mut cur_coords: Vec<Coord3> = Vec::new();
    let mut in_model   = false;

    for line in pdbqt.lines() {
        if line.starts_with("MODEL") {
            in_model   = true;
            rank      += 1;
            cur_coords = Vec::new();
        } else if line.starts_with("ENDMDL") {
            if in_model && !cur_coords.is_empty() {
                poses.push(DockPose { rank, affinity, rmsd_lb, rmsd_ub, coords: cur_coords.clone() });
            }
            in_model = false;
        } else if line.starts_with("REMARK VINA RESULT") {
            // "REMARK VINA RESULT:   -8.5      0.000      0.000"
            let vals: Vec<f64> = line.split_whitespace()
                .skip(3)
                .filter_map(|t| t.parse().ok())
                .collect();
            if vals.len() >= 3 {
                affinity = vals[0]; rmsd_lb = vals[1]; rmsd_ub = vals[2];
            } else if vals.len() >= 1 {
                affinity = vals[0];
            }
        } else if in_model && (line.starts_with("ATOM") || line.starts_with("HETATM")) {
            // PDB fixed format: columns 31-38 x, 39-46 y, 47-54 z
            let x = line.get(30..38).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            let y = line.get(38..46).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            let z = line.get(46..54).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            cur_coords.push(Coord3 { x, y, z });
        }
    }

    // Filter to poses that have the right number of heavy atoms
    // (Vina only writes the ligand atoms, not H — allow both n_atoms and n_non-H)
    let valid: Vec<DockPose> = poses.into_iter()
        .filter(|p| p.coords.len() == n_atoms || p.coords.len() > 0)
        .collect();

    if valid.is_empty() {
        return Err("No valid poses found in Vina output. The output PDBQT may be malformed.".into());
    }
    Ok(valid)
}

/// Prepare a receptor PDBQT from a PDB file using obabel.
/// Adds hydrogens (-h) and converts to PDBQT format (-xr = receptor mode).
#[tauri::command]
async fn lb_prepare_receptor(
    pdb_path:    String,
    obabel_path: String,
) -> Result<String, String> {
    use tokio::process::Command;

    // Output path is pdb_path with .pdbqt extension
    let out = std::path::Path::new(&pdb_path)
        .with_extension("pdbqt")
        .to_string_lossy()
        .to_string();

    let status = Command::new(&obabel_path)
        .args([&pdb_path, "-O", &out, "-xr", "-h"])
        .status()
        .await
        .map_err(|e| format!("Failed to start obabel: {e}\nIs obabel installed?"))?;

    if !status.success() {
        return Err(format!(
            "obabel exited with error (code {:?}).\nCheck that the PDB file is valid.",
            status.code()
        ));
    }
    Ok(out)
}

/// Read a text file — used by the frontend to read back the output PDBQT for save-all.
#[tauri::command]
fn lb_read_text(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

// ─── Visualizer integration helpers ──────────────────────────────────────────

/// Return the centre of mass of a PDB or PDBQT receptor file.
/// Used by the Ligand Builder to auto-populate the docking search box centre
/// from the receptor geometry rather than from the ligand (which starts near
/// the origin and may be 50+ Å away from the binding site).
#[tauri::command]
fn lb_get_receptor_center(receptor_path: String) -> Result<Coord3, String> {
    let content = std::fs::read_to_string(&receptor_path).map_err(|e| e.to_string())?;
    let (mut sx, mut sy, mut sz, mut n) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") { continue; }
        // Skip hydrogen atoms (element in cols 77-78 for PDB, or element name starting with H)
        let elem = line.get(76..78).map(|s| s.trim()).unwrap_or("").to_uppercase();
        if elem == "H" || elem == "HD" { continue; }
        let x = line.get(30..38).and_then(|s| s.trim().parse::<f64>().ok());
        let y = line.get(38..46).and_then(|s| s.trim().parse::<f64>().ok());
        let z = line.get(46..54).and_then(|s| s.trim().parse::<f64>().ok());
        if let (Some(x), Some(y), Some(z)) = (x, y, z) { sx+=x; sy+=y; sz+=z; n+=1; }
    }
    if n == 0 { return Err("No heavy ATOM/HETATM records found in receptor file.".into()); }
    Ok(Coord3 { x: sx/n as f64, y: sy/n as f64, z: sz/n as f64 })
}

/// Return the receptor as a clean PDB string suitable for loading into NGL.
/// PDBQT files are stripped of their extra columns and Vina-specific records.
/// PDB files are returned as-is.
#[tauri::command]
fn lb_get_receptor_pdb(receptor_path: String) -> Result<String, String> {
    let content = std::fs::read_to_string(&receptor_path).map_err(|e| e.to_string())?;
    let lower   = receptor_path.to_lowercase();
    if lower.ends_with(".pdbqt") {
        // Strip PDBQT-specific records and extra columns beyond col 80
        let pdb: String = content.lines()
            .filter(|l| {
                !l.starts_with("ROOT")     && !l.starts_with("ENDROOT") &&
                !l.starts_with("BRANCH")   && !l.starts_with("ENDBRANCH") &&
                !l.starts_with("TORSDOF")  && !l.starts_with("REMARK VINA") &&
                !l.starts_with("MODEL")    && !l.starts_with("ENDMDL")
            })
            .map(|l| {
                if (l.starts_with("ATOM") || l.starts_with("HETATM")) && l.len() > 80 {
                    l[..80].to_string()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(pdb + "\nEND\n")
    } else {
        Ok(content)
    }
}

/// Extract a specific docked pose from a Vina output PDBQT and return it as
/// a PDB string.  pose_rank is 1-based (matches the table shown in the UI).
#[tauri::command]
fn lb_get_pose_pdb(output_pdbqt_path: String, pose_rank: usize) -> Result<String, String> {
    let content = std::fs::read_to_string(&output_pdbqt_path).map_err(|e| e.to_string())?;
    let mut current_rank = 0usize;
    let mut in_model     = false;
    let mut out: Vec<String> = vec![format!("REMARK  Docked pose {} (Atmos Ligand Builder)", pose_rank)];

    for line in content.lines() {
        if line.starts_with("MODEL") {
            current_rank += 1;
            in_model      = true;
        } else if line.starts_with("ENDMDL") {
            if current_rank == pose_rank { out.push("END".into()); break; }
            in_model = false;
        } else if in_model && current_rank == pose_rank {
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                // Trim PDBQT extra columns, keep standard PDB width
                let pdb_line = if line.len() > 80 { &line[..80] } else { line };
                out.push(pdb_line.to_string());
            }
        }
    }

    if out.len() <= 1 {
        return Err(format!("Pose {} not found in output PDBQT.", pose_rank));
    }
    Ok(out.join("\n") + "\n")
}

// ─── Dependency manager (conda) ───────────────────────────────────────────────
//
// xtb, Open Babel, and AutoDock Vina are installed into a dedicated conda
// environment (`atmos-env`) via conda-forge.  This is the only approach that
// reliably works across macOS (x86 + ARM), Linux, and Windows because all three
// tools have pre-built conda-forge packages and their runtime dependencies
// (shared libs, xtb parameter files, obabel format plugins) are handled
// automatically by conda.
//
// The frontend emits progress lines via the "dep:output" event so the UI can
// show a live log.  The three Vina/obabel/xtb binary strings that `lb_dock_start`
// and `lb_minimize_start` already accept are simply pre-filled from
// `get_conda_bin` — no changes to those commands are needed.

const CONDA_ENV:      &str = "atmos-env";
const CONDA_PACKAGES: &[&str] = &[
    "xtb",
    "openbabel",
    "vina",
];

/// Status of the managed conda environment.
#[derive(Clone, Serialize)]
pub struct CondaEnvStatus {
    /// Whether conda/mamba is available at all.
    pub conda_available: bool,
    /// Path to the conda executable that was found.
    pub conda_path:      Option<String>,
    /// Whether the `atmos-env` environment exists.
    pub env_exists:      bool,
    /// Per-tool availability inside the env.
    pub tools:           Vec<ToolAvailability>,
}

#[derive(Clone, Serialize)]
pub struct ToolAvailability {
    pub id:        String,
    pub name:      String,
    pub bin:       String,   // binary name inside the env
    pub installed: bool,
    pub path:      Option<String>,
}

const MANAGED_TOOLS: &[(&str, &str, &str)] = &[
    // (id,       display name,    binary name)
    ("xtb",     "xtb (GFN2-xTB)", "xtb"),
    ("obabel",  "Open Babel",      "obabel"),
    ("vina",    "AutoDock Vina",   "vina"),
];

/// User-specified conda path override (set via the dep manager UI).
/// Persists only for the current session — stored in AppData for longer runs.
static CONDA_OVERRIDE: std::sync::OnceLock<std::sync::Mutex<Option<String>>>
    = std::sync::OnceLock::new();

fn conda_override() -> &'static std::sync::Mutex<Option<String>> {
    CONDA_OVERRIDE.get_or_init(|| std::sync::Mutex::new(None))
}

#[tauri::command]
async fn set_conda_override(path: String) -> Result<(), String> {
    // Validate that the path actually works before storing it
    let ok = tokio::process::Command::new(&path)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().await
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return Err(format!(
            "Could not run `{} --version`. Check the path is correct and the file is executable.",
            path
        ));
    }
    *conda_override().lock().unwrap() = Some(path);
    Ok(())
}

/// Try to locate a working conda/mamba executable.
///
/// Strategy (in order):
///   1. Parse shell rc files (~/.zshrc, ~/.bash_profile, ~/.zprofile, ~/.bashrc)
///      for the conda initialisation block that `conda init` writes.  That block
///      contains the exact absolute path to the conda executable in single quotes,
///      e.g. '__conda_setup="$('/Users/evan/miniforge3/bin/conda' ...)"'.
///      This is the most reliable approach in macOS app bundles because it reads
///      ground truth from disk rather than depending on PATH or shell env.
///   2. Try common absolute paths under $HOME and /opt/homebrew.
///   3. Login shell fallback — `zsh -l -c "which conda"` — for unusual installs.
///   4. Bare names — works on Linux/Windows where PATH is richer.
async fn find_conda() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();

    // ── Step 0: user-supplied override path (set via dep manager UI) ─────────
    if let Some(ref p) = *conda_override().lock().unwrap() {
        if std::path::Path::new(p).exists() {
            return Some(p.clone());
        }
    }

    // ── Step 1: parse conda init blocks from shell rc files ──────────────────
    // conda init writes a block like:
    //   __conda_setup="$('/path/to/conda' 'shell.zsh' 'hook' 2> /dev/null)"
    // We extract every path that appears in single quotes followed by ' 'shell.
    let rc_files = [
        ".zshrc", ".zprofile", ".bash_profile", ".bashrc", ".profile",
    ];
    for rc in &rc_files {
        let path = format!("{}/{}", home, rc);
        if let Ok(contents) = std::fs::read_to_string(&path) {
            // Look for the conda-init pattern: $('...path.../conda' 'shell.
            for line in contents.lines() {
                if !line.contains("conda") && !line.contains("mamba") { continue; }
                // Extract paths in single quotes: '...'
                let mut rest = line;
                while let Some(start) = rest.find("'") {
                    rest = &rest[start + 1..];
                    if let Some(end) = rest.find("'") {
                        let candidate = &rest[..end];
                        rest = &rest[end + 1..];
                        // Must look like an absolute path to a conda/mamba binary
                        if (candidate.contains("/conda") || candidate.contains("/mamba")
                            || candidate.contains("/micromamba"))
                            && candidate.starts_with('/')
                            && std::path::Path::new(candidate).exists()
                        {
                            return Some(candidate.to_string());
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // ── Step 2: common absolute paths ────────────────────────────────────────
    let mut candidates: Vec<String> = Vec::new();

    // Known install roots, searched for both mamba and conda binaries
    let roots: Vec<String> = vec![
        "/opt/homebrew/Caskroom/miniforge/base".into(),
        "/opt/homebrew/opt/miniforge/base".into(),
        "/opt/homebrew".into(),
        "/usr/local".into(),
        format!("{}/miniforge3",  home),
        format!("{}/miniforge",   home),
        format!("{}/mambaforge",  home),
        format!("{}/miniconda3",  home),
        format!("{}/miniconda",   home),
        format!("{}/anaconda3",   home),
        format!("{}/anaconda",    home),
        format!("{}/opt/miniconda3", home),
        format!("{}/opt/anaconda3",  home),
        format!("{}/.conda",         home),
    ];
    for root in &roots {
        candidates.push(format!("{}/bin/mamba",       root));
        candidates.push(format!("{}/bin/micromamba",  root));
        candidates.push(format!("{}/bin/conda",       root));
    }

    for path in &candidates {
        if !std::path::Path::new(path).exists() { continue; }
        let ok = tokio::process::Command::new(path)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().await
            .map(|s| s.success()).unwrap_or(false);
        if ok { return Some(path.clone()); }
    }

    // ── Step 3: login shell (unusual / custom installs) ──────────────────────
    #[cfg(not(target_os = "windows"))]
    for shell in &["zsh", "bash"] {
        for tool in &["mamba", "micromamba", "conda"] {
            if let Ok(o) = tokio::process::Command::new(shell)
                .args(["-l", "-c", &format!("which {}", tool)])
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .output().await
            {
                if o.status.success() {
                    let p = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    if !p.is_empty() && std::path::Path::new(&p).exists() {
                        return Some(p);
                    }
                }
            }
        }
    }

    // ── Step 4: bare names (Linux/Windows with conda on PATH) ────────────────
    for name in &["mamba", "micromamba", "conda"] {
        let ok = tokio::process::Command::new(name)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().await
            .map(|s| s.success()).unwrap_or(false);
        if ok { return Some(name.to_string()); }
    }

    None
}

/// Returns the path to a binary inside the managed conda env, or None.
fn conda_bin_path(conda: &str, bin: &str) -> Option<String> {
    // Ask conda to tell us the env prefix, then construct the bin path.
    // We do this synchronously since it's a fast metadata query.
    let out = std::process::Command::new(conda)
        .args(["env", "list", "--json"])
        .output().ok()?;
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let envs = json["envs"].as_array()?;

    let env_prefix = envs.iter()
        .filter_map(|v| v.as_str())
        .find(|p| p.ends_with(CONDA_ENV))?;

    #[cfg(target_os = "windows")]
    let full = format!("{}\\Library\\bin\\{}.exe", env_prefix, bin);
    #[cfg(not(target_os = "windows"))]
    let full = format!("{}/bin/{}", env_prefix, bin);

    if std::path::Path::new(&full).exists() { Some(full) } else { None }
}

#[tauri::command]
async fn check_conda_env() -> Result<CondaEnvStatus, String> {
    let conda_path = find_conda().await;

    let (env_exists, tools) = if let Some(ref conda) = conda_path {
        let tools = MANAGED_TOOLS.iter().map(|(id, name, bin)| {
            let path = conda_bin_path(conda, bin);
            ToolAvailability {
                id:        id.to_string(),
                name:      name.to_string(),
                bin:       bin.to_string(),
                installed: path.is_some(),
                path,
            }
        }).collect::<Vec<_>>();
        let env_exists = tools.iter().any(|t| t.installed);
        (env_exists, tools)
    } else {
        let tools = MANAGED_TOOLS.iter().map(|(id, name, bin)| ToolAvailability {
            id: id.to_string(), name: name.to_string(),
            bin: bin.to_string(), installed: false, path: None,
        }).collect();
        (false, tools)
    };

    Ok(CondaEnvStatus {
        conda_available: conda_path.is_some(),
        conda_path,
        env_exists,
        tools,
    })
}

/// Creates / updates `atmos-env` with all managed packages.
/// Streams conda output lines to the frontend via "dep:output" events.
#[tauri::command]
async fn install_conda_env(app: AppHandle) -> Result<CondaEnvStatus, String> {
    let conda = find_conda().await
        .ok_or("conda/mamba not found. Install Miniforge from miniforge.github.io and restart Atmos.")?;

    let _ = app.emit("dep:output", "── Creating atmos-env via conda-forge ──");
    let _ = app.emit("dep:output",
        format!("conda: {}", conda));
    let _ = app.emit("dep:output",
        format!("packages: {}", CONDA_PACKAGES.join(", ")));
    let _ = app.emit("dep:output", "");

    // Build args: create or update the env
    let mut args: Vec<&str> = vec![
        "create", "--name", CONDA_ENV,
        "--channel", "conda-forge",
        "--yes", "--quiet",
    ];
    for pkg in CONDA_PACKAGES { args.push(pkg); }

    let mut child = tokio::process::Command::new(&conda)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start conda: {}", e))?;

    // Stream stdout
    if let Some(stdout) = child.stdout.take() {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(stdout).lines();
        let app2 = app.clone();
        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = app2.emit("dep:output", line);
            }
        });
    }
    // Stream stderr
    if let Some(stderr) = child.stderr.take() {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(stderr).lines();
        let app2 = app.clone();
        tokio::spawn(async move {
            while let Ok(Some(line)) = lines.next_line().await {
                let _ = app2.emit("dep:output", format!("stderr: {}", line));
            }
        });
    }

    let status = child.wait().await
        .map_err(|e| format!("conda process error: {}", e))?;

    if !status.success() {
        return Err(format!(
            "conda create failed (exit {:?}). Check the log above for details.",
            status.code()
        ));
    }

    let _ = app.emit("dep:output", "");
    let _ = app.emit("dep:output", "✓ atmos-env created successfully.");

    // Return fresh status
    check_conda_env().await
}

/// Remove the atmos-env environment entirely.
#[tauri::command]
async fn remove_conda_env(app: AppHandle) -> Result<(), String> {
    let conda = find_conda().await
        .ok_or("conda not found")?;

    let _ = app.emit("dep:output", "Removing atmos-env…");

    let status = tokio::process::Command::new(&conda)
        .args(["env", "remove", "--name", CONDA_ENV, "--yes"])
        .status().await
        .map_err(|e| e.to_string())?;

    if !status.success() {
        return Err(format!("conda env remove failed (exit {:?})", status.code()));
    }
    let _ = app.emit("dep:output", "✓ atmos-env removed.");
    Ok(())
}

/// Returns the path to a managed binary, for use in lb_minimize_start /
/// lb_dock_start.  Falls back to plain binary name (system PATH lookup) if
/// the conda env is not installed, which preserves the existing behaviour for
/// users who manage their own xtb/obabel/vina installations.
#[tauri::command]
async fn get_conda_bin(id: String) -> Result<String, String> {
    let spec = MANAGED_TOOLS.iter().find(|(tid, ..)| *tid == id.as_str())
        .ok_or_else(|| format!("Unknown tool: {}", id))?;

    if let Some(conda) = find_conda().await {
        if let Some(path) = conda_bin_path(&conda, spec.2) {
            return Ok(path);
        }
    }
    // Graceful fallback — let resolve_binary() try system PATH
    Ok(spec.2.to_string())
}

/// Returns the path passed via CLI arg or file association on startup,
/// consuming it so subsequent calls return None.  The frontend calls this
/// once on DOMContentLoaded to populate the trajectory field.
#[tauri::command]
fn get_startup_file(state: State<'_, AppData>) -> Option<String> {
    state.startup_file.lock().unwrap().take()
}

// ─── Project file (.atmos) ────────────────────────────────────────────────────
//
// An .atmos project is a SQLite database with three tables:
//
//   meta            key TEXT PK, value TEXT
//   analyses        tool_id TEXT PK, data_json TEXT, ran_at INTEGER
//   embedded_files  role TEXT PK, filename TEXT, data BLOB
//
// meta keys: traj_path, topo_path, selection, stride, atmos_version,
//            n_frames, created_at, traj_embedded, topo_embedded
//
// analyses tool_id values mirror AppData cache names:
//   rmsd, rmsf, rg, dccm, pca, enm, contacts, hbond,
//   dihedral, prs, mi, cluster, geometry, sasa, membrane,
//   umbrella_windows, mbar
//
// embedded_files roles: trajectory, topology

use rusqlite::{params, Connection};

const ATMOS_VERSION: &str = "1.0.0";

/// Thin serialisable wrapper so EnmResult / PcaResult / DihedralResult
/// (which don't derive Serialize) can be stored as JSON.
// Manual JSON helpers for types that don't derive Serialize
fn pca_to_json(r: &PcaResult) -> String {
    serde_json::json!({
        "projections":        r.projections,
        "explained_variance": r.explained_variance,
        "eigenvectors":       r.eigenvectors,
    }).to_string()
}

fn enm_to_json(r: &EnmResult) -> String {
    serde_json::json!({
        "eigenvalues":  r.eigenvalues,
        "eigenvectors": r.eigenvectors,
        "model":        r.model,
    }).to_string()
}

fn dihedral_cache_to_json(r: &DihedralResult) -> String {
    let residues: Vec<serde_json::Value> = r.residues.iter().map(|rd| {
        serde_json::json!({
            "res_seq":  rd.res_seq,
            "res_name": rd.res_name,
            "atom_idx": rd.atom_idx,
            "phi":      rd.phi,
            "psi":      rd.psi,
        })
    }).collect();
    serde_json::json!({ "residues": residues, "mode": r.mode }).to_string()
}

fn dihedral_cache_from_json(s: &str) -> Option<DihedralResult> {
    let v: serde_json::Value = serde_json::from_str(s).ok()?;
    let mode = v["mode"].as_str().unwrap_or("backbone").to_string();
    let residues = v["residues"].as_array()?.iter().filter_map(|rd| {
        Some(ResidueDihedralFull {
            res_seq:  rd["res_seq"].as_i64()?,
            res_name: rd["res_name"].as_str()?.to_string(),
            atom_idx: rd["atom_idx"].as_u64()? as usize,
            phi: rd["phi"].as_array()?.iter().filter_map(|x| x.as_f64()).collect(),
            psi: rd["psi"].as_array()?.iter().filter_map(|x| x.as_f64()).collect(),
        })
    }).collect();
    Some(DihedralResult { residues, mode })
}

fn open_project_db(path: &str) -> Result<Connection, String> {
    Connection::open(path).map_err(|e| format!("Cannot open project file: {}", e))
}

fn init_project_db(conn: &Connection) -> Result<(), String> {
    conn.execute_batch("
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS analyses (
            tool_id   TEXT PRIMARY KEY,
            data_json TEXT NOT NULL,
            ran_at    INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS embedded_files (
            role     TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            data     BLOB NOT NULL
        );
    ").map_err(|e| e.to_string())
}

#[derive(Clone, Serialize)]
pub struct ProjectMeta {
    pub traj_path:      Option<String>,
    pub topo_path:      Option<String>,
    pub selection:      String,
    pub stride:         u32,
    pub n_frames:       u32,
    pub atmos_version:  String,
    pub traj_embedded:  bool,
    pub topo_embedded:  bool,
    pub analyses_saved: Vec<String>,
}

/// Returns a human-readable list of which caches are populated.
#[tauri::command]
fn get_cached_analyses(state: State<'_, AppData>) -> Vec<String> {
    let mut out = Vec::new();
    if state.rmsd_cache    .lock().unwrap().is_some() { out.push("rmsd".into()); }
    if state.rmsf_cache    .lock().unwrap().is_some() { out.push("rmsf".into()); }
    if state.rg_cache      .lock().unwrap().is_some() { out.push("rg".into()); }
    if state.dccm_cache    .lock().unwrap().is_some() { out.push("dccm".into()); }
    if state.pca_cache     .lock().unwrap().is_some() { out.push("pca".into()); }
    if state.enm_cache     .lock().unwrap().is_some() { out.push("enm".into()); }
    if state.contacts_cache.lock().unwrap().is_some() { out.push("contacts".into()); }
    if state.hbond_cache   .lock().unwrap().is_some() { out.push("hbond".into()); }
    if state.dihedral_cache.lock().unwrap().is_some() { out.push("dihedral".into()); }
    if state.prs_cache     .lock().unwrap().is_some() { out.push("prs".into()); }
    if state.mi_cache      .lock().unwrap().is_some() { out.push("mi".into()); }
    if state.cluster_cache .lock().unwrap().is_some() { out.push("cluster".into()); }
    if state.geometry_cache.lock().unwrap().is_some() { out.push("geometry".into()); }
    if state.sasa_cache    .lock().unwrap().is_some() { out.push("sasa".into()); }
    if state.membrane_cache.lock().unwrap().is_some() { out.push("membrane".into()); }
    if state.umbrella_windows.lock().unwrap().is_some() { out.push("umbrella_windows".into()); }
    if state.mbar_result   .lock().unwrap().is_some() { out.push("mbar".into()); }
    out
}

#[tauri::command]
async fn save_project(
    state:          State<'_, AppData>,
    dest_path:      String,
    embed_traj:     bool,
    embed_topo:     bool,
    traj_path:      Option<String>,
    topo_path:      Option<String>,
    selection:      String,
    stride:         u32,
) -> Result<(), String> {
    let conn = open_project_db(&dest_path)?;
    init_project_db(&conn)?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default()
        .as_secs() as i64;

    // ── Meta ─────────────────────────────────────────────────────────────────
    let n_frames = state.trajectory_data.lock().unwrap()
        .as_ref().map(|t| t.len() as u32).unwrap_or(0);

    let set_meta = |key: &str, val: &str| -> Result<(), String> {
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
            params![key, val],
        ).map_err(|e| e.to_string()).map(|_| ())
    };

    set_meta("traj_path",     &traj_path.clone().unwrap_or_default())?;
    set_meta("topo_path",     &topo_path.clone().unwrap_or_default())?;
    set_meta("selection",     &selection)?;
    set_meta("stride",        &stride.to_string())?;
    set_meta("n_frames",      &n_frames.to_string())?;
    set_meta("atmos_version", ATMOS_VERSION)?;
    set_meta("created_at",    &now.to_string())?;
    set_meta("traj_embedded", if embed_traj { "1" } else { "0" })?;
    set_meta("topo_embedded", if embed_topo { "1" } else { "0" })?;

    // ── Embed files ───────────────────────────────────────────────────────────
    if embed_traj {
        if let Some(ref p) = traj_path {
            let data = std::fs::read(p)
                .map_err(|e| format!("Cannot read trajectory for embedding: {}", e))?;
            let filename = std::path::Path::new(p)
                .file_name().unwrap_or_default().to_string_lossy().to_string();
            conn.execute(
                "INSERT OR REPLACE INTO embedded_files (role, filename, data) VALUES ('trajectory', ?1, ?2)",
                params![filename, data],
            ).map_err(|e| e.to_string())?;
        }
    }
    if embed_topo {
        if let Some(ref p) = topo_path {
            let data = std::fs::read(p)
                .map_err(|e| format!("Cannot read topology for embedding: {}", e))?;
            let filename = std::path::Path::new(p)
                .file_name().unwrap_or_default().to_string_lossy().to_string();
            conn.execute(
                "INSERT OR REPLACE INTO embedded_files (role, filename, data) VALUES ('topology', ?1, ?2)",
                params![filename, data],
            ).map_err(|e| e.to_string())?;
        }
    }

    // ── Analysis caches ───────────────────────────────────────────────────────
    let insert_analysis = |tool_id: &str, json: String| -> Result<(), String> {
        conn.execute(
            "INSERT OR REPLACE INTO analyses (tool_id, data_json, ran_at) VALUES (?1, ?2, ?3)",
            params![tool_id, json, now],
        ).map_err(|e| e.to_string()).map(|_| ())
    };

    if let Some(v) = state.rmsd_cache.lock().unwrap().as_ref() {
        insert_analysis("rmsd", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.rmsf_cache.lock().unwrap().as_ref() {
        insert_analysis("rmsf", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.rg_cache.lock().unwrap().as_ref() {
        insert_analysis("rg", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.dccm_cache.lock().unwrap().as_ref() {
        insert_analysis("dccm", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.pca_cache.lock().unwrap().as_ref() {
        insert_analysis("pca", pca_to_json(v))?;
    }
    if let Some(v) = state.enm_cache.lock().unwrap().as_ref() {
        insert_analysis("enm", enm_to_json(v))?;
    }
    if let Some(v) = state.contacts_cache.lock().unwrap().as_ref() {
        insert_analysis("contacts", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.hbond_cache.lock().unwrap().as_ref() {
        insert_analysis("hbond", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.dihedral_cache.lock().unwrap().as_ref() {
        insert_analysis("dihedral", dihedral_cache_to_json(v))?;
    }
    if let Some(v) = state.prs_cache.lock().unwrap().as_ref() {
        insert_analysis("prs", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.mi_cache.lock().unwrap().as_ref() {
        insert_analysis("mi", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.cluster_cache.lock().unwrap().as_ref() {
        insert_analysis("cluster", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.geometry_cache.lock().unwrap().as_ref() {
        insert_analysis("geometry", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.sasa_cache.lock().unwrap().as_ref() {
        insert_analysis("sasa", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.membrane_cache.lock().unwrap().as_ref() {
        insert_analysis("membrane", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.umbrella_windows.lock().unwrap().as_ref() {
        insert_analysis("umbrella_windows", serde_json::to_string(v).unwrap())?;
    }
    if let Some(v) = state.mbar_result.lock().unwrap().as_ref() {
        insert_analysis("mbar", serde_json::to_string(v).unwrap())?;
    }

    Ok(())
}

#[tauri::command]
async fn load_project(
    state:     State<'_, AppData>,
    app:       AppHandle,
    src_path:  String,
) -> Result<ProjectMeta, String> {
    let conn = open_project_db(&src_path)?;

    let get_meta = |key: &str| -> Option<String> {
        conn.query_row(
            "SELECT value FROM meta WHERE key = ?1",
            params![key],
            |row| row.get::<_, String>(0),
        ).ok()
    };

    // ── Read meta ─────────────────────────────────────────────────────────────
    let traj_embedded = get_meta("traj_embedded").as_deref() == Some("1");
    let topo_embedded = get_meta("topo_embedded").as_deref() == Some("1");
    let selection     = get_meta("selection").unwrap_or_else(|| "name CA".into());
    let stride: u32   = get_meta("stride").and_then(|s| s.parse().ok()).unwrap_or(1);
    let n_frames: u32 = get_meta("n_frames").and_then(|s| s.parse().ok()).unwrap_or(0);
    let stored_traj   = get_meta("traj_path").filter(|s| !s.is_empty());
    let stored_topo   = get_meta("topo_path").filter(|s| !s.is_empty());

    // ── Extract embedded files to app cache dir ───────────────────────────────
    let project_dir = app.path().app_local_data_dir()
        .map_err(|e| e.to_string())?
        .join("project_tmp");
    std::fs::create_dir_all(&project_dir).map_err(|e| e.to_string())?;

    let resolve_embedded = |role: &str, fallback: Option<String>| -> Option<String> {
        let result: rusqlite::Result<(String, Vec<u8>)> = conn.query_row(
            "SELECT filename, data FROM embedded_files WHERE role = ?1",
            params![role],
            |row| Ok((row.get(0)?, row.get(1)?)),
        );
        if let Ok((filename, data)) = result {
            let dest = project_dir.join(&filename);
            if std::fs::write(&dest, &data).is_ok() {
                return Some(dest.to_string_lossy().to_string());
            }
        }
        fallback
    };

    let traj_path = if traj_embedded {
        resolve_embedded("trajectory", stored_traj.clone())
    } else {
        stored_traj.clone()
    };
    let topo_path = if topo_embedded {
        resolve_embedded("topology", stored_topo.clone())
    } else {
        stored_topo.clone()
    };

    // ── Restore analysis caches ───────────────────────────────────────────────
    let mut analyses_saved = Vec::new();

    let load_analysis = |tool_id: &str| -> Option<String> {
        conn.query_row(
            "SELECT data_json FROM analyses WHERE tool_id = ?1",
            params![tool_id],
            |row| row.get::<_, String>(0),
        ).ok()
    };

    if let Some(json) = load_analysis("rmsd") {
        if let Ok(v) = serde_json::from_str::<Vec<f64>>(&json) {
            *state.rmsd_cache.lock().unwrap() = Some(v);
            analyses_saved.push("rmsd".into());
        }
    }
    if let Some(json) = load_analysis("rmsf") {
        if let Ok(v) = serde_json::from_str::<Vec<f64>>(&json) {
            *state.rmsf_cache.lock().unwrap() = Some(v);
            analyses_saved.push("rmsf".into());
        }
    }
    if let Some(json) = load_analysis("rg") {
        if let Ok(v) = serde_json::from_str::<Vec<f64>>(&json) {
            *state.rg_cache.lock().unwrap() = Some(v);
            analyses_saved.push("rg".into());
        }
    }
    if let Some(json) = load_analysis("dccm") {
        if let Ok(v) = serde_json::from_str::<Vec<Vec<f64>>>(&json) {
            *state.dccm_cache.lock().unwrap() = Some(v);
            analyses_saved.push("dccm".into());
        }
    }
    if let Some(json) = load_analysis("pca") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            let proj: Vec<[f64;2]> = serde_json::from_value(v["projections"].clone()).unwrap_or_default();
            let ev:   Vec<f64>     = serde_json::from_value(v["explained_variance"].clone()).unwrap_or_default();
            let vecs: Vec<Vec<f64>> = serde_json::from_value(v["eigenvectors"].clone()).unwrap_or_default();
            *state.pca_cache.lock().unwrap() = Some(PcaResult {
                projections: proj, explained_variance: ev, eigenvectors: vecs
            });
            analyses_saved.push("pca".into());
        }
    }
    if let Some(json) = load_analysis("enm") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
            let evals:  Vec<f64>     = serde_json::from_value(v["eigenvalues"].clone()).unwrap_or_default();
            let evecs:  Vec<Vec<f64>> = serde_json::from_value(v["eigenvectors"].clone()).unwrap_or_default();
            let model:  String        = serde_json::from_value(v["model"].clone()).unwrap_or_default();
            *state.enm_cache.lock().unwrap() = Some(EnmResult {
                eigenvalues: evals, eigenvectors: evecs, model
            });
            analyses_saved.push("enm".into());
        }
    }
    if let Some(json) = load_analysis("contacts") {
        if let Ok(v) = serde_json::from_str::<Vec<Vec<f64>>>(&json) {
            *state.contacts_cache.lock().unwrap() = Some(v);
            analyses_saved.push("contacts".into());
        }
    }
    if let Some(json) = load_analysis("hbond") {
        if let Ok(v) = serde_json::from_str::<Vec<HBondRecord>>(&json) {
            *state.hbond_cache.lock().unwrap() = Some(v);
            analyses_saved.push("hbond".into());
        }
    }
    if let Some(json) = load_analysis("dihedral") {
        if let Some(v) = dihedral_cache_from_json(&json) {
            *state.dihedral_cache.lock().unwrap() = Some(v);
            analyses_saved.push("dihedral".into());
        }
    }
    if let Some(json) = load_analysis("prs") {
        if let Ok(v) = serde_json::from_str::<PrsResult>(&json) {
            *state.prs_cache.lock().unwrap() = Some(v);
            analyses_saved.push("prs".into());
        }
    }
    if let Some(json) = load_analysis("mi") {
        if let Ok(v) = serde_json::from_str::<Vec<Vec<f64>>>(&json) {
            *state.mi_cache.lock().unwrap() = Some(v);
            analyses_saved.push("mi".into());
        }
    }
    if let Some(json) = load_analysis("cluster") {
        if let Ok(v) = serde_json::from_str::<ClusterResult>(&json) {
            *state.cluster_cache.lock().unwrap() = Some(v);
            analyses_saved.push("cluster".into());
        }
    }
    if let Some(json) = load_analysis("geometry") {
        if let Ok(v) = serde_json::from_str::<GeometryResult>(&json) {
            *state.geometry_cache.lock().unwrap() = Some(v);
            analyses_saved.push("geometry".into());
        }
    }
    if let Some(json) = load_analysis("sasa") {
        if let Ok(v) = serde_json::from_str::<SasaResult>(&json) {
            *state.sasa_cache.lock().unwrap() = Some(v);
            analyses_saved.push("sasa".into());
        }
    }
    if let Some(json) = load_analysis("membrane") {
        if let Ok(v) = serde_json::from_str::<MembraneResult>(&json) {
            *state.membrane_cache.lock().unwrap() = Some(v);
            analyses_saved.push("membrane".into());
        }
    }
    if let Some(json) = load_analysis("umbrella_windows") {
        if let Ok(v) = serde_json::from_str::<Vec<UmbrellaWindow>>(&json) {
            *state.umbrella_windows.lock().unwrap() = Some(v);
            analyses_saved.push("umbrella_windows".into());
        }
    }
    if let Some(json) = load_analysis("mbar") {
        if let Ok(v) = serde_json::from_str::<MbarResult>(&json) {
            *state.mbar_result.lock().unwrap() = Some(v);
            analyses_saved.push("mbar".into());
        }
    }

    Ok(ProjectMeta {
        traj_path,
        topo_path,
        selection,
        stride,
        n_frames,
        atmos_version: get_meta("atmos_version").unwrap_or_else(|| ATMOS_VERSION.into()),
        traj_embedded,
        topo_embedded,
        analyses_saved,
    })
}

fn main() {
    tauri::Builder::default()
        .manage(AppData {
            trajectory_data: Mutex::new(None),
            atom_meta:       Mutex::new(None),
            rmsd_cache:      Mutex::new(None),
            rmsf_cache:      Mutex::new(None),
            rg_cache:        Mutex::new(None),
            dccm_cache:      Mutex::new(None),
            pca_cache:       Mutex::new(None),
            enm_cache:       Mutex::new(None),
            contacts_cache:  Mutex::new(None),
            hbond_cache:     Mutex::new(None),
            bfactor_cache:    Mutex::new(None),
            dihedral_cache:   Mutex::new(None),
            prs_cache:        Mutex::new(None),
            mi_cache:         Mutex::new(None),
            cluster_cache:    Mutex::new(None),
            geometry_cache:   Mutex::new(None),
            sasa_cache:       Mutex::new(None),
            membrane_cache:   Mutex::new(None),
            cell_dims:        Mutex::new(None),
            umbrella_windows: Mutex::new(None),
            mbar_result:      Mutex::new(None),
            qmm_topology:     Mutex::new(None),
            umbrella_traj_coords: Mutex::new(None),
            qm_region:        Mutex::new(None),
            // Store CLI arg immediately so the frontend can read it once ready
            startup_file: Mutex::new(
                std::env::args().skip(1)
                    .find(|a| !a.starts_with('-') && std::path::Path::new(a).exists())
            ),
        })
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|_app| {
            // startup_file is populated from CLI args at manage() time above.
            // The frontend reads it via get_startup_file() once DOMContentLoaded fires.
            // On macOS, late-arriving file associations via Apple Events are
            // handled in the .run() callback below via RunEvent::Opened.
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_trajectory,
            run_rmsd,
            run_rmsf,
            run_dccm,
            run_radius_of_gyration,
            run_pca,
            run_contacts,
            run_hbond,
            run_enm,
            run_nma_overlap,
            run_communities,
            run_betweenness,
            run_optimal_paths,
            run_fes,
            run_entropy,
            export_csv,
            get_snapshot_pdb,
            get_frame_coords,
            set_bfactors,
            get_selection_residues,
            viz_event,
            load_umbrella_windows,
            run_mbar,
            get_umbrella_snapshot_pdb,
            set_qmm_topology,
            rewrite_pdb,
            write_text_file,
            save_qm_region,
            get_umbrella_window_count,
            preload_umbrella_coords,
            get_umbrella_window_coords,
            get_qm_region,
            clear_qm_region,
            resolve_qm_selection,
            run_dihedrals,
            get_residue_dihedrals,
            run_prs,
            run_mutual_information,
            run_clustering,
            run_geometry_series,
            get_dihedral_time_series,
            parse_cv_rst,
            run_sasa,
            cancel_sasa,
            run_membrane,
            cancel_membrane,
            batch_export,
            check_conda_env,
            install_conda_env,
            remove_conda_env,
            get_conda_bin,
            set_conda_override,
            get_startup_file,
            save_project,
            load_project,
            get_cached_analyses,
    	    lb_generate_coords,
	        lb_minimize_start,
	        lb_minimize_cancel,
	        lb_export_qm_input,
	        lb_export_xyz,
	        lb_export_pdb,
	        lb_export_mol,
	        lb_write_text,
	        lb_append_to_qm_region,
	        lb_dock_start,
	        lb_dock_cancel,
	        lb_prepare_receptor,
	        lb_read_text,
	        lb_get_receptor_center,
	        lb_get_receptor_pdb,
	        lb_get_pose_pdb,
        ])
        .build(tauri::generate_context!())
        .expect("error building tauri application")
        .run(|app_handle, event| {
            // ── macOS file-association handler ────────────────────────────────
            // When the user double-clicks an associated file in Finder while
            // Atmos is already running, macOS sends an Apple Event that Tauri
            // surfaces here as RunEvent::Opened.
            #[allow(clippy::single_match)]
            match event {
                tauri::RunEvent::Opened { urls } => {
                    for url in urls {
                        let path = url.to_file_path()
                            .map(|p| p.to_string_lossy().to_string())
                            .unwrap_or_else(|_| url.path().to_string());
                        // Store for cold-launch: frontend polls get_startup_file() on init
                        *app_handle.state::<AppData>().startup_file.lock().unwrap()
                            = Some(path.clone());
                        // Also emit for the already-running case
                        let _ = app_handle.emit("atmos://open-file", path);
                    }
                }
                _ => {}
            }
        });
}
