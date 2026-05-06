#![allow(clippy::too_many_arguments)]

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use tower_http::{cors::CorsLayer, services::ServeDir};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use chemfiles::{Frame, Selection, Trajectory};
use ndarray::prelude::*;
use ndarray_linalg::Eigh;
use rayon::prelude::*;
use std::fs::File;
use std::collections::{HashMap, VecDeque};
extern crate glob;

// ─── Progress event ──────────────────────────────────────────────────────────
//
// Broadcast over the WebSocket channel to all connected clients.
// Shape on the wire: { "tool": "pca", "pct": 50.0 }

#[derive(Clone, Serialize)]
pub struct ProgressEvent {
    tool: String,
    pct:  f64,
}

fn emit_progress(tx: &broadcast::Sender<ProgressEvent>, tool: &str, pct: f64) {
    let _ = tx.send(ProgressEvent { tool: tool.to_string(), pct });
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

pub struct AppData {
    pub trajectory_data:      Mutex<Option<Vec<Vec<[f64; 3]>>>>,
    pub atom_meta:            Mutex<Option<Vec<AtomMeta>>>,
    pub rmsd_cache:           Mutex<Option<Vec<f64>>>,
    pub rmsf_cache:           Mutex<Option<Vec<f64>>>,
    pub rg_cache:             Mutex<Option<Vec<f64>>>,
    pub dccm_cache:           Mutex<Option<Vec<Vec<f64>>>>,
    pub pca_cache:            Mutex<Option<PcaResult>>,
    pub enm_cache:            Mutex<Option<EnmResult>>,
    pub contacts_cache:       Mutex<Option<Vec<Vec<f64>>>>,
    pub hbond_cache:          Mutex<Option<Vec<HBondRecord>>>,
    pub bfactor_cache:        Mutex<Option<Vec<f64>>>,
    pub dihedral_cache:       Mutex<Option<DihedralResult>>,
    pub prs_cache:            Mutex<Option<PrsResult>>,
    pub mi_cache:             Mutex<Option<Vec<Vec<f64>>>>,
    pub cluster_cache:        Mutex<Option<ClusterResult>>,
    pub geometry_cache:       Mutex<Option<GeometryResult>>,
    pub sasa_cache:           Mutex<Option<SasaResult>>,
    pub umbrella_windows:     Mutex<Option<Vec<UmbrellaWindow>>>,
    pub mbar_result:          Mutex<Option<MbarResult>>,
    pub qmm_topology:         Mutex<Option<String>>,
    pub umbrella_traj_coords: Mutex<Option<Vec<Vec<f32>>>>,
    pub qm_region:            Mutex<Option<QmRegion>>,
    // Progress broadcast channel
    pub progress_tx:          broadcast::Sender<ProgressEvent>,
}

// ─── Internal cached types ────────────────────────────────────────────────────

#[derive(Clone)]
pub struct PcaResult {
    pub projections:        Vec<[f64; 2]>,
    pub explained_variance: Vec<f64>,
    pub eigenvectors:       Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct EnmResult {
    pub eigenvalues:  Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>,
    pub model:        String,
}

#[derive(Clone, Serialize)]
pub struct HBondRecord {
    pub donor:     usize,
    pub acceptor:  usize,
    pub occupancy: f64,
    pub mean_dist: f64,
}

#[derive(Clone, Serialize)]
pub struct UmbrellaWindow {
    pub index:    usize,
    pub val0:     f64,
    pub samples:  Vec<f64>,
    pub cv_file:  String,
    pub rst_file: Option<String>,
}

#[derive(Clone, Serialize)]
pub struct MbarResult {
    pub bin_centers: Vec<f64>,
    pub pmf:         Vec<f64>,
    pub pmf_err:     Vec<f64>,
    pub delta_g:     f64,
    pub delta_g_err: f64,
    pub n_windows:   usize,
    pub window_val0: Vec<f64>,
    pub kde_x:       Vec<Vec<f64>>,
    pub kde_y:       Vec<Vec<f64>>,
}

#[derive(Clone, Serialize)]
pub struct QmAtom {
    pub serial:    usize,
    pub atom_name: String,
    pub res_name:  String,
    pub res_seq:   i64,
    pub chain_id:  char,
    pub element:   String,
    pub x: f64, pub y: f64, pub z: f64,
}

#[derive(Clone, Serialize)]
pub struct QmRegion {
    pub atoms:      Vec<QmAtom>,
    pub amber_mask: String,
    pub window_idx: usize,
    pub n_atoms:    usize,
}

// ─── Serialisable return types ────────────────────────────────────────────────

#[derive(Serialize)]
struct AnalysisResult<T: Serialize> {
    data:    T,
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

#[derive(Clone)]
pub struct ResidueDihedralFull {
    pub res_seq:  i64,
    pub res_name: String,
    pub atom_idx: usize,
    pub phi:      Vec<f64>,
    pub psi:      Vec<f64>,
}

#[derive(Clone)]
pub struct DihedralResult {
    pub residues: Vec<ResidueDihedralFull>,
    pub mode:     String,
}

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
    density:  Vec<Vec<f64>>,
    residues: Vec<ResidueDihedralSummary>,
    mode:     String,
    n_frames: usize,
}

#[derive(Serialize)]
struct SingleResidueDihedrals {
    res_seq:  i64,
    res_name: String,
    phi:      Vec<f64>,
    psi:      Vec<f64>,
    mode:     String,
}

// ─── PRS / MI types ───────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
pub struct PrsResult {
    pub matrix:        Vec<Vec<f64>>,
    pub effectiveness: Vec<f64>,
    pub sensitivity:   Vec<f64>,
}

// ─── Request body for get_residue_dihedrals ───────────────────────────────────

#[derive(Deserialize)]
struct AtomIdxReq { atom_idx: usize }

// ─── MI type — stored as Vec<Vec<f64>> directly in the cache ─────────────────

// ─── Clustering types ─────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ClusterResult {
    pub assignments:        Vec<usize>,
    pub centers:            Vec<[f64; 2]>,
    pub populations:        Vec<f64>,
    pub method:             String,
    pub n_clusters:         usize,
    pub implied_timescales: Vec<f64>,
    pub pcca_membership:    Vec<Vec<f64>>,
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

#[derive(Clone, Serialize)]
pub struct GeometrySeries {
    pub label:  String,
    pub kind:   String,   // "distance", "angle", or "composite"
    pub values: Vec<f64>,
    pub unit:   String,
}

#[derive(Clone, Serialize)]
pub struct GeometryResult {
    pub series:   Vec<GeometrySeries>,
    pub n_frames: usize,
    pub source:   String,
}

#[derive(Serialize)]
struct CvRstBlock {
    block_idx: usize,
    iat:       Vec<i64>,
    rstwt:     Vec<f64>,
    r2:        f64,
    rk2:       f64,
    comment:   String,
    cv_label:  String,
}

// ─── New request body types ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct RunClusteringReq {
    n_clusters: usize,
    method:     Option<String>,
    lag:        Option<usize>,
    n_macro:    Option<usize>,
}

#[derive(Deserialize)]
struct RunGeometryReq {
    pairs:             Vec<[usize; 2]>,
    triplets:          Vec<[usize; 3]>,
    source:            Option<String>,
    labels:            Option<Vec<String>>,
    composites:        Option<Vec<[usize; 2]>>,
    composite_weights: Option<Vec<f64>>,
    composite_labels:  Option<Vec<String>>,
}

#[derive(Deserialize)]
struct GetDihedralTsReq { atom_indices: Vec<usize> }

#[derive(Deserialize)]
struct ParseCvRstReq { path: String }

fn ok<T: Serialize>(payload: T) -> Json<Value> {
    Json(serde_json::json!({ "ok": true, "payload": payload }))
}

fn err(msg: impl std::fmt::Display) -> Json<Value> {
    Json(serde_json::json!({ "ok": false, "error": msg.to_string() }))
}

// ─── SASA type ────────────────────────────────────────────────────────────────

#[derive(Clone, Serialize)]
pub struct SasaResult {
    pub per_residue_mean: Vec<f64>,
    pub per_residue_std:  Vec<f64>,
    pub total_per_frame:  Vec<f64>,
    pub res_labels:       Vec<String>,
}

// ─── Request body types ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct LoadTrajectoryReq {
    path:          String,
    top_path:      Option<String>,
    selection_str: String,
    stride:        Option<usize>,
}

#[derive(Deserialize)]
struct RunSasaReq { probe: Option<f64> }

// Batch export: client sends a server-side absolute path + optional SVG strings.
// No dialog needed — the server writes directly to the cluster filesystem.
#[derive(Deserialize)]
struct BatchExportReq {
    dir:      String,
    svg_data: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct FrameIdxReq { frame_idx: Option<usize> }
#[derive(Deserialize)]
struct FrameIdxReqRequired { frame_idx: usize }
#[derive(Deserialize)]
struct WindowIdxReq { window_idx: usize }
#[derive(Deserialize)]
struct SetBfactorsReq { values: Vec<f64> }
#[derive(Deserialize)]
struct VizEventReq { event: String, payload: String }
#[derive(Deserialize)]
struct SetQmmTopoReq { path: String }
#[derive(Deserialize)]
struct WriteTextFileReq { path: String, contents: String }
#[derive(Deserialize)]
struct SaveQmRegionReq { serials: Vec<usize>, window_idx: usize }
#[derive(Deserialize)]
struct ResolveQmSelReq { selection_str: String, window_idx: usize }
#[derive(Deserialize)]
struct RewritePdbReq { path: String, topo_path: Option<String> }

#[derive(Deserialize)]
struct RunContactsReq { cutoff: Option<f64> }
#[derive(Deserialize)]
struct RunHbondReq { cutoff_dist: Option<f64>, min_occupancy: Option<f64> }
#[derive(Deserialize)]
struct RunEnmReq { cutoff: Option<f64>, n_modes: Option<usize>, model: Option<String> }
#[derive(Deserialize)]
struct RunCommunitiesReq { threshold: Option<f64>, max_communities: Option<usize> }
#[derive(Deserialize)]
struct RunBetweennessReq { threshold: Option<f64> }
#[derive(Deserialize)]
struct RunOptimalPathsReq { source: usize, sink: usize, threshold: Option<f64> }
#[derive(Deserialize)]
struct RunFesReq { n_bins: Option<usize> }
#[derive(Deserialize)]
struct RunMbarReq {
    fc:     f64,
    temp:   Option<f64>,
    n_bins: Option<usize>,
    n_boot: Option<usize>,
}
#[derive(Deserialize)]
struct LoadUmbrellaReq {
    cv_pattern:  String,
    n_windows:   usize,
    val_min:     f64,
    val_max:     f64,
    cv_col:      usize,
    rst_pattern: Option<String>,
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

// ─── load_trajectory ──────────────────────────────────────────────────────────

async fn api_load_trajectory(
    State(app): State<Arc<AppData>>,
    Json(req): Json<LoadTrajectoryReq>,
) -> impl IntoResponse {
    let tx     = app.progress_tx.clone();
    let stride = req.stride.unwrap_or(1).max(1);
    let result = tokio::task::spawn_blocking(move || {
        load_trajectory_inner(&*app, req.path, req.top_path, req.selection_str, stride, &tx)
    }).await;
    match result {
        Ok(Ok(msg)) => ok(msg),
        Ok(Err(e))  => err(e),
        Err(e)      => err(e.to_string()),
    }
}

fn load_trajectory_inner(
    state: &AppData,
    path: String,
    top_path: Option<String>,
    selection_str: String,
    stride: usize,
    tx: &broadcast::Sender<ProgressEvent>,
) -> Result<String, String> {
    let mut traj = Trajectory::open(&path, 'r').map_err(|e| e.to_string())?;
    if let Some(ref topo_path) = top_path {
        if topo_path.ends_with(".parm7") || topo_path.ends_with(".prmtop") {
            traj.set_topology_with_format(topo_path, "Amber Topology").map_err(|e| e.to_string())?;
        } else {
            traj.set_topology_file(topo_path).map_err(|e| e.to_string())?;
        }
    }
    let mut sel = Selection::new(selection_str.as_str())
        .map_err(|_| format!("Invalid selection: '{}'", selection_str))?;
    let mut frame = Frame::new();
    let n_frames_hint = traj.nsteps();
    let mut all_coords: Vec<Vec<[f64; 3]>> = Vec::with_capacity(n_frames_hint / stride + 1);
    let mut atom_meta_store: Option<Vec<AtomMeta>> = None;
    let report_every = (n_frames_hint / 20).max(1);
    let mut frame_idx = 0usize;

    emit_progress(tx, "load", 0.0);
    while traj.read(&mut frame).is_ok() {
        if frame_idx % stride != 0 { frame_idx += 1; continue; }
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

        let n = all_coords.len();
        if n % report_every == 0 {
            let pct = if n_frames_hint > 0 {
                (n as f64 / n_frames_hint as f64 * 99.0).min(99.0)
            } else { 50.0 };
            emit_progress(tx, "load", pct);
        }
    }
    let frame_count = all_coords.len();
    if frame_count == 0 { return Err("No frames read — check trajectory and selection.".into()); }
    let n_atoms = all_coords[0].len();
    *state.trajectory_data.lock().unwrap() = Some(all_coords);
    *state.atom_meta.lock().unwrap()       = atom_meta_store;
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
    emit_progress(tx, "load", 100.0);
    let stride_msg = if stride > 1 { format!(" (stride {})", stride) } else { String::new() };
    Ok(format!("Loaded {} frames, {} atoms ({}){} into memory.", frame_count, n_atoms, selection_str, stride_msg))
}

// ─── Structural analyses ──────────────────────────────────────────────────────

async fn api_run_rmsd(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    let lock = app.trajectory_data.lock().unwrap();
    let traj = match lock.as_ref() { Some(t) => t, None => return err("No trajectory loaded") };
    if let Some(c) = app.rmsd_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "RMSD (cached)".into() });
    }
    let n_frames = traj.len();
    let n_atoms  = traj[0].len();
    let ref_flat: Vec<f64> = traj[0].iter().flat_map(|a| a.iter().copied()).collect();
    let ref_arr  = match Array2::from_shape_vec((n_atoms, 3), ref_flat) {
        Ok(a) => a, Err(e) => return err(e),
    };
    let mut results = vec![0.0f64; n_frames];
    results.par_iter_mut().enumerate().for_each(|(f, val)| {
        let mut sum = 0.0f64;
        for i in 0..n_atoms { for c in 0..3 { let d = traj[f][i][c] - ref_arr[[i, c]]; sum += d*d; } }
        *val = (sum / n_atoms as f64).sqrt();
    });
    drop(lock);
    *app.rmsd_cache.lock().unwrap() = Some(results.clone());
    ok(AnalysisResult { data: results, message: format!("RMSD over {n_frames} frames computed.") })
}

async fn api_run_rmsf(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    if let Some(c) = app.rmsf_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "RMSF (cached)".into() });
    }
    let lock = app.trajectory_data.lock().unwrap();
    let traj = match lock.as_ref() { Some(t) => t, None => return err("No trajectory loaded") };
    let n_frames = traj.len() as f64;
    let n_atoms  = traj[0].len();
    let mean     = mean_positions(traj);
    let rmsf: Vec<f64> = (0..n_atoms).into_par_iter().map(|i| {
        let mut sum = 0.0f64;
        for frame in traj { for c in 0..3 { let d = frame[i][c] - mean[i][c]; sum += d*d; } }
        (sum / n_frames).sqrt()
    }).collect();
    drop(lock);
    *app.rmsf_cache.lock().unwrap() = Some(rmsf.clone());
    ok(AnalysisResult { data: rmsf, message: format!("RMSF for {n_atoms} atoms computed.") })
}

async fn api_run_radius_of_gyration(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    if let Some(c) = app.rg_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "Rg (cached)".into() });
    }
    let lock = app.trajectory_data.lock().unwrap();
    let traj = match lock.as_ref() { Some(t) => t, None => return err("No trajectory loaded") };
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
    drop(lock);
    *app.rg_cache.lock().unwrap() = Some(rg.clone());
    ok(AnalysisResult { data: rg, message: format!("Radius of gyration over {n_frames} frames computed.") })
}

async fn api_run_pca(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    if let Some(ref c) = *app.pca_cache.lock().unwrap() {
        return ok(AnalysisResult {
            data: PcaData { projections: c.projections.clone(), explained_variance: c.explained_variance.clone() },
            message: "PCA (cached)".into(),
        });
    }
    let traj: Vec<Vec<[f64; 3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let tx = app.progress_tx.clone();
    let result = tokio::task::spawn_blocking(move || {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let dim      = n_atoms * 3;
        emit_progress(&tx, "pca", 5.0);
        let mean     = mean_positions(&traj);
        let flat_mean: Vec<f64> = mean.iter().flat_map(|a| a.iter().copied()).collect();
        let data_flat: Vec<f64> = traj.iter()
            .flat_map(|f| f.iter().flat_map(|a| a.iter().copied())).collect();
        emit_progress(&tx, "pca", 20.0);
        let mut x = Array2::from_shape_vec((n_frames, dim), data_flat).map_err(|e| e.to_string())?;
        for mut row in x.rows_mut() {
            for (v, m) in row.iter_mut().zip(&flat_mean) { *v -= m; }
        }
        let c = x.t().dot(&x) / n_frames as f64;
        emit_progress(&tx, "pca", 50.0);
        let (eigenvalues, eigenvectors) = c.eigh(ndarray_linalg::UPLO::Upper)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;
        emit_progress(&tx, "pca", 85.0);
        let total_var: f64 = eigenvalues.iter().sum();
        let n = eigenvalues.len();
        let explained: Vec<f64> = (0..n).rev()
            .map(|i| eigenvalues[i] / total_var).collect();
        let projections: Vec<[f64; 2]> = (0..n_frames).map(|f| {
            let row: Vec<f64> = traj[f].iter().flat_map(|a| a.iter().copied()).collect();
            let centered: Vec<f64> = row.iter().zip(&flat_mean).map(|(v, m)| v - m).collect();
            let pc1: f64 = centered.iter().zip(eigenvectors.column(n-1).iter()).map(|(v,e)| v*e).sum();
            let pc2: f64 = centered.iter().zip(eigenvectors.column(n-2).iter()).map(|(v,e)| v*e).sum();
            [pc1, pc2]
        }).collect();
        let evecs: Vec<Vec<f64>> = (0..n).map(|k| eigenvectors.column(k).iter().copied().collect()).collect();
        emit_progress(&tx, "pca", 100.0);
        Ok::<_, String>((projections, explained, evecs))
    }).await.map_err(|e| e.to_string());
    match result {
        Err(e)  => err(e),
        Ok(Err(e)) => err(e),
        Ok(Ok((projections, explained, evecs))) => {
            *app.pca_cache.lock().unwrap() = Some(PcaResult {
                projections: projections.clone(),
                explained_variance: explained.iter().copied().take(20).collect(),
                eigenvectors: evecs,
            });
            ok(AnalysisResult {
                data: PcaData { projections, explained_variance: explained.iter().copied().take(10).collect() },
                message: format!("PCA complete. PC1={:.1}%, PC2={:.1}% of variance.",
                    explained[0]*100.0, explained[1]*100.0),
            })
        }
    }
}

async fn api_run_dccm(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    if let Some(c) = app.dccm_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "DCCM (cached)".into() });
    }
    let traj: Vec<Vec<[f64; 3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let tx = app.progress_tx.clone();
    let result = tokio::task::spawn_blocking(move || {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let mean     = mean_positions(&traj);
        emit_progress(&tx, "dccm", 5.0);
        let norms: Vec<f64> = (0..n_atoms).into_par_iter().map(|i| {
            let sum: f64 = traj.iter().map(|f|
                (0..3usize).map(|c| { let d = f[i][c]-mean[i][c]; d*d }).sum::<f64>()
            ).sum();
            sum.sqrt()
        }).collect();
        emit_progress(&tx, "dccm", 15.0);
        let chunk = (n_atoms / 10).max(1);
        let matrix: Vec<Vec<f64>> = (0..n_atoms).into_par_iter().map(|i| {
            if i % chunk == 0 {
                let pct = 15.0 + (i as f64 / n_atoms as f64) * 80.0;
                emit_progress(&tx, "dccm", pct);
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
        emit_progress(&tx, "dccm", 100.0);
        Ok::<_, String>((matrix, n_atoms, n_frames))
    }).await.map_err(|e| e.to_string());
    match result {
        Err(e)     => err(e),
        Ok(Err(e)) => err(e),
        Ok(Ok((matrix, n_atoms, n_frames))) => {
            *app.dccm_cache.lock().unwrap() = Some(matrix.clone());
            ok(AnalysisResult {
                data: matrix,
                message: format!("DCCM ({n_atoms}×{n_atoms}) over {n_frames} frames computed."),
            })
        }
    }
}

async fn api_run_contacts(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunContactsReq>,
) -> impl IntoResponse {
    if let Some(c) = app.contacts_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "Contact map (cached)".into() });
    }
    let traj: Vec<Vec<[f64; 3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let res_seqs: Vec<i64> = app.atom_meta.lock().unwrap()
        .as_ref()
        .map(|m| m.iter().map(|a| a.res_seq).collect())
        .unwrap_or_else(|| (0..traj[0].len() as i64).collect());
    let tx = app.progress_tx.clone();
    let result = tokio::task::spawn_blocking(move || {
        let n_frames = traj.len() as f64;
        let n_atoms  = traj[0].len();
        let cut2     = (req.cutoff.unwrap_or(8.0)).powi(2);
        let chunk    = (traj.len() / 10).max(1);
        let mut counts = vec![vec![0.0f64; n_atoms]; n_atoms];
        for (fi, frame) in traj.iter().enumerate() {
            if fi % chunk == 0 { emit_progress(&tx, "contacts", (fi as f64 / traj.len() as f64) * 95.0); }
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
            .map(|row| row.iter().map(|&v| v / n_frames).collect()).collect();
        emit_progress(&tx, "contacts", 100.0);
        Ok::<_, String>((matrix, n_atoms, req.cutoff.unwrap_or(8.0)))
    }).await.map_err(|e| e.to_string());
    match result {
        Err(e)     => err(e),
        Ok(Err(e)) => err(e),
        Ok(Ok((matrix, n_atoms, cut))) => {
            *app.contacts_cache.lock().unwrap() = Some(matrix.clone());
            ok(AnalysisResult {
                data: matrix,
                message: format!("Contact map ({n_atoms}×{n_atoms}) computed at {cut:.1} Å cutoff."),
            })
        }
    }
}

async fn api_run_hbond(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunHbondReq>,
) -> impl IntoResponse {
    let lock      = app.trajectory_data.lock().unwrap();
    let traj      = match lock.as_ref() { Some(t) => t, None => return err("No trajectory loaded") };
    let meta_lock = app.atom_meta.lock().unwrap();
    let n_frames  = traj.len() as f64;
    let n_atoms   = traj[0].len();
    let cut       = req.cutoff_dist.unwrap_or(3.5);
    let cut2      = cut * cut;
    let min_occ   = req.min_occupancy.unwrap_or(0.05);
    let is_polar: Vec<bool> = meta_lock.as_ref()
        .map(|m| m.iter().map(|a| a.element == "N" || a.element == "O").collect())
        .unwrap_or_else(|| vec![false; n_atoms]);
    let polar_indices: Vec<usize> = is_polar.iter().enumerate()
        .filter_map(|(i, &p)| if p { Some(i) } else { None }).collect();
    let has_polar = polar_indices.iter().any(|&i| is_polar[i]);
    let ca_only   = !has_polar;
    let (donors, acceptors): (Vec<usize>, Vec<usize>) = if ca_only {
        let all: Vec<usize> = (0..n_atoms).collect();
        (all.clone(), all)
    } else {
        (polar_indices.clone(), polar_indices)
    };
    let records: Vec<HBondRecord> = donors.par_iter().flat_map(|&d| {
        let mut local = Vec::new();
        for &a in &acceptors {
            if a <= d { continue; }
            let mut count    = 0usize;
            let mut dist_sum = 0.0f64;
            for frame in traj.iter() {
                let dx = frame[d][0] - frame[a][0];
                let dy = frame[d][1] - frame[a][1];
                let dz = frame[d][2] - frame[a][2];
                let d2 = dx*dx + dy*dy + dz*dz;
                if d2 <= cut2 { count += 1; dist_sum += d2.sqrt(); }
            }
            let occ = count as f64 / n_frames;
            if occ >= min_occ {
                local.push(HBondRecord {
                    donor: d, acceptor: a, occupancy: occ,
                    mean_dist: if count > 0 { dist_sum / count as f64 } else { 0.0 },
                });
            }
        }
        local
    }).collect();
    let mut records = records;
    records.sort_by(|a, b| b.occupancy.partial_cmp(&a.occupancy).unwrap());
    let n_found = records.len();
    let warn = if ca_only { " (Cα-only fallback — load full-atom selection for proper H-bond analysis)" } else { "" };
    *app.hbond_cache.lock().unwrap() = Some(records.clone());
    let occupancies: Vec<f64> = records.iter().map(|r| r.occupancy).collect();
    ok(AnalysisResult {
        data: occupancies,
        message: format!("Found {n_found} contacts with occupancy ≥ {:.0}%.{warn}", min_occ * 100.0),
    })
}

async fn api_run_enm(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunEnmReq>,
) -> impl IntoResponse {
    let traj: Vec<Vec<[f64; 3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let tx = app.progress_tx.clone();
    let result = tokio::task::spawn_blocking(move || {
        let n_atoms   = traj[0].len();
        let cut       = req.cutoff.unwrap_or(7.5);
        let cut2      = cut * cut;
        let n_req     = req.n_modes.unwrap_or(20).min(n_atoms.saturating_sub(1));
        let model_str = req.model.as_deref().unwrap_or("ANM").to_uppercase();
        let mean      = mean_positions(&traj);
        emit_progress(&tx, "enm", 5.0);

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
                emit_progress(&tx, "enm", 40.0);
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
                        emit_progress(&tx, "enm", pct);
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
                emit_progress(&tx, "enm", 40.0);
                let (vals, vecs) = hessian.eigh(ndarray_linalg::UPLO::Upper)
                    .map_err(|e| format!("ANM eigendecomposition failed: {e}"))?;
                let vecs_cols: Vec<Vec<f64>> = (0..dim)
                    .map(|k| vecs.column(k).iter().copied().collect()).collect();
                (vals.to_vec(), vecs_cols, 6usize)
            },
        };
        emit_progress(&tx, "enm", 80.0);
        let mode_vals: Vec<f64> = all_eigenvalues[skip..].iter().take(n_req).copied().collect();
        let _mode_vecs: Vec<Vec<f64>> = all_eigenvectors[skip..].iter().take(n_req).cloned().collect();
        let bfactors: Vec<f64> = match model_str.as_str() {
            "GNM" => {
                let mut b = vec![0.0f64; n_atoms];
                for (&lam, evec) in all_eigenvalues[skip..].iter()
                        .zip(all_eigenvectors[skip..].iter()).take(n_req) {
                    if lam.abs() < 1e-10 { continue; }
                    for (i, &v) in evec.iter().enumerate() { b[i] += v * v / lam; }
                }
                b
            },
            _ => {
                let mut b = vec![0.0f64; n_atoms];
                for (&lam, evec) in all_eigenvalues[skip..].iter()
                        .zip(all_eigenvectors[skip..].iter()).take(n_req) {
                    if lam.abs() < 1e-10 { continue; }
                    for i in 0..n_atoms {
                        let vx = evec[i*3]; let vy = evec[i*3+1]; let vz = evec[i*3+2];
                        b[i] += (vx*vx + vy*vy + vz*vz) / lam;
                    }
                }
                b
            },
        };
        emit_progress(&tx, "enm", 100.0);
        Ok::<_, String>((mode_vals, model_str.clone(), bfactors, n_atoms))
    }).await.map_err(|e| e.to_string());
    match result {
        Err(e)     => err(e),
        Ok(Err(e)) => err(e),
        Ok(Ok((mode_vals, model_str, bfactors, n_atoms))) => {
            *app.bfactor_cache.lock().unwrap() = Some(bfactors.clone());
            ok(AnalysisResult {
                data: bfactors,
                message: format!("{model_str} ENM: {} modes for {n_atoms} atoms.", mode_vals.len()),
            })
        }
    }
}

async fn api_run_nma_overlap(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    let pca = match app.pca_cache.lock().unwrap().clone() {
        Some(p) => p, None => return err("Run PCA first."),
    };
    let enm = match app.enm_cache.lock().unwrap().clone() {
        Some(e) => e, None => return err("Run ENM first."),
    };
    let n_modes = enm.eigenvalues.len();
    let overlaps: Vec<f64> = (0..n_modes).map(|k| {
        let enm_vec = &enm.eigenvectors[k];
        let pca_vec = &pca.eigenvectors[0];
        let dot: f64 = enm_vec.iter().zip(pca_vec.iter()).map(|(a,b)| a*b).sum();
        let n1: f64 = enm_vec.iter().map(|v| v*v).sum::<f64>().sqrt();
        let n2: f64 = pca_vec.iter().map(|v| v*v).sum::<f64>().sqrt();
        if n1 > 1e-10 && n2 > 1e-10 { (dot / (n1 * n2)).powi(2) } else { 0.0 }
    }).collect();
    ok(AnalysisResult { data: overlaps.clone(), message: format!("Mode overlap computed for {n_modes} modes.") })
}

fn build_graph(dccm: &[Vec<f64>], threshold: f64) -> Vec<Vec<(usize, f64)>> {
    let n = dccm.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i+1)..n {
            let w = dccm[i][j].abs();
            if w >= threshold { adj[i].push((j, w)); adj[j].push((i, w)); }
        }
    }
    adj
}

async fn api_run_communities(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunCommunitiesReq>,
) -> impl IntoResponse {
    let dccm: Vec<Vec<f64>> = match app.dccm_cache.lock().unwrap().clone() {
        Some(d) => d, None => return err("Run DCCM first."),
    };
    let tx = app.progress_tx.clone();
    let result = tokio::task::spawn_blocking(move || {
        let n    = dccm.len();
        let thresh   = req.threshold.unwrap_or(0.6);
        let max_comm = req.max_communities.unwrap_or(10);
        let mut weight: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i+1)..n {
                let w = dccm[i][j].abs();
                if w >= thresh { weight[i][j] = w; weight[j][i] = w; }
            }
        }
        fn edge_betweenness(weight: &[Vec<f64>], n: usize) -> HashMap<(usize,usize), f64> {
            let mut eb: HashMap<(usize,usize), f64> = HashMap::new();
            for s in 0..n {
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
                        if dist[j] < 0 { queue.push_back(j); dist[j] = dist[v] + 1; }
                        if dist[j] == dist[v] + 1 { sigma[j] += sigma[v]; pred[j].push(v); }
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
                            label[j] = comp; queue.push_back(j);
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
            let &(ri, rj) = eb.iter().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            weight[ri][rj] = 0.0; weight[rj][ri] = 0.0;
            labels = components(&weight, n);
            let new_comp = *labels.iter().max().unwrap_or(&0) + 1;
            if new_comp > n_comp { n_comp = new_comp; }
            if n_comp >= max_comm { break; }
        }
        let mut sizes = vec![0usize; n_comp];
        for &l in &labels { if l < n_comp { sizes[l] += 1; } }
        let sizes_f: Vec<f64> = sizes.iter().map(|&s| s as f64).collect();
        emit_progress(&tx, "communities", 100.0);
        Ok::<_, String>((sizes_f, n_comp, thresh))
    }).await.map_err(|e| e.to_string());
    match result {
        Err(e)     => err(e),
        Ok(Err(e)) => err(e),
        Ok(Ok((sizes_f, n_comp, thresh))) =>
            ok(AnalysisResult { data: sizes_f, message: format!("Found {n_comp} communities (threshold={thresh:.2}).") }),
    }
}

async fn api_run_betweenness(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunBetweennessReq>,
) -> impl IntoResponse {
    let dccm_lock = app.dccm_cache.lock().unwrap();
    let dccm      = match dccm_lock.as_ref() { Some(d) => d, None => return err("Run DCCM first.") };
    let n         = dccm.len();
    let thresh    = req.threshold.unwrap_or(0.4);
    let adj       = build_graph(dccm, thresh);
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
    let norm = ((n - 1) * (n - 2)) as f64;
    if norm > 0.0 { for c in &mut centrality { *c /= norm; } }
    ok(AnalysisResult {
        data: centrality,
        message: format!("Betweenness centrality computed for {n} nodes (threshold={thresh:.2})."),
    })
}

async fn api_run_optimal_paths(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunOptimalPathsReq>,
) -> impl IntoResponse {
    let dccm_lock = app.dccm_cache.lock().unwrap();
    let dccm      = match dccm_lock.as_ref() { Some(d) => d, None => return err("Run DCCM first.") };
    let n         = dccm.len();
    let thresh    = req.threshold.unwrap_or(0.3);
    if req.source >= n { return err(format!("Source {} out of range (0–{}).", req.source, n-1)); }
    if req.sink   >= n { return err(format!("Sink {} out of range (0–{}).", req.sink, n-1)); }
    let inf = f64::INFINITY;
    let mut dist = vec![inf; n];
    let mut prev = vec![usize::MAX; n];
    dist[req.source] = 0.0;
    let mut visited = vec![false; n];
    for _ in 0..n {
        let u = (0..n).filter(|&i| !visited[i])
            .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap());
        let u = match u { Some(x) => x, None => break };
        if dist[u] == inf { break; }
        visited[u] = true;
        if u == req.sink { break; }
        for j in 0..n {
            if visited[j] { continue; }
            let w = dccm[u][j].abs();
            if w < thresh { continue; }
            let edge_cost = -(w.ln());
            let new_dist  = dist[u] + edge_cost;
            if new_dist < dist[j] { dist[j] = new_dist; prev[j] = u; }
        }
    }
    if dist[req.sink] == inf {
        return err(format!("No path found from {} to {} at threshold {thresh:.2}. Try lowering the threshold.", req.source, req.sink));
    }
    let mut path = Vec::new();
    let mut cur  = req.sink;
    while cur != usize::MAX { path.push(cur); cur = prev[cur]; }
    path.reverse();
    let costs: Vec<f64> = path.windows(2)
        .map(|w| { let c = dccm[w[0]][w[1]].abs(); -(c.ln()) }).collect();
    let cumulative: Vec<f64> = costs.iter()
        .scan(0.0f64, |acc, &v| { *acc += v; Some(*acc) }).collect();
    let path_str: Vec<String> = path.iter().map(|&i| i.to_string()).collect();
    ok(AnalysisResult {
        data: cumulative,
        message: format!("Path {}→{}: [{}]  total cost={:.3}", req.source, req.sink, path_str.join("→"), dist[req.sink]),
    })
}

async fn api_run_fes(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunFesReq>,
) -> impl IntoResponse {
    let pca_lock = app.pca_cache.lock().unwrap();
    let pca      = match pca_lock.as_ref() { Some(p) => p, None => return err("Run PCA first.") };
    let bins     = req.n_bins.unwrap_or(50);
    let pts      = &pca.projections;
    if pts.is_empty() { return err("PCA projections are empty."); }
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
        let xi = xi.min(bins - 1); let yi = yi.min(bins - 1);
        counts[yi][xi] += 1;
    }
    let n_total = pts.len() as f64;
    let f_max   = -(1.0f64 / n_total).ln();
    let fes: Vec<Vec<f64>> = counts.iter()
        .map(|row| row.iter().map(|&c| if c == 0 { f_max } else { -(c as f64 / n_total).ln() }).collect())
        .collect();
    let f_min: f64 = fes.iter().flat_map(|r| r.iter()).cloned().fold(f64::INFINITY, f64::min);
    let fes: Vec<Vec<f64>> = fes.iter()
        .map(|row| row.iter().map(|&v| v - f_min).collect()).collect();
    ok(AnalysisResult {
        data: fes,
        message: format!("Free energy surface ({bins}×{bins} bins) computed from {} frames.", pts.len()),
    })
}

async fn api_run_entropy(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    let lock = app.trajectory_data.lock().unwrap();
    let traj = match lock.as_ref() { Some(t) => t, None => return err("No trajectory loaded") };
    let cov = match build_covariance(traj) { Ok(c) => c, Err(e) => return err(e) };
    let (eigenvalues, _) = match cov.eigh(ndarray_linalg::UPLO::Upper) {
        Ok(r) => r, Err(e) => return err(format!("Entropy eigendecomposition failed: {e}")),
    };
    const KB: f64 = 1.380649e-23;
    const NA: f64 = 6.02214076e23;
    const H_BAR: f64 = 1.054571817e-34;
    const T: f64 = 300.0;
    const M_CA_KG: f64 = 12.011 * 1.66054e-27;
    let threshold = 1e-6 * eigenvalues.iter().cloned().fold(0.0f64, f64::max);
    let valid_evals: Vec<f64> = eigenvalues.iter().filter(|&&v| v > threshold).copied().collect();
    let n_modes = valid_evals.len();
    let angstrom2_to_m2 = 1e-20_f64;
    let entropy_per_mode: Vec<f64> = valid_evals.iter().map(|&sigma2_ang| {
        let sigma2_m2 = sigma2_ang * angstrom2_to_m2;
        let arg = (2.0 * std::f64::consts::PI * std::f64::consts::E * KB * T * M_CA_KG / (H_BAR * H_BAR)) * sigma2_m2;
        if arg > 1.0 { KB / 2.0 * arg.ln() } else { 0.0 }
    }).collect();
    let entropy_j_k = entropy_per_mode.iter().sum::<f64>();
    let entropy_j_per_mol_k = entropy_j_k * NA;
    ok(AnalysisResult {
        data: EntropyResult { entropy_j_per_mol_k, n_modes_used: n_modes },
        message: format!("Quasi-harmonic entropy: {:.2} J/(mol·K) from {n_modes} modes (T=300 K).", entropy_j_per_mol_k),
    })
}

// ─── PDB / snapshot helpers ───────────────────────────────────────────────────

const STANDARD_RES: &[&str] = &[
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HIE","HID","HIP","CYX","ASH","GLH","LYN","CYM",
    "ACE","NME","NHE","WAT","HOH","SOL","TIP","TIP3","TP3",
];

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
            "{}{:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}\n",
            record, i+1, pdb_name, res3, chain_id, res_seq,
            p[0] as f64, p[1] as f64, p[2] as f64, b, elem2
        ));
    }
    let bonds = topo.bonds();
    if !bonds.is_empty() {
        let mut neighbours: Vec<Vec<usize>> = vec![vec![]; n];
        for bond in &bonds {
            let (a, b) = (bond[0], bond[1]);
            if a < n && b < n && (is_hetatm[a] || is_hetatm[b]) {
                neighbours[a].push(b + 1); neighbours[b].push(a + 1);
            }
        }
        for (i, partners) in neighbours.iter().enumerate() {
            if partners.is_empty() { continue; }
            for chunk in partners.chunks(4) {
                let fields: String = chunk.iter().map(|p| format!("{:5}", p)).collect::<Vec<_>>().join("");
                pdb.push_str(&format!("CONECT{:5}{}\n", i + 1, fields));
            }
        }
    }
    pdb.push_str("TER\nEND\n");
    pdb
}

async fn api_get_snapshot_pdb(
    State(app): State<Arc<AppData>>,
    Json(req): Json<FrameIdxReq>,
) -> impl IntoResponse {
    let idx = req.frame_idx.unwrap_or(0);
    let data_lock = app.trajectory_data.lock().unwrap();
    let frames = match data_lock.as_ref() { Some(f) => f, None => return err("No trajectory loaded") };
    if idx >= frames.len() {
        return err(format!("Frame {} out of range (0–{})", idx, frames.len() - 1));
    }
    let coords    = &frames[idx];
    let meta_lock = app.atom_meta.lock().unwrap();
    let bf_lock   = app.bfactor_cache.lock().unwrap();
    let std_res: &[&str] = &[
        "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
        "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        "HIE","HID","HIP","CYX","ASH","GLH","LYN","CYM",
        "ACE","NME","NHE","WAT","HOH","SOL","TIP","TIP3","TP3",
    ];
    let mut pdb = String::with_capacity(coords.len() * 82);
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
        let bf = bf_lock.as_ref().and_then(|v| v.get(i).copied()).unwrap_or(0.0);
        pdb.push_str(&format!(
            "{}{:5} {:<4} {:3} {:1}{:4}    {:8.3}{:8.3}{:8.3}  1.00{:6.2}          {:>2}\n",
            record, i+1, pdb_name, res3, chain_id, res_seq, pos[0], pos[1], pos[2], bf, pdb_elem,
        ));
    }
    pdb.push_str("TER\nEND\n");
    ok(pdb)
}

async fn api_get_frame_coords(
    State(app): State<Arc<AppData>>,
    Json(req): Json<FrameIdxReqRequired>,
) -> impl IntoResponse {
    let data_lock = app.trajectory_data.lock().unwrap();
    let frames = match data_lock.as_ref() { Some(f) => f, None => return err("No trajectory loaded") };
    if req.frame_idx >= frames.len() {
        return err(format!("Frame {} out of range (0–{})", req.frame_idx, frames.len() - 1));
    }
    let flat: Vec<f64> = frames[req.frame_idx].iter().flat_map(|c| c.iter().cloned()).collect();
    ok(flat)
}

async fn api_set_bfactors(
    State(app): State<Arc<AppData>>,
    Json(req): Json<SetBfactorsReq>,
) -> impl IntoResponse {
    let n = req.values.len();
    if n == 0 { return err("Empty B-factor array."); }
    let min = req.values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = req.values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let normalised: Vec<f64> = if range < 1e-10 {
        vec![50.0; n]
    } else {
        req.values.iter().map(|&v| (v - min) / range * 100.0).collect()
    };
    *app.bfactor_cache.lock().unwrap() = Some(normalised);
    ok(format!("B-factor data set for {n} atoms (range {min:.3}–{max:.3})."))
}

// ─── viz_event — relay a named event to all WebSocket clients ─────────────────
// In Tauri this targeted the visualizer window; in the web version we broadcast
// to all connected clients.  The visualizer tab's listen() calls receive it.

async fn api_viz_event(
    State(app): State<Arc<AppData>>,
    Json(req): Json<VizEventReq>,
) -> impl IntoResponse {
    let payload: Value = match serde_json::from_str(&req.payload) {
        Ok(v) => v, Err(e) => return err(format!("Invalid payload JSON: {e}")),
    };
    // Build a generic event message and broadcast it
    let msg = serde_json::json!({ "event": req.event, "payload": payload });
    // We repurpose the progress channel for all events since the shape is compatible
    // (clients check msg.event or msg.tool to dispatch)
    let _ = app.progress_tx.send(ProgressEvent { tool: msg.to_string(), pct: -1.0 });
    ok(())
}

// ─── export_csv — returns CSV as a string (browser downloads it) ──────────────

async fn api_export_csv(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    let rmsd = app.rmsd_cache.lock().unwrap().clone();
    let rmsf = app.rmsf_cache.lock().unwrap().clone();
    let rg   = app.rg_cache.lock().unwrap().clone();
    if rmsd.is_none() && rmsf.is_none() && rg.is_none() {
        return err("No per-frame analyses have been run yet.");
    }
    let mut out = String::new();
    let mut headers = vec!["frame"];
    if rmsd.is_some() { headers.push("rmsd_angstrom"); }
    if rmsf.is_some() { headers.push("rmsf_angstrom"); }
    if rg.is_some()   { headers.push("rg_angstrom"); }
    out.push_str(&headers.join(","));
    out.push('\n');
    let n = [&rmsd, &rmsf, &rg].iter()
        .filter_map(|v| v.as_ref()).map(|v| v.len()).max().unwrap_or(0);
    for i in 0..n {
        let mut row = vec![i.to_string()];
        if let Some(ref v) = rmsd { row.push(format!("{:.6}", v.get(i).unwrap_or(&f64::NAN))); }
        if let Some(ref v) = rmsf { row.push(format!("{:.6}", v.get(i).unwrap_or(&f64::NAN))); }
        if let Some(ref v) = rg   { row.push(format!("{:.6}", v.get(i).unwrap_or(&f64::NAN))); }
        out.push_str(&row.join(","));
        out.push('\n');
    }
    ok(out)
}

// ─── write_text_file — write any string to a cluster path ─────────────────────

async fn api_write_text_file(
    _: State<Arc<AppData>>,
    Json(req): Json<WriteTextFileReq>,
) -> impl IntoResponse {
    match std::fs::write(&req.path, req.contents) {
        Ok(()) => ok(()),
        Err(e) => err(e),
    }
}

// ─── load_umbrella_windows ────────────────────────────────────────────────────

async fn api_load_umbrella_windows(
    State(app): State<Arc<AppData>>,
    Json(req): Json<LoadUmbrellaReq>,
) -> impl IntoResponse {
    use std::io::{BufRead, BufReader};
    let n_windows = req.n_windows;
    let val0_k: Vec<f64> = (0..n_windows)
        .map(|i| req.val_min + (req.val_max - req.val_min) * i as f64 / (n_windows - 1).max(1) as f64)
        .collect();
    let mut windows: Vec<UmbrellaWindow> = Vec::with_capacity(n_windows);
    let mut total_samples = 0usize;
    for i in 0..n_windows {
        let idx_str  = format!("{:02}", i);
        let cv_path  = req.cv_pattern.replace("{window}", &idx_str);
        let rst_path = req.rst_pattern.as_ref().map(|p| p.replace("{window}", &idx_str));
        let expanded: Vec<std::path::PathBuf> = {
            let mut v = Vec::new();
            if let Ok(entries) = glob::glob(&cv_path) {
                for entry in entries.flatten() { v.push(entry); }
            }
            if v.is_empty() {
                let p = std::path::PathBuf::from(&cv_path);
                if p.exists() { v.push(p); }
            }
            v.sort(); v
        };
        if expanded.is_empty() {
            return err(format!("Window {:02}: no files found matching '{}'", i, cv_path));
        }
        let mut samples: Vec<f64> = Vec::new();
        for path in &expanded {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => return err(format!("Window {:02}: cannot open '{}': {}", i, path.display(), e)),
            };
            for line in BufReader::new(file).lines() {
                let line = match line { Ok(l) => l, Err(e) => return err(e.to_string()) };
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('@') { continue; }
                let cols: Vec<&str> = trimmed.split_whitespace().collect();
                if let Some(val_str) = cols.get(req.cv_col) {
                    if let Ok(val) = val_str.parse::<f64>() { samples.push(val); }
                }
            }
        }
        if samples.is_empty() {
            return err(format!("Window {:02}: no numeric data found in column {}", i, req.cv_col));
        }
        total_samples += samples.len();
        let rst = rst_path.filter(|p| std::path::Path::new(p).exists());
        windows.push(UmbrellaWindow {
            index: i, val0: val0_k[i], samples, cv_file: expanded[0].display().to_string(), rst_file: rst,
        });
    }
    *app.umbrella_windows.lock().unwrap() = Some(windows);
    *app.mbar_result.lock().unwrap()      = None;
    ok(format!("Loaded {} umbrella windows ({} total CV samples).", n_windows, total_samples))
}

// ─── MBAR helpers ─────────────────────────────────────────────────────────────

const R_GAS:     f64 = 1.987_204_258e-3;
const MBAR_TOL:  f64 = 1e-8;
const MBAR_ITER: usize = 500;

fn compute_bias_matrix(samples: &[f64], val0_k: &[f64], fc: f64, beta: f64) -> Vec<Vec<f64>> {
    val0_k.iter().map(|&x0| {
        samples.par_iter().map(|&x| { let d = x - x0; beta * 0.5 * fc * d * d }).collect()
    }).collect()
}

fn mbar_sweep(f: &[f64], u_kn: &[Vec<f64>], n_k: &[usize]) -> Vec<f64> {
    let k = f.len(); let n_total = u_kn[0].len();
    let log_denom: Vec<f64> = (0..n_total).into_par_iter().map(|n| {
        let max_t = (0..k).map(|ki| (n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..k).map(|ki| ((n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n] - max_t).exp()).sum::<f64>();
        max_t + sum.ln()
    }).collect();
    (0..k).map(|ki| {
        let max_t = (0..n_total).map(|n| -u_kn[ki][n] - log_denom[n])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..n_total).map(|n| (-u_kn[ki][n] - log_denom[n] - max_t).exp()).sum::<f64>();
        -(max_t + sum.ln())
    }).collect()
}

fn solve_mbar(u_kn: &[Vec<f64>], n_k: &[usize]) -> Vec<f64> {
    let k = n_k.len();
    let mut f = vec![0.0f64; k];
    for _ in 0..MBAR_ITER {
        let mut f_new = mbar_sweep(&f, u_kn, n_k);
        let shift = f_new[0];
        f_new.iter_mut().for_each(|v| *v -= shift);
        let max_d = f_new.iter().zip(f.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        f = f_new;
        if max_d < MBAR_TOL { break; }
    }
    f
}

fn compute_pmf(samples: &[f64], f: &[f64], u_kn: &[Vec<f64>], n_k: &[usize],
               val_min: f64, val_max: f64, nbins: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let k = f.len(); let n_total = samples.len();
    let bin_w = (val_max - val_min) / nbins as f64;
    let log_denom: Vec<f64> = (0..n_total).into_par_iter().map(|n| {
        let max_t = (0..k).map(|ki| (n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n])
            .fold(f64::NEG_INFINITY, f64::max);
        let sum = (0..k).map(|ki| ((n_k[ki] as f64).ln() + f[ki] - u_kn[ki][n] - max_t).exp()).sum::<f64>();
        max_t + sum.ln()
    }).collect();
    let mut bin_lws: Vec<Vec<f64>> = vec![vec![]; nbins];
    let mut bin_counts = vec![0usize; nbins];
    for n in 0..n_total {
        let x = samples[n];
        if x < val_min || x > val_max { continue; }
        let bi = ((x - val_min) / bin_w) as usize; let bi = bi.min(nbins - 1);
        bin_lws[bi].push(-log_denom[n]); bin_counts[bi] += 1;
    }
    let pmf: Vec<f64> = bin_lws.iter().map(|lws| {
        if lws.is_empty() { return f64::NAN; }
        let max_lw = lws.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        -(max_lw + lws.iter().map(|&lw| (lw - max_lw).exp()).sum::<f64>().ln())
    }).collect();
    let bin_centers = (0..nbins).map(|b| val_min + (b as f64 + 0.5) * bin_w).collect();
    (bin_centers, pmf, bin_counts)
}

fn gaussian_kde(samples: &[f64], x_min: f64, x_max: f64, n_pts: usize) -> (Vec<f64>, Vec<f64>) {
    if samples.is_empty() { return (vec![], vec![]); }
    let n  = samples.len() as f64;
    let mu = samples.iter().sum::<f64>() / n;
    let sigma = (samples.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n).sqrt().max(1e-12);
    let bw = sigma * n.powf(-0.2);
    let xs: Vec<f64> = (0..n_pts)
        .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1) as f64).collect();
    let norm = n * bw * (2.0 * std::f64::consts::PI).sqrt();
    let ys: Vec<f64> = xs.par_iter().map(|&xi| {
        samples.iter().map(|&s| { let z = (xi - s) / bw; (-0.5 * z * z).exp() }).sum::<f64>() / norm
    }).collect();
    (xs, ys)
}

async fn api_run_mbar(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunMbarReq>,
) -> impl IntoResponse {
    let (all_samples, val0_k, n_k, n_windows) = {
        let lock    = app.umbrella_windows.lock().unwrap();
        let windows = match lock.as_ref() { Some(w) => w, None => return err("Load umbrella windows first.") };
        if windows.len() < 2 { return err("Need at least 2 windows for MBAR."); }
        let n_windows = windows.len();
        let val0_k: Vec<f64>   = windows.iter().map(|w| w.val0).collect();
        let n_k:    Vec<usize> = windows.iter().map(|w| w.samples.len()).collect();
        let all_samples: Vec<f64> = windows.iter().flat_map(|w| w.samples.iter().copied()).collect();
        (all_samples, val0_k, n_k, n_windows)
    };
    let beta   = 1.0 / (R_GAS * req.temp.unwrap_or(300.0));
    let nbins  = req.n_bins.unwrap_or(n_windows.saturating_sub(1)).max(2);
    let n_boot = req.n_boot.unwrap_or(50);
    let val_min = val0_k.iter().cloned().fold(f64::INFINITY,     f64::min);
    let val_max = val0_k.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_total = all_samples.len();
    let tx = app.progress_tx.clone();
    emit_progress(&tx, "mbar", 5.0);
    let u_kn = compute_bias_matrix(&all_samples, &val0_k, req.fc, beta);
    emit_progress(&tx, "mbar", 20.0);
    let f = {
        let u2 = u_kn.clone(); let nk2 = n_k.clone();
        match tokio::task::spawn_blocking(move || solve_mbar(&u2, &nk2)).await {
            Ok(f) => f, Err(e) => return err(e.to_string()),
        }
    };
    emit_progress(&tx, "mbar", 55.0);
    let (bin_centers, pmf_raw, _) = compute_pmf(&all_samples, &f, &u_kn, &n_k, val_min, val_max, nbins);
    let ref_half = (nbins / 2).max(1);
    let ref_val  = pmf_raw[..ref_half].iter().filter(|v| v.is_finite()).cloned()
        .fold(f64::INFINITY, f64::min);
    let ref_val  = if ref_val.is_finite() { ref_val } else {
        pmf_raw.iter().filter(|v| v.is_finite()).cloned().fold(f64::INFINITY, f64::min)
    };
    let pmf: Vec<f64> = pmf_raw.iter()
        .map(|v| if v.is_finite() { v - ref_val } else { f64::NAN }).collect();
    emit_progress(&tx, "mbar", 65.0);
    let n_blocks = 10usize;
    let all_ref = std::sync::Arc::new(all_samples.clone());
    let vk_ref  = std::sync::Arc::new(val0_k.clone());
    let pmf_boots: Vec<Vec<f64>> = (0..n_boot).into_par_iter().map(|rep| {
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
            for _ in 0..n_blocks {
                let pick = next(&mut rng) % n_blocks;
                let s = off + pick * bs;
                let e = (s + bs).min(off + nk);
                for idx in s..e { boot.push(all_ref[idx]); }
            }
            boot_nk[ki] = boot.len() - boot_nk[..ki].iter().sum::<usize>();
            off += nk;
        }
        let u_b = compute_bias_matrix(&boot, &vk_ref, req.fc, beta);
        let f_b = solve_mbar(&u_b, &boot_nk);
        let (_, pmf_b, _) = compute_pmf(&boot, &f_b, &u_b, &boot_nk, val_min, val_max, nbins);
        let ref_b = pmf_b[..ref_half].iter().filter(|v| v.is_finite())
            .cloned().fold(f64::INFINITY, f64::min);
        let ref_b = if ref_b.is_finite() { ref_b } else {
            pmf_b.iter().filter(|v| v.is_finite()).cloned().fold(f64::INFINITY, f64::min)
        };
        pmf_b.iter().map(|v| if v.is_finite() { v - ref_b } else { f64::NAN }).collect()
    }).collect();
    emit_progress(&tx, "mbar", 90.0);
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
    let kde_margin = (val_max - val_min) * 0.05;
    let mut off2 = 0usize;
    let (kde_x, kde_y): (Vec<_>, Vec<_>) = n_k.iter().map(|&nk| {
        let sl = &all_samples[off2..off2 + nk]; off2 += nk;
        gaussian_kde(sl, val_min - kde_margin, val_max + kde_margin, 60)
    }).unzip();
    emit_progress(&tx, "mbar", 100.0);
    let result = MbarResult {
        bin_centers, pmf, pmf_err, delta_g, delta_g_err,
        n_windows, window_val0: val0_k, kde_x, kde_y,
    };
    let msg = format!(
        "MBAR: ΔG‡ = {:.2} ± {:.2} kcal/mol  ({n_windows} windows · {nbins} bins · {n_boot} bootstrap).",
        delta_g, delta_g_err,
    );
    *app.mbar_result.lock().unwrap() = Some(result.clone());
    ok(AnalysisResult { data: result, message: msg })
}

async fn api_set_qmm_topology(
    State(app): State<Arc<AppData>>,
    Json(req): Json<SetQmmTopoReq>,
) -> impl IntoResponse {
    if !std::path::Path::new(&req.path).exists() {
        return err(format!("Topology file not found: {}", req.path));
    }
    let name = std::path::Path::new(&req.path)
        .file_name().unwrap_or_default().to_string_lossy().to_string();
    *app.qmm_topology.lock().unwrap() = Some(req.path);
    ok(format!("QM/MM topology set: {name}"))
}

async fn api_get_umbrella_snapshot_pdb(
    State(app): State<Arc<AppData>>,
    Json(req): Json<WindowIdxReq>,
) -> impl IntoResponse {
    let lock    = app.umbrella_windows.lock().unwrap();
    let windows = match lock.as_ref() { Some(w) => w, None => return err("No umbrella windows loaded.") };
    let win     = match windows.get(req.window_idx) {
        Some(w) => w, None => return err(format!("Window index {} out of range.", req.window_idx)),
    };
    let rst_path = match win.rst_file.as_ref() {
        Some(p) => p.clone(),
        None    => return err(format!("No restart file found for window {:02}. Set rst_pattern when loading.", req.window_idx)),
    };
    let topo_path = app.qmm_topology.lock().unwrap().clone();
    drop(lock);
    let mut traj = match chemfiles::Trajectory::open(&rst_path, 'r') {
        Ok(t) => t, Err(e) => return err(e.to_string()),
    };
    if let Some(ref tp) = topo_path {
        let r = if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
            traj.set_topology_with_format(tp, "Amber Topology")
        } else { traj.set_topology_file(tp) };
        if let Err(e) = r { return err(e.to_string()); }
    }
    let mut frame = chemfiles::Frame::new();
    if let Err(e) = traj.read(&mut frame) { return err(e.to_string()); }
    ok(frame_to_pdb(&frame, None))
}

async fn api_rewrite_pdb(
    _: State<Arc<AppData>>,
    Json(req): Json<RewritePdbReq>,
) -> impl IntoResponse {
    let mut traj = match chemfiles::Trajectory::open(&req.path, 'r') {
        Ok(t) => t, Err(e) => return err(e.to_string()),
    };
    if let Some(ref tp) = req.topo_path {
        let r = if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
            traj.set_topology_with_format(tp, "Amber Topology")
        } else { traj.set_topology_file(tp) };
        if let Err(e) = r { return err(e.to_string()); }
    }
    let mut frame = chemfiles::Frame::new();
    if let Err(e) = traj.read(&mut frame) { return err(e.to_string()); }
    ok(frame_to_pdb(&frame, None))
}

async fn api_save_qm_region(
    State(app): State<Arc<AppData>>,
    Json(req): Json<SaveQmRegionReq>,
) -> impl IntoResponse {
    let lock    = app.umbrella_windows.lock().unwrap();
    let windows = match lock.as_ref() { Some(w) => w, None => return err("No umbrella windows loaded.") };
    let win     = match windows.get(req.window_idx) { Some(w) => w, None => return err("Window index out of range.") };
    let rst_path = match win.rst_file.as_ref() { Some(p) => p.clone(), None => return err("No restart file for this window.") };
    let topo_path = app.qmm_topology.lock().unwrap().clone();
    drop(lock);
    let mut traj = match chemfiles::Trajectory::open(&rst_path, 'r') { Ok(t) => t, Err(e) => return err(e.to_string()) };
    if let Some(ref tp) = topo_path {
        let r = if tp.ends_with(".parm7") || tp.ends_with(".prmtop") { traj.set_topology_with_format(tp, "Amber Topology") }
                else { traj.set_topology_file(tp) };
        if let Err(e) = r { return err(e.to_string()); }
    }
    let mut frame = chemfiles::Frame::new();
    if let Err(e) = traj.read(&mut frame) { return err(e.to_string()); }
    let pos  = frame.positions();
    let topo = frame.topology();
    let mut atoms: Vec<QmAtom> = Vec::new();
    for &serial in &req.serials {
        if serial == 0 || serial > pos.len() { continue; }
        let i = serial - 1;
        let atom      = topo.atom(i);
        let atom_name = { let n = atom.name().to_string(); if n.is_empty() { format!("X{i}") } else { n } };
        let element   = { let e = atom.atomic_type().to_string();
            if e.is_empty() { atom_name.chars().find(|c| c.is_ascii_uppercase()).map(|c| c.to_string()).unwrap_or("C".into()) } else { e } };
        let (res_name, chain_id, res_seq) =
            if let Some(res) = topo.residue_for_atom(i) {
                let chain = res.get("chainname")
                    .and_then(|p| if let chemfiles::Property::String(s) = p { s.chars().next() } else { None })
                    .unwrap_or('A');
                let rn = res.name().to_string();
                (if rn.is_empty() { "UNK".into() } else { rn }, chain, res.id().unwrap_or(i as i64 + 1))
            } else { ("UNK".into(), 'A', i as i64 + 1) };
        atoms.push(QmAtom { serial, atom_name, res_name, res_seq, chain_id, element,
            x: pos[i][0] as f64, y: pos[i][1] as f64, z: pos[i][2] as f64 });
    }
    if atoms.is_empty() { return err("No valid atoms in selection."); }
    let protein_resnames = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                             "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                             "HIE","HID","HIP","CYX","ASH","GLH"];
    let mut res_ids: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    let mut het_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for a in &atoms {
        if protein_resnames.contains(&a.res_name.as_str()) { res_ids.insert(a.res_seq); }
        else { het_names.insert(a.res_name.clone()); }
    }
    let mut mask_parts: Vec<String> = Vec::new();
    if !res_ids.is_empty() {
        let ids: Vec<String> = res_ids.iter().map(|id| id.to_string()).collect();
        mask_parts.push(format!(":{}", ids.join(",")));
    }
    for name in &het_names { mask_parts.push(format!(":{name}")); }
    let amber_mask = mask_parts.join(" | ");
    let n_atoms = atoms.len();
    let region = QmRegion { atoms, amber_mask, window_idx: req.window_idx, n_atoms };
    *app.qm_region.lock().unwrap() = Some(region.clone());
    ok(region)
}

async fn api_get_umbrella_window_count(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    let count = app.umbrella_windows.lock().unwrap().as_ref().map(|w| w.len()).unwrap_or(0);
    ok(count)
}

async fn api_preload_umbrella_coords(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    let (rst_paths, topo_path) = {
        let lock    = app.umbrella_windows.lock().unwrap();
        let windows = match lock.as_ref() { Some(w) => w, None => return err("No umbrella windows loaded.") };
        let paths: Vec<Option<String>> = windows.iter().map(|w| w.rst_file.clone()).collect();
        (paths, app.qmm_topology.lock().unwrap().clone())
    };
    let n = rst_paths.len();
    if n == 0 { return err("No umbrella windows loaded."); }
    let tx = app.progress_tx.clone();
    emit_progress(&tx, "preload", 0.0);
    let all: Vec<Vec<f32>> = match tokio::task::spawn_blocking(move || {
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (i, rst_opt) in rst_paths.iter().enumerate() {
            let rst = rst_opt.as_ref().ok_or(format!("Window {i}: no restart file set."))?;
            let mut traj = chemfiles::Trajectory::open(rst, 'r').map_err(|e| format!("Window {i}: {e}"))?;
            if let Some(ref tp) = topo_path {
                let r = if tp.ends_with(".parm7") || tp.ends_with(".prmtop") {
                    traj.set_topology_with_format(tp, "Amber Topology")
                } else { traj.set_topology_file(tp) };
                r.map_err(|e| format!("Window {i} topo: {e}"))?;
            }
            let mut frame = chemfiles::Frame::new();
            traj.read(&mut frame).map_err(|e| format!("Window {i} read: {e}"))?;
            let flat: Vec<f32> = frame.positions().iter()
                .flat_map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
            out.push(flat);
            emit_progress(&tx, "preload", (i + 1) as f64 / n as f64 * 100.0);
        }
        Ok::<Vec<Vec<f32>>, String>(out)
    }).await {
        Ok(Ok(v))  => v,
        Ok(Err(e)) => return err(e),
        Err(e)     => return err(e.to_string()),
    };
    let msg = format!("Loaded {} window snapshots into trajectory cache.", n);
    *app.umbrella_traj_coords.lock().unwrap() = Some(all);
    ok(msg)
}

async fn api_get_umbrella_window_coords(
    State(app): State<Arc<AppData>>,
    Json(req): Json<WindowIdxReq>,
) -> impl IntoResponse {
    let lock = app.umbrella_traj_coords.lock().unwrap();
    let all  = match lock.as_ref() { Some(a) => a, None => return err("Coords not pre-loaded. Call preload_umbrella_coords first.") };
    match all.get(req.window_idx) {
        Some(v) => ok(v.clone()),
        None    => err(format!("Window {} out of range.", req.window_idx)),
    }
}

async fn api_get_qm_region(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    ok(app.qm_region.lock().unwrap().clone())
}

async fn api_clear_qm_region(State(app): State<Arc<AppData>>, _: Json<Value>) -> impl IntoResponse {
    *app.qm_region.lock().unwrap() = None;
    ok(())
}

async fn api_resolve_qm_selection(
    State(app): State<Arc<AppData>>,
    Json(req): Json<ResolveQmSelReq>,
) -> impl IntoResponse {
    let lock    = app.umbrella_windows.lock().unwrap();
    let windows = match lock.as_ref() { Some(w) => w, None => return err("No umbrella windows loaded.") };
    let win     = match windows.get(req.window_idx) { Some(w) => w, None => return err("Window index out of range.") };
    let rst_path = match win.rst_file.as_ref() { Some(p) => p.clone(), None => return err("No restart file for this window.") };
    let topo_path = app.qmm_topology.lock().unwrap().clone();
    drop(lock);
    let mut traj = match chemfiles::Trajectory::open(&rst_path, 'r') { Ok(t) => t, Err(e) => return err(e.to_string()) };
    if let Some(ref tp) = topo_path {
        let r = if tp.ends_with(".parm7") || tp.ends_with(".prmtop") { traj.set_topology_with_format(tp, "Amber Topology") }
                else { traj.set_topology_file(tp) };
        if let Err(e) = r { return err(e.to_string()); }
    }
    let mut frame = chemfiles::Frame::new();
    if let Err(e) = traj.read(&mut frame) { return err(e.to_string()); }
    let topo    = frame.topology();
    let n_atoms = frame.positions().len();
    let mut serials: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for token in req.selection_str.split_whitespace() {
        if token.starts_with(':') {
            let inner = &token[1..];
            for part in inner.split(',') {
                let part = part.trim();
                if let Ok(resno) = part.parse::<i64>() {
                    for i in 0..n_atoms {
                        if let Some(res) = topo.residue_for_atom(i) {
                            if res.id().unwrap_or(-1) == resno { serials.insert(i + 1); }
                        }
                    }
                } else {
                    for i in 0..n_atoms {
                        if let Some(res) = topo.residue_for_atom(i) {
                            if res.name().eq_ignore_ascii_case(part) { serials.insert(i + 1); }
                        }
                    }
                }
            }
        } else if token.starts_with('@') {
            let inner = &token[1..];
            for part in inner.split(',') {
                if let Ok(serial) = part.trim().parse::<usize>() {
                    if serial >= 1 && serial <= n_atoms { serials.insert(serial); }
                }
            }
        }
    }
    if serials.is_empty() { return err(format!("Selection '{}' matched no atoms.", req.selection_str)); }
    ok(serials.into_iter().collect::<Vec<_>>())
}

// ─── WebSocket — progress + viz events ───────────────────────────────────────

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(app): State<Arc<AppData>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, app))
}

async fn handle_ws(mut socket: WebSocket, app: Arc<AppData>) {
    let mut rx = app.progress_tx.subscribe();
    loop {
        match rx.recv().await {
            Ok(event) => {
                // Deserialise: if pct == -1 the "tool" field holds a full JSON event object
                let text = if event.pct < 0.0 {
                    event.tool   // already a serialised JSON object
                } else {
                    serde_json::to_string(&event).unwrap_or_default()
                };
                if socket.send(Message::Text(text)).await.is_err() { break; }
            }
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(_) => break,
        }
    }
}

// ─── get_selection_residues ───────────────────────────────────────────────────
// Returns per-atom residue info for every atom in the current selection.
// The viewer uses this to map selection atom indices → correct NGL residue
// selectors, regardless of whether Cα-only, full-atom, or a subregion was
// loaded.  Shape: [{ res_seq, res_name, chain_id }]  (one entry per atom).

#[derive(Serialize)]
struct ResidueInfo {
    res_seq:  i64,
    res_name: String,
    chain_id: char,
}

async fn api_get_selection_residues(
    State(app): State<Arc<AppData>>,
    _: Json<Value>,
) -> impl IntoResponse {
    let lock = app.atom_meta.lock().unwrap();
    match lock.as_ref() {
        None => err("No trajectory loaded."),
        Some(meta) => {
            let info: Vec<ResidueInfo> = meta.iter().map(|a| ResidueInfo {
                res_seq:  a.res_seq,
                res_name: a.res_name.clone(),
                chain_id: a.chain_id,
            }).collect();
            ok(info)
        }
    }
}

// ─── Dihedral geometry helpers ────────────────────────────────────────────────

fn cross3(a: [f64;3], b: [f64;3]) -> [f64;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
fn dot3(a: [f64;3], b: [f64;3]) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn norm3(a: [f64;3]) -> f64 { dot3(a,a).sqrt() }
fn sub3(a: [f64;3], b: [f64;3]) -> [f64;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }

fn dihedral_deg(p1: [f64;3], p2: [f64;3], p3: [f64;3], p4: [f64;3]) -> f64 {
    let b1 = sub3(p2, p1); let b2 = sub3(p3, p2); let b3 = sub3(p4, p3);
    let n1 = cross3(b1, b2); let n2 = cross3(b2, b3);
    let m1 = cross3(n1, b2); let b2n = norm3(b2);
    if b2n < 1e-10 { return f64::NAN; }
    let x = dot3(n1, n2); let y = dot3(m1, n2) / b2n;
    (-y.atan2(x)).to_degrees()
}

fn summarise_dihedrals(residues: &[ResidueDihedralFull]) -> Vec<ResidueDihedralSummary> {
    residues.iter().map(|r| {
        let phi_v: Vec<f64> = r.phi.iter().copied().filter(|v| v.is_finite()).collect();
        let psi_v: Vec<f64> = r.psi.iter().copied().filter(|v| v.is_finite()).collect();
        let ms = |v: &[f64]| -> (Option<f64>, Option<f64>) {
            if v.is_empty() { return (None, None); }
            let m = v.iter().sum::<f64>() / v.len() as f64;
            let s = (v.iter().map(|x| (x-m).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
            (Some(m), Some(s))
        };
        let (phi_mean, phi_std) = ms(&phi_v);
        let (psi_mean, psi_std) = ms(&psi_v);
        ResidueDihedralSummary {
            res_seq: r.res_seq, res_name: r.res_name.clone(), atom_idx: r.atom_idx,
            phi_mean, psi_mean, phi_std, psi_std, n_valid: phi_v.len().min(psi_v.len()),
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

// ─── api_run_dihedrals ────────────────────────────────────────────────────────

async fn api_run_dihedrals(
    State(app): State<Arc<AppData>>,
    _: Json<Value>,
) -> impl IntoResponse {
    // Serve from cache
    if let Some(ref c) = *app.dihedral_cache.lock().unwrap() {
        let summaries = summarise_dihedrals(&c.residues);
        let density   = build_rama_density(&c.residues);
        let n_frames  = c.residues.first().map(|r| r.phi.len()).unwrap_or(0);
        return ok(AnalysisResult {
            data: DihedralResultJson { density, residues: summaries, mode: c.mode.clone(), n_frames },
            message: "Dihedrals (cached)".into(),
        });
    }

    let traj: Vec<Vec<[f64;3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let meta: Vec<AtomMeta> = match app.atom_meta.lock().unwrap().clone() {
        Some(m) => m, None => return err("No atom metadata"),
    };
    let tx = app.progress_tx.clone();
    emit_progress(&tx, "dihedrals", 0.0);

    let result = tokio::task::spawn_blocking(move || {
        let n_frames = traj.len();
        let mut seen_keys: Vec<(i64, char)> = Vec::new();
        seen_keys.sort_by_key(|k| k.0);
        let mut res_map: std::collections::HashMap<(i64, char), Vec<(usize, String)>> = Default::default();
        for (i, m) in meta.iter().enumerate() {
            let key = (m.res_seq, m.chain_id);
            if !res_map.contains_key(&key) { seen_keys.push(key); }
            res_map.entry(key).or_default().push((i, m.atom_name.clone()));
        }
        let backbone_count = seen_keys.iter().filter(|k| {
            let atoms = &res_map[k];
            let names: Vec<&str> = atoms.iter().map(|(_, n)| n.as_str()).collect();
            names.contains(&"N") && names.contains(&"CA") && names.contains(&"C")
        }).count();
        let backbone_mode = backbone_count > seen_keys.len() / 2;
        let mode_str = if backbone_mode { "backbone" } else { "pseudodihedral" };
        let n_res = seen_keys.len();
        let chunk = (n_res / 20).max(1);

        let residues: Vec<ResidueDihedralFull> = if backbone_mode {
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
                if ri % chunk == 0 { emit_progress(&tx, "dihedrals", ri as f64 / nb as f64 * 95.0); }
                let mut phi = vec![f64::NAN; n_frames];
                let mut psi = vec![f64::NAN; n_frames];
                for fi in 0..n_frames {
                    let f = &traj[fi];
                    if ri > 0 {
                        phi[fi] = dihedral_deg(f[bb_list[ri-1].c], f[bb_list[ri].n],
                                               f[bb_list[ri].ca],  f[bb_list[ri].c]);
                    }
                    if ri < nb - 1 {
                        psi[fi] = dihedral_deg(f[bb_list[ri].n],    f[bb_list[ri].ca],
                                               f[bb_list[ri].c],    f[bb_list[ri+1].n]);
                    }
                }
                ResidueDihedralFull { res_seq: bb_list[ri].res_seq, res_name: bb_list[ri].res_name.clone(),
                    atom_idx: bb_list[ri].ca, phi, psi }
            }).collect()
        } else {
            let ca_list: Vec<(i64, String, usize)> = seen_keys.iter().map(|k| {
                let atoms = &res_map[k];
                let ca_idx = atoms.iter().find(|(_, nm)| nm == "CA").map(|(i,_)| *i)
                    .unwrap_or(atoms[0].0);
                (k.0, meta[ca_idx].res_name.clone(), ca_idx)
            }).collect();
            let nc = ca_list.len();
            (0..nc).map(|ri| {
                if ri % chunk == 0 { emit_progress(&tx, "dihedrals", ri as f64 / nc as f64 * 95.0); }
                let mut phi = vec![f64::NAN; n_frames];
                let psi     = vec![f64::NAN; n_frames];
                if ri > 0 && ri + 2 < nc {
                    for fi in 0..n_frames {
                        let f = &traj[fi];
                        phi[fi] = dihedral_deg(f[ca_list[ri-1].2], f[ca_list[ri].2],
                                               f[ca_list[ri+1].2], f[ca_list[ri+2].2]);
                    }
                }
                ResidueDihedralFull { res_seq: ca_list[ri].0, res_name: ca_list[ri].1.clone(),
                    atom_idx: ca_list[ri].2, phi, psi }
            }).collect()
        };

        emit_progress(&tx, "dihedrals", 100.0);
        let summaries  = summarise_dihedrals(&residues);
        let density    = build_rama_density(&residues);
        let n_res_out  = residues.len();
        let mode_s     = mode_str.to_string();
        (DihedralResult { residues, mode: mode_s.clone() }, summaries, density, n_res_out, n_frames, mode_s)
    }).await;

    match result {
        Err(e) => err(e.to_string()),
        Ok((dihedral_result, summaries, density, n_res_out, n_frames, mode_s)) => {
            *app.dihedral_cache.lock().unwrap() = Some(dihedral_result);
            ok(AnalysisResult {
                data: DihedralResultJson { density, residues: summaries, mode: mode_s.clone(), n_frames },
                message: format!("Dihedral angles computed for {n_res_out} residues over {n_frames} frames ({mode_s} mode)."),
            })
        }
    }
}

// ─── api_get_residue_dihedrals ────────────────────────────────────────────────

async fn api_get_residue_dihedrals(
    State(app): State<Arc<AppData>>,
    Json(req): Json<AtomIdxReq>,
) -> impl IntoResponse {
    let lock  = app.dihedral_cache.lock().unwrap();
    let cache = match lock.as_ref() { Some(c) => c, None => return err("Run dihedral analysis first.") };
    match cache.residues.iter().find(|r| r.atom_idx == req.atom_idx) {
        None => err(format!("No dihedral data for atom index {}.", req.atom_idx)),
        Some(res) => ok(SingleResidueDihedrals {
            res_seq:  res.res_seq,
            res_name: res.res_name.clone(),
            phi:      res.phi.clone(),
            psi:      res.psi.clone(),
            mode:     cache.mode.clone(),
        }),
    }
}

// ─── api_run_prs ──────────────────────────────────────────────────────────────

async fn api_run_prs(
    State(app): State<Arc<AppData>>,
    _: Json<Value>,
) -> impl IntoResponse {
    if let Some(c) = app.prs_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "PRS (cached)".into() });
    }
    let traj: Vec<Vec<[f64;3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let tx = app.progress_tx.clone();
    emit_progress(&tx, "prs", 2.0);

    let result = tokio::task::spawn_blocking(move || {
        let n_frames = traj.len() as f64;
        let n_atoms  = traj[0].len();
        let mean     = mean_positions(&traj);

        let cov: Vec<Vec<f64>> = (0..n_atoms).into_par_iter().map(|i| {
            (0..n_atoms).map(|j| {
                let mut s = 0.0f64;
                for frame in traj.iter() {
                    let dix = frame[i][0]-mean[i][0]; let diy = frame[i][1]-mean[i][1]; let diz = frame[i][2]-mean[i][2];
                    let djx = frame[j][0]-mean[j][0]; let djy = frame[j][1]-mean[j][1]; let djz = frame[j][2]-mean[j][2];
                    s += dix*djx + diy*djy + diz*djz;
                }
                s / n_frames
            }).collect()
        }).collect();

        let variances: Vec<f64> = (0..n_atoms).map(|i| cov[i][i].max(1e-12)).collect();
        let matrix: Vec<Vec<f64>> = (0..n_atoms).map(|i|
            (0..n_atoms).map(|j| cov[i][j].abs() / (variances[i]*variances[j]).sqrt()).collect()
        ).collect();
        let effectiveness: Vec<f64> = (0..n_atoms).map(|i| {
            (0..n_atoms).filter(|&j| j!=i).map(|j| cov[i][j].abs()).sum::<f64>() / variances[i]
        }).collect();
        let sensitivity: Vec<f64> = (0..n_atoms).map(|j| {
            (0..n_atoms).filter(|&i| i!=j).map(|i| cov[i][j].abs()).sum::<f64>() / variances[j]
        }).collect();
        let norm = |v: Vec<f64>| -> Vec<f64> {
            let mx = v.iter().cloned().fold(0.0f64, f64::max);
            if mx < 1e-12 { return v; }
            v.iter().map(|&x| x/mx).collect()
        };
        (PrsResult { matrix, effectiveness: norm(effectiveness), sensitivity: norm(sensitivity) },
         n_atoms, n_frames as usize)
    }).await;

    emit_progress(&app.progress_tx.clone(), "prs", 100.0);
    match result {
        Err(e) => err(e.to_string()),
        Ok((prs_result, n_atoms, n_frames)) => {
            *app.prs_cache.lock().unwrap() = Some(prs_result.clone());
            ok(AnalysisResult {
                data: prs_result,
                message: format!("PRS computed for {n_atoms} residues over {n_frames} frames."),
            })
        }
    }
}

// ─── api_run_mutual_information ───────────────────────────────────────────────

async fn api_run_mutual_information(
    State(app): State<Arc<AppData>>,
    _: Json<Value>,
) -> impl IntoResponse {
    if let Some(c) = app.mi_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "Mutual information (cached)".into() });
    }
    let traj: Vec<Vec<[f64;3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let tx = app.progress_tx.clone();
    emit_progress(&tx, "mi", 2.0);

    let result = tokio::task::spawn_blocking(move || {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let mean     = mean_positions(&traj);
        let n_bins   = ((n_frames as f64).sqrt() as usize).min(50).max(5);

        let displacements: Vec<Vec<f64>> = (0..n_atoms).map(|i|
            traj.iter().map(|frame| {
                let dx=frame[i][0]-mean[i][0]; let dy=frame[i][1]-mean[i][1]; let dz=frame[i][2]-mean[i][2];
                (dx*dx+dy*dy+dz*dz).sqrt()
            }).collect()
        ).collect();

        let ranges: Vec<(f64,f64)> = (0..n_atoms).map(|i| {
            let lo = displacements[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = displacements[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let eps = (hi-lo)*0.01 + 1e-12;
            (lo-eps, hi+eps)
        }).collect();

        let marginal_entropy: Vec<f64> = (0..n_atoms).map(|i| {
            let (lo, hi) = ranges[i];
            let bw = (hi-lo) / n_bins as f64;
            let mut hist = vec![0u32; n_bins];
            for &d in &displacements[i] { let b = ((d-lo)/bw) as usize; hist[b.min(n_bins-1)] += 1; }
            let n = n_frames as f64;
            hist.iter().filter(|&&c| c>0).map(|&c| { let p=c as f64/n; -p*p.ln() }).sum::<f64>()
        }).collect();

        let matrix: Vec<Vec<f64>> = (0..n_atoms).into_par_iter().map(|i| {
            let (loi, hii) = ranges[i]; let bwi = (hii-loi)/n_bins as f64;
            (0..n_atoms).map(|j| {
                if i == j { return 1.0; }
                let (loj, hij) = ranges[j]; let bwj = (hij-loj)/n_bins as f64;
                let mut joint = vec![vec![0u32; n_bins]; n_bins];
                for t in 0..n_frames {
                    let bi = ((displacements[i][t]-loi)/bwi) as usize;
                    let bj = ((displacements[j][t]-loj)/bwj) as usize;
                    joint[bi.min(n_bins-1)][bj.min(n_bins-1)] += 1;
                }
                let n = n_frames as f64;
                let jh: f64 = joint.iter().flat_map(|r| r.iter())
                    .filter(|&&c| c>0).map(|&c| { let p=c as f64/n; -p*p.ln() }).sum();
                let mi = marginal_entropy[i] + marginal_entropy[j] - jh;
                let denom = (marginal_entropy[i]*marginal_entropy[j]).sqrt();
                if denom < 1e-12 { 0.0 } else { (mi/denom).max(0.0).min(1.0) }
            }).collect()
        }).collect();

        emit_progress(&tx, "mi", 100.0);
        (matrix, n_atoms, n_bins)
    }).await;

    match result {
        Err(e) => err(e.to_string()),
        Ok((matrix, n_atoms, n_bins)) => {
            *app.mi_cache.lock().unwrap() = Some(matrix.clone());
            ok(AnalysisResult {
                data: matrix,
                message: format!("NMI matrix ({n_atoms}×{n_atoms}) computed with {n_bins} histogram bins."),
            })
        }
    }
}

// ─── api_parse_cv_rst ─────────────────────────────────────────────────────────

async fn api_parse_cv_rst(
    _: State<Arc<AppData>>,
    Json(req): Json<ParseCvRstReq>,
) -> impl IntoResponse {
    let content = match std::fs::read_to_string(&req.path) {
        Ok(c) => c, Err(e) => return err(e.to_string()),
    };
    let mut blocks: Vec<CvRstBlock> = Vec::new();
    let mut pending_comment = String::new();
    let mut in_block = false;
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
                let block_idx = blocks.len();
                let cv_label = if iat.len() == 4 && rstwt.len() == 2 {
                    let sign = if rstwt[1] < 0.0 { "−" } else { "+" };
                    format!("r1{}r2 (col {})", sign, block_idx + 1)
                } else {
                    format!("CV col {} ({} atoms)", block_idx + 1, iat.len())
                };
                blocks.push(CvRstBlock {
                    block_idx, iat: iat.clone(), rstwt: rstwt.clone(),
                    r2, rk2, comment: pending_comment.clone(), cv_label,
                });
                pending_comment.clear();
            }
            in_block = false;
            continue;
        }
        if !in_block { continue; }
        let clean = line.replace(' ', "");
        for token in clean.split(',').filter(|t| t.contains('=')) {
            let mut parts = token.splitn(2, '=');
            let key = parts.next().unwrap_or("").to_lowercase();
            let val = parts.next().unwrap_or("").trim_end_matches(',');
            match key.as_str() {
                "iat" => {
                    for v in line.split(|c: char| !c.is_ascii_digit() && c != '-') {
                        if let Ok(n) = v.parse::<i64>() { iat.push(n); }
                    }
                }
                "rstwt" => { if let Ok(f) = val.parse::<f64>() { rstwt.push(f); } }
                "r2" | "r3" => { if r2 == 0.0 { r2 = val.parse::<f64>().unwrap_or(0.0); } }
                "rk2" | "rk3" => { if rk2 == 0.0 { rk2 = val.parse::<f64>().unwrap_or(0.0); } }
                _ => {}
            }
        }
    }
    if blocks.is_empty() { return err("No &rst blocks found — check the file format."); }
    ok(blocks)
}

// ─── api_run_clustering ───────────────────────────────────────────────────────

async fn api_run_clustering(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunClusteringReq>,
) -> impl IntoResponse {
    let pca = match app.pca_cache.lock().unwrap().clone() {
        Some(p) => p,
        None    => return err("Run PCA first — clustering uses PCA projections."),
    };
    let k       = req.n_clusters.max(2);
    let method  = req.method.as_deref().unwrap_or("kmeans").to_lowercase();
    let lag     = req.lag.unwrap_or(1).max(1);
    let n_macro = req.n_macro.unwrap_or(k).max(2).min(k);
    let pts     = pca.projections.clone();
    let n_frames = pts.len();
    if n_frames < k { return err(format!("Need at least {k} frames for {k} clusters.")); }

    let tx           = app.progress_tx.clone();
    let method_clone = method.clone();
    emit_progress(&tx, "cluster", 2.0);

    let result = tokio::task::spawn_blocking(move || -> Result<_,String> {
        let method = method_clone;

        fn dist2(a: [f64;2], b: [f64;2]) -> f64 {
            let dx = a[0]-b[0]; let dy = a[1]-b[1]; dx*dx + dy*dy
        }
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

        let mut assignments = vec![0usize; n_frames];
        for _ in 0..200 {
            let changed: usize = pts.par_iter().zip(assignments.par_iter_mut()).map(|(&p, a)| {
                let new = (0..k).min_by(|&i,&j| dist2(p,centers[i]).partial_cmp(&dist2(p,centers[j])).unwrap()).unwrap();
                let changed = if new != *a { 1 } else { 0 };
                *a = new; changed
            }).sum();
            let mut sums   = vec![[0.0f64;2]; k];
            let mut counts = vec![0usize; k];
            for (i, &a) in assignments.iter().enumerate() {
                sums[a][0] += pts[i][0]; sums[a][1] += pts[i][1]; counts[a] += 1;
            }
            for c in 0..k {
                if counts[c] > 0 {
                    centers[c] = [sums[c][0]/counts[c] as f64, sums[c][1]/counts[c] as f64];
                }
            }
            if changed == 0 { break; }
        }
        emit_progress(&tx, "cluster", 50.0);

        let populations: Vec<f64> = {
            let mut cnt = vec![0usize; k];
            for &a in &assignments { cnt[a] += 1; }
            cnt.iter().map(|&c| c as f64 / n_frames as f64).collect()
        };

        let mut its_out:  Vec<f64>       = Vec::new();
        let mut pcca_out: Vec<Vec<f64>>  = vec![vec![0.0; n_macro]; n_frames];

        if method == "msm" {
            let mut count = vec![vec![0u32; k]; k];
            for t in 0..(n_frames.saturating_sub(lag)) {
                count[assignments[t]][assignments[t+lag]] += 1;
            }
            let mut t_mat = vec![vec![0.0f64; k]; k];
            for i in 0..k {
                let mut row_sum = 0.0f64;
                for j in 0..k {
                    let sym = (count[i][j]+count[j][i]) as f64 * 0.5;
                    t_mat[i][j] = sym; row_sum += sym;
                }
                if row_sum > 0.0 { for j in 0..k { t_mat[i][j] /= row_sum; } }
                else { t_mat[i][i] = 1.0; }
            }
            emit_progress(&tx, "cluster", 70.0);
            let t_arr = Array2::from_shape_vec((k,k),
                t_mat.iter().flat_map(|r| r.iter().copied()).collect::<Vec<f64>>())
                .map_err(|e| e.to_string())?;
            let (eigenvalues, eigenvectors) = t_arr.eigh(ndarray_linalg::UPLO::Upper)
                .map_err(|e| format!("MSM eigendecomposition failed: {e}"))?;
            let n_ev = eigenvalues.len();
            its_out = (0..(n_ev.saturating_sub(1))).rev().take(n_macro.saturating_sub(1))
                .map(|i| {
                    let lam = eigenvalues[n_ev-1-i].abs().max(1e-10).min(1.0-1e-10);
                    -(lag as f64) / lam.ln().abs()
                }).collect();
            let top_evecs: Vec<Vec<f64>> = (1..=n_macro.min(n_ev-1)).map(|m| {
                eigenvectors.column(n_ev-1-m).iter().copied().collect()
            }).collect();
            for (f, &micro) in assignments.iter().enumerate() {
                let weights: Vec<f64> = top_evecs.iter().map(|ev| ev[micro].abs()+1e-12).collect();
                let total: f64 = weights.iter().sum();
                for m in 0..n_macro {
                    pcca_out[f][m] = if m < weights.len() { weights[m]/total } else { 0.0 };
                }
            }
        }
        emit_progress(&tx, "cluster", 100.0);
        Ok((assignments, centers, populations, its_out, pcca_out))
    }).await;

    match result {
        Err(e) => err(e.to_string()),
        Ok(Err(e)) => err(e),
        Ok(Ok((assignments, centers, populations, its, pcca_mem))) => {
            let msg = if method == "msm" && !its.is_empty() {
                format!("{k} clusters · MSM lag={lag} · ITS₁={:.1} frames", its[0])
            } else {
                format!("K-means: {k} clusters over {n_frames} frames.")
            };
            let result = ClusterResult {
                assignments: assignments.clone(), centers: centers.clone(),
                populations: populations.clone(), method: method.clone(),
                n_clusters: k, implied_timescales: its.clone(), pcca_membership: pcca_mem.clone(),
            };
            *app.cluster_cache.lock().unwrap() = Some(result);
            ok(AnalysisResult {
                data: ClusterResultJson { assignments, centers, populations, method,
                                          n_clusters: k, implied_timescales: its,
                                          pcca_membership: pcca_mem },
                message: msg,
            })
        }
    }
}

// ─── api_run_geometry_series ──────────────────────────────────────────────────

async fn api_run_geometry_series(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunGeometryReq>,
) -> impl IntoResponse {
    let source            = req.source.as_deref().unwrap_or("trajectory");
    let labels            = req.labels.unwrap_or_default();
    let composites        = req.composites.unwrap_or_default();
    let composite_weights = req.composite_weights.unwrap_or_default();
    let composite_labels  = req.composite_labels.unwrap_or_default();

    // Helper closures
    let apply_labels = |series: &mut Vec<GeometrySeries>| {
        for (i, s) in series.iter_mut().enumerate() {
            if let Some(lbl) = labels.get(i) {
                if !lbl.is_empty() { s.label = lbl.clone(); }
            }
        }
    };
    let build_composites = |series: &mut Vec<GeometrySeries>| {
        for (ci, &[si, sj]) in composites.iter().enumerate() {
            if si >= series.len() || sj >= series.len() { continue; }
            let w  = composite_weights.get(ci).copied().unwrap_or(-1.0);
            let n  = series[si].values.len().min(series[sj].values.len());
            let values: Vec<f64> = (0..n).map(|t| {
                let a = series[si].values[t]; let b = series[sj].values[t];
                if a.is_finite() && b.is_finite() { a + w*b } else { f64::NAN }
            }).collect();
            let label = composite_labels.get(ci).cloned()
                .unwrap_or_else(|| format!("RC({}{}{sj})", si, if w < 0.0 {"−"} else {"+"}));
            series.push(GeometrySeries { label, kind: "composite".into(), values, unit: "Å".into() });
        }
    };

    if source == "umbrella" {
        let windows_lock = app.umbrella_windows.lock().unwrap();
        let windows = match windows_lock.as_ref() { Some(w) => w, None => return err("Load umbrella windows first.") };
        let coords_lock = app.umbrella_traj_coords.lock().unwrap();
        let coords  = match coords_lock.as_ref() { Some(c) => c, None => return err("Pre-load umbrella coords first.") };
        let n_win   = windows.len().min(coords.len());
        drop(windows_lock);

        let mut series: Vec<GeometrySeries> = Vec::new();

        for &[i, j] in &req.pairs {
            let values: Vec<f64> = (0..n_win).map(|wi| {
                let flat = &coords[wi]; let n3 = flat.len();
                if i*3+2 >= n3 || j*3+2 >= n3 { return f64::NAN; }
                let dx = flat[i*3] as f64 - flat[j*3] as f64;
                let dy = flat[i*3+1] as f64 - flat[j*3+1] as f64;
                let dz = flat[i*3+2] as f64 - flat[j*3+2] as f64;
                (dx*dx+dy*dy+dz*dz).sqrt()
            }).collect();
            series.push(GeometrySeries { label: format!("d({i},{j})"), kind: "distance".into(), values, unit: "Å".into() });
        }

        for &[i, j, k] in &req.triplets {
            let values: Vec<f64> = (0..n_win).map(|wi| {
                let flat = &coords[wi]; let n3 = flat.len();
                if i*3+2 >= n3 || j*3+2 >= n3 || k*3+2 >= n3 { return f64::NAN; }
                let vji = [flat[i*3]-flat[j*3], flat[i*3+1]-flat[j*3+1], flat[i*3+2]-flat[j*3+2]];
                let vjk = [flat[k*3]-flat[j*3], flat[k*3+1]-flat[j*3+1], flat[k*3+2]-flat[j*3+2]];
                let dot = (vji[0]*vjk[0]+vji[1]*vjk[1]+vji[2]*vjk[2]) as f64;
                let ni  = (vji[0]*vji[0]+vji[1]*vji[1]+vji[2]*vji[2]).sqrt() as f64;
                let nk  = (vjk[0]*vjk[0]+vjk[1]*vjk[1]+vjk[2]*vjk[2]).sqrt() as f64;
                if ni < 1e-10 || nk < 1e-10 { return f64::NAN; }
                (dot/(ni*nk)).clamp(-1.0,1.0).acos().to_degrees()
            }).collect();
            series.push(GeometrySeries { label: format!("a({i},{j},{k})"), kind: "angle".into(), values, unit: "°".into() });
        }

        let n = series.first().map(|s| s.values.len()).unwrap_or(0);
        apply_labels(&mut series);
        build_composites(&mut series);
        let result = GeometryResult { series: series.clone(), n_frames: n, source: "umbrella".into() };
        *app.geometry_cache.lock().unwrap() = Some(result.clone());
        return ok(AnalysisResult {
            data: result,
            message: format!("{} geometry series over {n_win} umbrella windows.", series.len()),
        });
    }

    // Trajectory source
    let traj: Vec<Vec<[f64;3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded."),
    };
    let n_frames = traj.len();
    let n_atoms  = traj[0].len();
    for &[i,j] in &req.pairs {
        if i >= n_atoms || j >= n_atoms {
            return err(format!("Atom index out of range: max is {} (0-based).", n_atoms-1));
        }
    }
    let tx     = app.progress_tx.clone();
    let pairs  = req.pairs.clone();
    let triplets = req.triplets.clone();
    emit_progress(&tx, "geometry", 5.0);

    let series = tokio::task::spawn_blocking(move || -> Vec<GeometrySeries> {
        let mut out = Vec::new();
        let chunk = (n_frames / 20).max(1);
        for &[i,j] in &pairs {
            let values: Vec<f64> = (0..n_frames).map(|f| {
                if f % chunk == 0 { emit_progress(&tx, "geometry", 5.0 + (f as f64/n_frames as f64)*88.0); }
                let dx = traj[f][i][0]-traj[f][j][0];
                let dy = traj[f][i][1]-traj[f][j][1];
                let dz = traj[f][i][2]-traj[f][j][2];
                (dx*dx+dy*dy+dz*dz).sqrt()
            }).collect();
            out.push(GeometrySeries { label: format!("d({i},{j})"), kind: "distance".into(), values, unit: "Å".into() });
        }
        for &[i,j,k] in &triplets {
            let values: Vec<f64> = (0..n_frames).map(|f| {
                let vji = [traj[f][i][0]-traj[f][j][0], traj[f][i][1]-traj[f][j][1], traj[f][i][2]-traj[f][j][2]];
                let vjk = [traj[f][k][0]-traj[f][j][0], traj[f][k][1]-traj[f][j][1], traj[f][k][2]-traj[f][j][2]];
                let dot = vji[0]*vjk[0]+vji[1]*vjk[1]+vji[2]*vjk[2];
                let ni  = (vji[0]*vji[0]+vji[1]*vji[1]+vji[2]*vji[2]).sqrt();
                let nk  = (vjk[0]*vjk[0]+vjk[1]*vjk[1]+vjk[2]*vjk[2]).sqrt();
                if ni < 1e-10 || nk < 1e-10 { return f64::NAN; }
                (dot/(ni*nk)).clamp(-1.0,1.0).acos().to_degrees()
            }).collect();
            out.push(GeometrySeries { label: format!("a({i},{j},{k})"), kind: "angle".into(), values, unit: "°".into() });
        }
        emit_progress(&tx, "geometry", 100.0);
        out
    }).await.unwrap_or_default();

    let n = series.first().map(|s| s.values.len()).unwrap_or(0);
    let mut series = series;
    apply_labels(&mut series);
    build_composites(&mut series);
    let result = GeometryResult { series: series.clone(), n_frames: n, source: "trajectory".into() };
    *app.geometry_cache.lock().unwrap() = Some(result.clone());
    ok(AnalysisResult {
        data: result,
        message: format!("{} geometry series over {n_frames} frames.", series.len()),
    })
}

// ─── api_get_dihedral_time_series ─────────────────────────────────────────────

async fn api_get_dihedral_time_series(
    State(app): State<Arc<AppData>>,
    Json(req): Json<GetDihedralTsReq>,
) -> impl IntoResponse {
    let lock  = app.dihedral_cache.lock().unwrap();
    let cache = match lock.as_ref() { Some(c) => c, None => return err("Run Ramachandran / Dihedrals analysis first.") };
    let result: Vec<_> = req.atom_indices.iter().filter_map(|&idx| {
        cache.residues.iter().find(|r| r.atom_idx == idx).map(|r|
            serde_json::json!({
                "res_seq":  r.res_seq,
                "res_name": r.res_name,
                "phi":      r.phi,
                "psi":      r.psi,
                "mode":     cache.mode,
            })
        )
    }).collect();
    if result.is_empty() { return err("No dihedral data found for the given atom indices."); }
    ok(result)
}

// ─── api_run_sasa ─────────────────────────────────────────────────────────────
//
// Shrake-Rupley SASA, 92-point Fibonacci sphere, probe 1.4 Å by default.
// Per-atom results aggregated to per-residue groups.

async fn api_run_sasa(
    State(app): State<Arc<AppData>>,
    Json(req): Json<RunSasaReq>,
) -> impl IntoResponse {
    if let Some(c) = app.sasa_cache.lock().unwrap().clone() {
        return ok(AnalysisResult { data: c, message: "SASA (cached)".into() });
    }
    let traj: Vec<Vec<[f64;3]>> = match app.trajectory_data.lock().unwrap().clone() {
        Some(t) => t, None => return err("No trajectory loaded"),
    };
    let meta: Vec<AtomMeta> = match app.atom_meta.lock().unwrap().clone() {
        Some(m) => m, None => return err("No atom metadata"),
    };
    let probe_r = req.probe.unwrap_or(1.4);
    let tx = app.progress_tx.clone();
    emit_progress(&tx, "sasa", 2.0);

    let result = tokio::task::spawn_blocking(move || -> SasaResult {
        let n_frames = traj.len();
        let n_atoms  = traj[0].len();
        let vdw = |elem: &str| -> f64 {
            match elem {
                "C"  => 1.70, "N"  => 1.55, "O"  => 1.52, "S"  => 1.80,
                "P"  => 1.80, "H"  => 1.20, "F"  => 1.47,
                "CL"|"Cl" => 1.75, "BR"|"Br" => 1.85, "I" => 1.98,
                "CA"|"Ca" => 2.31, "ZN"|"Zn" => 1.39,
                "MG"|"Mg" => 1.73, "FE"|"Fe" => 1.52, _ => 1.70,
            }
        };
        let radii: Vec<f64> = meta.iter().map(|m| vdw(&m.element)).collect();

        const N: usize = 92;
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        let sphere: Vec<[f64;3]> = (0..N).map(|i| {
            let y = 1.0 - (i as f64 / (N as f64 - 1.0)) * 2.0;
            let r = (1.0 - y*y).max(0.0).sqrt();
            let p = golden * i as f64;
            [r*p.cos(), y, r*p.sin()]
        }).collect();

        let chunk = (n_frames / 20).max(1);
        let per_atom: Vec<Vec<f64>> = (0..n_frames).into_par_iter().map(|fi| {
            if fi % chunk == 0 {
                emit_progress(&tx, "sasa", 5.0 + (fi as f64 / n_frames as f64) * 88.0);
            }
            let frame = &traj[fi];
            (0..n_atoms).map(|i| {
                let ri = radii[i] + probe_r;
                let cut2 = (ri + 1.98 + 2.0*probe_r).powi(2);
                let nb: Vec<(usize,f64)> = (0..n_atoms).filter(|&j| {
                    if j == i { return false; }
                    let dx=frame[i][0]-frame[j][0]; let dy=frame[i][1]-frame[j][1]; let dz=frame[i][2]-frame[j][2];
                    dx*dx+dy*dy+dz*dz <= cut2
                }).map(|j| (j, radii[j]+probe_r)).collect();
                let exp = sphere.iter().filter(|&&pt| {
                    let px=frame[i][0]+ri*pt[0]; let py=frame[i][1]+ri*pt[1]; let pz=frame[i][2]+ri*pt[2];
                    !nb.iter().any(|&(j,rj)| {
                        let dx=px-frame[j][0]; let dy=py-frame[j][1]; let dz=pz-frame[j][2];
                        dx*dx+dy*dy+dz*dz < rj*rj
                    })
                }).count();
                4.0 * std::f64::consts::PI * ri*ri * (exp as f64 / N as f64)
            }).collect()
        }).collect();

        // Aggregate by residue
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, m) in meta.iter().enumerate() {
            let lbl = format!("{} {}", m.res_name, m.res_seq);
            if let Some(last) = groups.last_mut() {
                if last.0 == lbl { last.1.push(i); continue; }
            }
            groups.push((lbl, vec![i]));
        }
        let nr = groups.len();
        let mut mean = vec![0.0f64; nr]; let mut std = vec![0.0f64; nr];
        let mut total = vec![0.0f64; n_frames]; let mut labels = Vec::with_capacity(nr);
        for (ri, (lbl, idxs)) in groups.iter().enumerate() {
            labels.push(lbl.clone());
            let vals: Vec<f64> = (0..n_frames).map(|fi| idxs.iter().map(|&ai| per_atom[fi][ai]).sum()).collect();
            let m = vals.iter().sum::<f64>() / n_frames as f64;
            let s = (vals.iter().map(|v|(v-m).powi(2)).sum::<f64>() / n_frames as f64).sqrt();
            mean[ri] = m; std[ri] = s;
            for (fi,&v) in vals.iter().enumerate() { total[fi] += v; }
        }
        emit_progress(&tx, "sasa", 100.0);
        SasaResult { per_residue_mean: mean, per_residue_std: std, total_per_frame: total, res_labels: labels }
    }).await.unwrap();

    let nr = result.per_residue_mean.len();
    let nf = result.total_per_frame.len();
    let msg = format!("SASA computed for {nr} residues over {nf} frames (probe={probe_r:.1} Å).");
    *app.sasa_cache.lock().unwrap() = Some(result.clone());
    ok(AnalysisResult { data: result, message: msg })
}

// ─── api_batch_export ─────────────────────────────────────────────────────────
//
// HPC batch export: the client sends an absolute server-side directory path.
// The server creates the directory if needed and writes every populated cache
// to a named CSV.  SVG strings serialized on the browser are also written.
// Returns the list of filenames created so the client can confirm what landed.

async fn api_batch_export(
    State(app): State<Arc<AppData>>,
    Json(req): Json<BatchExportReq>,
) -> impl IntoResponse {
    use std::io::Write;
    let dir = std::path::Path::new(&req.dir);
    if let Err(e) = std::fs::create_dir_all(dir) {
        return err(format!("Cannot create '{}': {}", req.dir, e));
    }
    let mut written: Vec<String> = Vec::new();

    macro_rules! csv {
        ($name:literal, $hdr:expr, $rows:expr) => {{
            let p = dir.join($name);
            if let Ok(mut f) = File::create(&p) {
                let _ = writeln!(f, "{}", $hdr);
                for row in $rows { let _ = writeln!(f, "{}", row); }
                written.push($name.to_string());
            }
        }};
    }
    macro_rules! matrix {
        ($name:literal, $m:expr) => {{
            let p = dir.join($name);
            if let Ok(mut f) = File::create(&p) {
                for row in $m {
                    let _ = writeln!(f, "{}", row.iter().map(|v: &f64| format!("{:.4}", v)).collect::<Vec<_>>().join(","));
                }
                written.push($name.to_string());
            }
        }};
    }

    if let Some(v) = app.rmsd_cache.lock().unwrap().as_ref() {
        csv!("rmsd.csv", "frame,rmsd_angstrom",
             v.iter().enumerate().map(|(i,x)| format!("{},{:.6}",i,x)).collect::<Vec<_>>());
    }
    if let Some(v) = app.rmsf_cache.lock().unwrap().as_ref() {
        csv!("rmsf.csv", "atom_index,rmsf_angstrom",
             v.iter().enumerate().map(|(i,x)| format!("{},{:.6}",i,x)).collect::<Vec<_>>());
    }
    if let Some(v) = app.rg_cache.lock().unwrap().as_ref() {
        csv!("rg.csv", "frame,rg_angstrom",
             v.iter().enumerate().map(|(i,x)| format!("{},{:.6}",i,x)).collect::<Vec<_>>());
    }
    if let Some(p) = app.pca_cache.lock().unwrap().as_ref() {
        csv!("pca.csv", "frame,pc1,pc2",
             p.projections.iter().enumerate().map(|(i,v)| format!("{},{:.6},{:.6}",i,v[0],v[1])).collect::<Vec<_>>());
        csv!("pca_variance.csv", "component,explained_variance",
             p.explained_variance.iter().enumerate().map(|(i,v)| format!("{},{:.6}",i+1,v)).collect::<Vec<_>>());
    }
    if let Some(m) = app.dccm_cache.lock().unwrap().as_ref()     { matrix!("dccm.csv", m); }
    if let Some(m) = app.contacts_cache.lock().unwrap().as_ref() { matrix!("contacts.csv", m); }
    if let Some(m) = app.mi_cache.lock().unwrap().as_ref()       { matrix!("mutual_information.csv", m); }
    if let Some(p) = app.prs_cache.lock().unwrap().as_ref() {
        csv!("prs_effectiveness.csv", "atom_index,effectiveness",
             p.effectiveness.iter().enumerate().map(|(i,v)| format!("{},{:.6}",i,v)).collect::<Vec<_>>());
        csv!("prs_sensitivity.csv", "atom_index,sensitivity",
             p.sensitivity.iter().enumerate().map(|(i,v)| format!("{},{:.6}",i,v)).collect::<Vec<_>>());
    }
    if let Some(s) = app.sasa_cache.lock().unwrap().as_ref() {
        csv!("sasa_per_residue.csv", "residue,mean_sasa_a2,std_sasa_a2",
             s.per_residue_mean.iter().zip(&s.per_residue_std).zip(&s.res_labels)
               .map(|((m,sd),l)| format!("{},{:.4},{:.4}",l,m,sd)).collect::<Vec<_>>());
        csv!("sasa_total.csv", "frame,total_sasa_a2",
             s.total_per_frame.iter().enumerate().map(|(i,v)| format!("{},{:.4}",i,v)).collect::<Vec<_>>());
    }
    if let Some(d) = app.dihedral_cache.lock().unwrap().as_ref() {
        let p = dir.join("dihedrals.csv");
        if let Ok(mut f) = File::create(&p) {
            let _ = writeln!(f, "atom_idx,res_name,res_seq,phi_mean,psi_mean,phi_std,psi_std");
            for r in &d.residues {
                let pf: Vec<f64> = r.phi.iter().copied().filter(|v| v.is_finite()).collect();
                let ps: Vec<f64> = r.psi.iter().copied().filter(|v| v.is_finite()).collect();
                let pm  = if pf.is_empty() { f64::NAN } else { pf.iter().sum::<f64>()/pf.len() as f64 };
                let qm  = if ps.is_empty() { f64::NAN } else { ps.iter().sum::<f64>()/ps.len() as f64 };
                let ps2 = if pf.len()<2 { f64::NAN } else { (pf.iter().map(|v|(v-pm).powi(2)).sum::<f64>()/pf.len() as f64).sqrt() };
                let qs2 = if ps.len()<2 { f64::NAN } else { (ps.iter().map(|v|(v-qm).powi(2)).sum::<f64>()/ps.len() as f64).sqrt() };
                let _ = writeln!(f, "{},{},{},{:.4},{:.4},{:.4},{:.4}", r.atom_idx, r.res_name, r.res_seq, pm, qm, ps2, qs2);
            }
            written.push("dihedrals.csv".to_string());
        }
    }
    if let Some(c) = app.cluster_cache.lock().unwrap().as_ref() {
        csv!("clustering.csv", "frame,cluster_id",
             c.assignments.iter().enumerate().map(|(i,k)| format!("{},{}",i,k)).collect::<Vec<_>>());
        csv!("cluster_populations.csv", "cluster_id,population",
             c.populations.iter().enumerate().map(|(i,p)| format!("{},{:.6}",i,p)).collect::<Vec<_>>());
    }
    if let Some(g) = app.geometry_cache.lock().unwrap().as_ref() {
        let p = dir.join("geometry.csv");
        if let Ok(mut f) = File::create(&p) {
            let hdr = std::iter::once("index".to_string())
                .chain(g.series.iter().map(|s| s.label.clone()))
                .collect::<Vec<_>>().join(",");
            let _ = writeln!(f, "{}", hdr);
            let n = g.series.iter().map(|s| s.values.len()).max().unwrap_or(0);
            for i in 0..n {
                let mut row = vec![i.to_string()];
                for s in &g.series { row.push(format!("{:.6}", s.values.get(i).copied().unwrap_or(f64::NAN))); }
                let _ = writeln!(f, "{}", row.join(","));
            }
            written.push("geometry.csv".to_string());
        }
    }

    // Write SVG strings sent from the browser
    if let Some(svgs) = &req.svg_data {
        for (i, svg) in svgs.iter().enumerate() {
            if svg.trim().is_empty() { continue; }
            let name = if svgs.len() == 1 { "chart.svg".to_string() } else { format!("chart_{}.svg", i+1) };
            let p = dir.join(&name);
            if let Ok(mut f) = File::create(&p) {
                use std::io::Write as _;
                let _ = f.write_all(svg.as_bytes());
                written.push(name);
            }
        }
    }

    ok(written)
}

#[tokio::main]
async fn main() {
    let (tx, _) = broadcast::channel(256);

    let app_data = Arc::new(AppData {
        trajectory_data:      Mutex::new(None),
        atom_meta:            Mutex::new(None),
        rmsd_cache:           Mutex::new(None),
        rmsf_cache:           Mutex::new(None),
        rg_cache:             Mutex::new(None),
        dccm_cache:           Mutex::new(None),
        pca_cache:            Mutex::new(None),
        enm_cache:            Mutex::new(None),
        contacts_cache:       Mutex::new(None),
        hbond_cache:          Mutex::new(None),
        bfactor_cache:        Mutex::new(None),
        dihedral_cache:       Mutex::new(None),
        prs_cache:            Mutex::new(None),
        mi_cache:             Mutex::new(None),
        cluster_cache:        Mutex::new(None),
        geometry_cache:       Mutex::new(None),
        sasa_cache:           Mutex::new(None),
        umbrella_windows:     Mutex::new(None),
        mbar_result:          Mutex::new(None),
        qmm_topology:         Mutex::new(None),
        umbrella_traj_coords: Mutex::new(None),
        qm_region:            Mutex::new(None),
        progress_tx:          tx,
    });

    // Determine where compiled frontend assets live.
    // Default: a `dist/` folder next to the binary (produced by `npm run build`).
    let dist_dir = std::env::var("DIST_DIR").unwrap_or_else(|_| "dist".to_string());

    let router = Router::new()
        // WebSocket for progress + viz events
        .route("/api/events",                          get(ws_handler))
        // Analysis commands
        .route("/api/load_trajectory",                 post(api_load_trajectory))
        .route("/api/run_rmsd",                        post(api_run_rmsd))
        .route("/api/run_rmsf",                        post(api_run_rmsf))
        .route("/api/run_radius_of_gyration",          post(api_run_radius_of_gyration))
        .route("/api/run_pca",                         post(api_run_pca))
        .route("/api/run_dccm",                        post(api_run_dccm))
        .route("/api/run_contacts",                    post(api_run_contacts))
        .route("/api/run_hbond",                       post(api_run_hbond))
        .route("/api/run_enm",                         post(api_run_enm))
        .route("/api/run_nma_overlap",                 post(api_run_nma_overlap))
        .route("/api/run_communities",                 post(api_run_communities))
        .route("/api/run_betweenness",                 post(api_run_betweenness))
        .route("/api/run_optimal_paths",               post(api_run_optimal_paths))
        .route("/api/run_fes",                         post(api_run_fes))
        .route("/api/run_entropy",                     post(api_run_entropy))
        .route("/api/export_csv",                      post(api_export_csv))
        .route("/api/write_text_file",                 post(api_write_text_file))
        // Visualizer helpers
        .route("/api/get_snapshot_pdb",                post(api_get_snapshot_pdb))
        .route("/api/get_frame_coords",                post(api_get_frame_coords))
        .route("/api/set_bfactors",                    post(api_set_bfactors))
        .route("/api/viz_event",                       post(api_viz_event))
        // Umbrella sampling
        .route("/api/load_umbrella_windows",           post(api_load_umbrella_windows))
        .route("/api/run_mbar",                        post(api_run_mbar))
        .route("/api/set_qmm_topology",                post(api_set_qmm_topology))
        .route("/api/get_umbrella_snapshot_pdb",       post(api_get_umbrella_snapshot_pdb))
        .route("/api/get_umbrella_window_count",       post(api_get_umbrella_window_count))
        .route("/api/preload_umbrella_coords",         post(api_preload_umbrella_coords))
        .route("/api/get_umbrella_window_coords",      post(api_get_umbrella_window_coords))
        // QM region
        .route("/api/save_qm_region",                  post(api_save_qm_region))
        .route("/api/get_qm_region",                   post(api_get_qm_region))
        .route("/api/clear_qm_region",                 post(api_clear_qm_region))
        .route("/api/resolve_qm_selection",            post(api_resolve_qm_selection))
        .route("/api/rewrite_pdb",                     post(api_rewrite_pdb))
        .route("/api/get_selection_residues",          post(api_get_selection_residues))
        .route("/api/run_dihedrals",                   post(api_run_dihedrals))
        .route("/api/get_residue_dihedrals",           post(api_get_residue_dihedrals))
        .route("/api/run_prs",                         post(api_run_prs))
        .route("/api/run_mutual_information",          post(api_run_mutual_information))
        .route("/api/run_clustering",                  post(api_run_clustering))
        .route("/api/run_geometry_series",             post(api_run_geometry_series))
        .route("/api/get_dihedral_time_series",        post(api_get_dihedral_time_series))
        .route("/api/parse_cv_rst",                    post(api_parse_cv_rst))
        .route("/api/run_sasa",                        post(api_run_sasa))
        .route("/api/batch_export",                    post(api_batch_export))
        // Static frontend (must be last — catch-all fallback)
        .nest_service("/", ServeDir::new(&dist_dir).append_index_html_on_directories(true))
        .layer(CorsLayer::permissive())
        .with_state(app_data);

    let port = std::env::var("PORT").unwrap_or_else(|_| "7272".to_string());
    let addr = format!("0.0.0.0:{port}");
    println!("md-server listening on http://{addr}");
    println!("SSH tunnel:  ssh -L {port}:localhost:{port} <user>@<cluster>");
    println!("Then open:   http://localhost:{port}");

    let listener = tokio::net::TcpListener::bind(&addr).await
        .expect("Failed to bind port — is another process using it?");
    axum::serve(listener, router).await.expect("Server error");
}
