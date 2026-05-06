# Atmos

**Analysis, Trajectory, Molecular & Optimization Suite**

A desktop application for interactive molecular dynamics trajectory analysis and ligand preparation. Built with **Tauri v2** (Rust backend) and a TypeScript/SVG frontend, Atmos loads trajectory files directly into memory and runs all analyses locally — no Python environment, no Jupyter notebooks, no data transfer.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Building from Source](#building-from-source)
- [Loading a Trajectory](#loading-a-trajectory)
- [Analysis Tools](#analysis-tools)
  - [Structural](#structural)
  - [Correlation & Dynamics](#correlation--dynamics)
  - [Normal Mode Analysis](#normal-mode-analysis)
  - [Network Analysis](#network-analysis)
  - [Membrane Analysis](#membrane-analysis)
  - [Thermodynamics](#thermodynamics)
  - [QM/MM & Umbrella Sampling](#qmmm--umbrella-sampling)
- [Ligand Builder](#ligand-builder)
- [3D Analysis Viewer](#3d-analysis-viewer)
- [Exporting Results](#exporting-results)
- [Optional Tools (conda)](#optional-tools-conda)
- [File Format Support](#file-format-support)
- [Atom Selection Syntax](#atom-selection-syntax)
- [Architecture](#architecture)
- [Dependencies](#dependencies)

---

## Features

- **Zero-copy trajectory indexing** — the full trajectory is loaded into RAM once; all subsequent analyses run from memory with no disk I/O
- **Parallel computation** — all heavy analyses (SASA, PCA, DCCM, MI, PRS, clustering, membrane) use Rayon for multi-core parallelism
- **Async GUI** — every long-running command runs in a dedicated thread pool via `spawn_blocking`; the UI stays responsive with a live progress bar and a **Cancel** button on long analyses
- **Session persistence** — trajectory path, topology path, atom selection, and stride are restored automatically on next launch
- **File associations** — double-click any `.nc`, `.xtc`, `.dcd`, or `.pdb` file to open it directly in Atmos
- **Integrated 3D viewer** — NGL-based structure viewer opens in a secondary window for every analysis result, with per-mode representations and colorschemes
- **Interactive SVG charts** — all charts are rendered as SVG with hover, click-to-3D, and per-chart CSV export
- **Batch export** — all cached analyses exported to a user-chosen folder as CSV in one click
- **Stride support** — load every Nth frame to reduce memory usage for very long trajectories. A memory estimate is shown in the log after loading
- **Membrane analysis** — bilayer thickness, area per lipid, leaflet z-density, and lipid order parameters (|SCD|) in one pass
- **Ligand Builder** — interactive 3D molecular editor with UFF, GFN-xTB, and DFT (ORCA) geometry optimization; ChemDraw-style 2D depiction; SMILES I/O; QM input file generation; direct QM region integration
- **Optional tool management** — xtb, Open Babel, and AutoDock Vina can be installed into a managed conda environment with one click from the dep manager (cat icon → Manage Dependencies)

---

## Requirements

- macOS 11+ (Apple Silicon or Intel), Windows 10+, or Linux (glibc 2.28+)
- OpenBLAS or Accelerate framework (macOS) for linear algebra
- 8 GB RAM recommended for typical all-atom trajectories; 16 GB+ for large systems
- **Optional:** [Miniforge](https://github.com/conda-forge/miniforge) for installing xtb, Open Babel, and AutoDock Vina via the built-in dependency manager

---

## Installation

Download the latest release from the [Releases](../../releases) page:

| Platform | File |
|---|---|
| macOS (Apple Silicon) | `Atmos_x.x.x_aarch64.dmg` |
| macOS (Intel) | `Atmos_x.x.x_x64.dmg` |
| Windows | `Atmos_x.x.x_x64-setup.exe` |
| Linux | `Atmos_x.x.x_amd64.AppImage` |

### macOS — first launch

Atmos is currently unsigned. On first launch, macOS Gatekeeper will block it. To bypass:

```bash
xattr -dr com.apple.quarantine /Applications/Atmos.app
```

Or: right-click the `.app` → **Open** → **Open** in the dialog.

---

## Building from Source

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Node.js (v18+) — https://nodejs.org

# macOS: Accelerate is included in Xcode Command Line Tools
xcode-select --install

# Linux: install OpenBLAS
sudo apt install libopenblas-dev   # Debian/Ubuntu
sudo dnf install openblas-devel    # Fedora/RHEL
```

### Build

```bash
git clone https://github.com/evanj/atmos
cd Atmos
npm install
npm run tauri build
```

The compiled application will be in `src-tauri/target/release/bundle/`.

### Development mode

```bash
npm run tauri dev
```

---

## Loading a Trajectory

1. Click **Select trajectory** and choose your trajectory file (or double-click a trajectory file in Finder/Explorer to open Atmos directly)
2. Optionally click **Select topology** — required for `.nc`/`.ncrst` files without embedded topology
3. Set the **Atom selection** (default: `name CA` — Cα atoms only). See [Atom Selection Syntax](#atom-selection-syntax)
4. Optionally set **Stride** — enter `10` to load every 10th frame, reducing memory usage 10×
5. Click **Load & Index**

The status bar shows the number of frames, atoms, and estimated memory usage. A ⚠ warning is shown if the load exceeds 1 GB.

After loading, click **Visualize Trajectory** to open the 3D frame scrubber.

**Session restore:** Atmos automatically remembers your last trajectory, topology, selection, and stride. On the next launch, the fields are pre-populated — just click **Load & Index**.

---

## Analysis Tools

All tools appear in the right panel, grouped by category. Select a tool, adjust parameters if needed, and click **Run**. Results are cached — re-running a tool instantly returns the cached result unless a new trajectory is loaded. Long-running tools (SASA, Membrane) show a **✕ Cancel** button.

### Structural

#### RMSD
Root-mean-square deviation of all selected atoms from the first frame. Click any point to open that frame in the 3D viewer.

#### RMSF
Per-atom root-mean-square fluctuation over the trajectory. Identifies flexible and rigid regions. Mapped to the B-factor channel for 3D coloring (blue = rigid → red = flexible). **Map to 3D** opens the colored structure.

#### Radius of Gyration
Global compactness (Rg) per frame. Tracks folding/unfolding events. Click any frame to view in 3D.

#### SASA — Solvent Accessible Surface Area
Shrake-Rupley algorithm with a 92-point Fibonacci sphere.

**Parameters:**
- *Probe radius (Å)* — default 1.4 Å (water molecule)

**Output:** Two-panel chart — per-residue mean SASA bar chart with ±1σ error bars (blue=buried → red=exposed) plus total SASA vs frame. Clicking a residue bar opens the 3D viewer with that residue highlighted. **Map to 3D** colors the structure by mean SASA.

**Van-der-Waals radii (Å):** C 1.70, N 1.55, O 1.52, S 1.80, P 1.80, H 1.20.

#### PCA — Principal Component Analysis
Covariance matrix from Cα displacements. 2D scatter shows frames projected onto PC1/PC2 — clusters indicate distinct conformational states. Click any frame to view in 3D. Explained variance for PC1 and PC2 is shown in the log.

> **Note:** PCA must be run before Clustering / MSM.

#### Ramachandran / Dihedrals
Backbone φ/ψ angles over the full trajectory.

- **Backbone mode** (auto-detected): uses N–Cα–C atom triplets when all backbone atoms are present
- **Pseudo-dihedral mode**: uses 4 consecutive Cα atoms when only Cα is available

**Output:** 2D density heatmap with per-residue mean positions as coloured dots. Allowed regions (α-helix, β-sheet, left-handed α) are outlined. Click any residue dot to open the Dihedral mode viewer with a floating φ/ψ time series overlay.

#### Clustering / MSM
K-means clustering of trajectory frames in PCA space, with optional Markov State Model.

**Parameters:**
- *Clusters (k)* — number of microstates
- *Method* — `kmeans` or `msm`
- *MSM lag (frames)* — transition counting lag time (MSM only)
- *PCCA macrostates* — spectral coarse-graining (MSM only)

> Requires PCA to be run first.

#### Dihedral Time Series
Full φ and ψ angle time series for selected residues. Shows rotameric transitions over time.

**Parameters:**
- *Atom indices (comma-separated)* — 0-based indices (e.g. `0,5,42`)

> Requires Ramachandran / Dihedrals to be run first.

---

### Correlation & Dynamics

#### DCCM — Dynamic Cross-Correlation Matrix
Linear pairwise correlation of atom displacements (−1 to +1). Click any cell to highlight the residue pair in 3D.

#### H-Bond Lifetime
Hydrogen bond occupancy over the trajectory.

**Parameters:**
- *Cutoff distance (Å)* — donor–acceptor heavy-atom distance (default 3.5 Å)
- *Min occupancy* — minimum fraction of frames to include (default 0.1)

#### Contact Map
Residue–residue contact frequency matrix.

**Parameters:**
- *Cutoff (Å)* — default 8.0 Å

#### Mutual Information
Normalised mutual information (NMI) between residue displacement magnitudes. Detects non-linear statistical coupling that DCCM misses. Values from 0 (independent) to 1 (perfectly coupled).

---

### Normal Mode Analysis

#### Elastic Network Model (ENM)
ANM or GNM built from the mean structure.

**Parameters:**
- *Cutoff (Å)* — residue pair interaction cutoff (default 7.5 Å)
- *Modes* — number of normal modes to compute (default 20)
- *Model* — `ANM` or `GNM`

**Map to 3D** colors residues by their contribution to mode 1.

#### Mode–Trajectory Overlap
Cumulative overlap of ENM modes with observed trajectory fluctuations.

> Requires ENM to be run first.

---

### Network Analysis

#### Community Detection
Girvan-Newman modularity-based community detection on the DCCM-derived residue contact graph.

**Parameters:**
- *Edge threshold* — minimum |DCCM| value to include an edge (default 0.6)

> Requires DCCM to be run first.

#### Perturbation Response Scanning (PRS)
Unit mechanical perturbation at each residue; measures propagation using the covariance matrix as compliance tensor.

**Output:** Response matrix heatmap, **Effectiveness** (allosteric drivers) and **Sensitivity** (allosteric sensors) bar charts. **Map to 3D** colors by effectiveness.

#### Betweenness Centrality
Edge betweenness centrality in the DCCM-derived network. Identifies communication bridges.

#### Optimal Paths
Shortest path between a source and sink residue through the residue network.

**Parameters:**
- *Source residue* — 0-based index
- *Sink residue* — 0-based index
- *Edge threshold* — minimum |DCCM| to include an edge

---

### Membrane Analysis

Membrane bilayer analysis — runs in a single parallel pass over all frames. Load a full-atom lipid selection for best results.

**Parameters:**
- *Headgroup atom name(s)* — comma-separated atom names used for leaflet assignment (default `P` — matches phosphorus in all common phospholipids including POPC LIPID17/LIPID21)
- *Membrane normal* — `0`=x, `1`=y, `2`=z (default `2`)

**Output:** Four-panel chart:

| Panel | Description |
|---|---|
| Thickness (Å) vs frame | Mean headgroup-to-headgroup bilayer thickness |
| Area per lipid (Ų) vs frame | Box XY area ÷ number of lipid molecules per leaflet. Requires periodic box dimensions (present in AMBER `.nc`) |
| Leaflet z-density | Normalised headgroup density distributions for upper (teal) and lower (purple) leaflets shown as a mirror plot — lower leaflet left, upper leaflet right, separated by the bilayer midplane |
| \|SCD\| order parameters | Mean carbon–carbon bond order parameter per chain carbon. Rendered as a polyline profile for full-atom selections or a colour-mapped bar chart for short chains |

**Leaflet assignment** is performed by median split of headgroup atom z-coordinates per frame — no residue-name assumptions, works with any AMBER lipid force field.

**APL** is calculated as box area divided by the number of unique lipid residues in the upper leaflet, not the number of atoms, so it is correct for full-atom selections.

> For AMBER `.nc` files, periodic box dimensions are read automatically. For files without box information, APL will not be available but thickness and density are unaffected.

---

### Thermodynamics

#### Free Energy Surface (FES)
2D free energy surface from the PCA projection histogram: ΔG(x,y) = −kT ln P(x,y).

**Parameters:**
- *Bins* — histogram bins per axis

#### Conformational Entropy
Quasi-harmonic configurational entropy estimate from covariance eigenvalues (Schlitter formula).

---

### QM/MM & Umbrella Sampling

Handles AMBER umbrella sampling workflows and QM/MM system preparation. Switch to the **QM/MM** tab in the sidebar.

#### Loading Umbrella Windows

1. **CV file pattern** — path with `{window}` placeholder, e.g. `../run_{window}/step5.cv`
2. **Windows** — total number of umbrella windows
3. **CV min / CV max** — reaction coordinate range
4. **CV column** — which column of the `.cv` file contains the collective variable
5. **Restart pattern** (optional) — `.ncrst` file pattern for 3D visualization
6. **Topology** (optional) — `.pdb` for restart file reading
7. **CV restraint file** (optional) — `.cv.rst` / `NMR.def` AMBER restraint file
8. Click **Scan & Load Windows**

#### CV Restraint File (cv.rst)

Loading a `.cv.rst` file parses all `&rst...&end` blocks and extracts atom indices, weights, and force constants. When running **Distance / Angle Monitor** with `Source: umbrella`, distances are auto-populated from the appropriate restraint block:

- **r1–r2 type** (4 `iat` values): produces `r1 (breaking)`, `r2 (forming)`, and `RC = r1 − r2` automatically
- **Simple distance** (2 `iat` values): produces a single distance series

Example:
```fortran
 &rst
  iat=7488,7496,7496,7164,
  rstwt=1.,-1.,
  r2=-0.950, r3=-0.950,
  rk2=150.0, rk3=150.0,
 &end
```

#### MBAR / PMF
Multistate Bennett Acceptance Ratio free energy calculation.

**Parameters:**
- *Force constant (kcal/mol/Å²)* — umbrella harmonic restraint force constant
- *Temperature (K)* — default 300 K
- *PMF bins* — histogram resolution
- *Bootstrap replicates* — for ΔG‡ uncertainty estimation

**Output:** PMF profile with bootstrap confidence band, ΔG‡ with uncertainty, KDE overlap histogram. Click any PMF point to open the nearest umbrella window structure in 3D.

#### Distance / Angle Monitor
Interatomic distances and/or angles over the trajectory or across umbrella windows.

**Parameters:**
- *Extra distance pairs* — 0-based atom index pairs in `i,j;i,j` format
- *Angle triplets* — 0-based triplets in `i,j,k` format
- *Source* — `trajectory` or `umbrella`

When `Source: umbrella` and a cv.rst file is loaded, r1/r2/RC series are auto-populated. An **Overlay on PMF** button appends the geometry panel below the existing PMF chart.

#### Window Trajectory Viewer
Plays back umbrella window restart files as a pseudo-trajectory in the 3D viewer.

#### QM Region Builder
Interactive tool for defining the QM region. Click atoms in the 3D viewer to add/remove them. The AMBER atom mask (`@1,2,3,…`) is generated automatically.

---

## Ligand Builder

Opens in its own window via the **⬡ Ligand Builder** button in the main header.

### 3D Editor

**Mouse controls:**

| Action | Result |
|---|---|
| Left drag | Rotate |
| Scroll | Zoom |
| Alt + drag / Middle drag | Pan |
| Right-click bond | Cycle bond order |

**Tools:** Select (V), Atom (A), Bond (B), Erase (E).

**Edit actions:** Undo/Redo (Ctrl+Z / Ctrl+Y), ⚡ Quick Minimize (200-step UFF), +H / −H, ✕ Clear.

### 2D Structure Depiction

Continuously updated ChemDraw-style 2D depiction computed from the molecular graph — not from projecting 3D coordinates. Stereo bonds (wedge/dash) are derived automatically from 3D z-coordinates after minimization.

### Minimization Methods

All methods run asynchronously with a live progress bar and **Cancel** button.

#### UFF — built-in, always available
Rust-native UFF minimizer. Bond stretching, angle bending, non-bonded repulsion, Armijo backtracking line search. No external dependencies.

#### GFN2-xTB / GFN1-xTB — requires `xtb`
Writes a temporary XYZ, invokes `xtb --opt`, streams per-cycle energies, reads back `xtbopt.xyz`. Install via the dependency manager (see [Optional Tools](#optional-tools-conda)).

**Parameters:** *Opt level* — `crude`, `normal`, `tight`, `vtight`.

#### ORCA — requires ORCA ≥ 5.0
Supported methods: B3LYP/def2-SVP, PBE0/def2-TZVP, ωB97X-D3/def2-TZVP. ORCA is free for academic use: [orcaforum.kofo.mpg.de](https://orcaforum.kofo.mpg.de).

**Parameters:** *orca binary* path, *Cores* (parallel MPI processes).

### SMILES Support
Paste a SMILES string into the header field and click **Load**. Handles the full organic subset including ring closures, aromatic atoms, and bracketed atoms with charge/isotope. **Copy SMILES** writes the current molecule to the clipboard.

### QM Input Generation

Generates ready-to-run input files for six QM packages: Q-Chem, ORCA, Gaussian 16, NWChem, Psi4, GAMESS. Supports SP, Opt, Freq, TS, and Scan job types. PCM solvents, custom basis sets, and extra keywords are configurable.

### Export Formats

| Format | Notes |
|---|---|
| XYZ | Standard XYZ with element symbols and Ångström coordinates |
| PDB | HETATM records |
| MOL | MDL MOL V2000 |
| SDF | MOL block with `$$$$` terminator |
| QM input | Format set by QM Input tab's Program selector |

---

## 3D Analysis Viewer

Secondary window opened by most analysis tools. NGL WebGL rendering.

| Mode | Description |
|---|---|
| **B-factor** | Color structure by mapped analysis value |
| **Frame** | Specific trajectory frame with scrub slider |
| **Residue** | Highlighted residue (ball+stick) against backbone |
| **Dihedral** | Residue highlight + floating Ramachandran overlay |
| **Pair** | Two residues with a distance line |
| **Cluster** | Representative frame of a cluster |
| **Umbrella** | Umbrella window structure with QM region overlay |

**Representations:** Cartoon, Backbone, Ball+Stick, Surface, Licorice.
**Color schemes:** Residue index, Element, B-factor, Secondary structure, Chain, Uniform.

---

## Exporting Results

### Per-chart CSV
Every chart has an **Export CSV** button in its title bar.

### Batch export
Click **Batch Export** in the bottom bar. All cached analyses are written to a chosen folder:

| File | Contents |
|---|---|
| `rmsd.csv` | Frame, RMSD (Å) |
| `rmsf.csv` | Atom, RMSF (Å) |
| `rg.csv` | Frame, Rg (Å) |
| `pca.csv` | Frame, PC1, PC2 |
| `pca_variance.csv` | Component, explained variance |
| `dccm.csv` | N×N correlation matrix |
| `contacts.csv` | N×N contact frequency matrix |
| `mutual_information.csv` | N×N NMI matrix |
| `prs_effectiveness.csv` | Atom, effectiveness |
| `prs_sensitivity.csv` | Atom, sensitivity |
| `sasa_per_residue.csv` | Residue, mean SASA (Å²), std SASA (Å²) |
| `sasa_total.csv` | Frame, total SASA (Å²) |
| `membrane_per_frame.csv` | Frame, thickness (Å), APL (Ų) |
| `membrane_order_params.csv` | Carbon, \|SCD\| |
| `membrane_z_density.csv` | Bin, upper density, lower density |
| `dihedrals.csv` | Atom, residue, φ mean/std, ψ mean/std |
| `clustering.csv` | Frame, cluster ID |
| `cluster_populations.csv` | Cluster ID, population fraction |
| `geometry.csv` | Index, distance/angle series |
| `chart.svg` | Current chart panel as SVG |

Analyses not yet run are silently skipped.

---

## Optional Tools (conda)

xtb, Open Babel, and AutoDock Vina can be installed into a managed conda environment (`atmos-env`) with one click.

### Installing Miniforge (required)

If conda is not already installed, download [Miniforge](https://github.com/conda-forge/miniforge) (recommended) or any conda distribution:

```bash
# macOS/Linux — install Miniforge
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh | bash
```

Restart Atmos after installing conda.

### Installing optional tools

1. Click the **cat icon** in the header bar → **Manage Dependencies**
2. If conda is detected, click **Install all tools via conda**
3. A live log shows conda output as the packages are installed

This creates an `atmos-env` conda environment containing:

| Tool | Package | Purpose |
|---|---|---|
| xtb | `xtb` | GFN1-xTB and GFN2-xTB geometry optimization |
| Open Babel | `openbabel` | Format conversion for docking (PDBQT prep) |
| AutoDock Vina | `vina` | Molecular docking |

Once installed, Atmos automatically detects and uses these tools — no manual path configuration needed.

### Manual path (advanced)

If conda is not found, the dep manager shows a text field where you can enter the full path to your conda or mamba executable directly (e.g. `/opt/homebrew/Caskroom/miniforge/base/bin/conda`). Atmos will validate the path before saving it.

If you manage your own xtb/obabel/vina installations outside conda, Atmos falls back to looking for these binaries on your system PATH.

---

## File Format Support

Atmos uses [chemfiles](https://chemfiles.org) for trajectory I/O.

| Format | Extensions | Notes |
|---|---|---|
| AMBER NetCDF | `.nc` | Requires topology; box dimensions read for APL |
| AMBER restart | `.ncrst` | Single-frame; used for umbrella window visualization |
| GROMACS XTC | `.xtc` | |
| GROMACS TRR | `.trr` | |
| CHARMM/NAMD DCD | `.dcd` | |
| PDB | `.pdb` | Trajectory or single frame |
| XYZ | `.xyz` | |

Topology formats: `.pdb`, `.psf` (CHARMM).

---

## Atom Selection Syntax

Selections use the chemfiles/VMD-style selection language:

```
name CA                          # Cα atoms only (default)
name CA CB                       # Cα and Cβ
name CA C N O                    # full backbone
resname ALA GLY                  # all atoms in alanine and glycine
(resid >= 10) and (resid <= 50)  # residues 10–50
not name H HA HB                 # exclude hydrogens
index 0:499                      # first 500 atoms (0-based)
```

The selection is applied once at load time. All analysis indices are 0-based within this selection.

**Recommendations by analysis:**
- RMSD, RMSF, PCA, clustering — `name CA` is sufficient
- SASA, contacts, MI — use `not name H*` (all heavy atoms) for accurate surface/contact geometry
- Membrane — full-atom lipid selection; use `name P` (or `name P31` for some force fields) for headgroup-only leaflet assignment
- Ramachandran — requires backbone atoms: `name CA C N O`

---

## Architecture

```
atmos/
├── src/
│   ├── main.ts                 # UI, tool registry, chart renderers, dep manager
│   ├── analysis_viewer.ts      # NGL-based 3D viewer (secondary window)
│   ├── visualizer.ts           # Trajectory scrubber viewer
│   ├── umbrella_viewer.ts      # Umbrella window trajectory viewer
│   └── ligand_builder.ts       # Ligand Builder — editor, 2D depiction, minimization UI
├── src-tauri/
│   ├── src/main.rs             # Rust backend — all analysis, ligand, dep manager commands
│   ├── capabilities/
│   │   └── default.json        # Tauri v2 permissions (fs, dialog, opener)
│   ├── Cargo.toml
│   └── tauri.conf.json
└── package.json
```

### Data flow

1. User selects a file via `@tauri-apps/plugin-dialog` (or double-clicks an associated file)
2. Frontend calls `invoke('load_trajectory', { path, topPath, selectionStr, stride })`
3. Rust reads with chemfiles, applies selection, stores `Vec<Vec<[f64;3]>>` + atom metadata + per-frame box dimensions in `AppData` (shared state in `Mutex`)
4. Analysis commands read from `AppData`, run on `spawn_blocking` threads, cache results back into `AppData`
5. Progress emitted via `app.emit("progress", ...)` drives the frontend progress bar
6. Results serialized to JSON, returned to frontend, rendered as SVG charts
7. Optional tools resolved from the managed `atmos-env` conda prefix; `XTBPATH` set automatically before invoking xtb

---

## Dependencies

### Rust

| Crate | Version | Purpose |
|---|---|---|
| `tauri` | 2 | Desktop app framework |
| `tauri-plugin-dialog` | 2 | Native file picker dialogs |
| `tauri-plugin-fs` | 2 | File write for batch export |
| `tauri-plugin-opener` | 2 | Open external URLs (dep manager) |
| `chemfiles` | 0.10.41 | Trajectory I/O (NC, XTC, DCD, …) |
| `ndarray` | 0.15 | N-dimensional arrays |
| `ndarray-linalg` | 0.16 | Eigendecomposition (LAPACK via OpenBLAS/Accelerate) |
| `rayon` | 1.8 | Data-parallel iterators |
| `serde` / `serde_json` | 1 | JSON serialization |
| `glob` | 0.3 | Umbrella window file pattern matching |
| `tempfile` | 3 | Temporary directory for xtb/ORCA I/O |
| `tokio` | 1 | Async subprocess execution |

### Frontend

| Library | Purpose |
|---|---|
| NGL Viewer | WebGL molecular graphics |
| Vite | Build tool and dev server |
| TypeScript | Type-safe frontend code |

All charts are rendered as inline SVG — no charting library dependency.

### Optional external tools

Installed via the built-in dependency manager (conda-forge). None are required to run the application — UFF minimization is always available without any external dependencies.

| Tool | conda package | Purpose |
|---|---|---|
| xtb | `xtb` | GFN1-xTB and GFN2-xTB geometry optimization |
| Open Babel | `openbabel` | Format conversion for docking |
| AutoDock Vina | `vina` | Molecular docking in the Ligand Builder |
| ORCA | (manual install) | DFT geometry optimization — free for academic use at [orcaforum.kofo.mpg.de](https://orcaforum.kofo.mpg.de) |
