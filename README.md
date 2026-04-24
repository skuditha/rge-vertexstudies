# RG-E Vertex Studies

Utilities for studying reconstructed vertex positions in CLAS12 Run Group E (RG-E) pass-1 data. The workflow converts HIPO files to compact ROOT ntuples, compares available reconstructed vertex definitions, fits target-related vertex peaks, monitors run dependence, and produces recommended LD2/solid-target vertex cuts.

The repository is organized around the RG-E target layouts used in the spring 2024 production, including LD2 + solid target configurations for C, Al, Cu, Sn, and Pb, solid-only runs, and an empty/wire reference run.

## What this repository does

The analysis pipeline is:

1. **Extract charged-track ntuples** from CLAS12 HIPO files using a C++ ROOT/HIPO extractor.
2. **Compare vertex sources**:
   - `particle`: `REC::Particle.vz`
   - `ftrack`: `REC::FTrack.vz` only
   - `hybrid`: `REC::FTrack.vz` when available and finite, otherwise `REC::Particle.vz`
3. **Fit the empty/wire reference run** to determine the reference-foil position.
4. **Produce production QA histograms** for LD2 + solid target runs.
5. **Fit LD2 and solid peaks** run-by-run, charge-by-charge, detector-region-by-detector-region, and, for forward tracks, sector-by-sector.
6. **Study run dependence** of fitted peak positions, widths, and entry fractions.
7. **Compare charge-only and PID-split selections** as a systematic cross-check.
8. **Extract recommended LD2/solid vertex cuts** and generate validation plots.

Generated ROOT files, plots, tables, and logs are written under `outputs/`, which is intentionally ignored by git.

## Repository layout

```text
.
├── configs/                       # YAML configuration files for runs, fits, QA, cuts, and plotting
├── cpp/                           # C++ HIPO-to-ROOT extractor
│   ├── src/hipo_to_root.cpp
│   ├── CMakeLists.txt
│   ├── Makefile
│   └── scripts/build.sh
├── python/rge_vertex/             # Reusable Python package
│   ├── cuts/                      # Cut extraction and recommendation logic
│   ├── fitting/                   # Local peak fitting models and fit helpers
│   ├── io/                        # YAML and ROOT loading helpers
│   ├── plotting/                  # Histogram and plotting helpers
│   ├── selections/                # Track/category selection logic
│   └── studies/                   # QA, run-dependence, PID, and validation studies
├── scripts/                       # Numbered command-line analysis steps
├── environment.yml                # Conda environment for the Python workflow
├── pyproject.toml                 # Editable-install metadata for the Python package
├── overnight_ld2_solid_batch.sh   # Parallel launcher for QA and LD2/solid fits
└── merge_after_overnight.sh       # Helper to merge per-run CSV outputs
```

## Requirements

### C++ extractor

The extractor requires:

- C++17 compiler
- ROOT with `root-config` available, or CMake ROOT discovery working
- HIPO4 headers and library
- `HIPO` environment variable pointing to the HIPO installation

On the JLab farm, load the relevant CLAS12/ROOT/HIPO environment before building. The exact module commands may depend on the production environment being used.

### Python analysis

The Python workflow requires Python 3.9 or newer and the packages listed in `environment.yml` / `pyproject.toml`:

- `numpy`
- `pandas`
- `pyyaml`
- `uproot`
- `awkward`
- `matplotlib`
- `iminuit`

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate rge-vertex
```

Install the local package in editable mode so the scripts can import `rge_vertex`:

```bash
pip install -e .
```

## Build the extractor

From the repository root:

```bash
cd cpp
./scripts/build.sh
cd ..
```

This creates:

```text
cpp/build/rge_hipo_to_root
```

A simple Makefile is also available:

```bash
cd cpp
make
cd ..
```

## Input configuration

Run metadata lives in YAML files under `configs/`.

Important files:

- `configs/runs.yaml`: full run list used by the main workflow.
- `configs/short_runs.yaml`: smaller run subset for testing.
- `configs/detector_regions.yaml`: detector-region convention.
- `configs/empty_wire_fit.yaml`: empty/wire local-fit configuration.
- `configs/production_qa.yaml`: production QA histogram configuration.
- `configs/ld2_solid_local_fit.yaml`: LD2 + solid local-fit configuration.
- `configs/pid_study.yaml`: charge-only vs PID-split comparison configuration.
- `configs/run_dependence.yaml`: run-dependence plotting configuration.
- `configs/cut_extraction.yaml`: final cut extraction/recommendation policy.
- `configs/cut_validation.yaml`: validation plotting configuration.

Each run entry typically defines:

```yaml
runs:
  "020026":
    label: "ld2_c_inbending"
    run_class: "ld2_solid"
    polarity: "inbending"
    target_config: "ld2_solid"
    solid_target: "C"
    beam_energy_gev: 10.6
    input_dir: "/cache/clas12/rg-e/production/spring2024/pass1/torus-1/C_D2/dst/recon/020026"
    output_root: "outputs/ntuples/020026.root"
    enabled: true
```

The extraction script can read either:

- `input_dir`: directory containing HIPO files,
- `input_files`: explicit list of HIPO files, or
- `input_list`: path to an existing text file containing HIPO file paths.

## Track and detector-region conventions

The C++ extractor keeps charged `REC::Particle` rows with:

```text
pid != 0
charge != 0
```

It stores one row per selected particle in a ROOT tree named `tracks`. `REC::FTrack` and `REC::Track` information is matched through `pindex` when available.

Detector regions are encoded as:

```text
0 = other    anything else
1 = forward  2000 < abs(status) < 4000
2 = central  abs(status) > 4000
```

Only the forward detector is treated as having a meaningful six-sector split. Central and other tracks are stored with `sector = 0` and are analyzed as sectorless categories.

## ROOT ntuple contents

The extractor writes two trees:

### `tracks`

Main per-track analysis tree. Important branches include:

```text
run, file_index, event_index, global_event_id, particle_index
pid, charge, status, sector, rec_track_detector, detector_region
chi2pid
px, py, pz, p, theta, phi
vx_particle, vy_particle, vz_particle
has_ftrack
vx_ftrack, vy_ftrack, vz_ftrack
ftrack_sector, ftrack_chi2, ftrack_ndf, ftrack_chi2_ndf
```

### `run_info`

Extractor counters:

```text
run
n_files_requested
n_files_opened
n_events_seen
n_tracks_seen
n_tracks_written
n_tracks_with_ftrack
```

## Basic workflow

The numbered scripts are intended to be run from the repository root.

### 1. Extract HIPO files to ROOT ntuples

Single run:

```bash
python scripts/01_extract_ntuples.py \
  --runs-config configs/runs.yaml \
  --run 020507
```

Small test using only the first HIPO file and a limited number of events:

```bash
python scripts/01_extract_ntuples.py \
  --runs-config configs/short_runs.yaml \
  --run 020507 \
  --max-files 1 \
  --max-events 10000
```

Dry run, printing the extractor command without running it:

```bash
python scripts/01_extract_ntuples.py \
  --runs-config configs/short_runs.yaml \
  --run 020507 \
  --dry-run
```

### 2. Make vertex-source comparison histograms

```bash
python scripts/02_make_vertex_histograms.py \
  --runs-config configs/runs.yaml \
  --run 020507
```

This compares `particle`, `ftrack`, and `hybrid` vertex definitions for positive/negative tracks, forward/central detector regions, and forward sectors.

Useful options:

```bash
--normalize              # normalize overlay histograms
--no-sector-plots        # only make all-sector forward plots, not sectors 1-6
--chi2pid-abs-max VALUE  # optional |chi2pid| selection
```

Main outputs:

```text
outputs/plots/<run>/vertex_histograms/...
outputs/tables/vertex_histogram_summary.csv
```

### 3. Fit the empty/wire reference run

The empty/wire run is used to determine the reference-foil position. The default reference run in the config is `020507`.

```bash
python scripts/03_fit_empty_wire.py \
  --runs-config configs/runs.yaml \
  --fit-config configs/empty_wire_fit.yaml \
  --run 020507
```

This performs local Poisson fits to configured empty/wire components:

- LD2 entrance
- LD2 exit
- reference foil
- wire

Main outputs:

```text
outputs/fits/empty_wire_local_peak_fit_results.csv
outputs/tables/reference_foil_positions.csv
outputs/tables/unresolved_empty_wire_categories.csv
outputs/plots/<run>/empty_wire_local_fits/...
```

### 4. Make production QA histograms

Production QA is designed to be parallel-safe: by default, each run gets its own summary CSV.

Single run:

```bash
python scripts/04_make_production_qa_histograms.py \
  --runs-config configs/runs.yaml \
  --qa-config configs/production_qa.yaml \
  --run 020026
```

Filter by target or polarity:

```bash
python scripts/04_make_production_qa_histograms.py \
  --solid-targets C,Al \
  --polarities inbending
```

Merge per-run QA summaries:

```bash
python scripts/04b_merge_production_qa_csvs.py \
  --qa-config configs/production_qa.yaml
```

Main outputs:

```text
outputs/plots/<run>/production_qa/...
outputs/tables/production_qa_per_run/<run>_production_qa_histogram_summary.csv
outputs/tables/production_qa_histogram_summary.csv
```

### 5. Fit LD2 + solid target peaks

Single run:

```bash
python scripts/05_fit_ld2_solid.py \
  --runs-config configs/runs.yaml \
  --fit-config configs/ld2_solid_local_fit.yaml \
  --run 020026
```

The default model is:

- LD2: `box_gaussian`
- solid target: `gaussian`
- background: local quadratic background, if enabled in config

The script fits positive and negative tracks, forward and central tracks, all configured vertex sources, and forward sectors. Central FTrack-only categories are skipped by default because the central detector is treated as sectorless and FTrack-only is not useful for that category in this workflow.

Merge per-run LD2/solid fit outputs:

```bash
python scripts/05b_merge_ld2_solid_fit_csvs.py \
  --fit-config configs/ld2_solid_local_fit.yaml
```

Main outputs:

```text
outputs/fits/ld2_solid_per_run/<run>_ld2_solid_local_fit_results.csv
outputs/tables/ld2_solid_per_run/<run>_unresolved_ld2_solid_categories.csv
outputs/tables/ld2_solid_per_run/<run>_ld2_solid_category_summary.csv
outputs/fits/ld2_solid_local_fit_results.csv
outputs/tables/unresolved_ld2_solid_categories.csv
outputs/tables/ld2_solid_category_summary.csv
outputs/plots/<run>/ld2_solid_local_fits/...
```

### 6. Make run-dependence plots

After merging LD2/solid fits:

```bash
python scripts/06_make_run_dependence_plots.py \
  --config configs/run_dependence.yaml
```

This plots configured quantities such as:

- `ld2_mean`
- `solid_mean`
- `ld2_sigma`
- `solid_sigma`
- `mean_gap_solid_minus_ld2`
- entry fractions from the production QA summary, if available

Main outputs:

```text
outputs/plots/run_dependence/...
outputs/tables/run_dependence_plot_groups.csv
```

### 7. Compare charge-only and PID-split fits

```bash
python scripts/07_compare_charge_vs_pid.py \
  --runs-config configs/runs.yaml \
  --config configs/pid_study.yaml \
  --run 020026
```

The default PID-study categories compare:

- negative charge-only vs electron and pi-minus selections
- positive charge-only vs proton and pi-plus selections

Merge PID-study outputs:

```bash
python scripts/07b_merge_charge_vs_pid_csvs.py \
  --config configs/pid_study.yaml
```

Main outputs:

```text
outputs/fits/pid_study_per_run/...
outputs/tables/pid_study_per_run/...
outputs/fits/pid_study_fit_results.csv
outputs/tables/pid_study_category_summary.csv
outputs/tables/charge_vs_pid_comparison_summary.csv
```

### 8. Extract recommended LD2/solid vertex cuts

This step uses:

- the empty/wire reference-foil table from step 3, and
- the merged LD2/solid category summary from step 5.

```bash
python scripts/08_extract_ld2_solid_cuts.py \
  --config configs/cut_extraction.yaml
```

The default policy uses the `hybrid` vertex source and clips LD2/solid cut candidates against the reference-foil forbidden region.

Main outputs:

```text
outputs/tables/ld2_solid_vertex_cuts_all.csv
outputs/tables/ld2_solid_vertex_cuts_recommended.csv
```

### 9. Make cut-validation plots

```bash
python scripts/09_make_cut_validation_plots.py \
  --config configs/cut_validation.yaml
```

This makes validation plots for the recommended cuts and can bundle them into a PDF.

Useful options:

```bash
--max-plots N
--solid-targets C,Al
--polarities inbending
--no-pdf
```

Main outputs:

```text
outputs/plots/cut_validation/...
outputs/plots/cut_validation/cut_validation_bundle.pdf
outputs/tables/cut_validation_plot_index.csv
```

## Parallel batch workflow

For large LD2/solid production processing, use the batch launcher from the repository root:

```bash
./overnight_ld2_solid_batch.sh \
  --runs-config configs/runs.yaml \
  --nproc 30
```

Include HIPO-to-ROOT extraction as phase 1:

```bash
./overnight_ld2_solid_batch.sh \
  --runs-config configs/runs.yaml \
  --nproc 30 \
  --do-extract
```

Restrict to selected targets or polarities:

```bash
./overnight_ld2_solid_batch.sh \
  --solid-targets C,Al \
  --polarities inbending \
  --nproc 20
```

Merge outputs after the batch run:

```bash
./merge_after_overnight.sh
```

The batch launcher writes logs under:

```text
outputs/logs/overnight_ld2_solid/
```

## Fit strategy

The local peak fitter uses a two-stage approach:

1. Find a peak candidate inside a configured search window using a smoothed histogram and simple height/prominence thresholds.
2. Fit the selected local window with a Poisson negative-log-likelihood model using `iminuit`.

Supported signal models include:

- `gaussian`
- `box_gaussian`

The local background is modeled as a quadratic polynomial in a scaled coordinate over the fit window.

Fit status values include categories such as:

```text
good
bad_fit
low_statistics
unresolved_peak
skipped_central_ftrack
```

Unresolved or skipped categories are written to explicit CSV tables so they can be audited rather than silently dropped.

## Common output tables

| File | Purpose |
| --- | --- |
| `outputs/tables/vertex_histogram_summary.csv` | Entry counts and histogram file paths for vertex-source comparison plots. |
| `outputs/tables/reference_foil_positions.csv` | Reference-foil fit results from the empty/wire run. |
| `outputs/tables/production_qa_histogram_summary.csv` | Merged production QA metrics and histogram paths. |
| `outputs/fits/ld2_solid_local_fit_results.csv` | Component-level LD2/solid fit results. |
| `outputs/tables/ld2_solid_category_summary.csv` | Category-level LD2/solid fit summary, including peak means, widths, and fit quality. |
| `outputs/tables/unresolved_ld2_solid_categories.csv` | Categories that were skipped, low-statistics, unresolved, or otherwise not good. |
| `outputs/tables/run_dependence_plot_groups.csv` | Index of generated run-dependence plots. |
| `outputs/tables/charge_vs_pid_comparison_summary.csv` | Charge-only vs PID-split comparison summary. |
| `outputs/tables/ld2_solid_vertex_cuts_all.csv` | All per-run candidate cuts. |
| `outputs/tables/ld2_solid_vertex_cuts_recommended.csv` | Final aggregated recommended cuts. |
| `outputs/tables/cut_validation_plot_index.csv` | Index of cut-validation plots. |

## Troubleshooting

### `No module named rge_vertex`

Install the local package from the repository root:

```bash
pip install -e .
```

### `Extractor not found: cpp/build/rge_hipo_to_root`

Build the C++ extractor:

```bash
cd cpp
./scripts/build.sh
cd ..
```

### `Environment variable HIPO is not set`

Load or configure the HIPO/CLAS12 software environment before building the extractor. The CMake build expects:

```bash
export HIPO=/path/to/hipo
```

### `root-config: command not found`

Load a ROOT environment before using the Makefile build, or make sure CMake can find ROOT.

### `No HIPO files found for run ...`

Check the run entry in `configs/runs.yaml` or `configs/short_runs.yaml`. In particular, verify `input_dir`, `input_files`, or `input_list`. If files are nested below the configured directory, use:

```bash
python scripts/01_extract_ntuples.py --run RUN --recursive
```

### Empty/wire fit config path

The empty/wire fit configuration in this checkout is:

```text
configs/empty_wire_fit.yaml
```

Use it explicitly when running step 3:

```bash
python scripts/03_fit_empty_wire.py --fit-config configs/empty_wire_fit.yaml
```

## Notes and assumptions

- The default run paths in `configs/runs.yaml` are JLab farm/pass-1 paths and may need to be edited for other systems.
- The code assumes the input HIPO files contain the CLAS12 banks used by the extractor: `REC::Particle`, `REC::Track`, and `REC::FTrack`.
- Duplicate `pindex` matches in `REC::FTrack` or `REC::Track` are resolved by keeping the first match encountered.
- Missing input HIPO files are skipped by the extractor with a warning.
- Central tracks are not sector-split in this analysis.
- The recommended final cut workflow currently favors the `hybrid` vertex source, as configured in `configs/cut_extraction.yaml`.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
