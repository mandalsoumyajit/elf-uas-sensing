# elf-uas-sensing

Code accompanying the paper **"Spatial ELF Sensing of Small UAS: Measured Signatures, Array Localization, and Vector Receiver Design"** (R. Hapuarachchi *et al.*, submitted to *IEEE Access*).

The paper studies passive reception of the quasi-static magnetic fields that small drones radiate through their motors, phase leads and speed controllers. It also asks what an array of loop or vector receivers can learn from these fields. This repository contains:
- the source, range, array and vector-node models;
- the processing applied to the four-loop grid recordings;
- the scripts that produce the paper's figures.

## Layout

| Folder | Contents | Paper section |
|---|---|---|
| `source_model/` | Equivalent-source model of four drone classes (`drone_source.py`); ground effect over a conductive half-space with `empymod` (`ground_effect.py`, `ground_tables.py`, `burch_check.py`, `check_green.py`); range by class and environment (`range_by_class.py`); curtain and site array models (`array_model*.py`, `site_model.py`); background cancellation (`site_witness.py`); ground-model bias (`ground_bias.py`); multiple drones (`multi_drone.py`); grid CRLB (`grid_crlb.py`) | II, V, VII |
| `range_budget/` | Link-budget detection-range model (`range_budget.py`), cancellation and near-field-phase calculations | V |
| `grid_analysis/` | Processing of the four-loop grid recordings: MATLAB export (`export_raw.m`), spectra and maps, background statistics (`env_*.py`, `classA_fit.py`, `detstat_*.py`), synchronous comb (`sync_comb.py`, `periodic_cancel.py`), phase-referenced coherent integration (`ref_tracking.py`, `crosstalk_test.py`, `joint_tracking.py`), held-out-cell localization (`raw_localize.py`, `complex_amps.py`, `fit_position*.py`) | III, IV |
| `vector_sim/` | Vector-node simulations: point-rotor forward model, features, estimators, the original studies 1–9, and the measurement-informed studies 10–12 (`realistic_model.py`, `study10_*`, `study11_*`, `study12_*`) | VI |
| `vlf_receiver/` | LTspice simulations of the ELF and VLF node receivers: coil model, VLF damping and noise, cost of co-locating the windings. Needs LTspice and three vendor device models that are not included (see `vlf_receiver/README.md`) | VIII |
| `figures/` | Block diagrams and data figures (`fig_diagrams.py`, `fig_experiment.py`, `fig_models.py`) | all |

## Requirements

Python 3.12 with the packages in `requirements.txt`. MATLAB is needed only for `grid_analysis/export_raw.m`, which converts the original `timeseries` recordings to plain arrays. LTspice is needed only for `vlf_receiver/`.

```
pip install -r requirements.txt
```

## Data

The grid recordings are available from the authors on reasonable request and are not included here.

1. Export the recordings with `grid_analysis/export_raw.m`. This produces one `cXX_Y.mat` file per cell (`XX`) and antenna (`Y`).
2. Pass the export folder as the first argument to the `grid_analysis` scripts.
3. For the figure scripts, set the export folder in the environment variable `ELF_RAW_EXPORT`.

`source_model/impulsive_penalty.json` holds the detector penalties derived from the recordings (`grid_analysis/detstat_report.py`, `clutter_vs_coverage.py`), so the array models run without the raw data.

## Typical use

```
cd source_model && python drone_source.py && python range_by_class.py && python ground_tables.py && python site_model.py
cd vector_sim && python study10_inclination_realistic.py && python study11_position_realistic.py
cd grid_analysis && python sync_comb.py <export_folder>
cd figures && python fig_models.py
```

Most array and simulation scripts use `concurrent.futures` with many workers. For reproducible timing, set `OMP_NUM_THREADS=1`.

## License

MIT; see `LICENSE`.

## Citation

If you use this code, please cite the paper (full reference to be added on publication).
