# vlf_receiver: ELF and VLF node receivers (paper Section VIII)

LTspice simulations of the printed-circuit-board vector receiver for the paper's two node types:
- **ELF node:** the published version-1 design, a 100-turn winding per axis, for the motor band (below 10 kHz).
- **VLF node:** the same front end with a low-turn winding, retuned active damping, rescaled output filters and a faster ADC, for the ESC PWM carriers (about 10–90 kHz).

The two node types are deployed separately. `coupling.py` and `coupled_study.py` quantify what co-locating the two windings on one node would cost, which is the argument for keeping them apart.

## Vendor device models (not included)

The netlists need three device models that are not redistributed here. Before running anything, place them in this folder under these exact names:

| File | Contents | Source |
|---|---|---|
| `ADA4511.lib` | `.subckt ADA4511` macro model | Analog Devices, ADA4511 product page (SPICE model) |
| `LSBF862_from_pdf.lib` | LSBF862 N-JFET `.model`, renamed to `LSBF862_UPDATED` | Linear Systems, LSBF862 SPICE model |
| `clamp_diode.lib` | the `.model 1N4148 D(...)` line | LTspice's `standard.dio` (installed with LTspice) |

`spice.py` checks for these files and stops with a message naming any that are missing. `*.lib` is git-ignored, so the files cannot be committed by accident.

## Requirements

- LTspice (tested with LTspice 26 on Windows). By default the driver looks in `%LOCALAPPDATA%\Programs\ADI\LTspice\LTspice.exe`; set the `LTSPICE` environment variable to use another location.
- Python packages as in the top-level `requirements.txt`.

LTspice batch jobs run one at a time, because concurrent runs in the same folder collide.

## Files

| File | Role |
|---|---|
| `core.inc` | Version-1 receiver front end (JFET preamplifier, buffer, active damping), as published |
| `bias_hints.inc` | Operating-point starting guesses |
| `output_filters.inc` | Version-1 multiple-feedback (MFB) Butterworth output filters: 200 Hz high-pass, 10 kHz low-pass |
| `output_filters_vlf.inc` | Same topology with rescaled capacitors: 10 kHz high-pass, 120 kHz low-pass |
| `published_coils.json` | Published R, L, C, area, gain and noise of the 48-, 56- and 100-turn windings |
| `spice.py` | Driver. Substitutes the coil, damping network, filters and op-amp into `core.inc`, runs LTspice in batch mode and reads the ASCII raw files |
| `check_published.py` | Reproduces the published gain and noise of the 48-, 56- and 100-turn designs |
| `coil_model.py` | Coil model: coaxial square filaments at 1 mm pitch with exact parallel-filament mutual inductances. One fitted trace radius reproduces the published inductances. Writes `coil_model.json` |
| `vlf_damping_study.py` | Undamped, passive-shunt and active damping, with a swept R_int, for 8–24-turn VLF windings |
| `coupling.py` | Coupling coefficient between ELF and VLF windings in three co-located arrangements |
| `coupled_study.py` | VLF receiver response and noise with a coupled ELF winding |
| `fig_node_types.py` | Paper figure `dual_channel`, written to `figures/output/` |

## Running

```
python coil_model.py
python check_published.py
python vlf_damping_study.py
python coupling.py
python coupled_study.py
python fig_node_types.py
```

`fig_node_types.py` reads the background models from `source_model/range_by_class.py` and the plot style from `vector_sim/plotstyle.py`.

## Expected results

- **Validation:** `check_published.py` gives the published gain (about 75 V/V at 1 kHz) and EMF noise (1.42–1.63 nV/√Hz). `coil_model.py` reproduces L(56) and L(100) within 0.1% after fitting the trace radius to L(48).
- **12-turn VLF node with active damping:** −3 dB band 9.8–89 kHz; field noise 1.4 / 0.7 / 0.4 fT/√Hz at 24 / 48 / 85 kHz.
- **Co-location:** windings sharing the same boards couple with k ≈ 0.5–0.6. That suppresses the VLF response by up to 9–12 dB.

## Limits

- The winding capacitance is the original 2-D estimate scaled per turn; it has not been measured.
- Layout-specific board and connector parasitics are not included.
- No transient or overload study has been done for the VLF node.
