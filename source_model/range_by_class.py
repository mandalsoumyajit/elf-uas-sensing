"""Detection range per drone class, background scenario, receiver band and processing mode.

Source lines : drone_source_results.json (per motor; lo/nominal/hi moment).
Background   : frequency-dependent ambient ASD scenarios (literature + our indoor data); every scenario is
               floored at the natural background; receiver noise (wire loop) added in quadrature.
Detection    : range_budget.max_range (Pd 0.9, Pfa 1e-3 per record over a 500 Hz search band).
Bands        : 'current' = lines in 100 Hz - 10 kHz (existing wire-loop chain: 20 kS/s, ~10 kHz low-pass;
               also the PCB receiver design band); 'wideband' = all lines incl. 16-48 kHz PWM/ESC lines.
Ground       : neglected (<~5% field change for ordinary soil at these ranges/frequencies; ground_effect.py).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'range_budget'))
from range_budget import max_range, wire_loop_noise

def natural(f):          # AWESOME/Chrissan & Fraser-Smith: 1-100 fT/rtHz natural; quiet sites ~10-40 fT/rtHz;
    return 20e-15 * np.maximum(1.0, (300.0 / f) ** 1.5)    # Burch field ~8 fT/rtHz at 500 Hz; rise below 300 Hz
def semi_urban(f):       # Zhou et al. 2024 mine-surface sites 0.38-3.14 pT/rtHz at 1 kHz; take 1 pT, ~1/f
    return np.maximum(natural(f), 1e-12 * (f / 1e3) ** -1.0)
def indoor_lab(f):       # this work (grid data, G=1000 assumed): ~2 pT @600 Hz, 0.2-0.9 pT @3 kHz, 20-60 fT @8 kHz;
    return np.maximum(np.maximum(natural(f), 30e-15), 2e-12 * (f / 600.0) ** -1.3)   # Virgo quiet indoor 10-100 pT @20-300 Hz
def indoor_noisy(f):     # Virgo: near racks/power supplies up to ~1000x quiet; take 30x
    return 30 * indoor_lab(f)
SCEN = {'quiet outdoor': natural, 'semi-urban outdoor': semi_urban, 'indoor lab': indoor_lab, 'indoor near electronics': indoor_noisy}
MODES = {'0.2 s': (0.2, 0.2), '10 s untracked': (10.0, 0.1), '60 s tracked': (60.0, 60.0)}
BANDS = {'current (<=10 kHz)': (100.0, 10e3), 'wideband': (20.0, 1e6)}

R = json.load(open(os.path.join(HERE, 'drone_source_results.json')))
out = {}
for cls, d in R.items():
    for s, nf in SCEN.items():
        for b, (flo, fhi) in BANDS.items():
            for mname, (T, Tc) in MODES.items():
                best = None
                for L in d['lines']:
                    if not (flo <= L['f'] <= fhi):
                        continue
                    n = np.hypot(nf(L['f']), wire_loop_noise(L['f'])[0])
                    rr = [max_range(m, n, T, Tc) for m in L['m']]
                    if best is None or rr[1] > best['range'][1]:
                        best = dict(line=L['source'], f=L['f'], n=float(n), range=rr)
                out[f'{cls} | {s} | {b} | {mname}'] = best
json.dump(out, open(os.path.join(HERE, 'range_by_class.json'), 'w'), indent=1, default=float)

for b in BANDS:
    for mname in MODES:
        print(f'\nRange (m), nominal [low-high], band: {b}, processing: {mname}')
        print(f'{"class":30s}' + ''.join(f'{s:>27s}' for s in SCEN))
        for cls in R:
            row = f'{cls:30s}'
            for s in SCEN:
                v = out[f'{cls} | {s} | {b} | {mname}']
                row += f'{v["range"][1]:>9.1f} [{v["range"][0]:4.1f}-{v["range"][2]:5.1f}]   '
            print(row)
        print('  best line: ' + '; '.join(f'{cls.split(" (")[0]}: ' + ', '.join(
            sorted({f'{out[f"{cls} | {s} | {b} | {mname}"]["line"]} @{out[f"{cls} | {s} | {b} | {mname}"]["f"]:.0f} Hz' for s in SCEN}))
            for cls in R))
