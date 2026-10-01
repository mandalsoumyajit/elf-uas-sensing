"""VLF node: damping options for low-turn windings with the unchanged version-1 receiver front end (LTspice).

For N = 8, 12, 16, 24 turns (coil model calibrated on the published 48/56/100-turn coils) and the VLF output
filters (10 kHz HP, 120 kHz LP), compare
  undamped            : Mode 0 (CF disconnected)
  active (retuned)    : Mode 1, CF scaled with the coil capacitance (8.6% as in the ELF design), RLEAK chosen so
                        the leak pole sits a decade below the band (1.6 kHz), RINT swept for the flattest response
  passive             : critical shunt resistor R_sh = 1/G_crit across the coil input, CF disconnected
Metrics from |V(filtered)/EMF|: passband gain at 30 kHz, peaking above the 30 kHz gain, -3 dB edges; peaking of
V(gate) and V(out) above their 30 kHz values (the output filters hide the self-resonance); input-referred
EMF noise and field noise n_B = e_n/(2 pi f N A) at 24, 48 and 85 kHz.
"""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import spice as S

COILS = {c['turns']: c for c in json.load(open(S.HERE / 'coil_model.json'))['vlf']}
FN = (24e3, 48e3, 85e3)

def metrics(a, z, coil):
    f = a['frequency'].real; g = np.abs(a['v(filtered)'])
    g30 = np.interp(3e4, f, g); band = (f > 3e3) & (f < 3e6)
    peak_db = 20 * np.log10(g[band].max() / g30)
    above = f[(g >= g30 / np.sqrt(2)) & band]
    fz = z['frequency'].real; en = z['v(inoise)'].real
    out = dict(gain30=float(g30), peak_db=float(peak_db), f_lo=float(above.min()), f_hi=float(above.max()))
    # the 120 kHz low-pass hides the self-resonance at V(filtered); peaking is judged at the gate and buffer output
    for node, key in (('v(gate)', 'gate'), ('v(out)', 'out')):
        gn = np.abs(a[node]); i = int(np.argmax(gn))
        out[f'{key}_peak_db'] = float(20 * np.log10(gn[i] / np.interp(3e4, f, gn)))
        if key == 'out':
            out['out_peak_f'] = float(f[i])
    for fx in FN:
        e = float(np.interp(fx, fz, en))
        out[f'en_{int(fx/1e3)}k_nV'] = e * 1e9
        out[f'nB_{int(fx/1e3)}k_fT'] = e / (2 * np.pi * fx * coil['area']) * 1e15
    return out

def case(args):
    n, kind, rint = args
    c = COILS[n]; cf = 0.086 * c['C']; ctot = c['C'] + 12e-12
    gcrit = c['R'] * ctot / c['L'] + 2 * np.sqrt(ctot / c['L'])
    name = f'vlf{n}_{kind}_{int(rint) if rint else 0}'
    if kind == 'passive':
        core = S.core_text(c['R'], c['L'], c['C'], 1e-15, 1e3, 1e4, 'output_filters_vlf.inc', rshunt=1 / gcrit); mode = 0
    elif kind == 'undamped':
        core = S.core_text(c['R'], c['L'], c['C'], 1e-15, 1e3, 1e4, 'output_filters_vlf.inc'); mode = 0
    else:
        core = S.core_text(c['R'], c['L'], c['C'], cf, rint, 1 / (2 * np.pi * 1.6e3 * 10e-9), 'output_filters_vlf.inc'); mode = 1
    a = S.run(name, core, 'ac', mode=mode, fstart=1e3, fstop=3e6)
    z = S.run(name, core, 'noise', mode=mode, fstart=1e3, fstop=3e6)
    return (n, kind, rint), metrics(a, z, c), dict(cf=cf, gcrit=gcrit, rsh=1 / gcrit)

if __name__ == '__main__':
    jobs = []
    for n in (8, 12, 16, 24):
        c = COILS[n]; cf = 0.086 * c['C']; ctot = c['C'] + cf + 12e-12
        g = c['R'] * ctot / c['L'] + 2 * np.sqrt(ctot / c['L'])
        r0 = 75 * cf / (g * 10e-9)                              # nominal R_int for critical damping (A ~ 75)
        jobs += [(n, 'undamped', None), (n, 'passive', None)] + [(n, 'active', r0 * k) for k in (0.25, 0.5, 1, 1.5, 2.5, 4)]
    with ThreadPoolExecutor(max_workers=1) as ex:
        R = list(ex.map(case, jobs))
    res = []
    for (n, kind, rint), m, p in R:
        res.append(dict(turns=n, kind=kind, rint=rint, **m, **p))
        print(f'N={n:2d} {kind:9s} RINT={rint or 0:8.0f}  gain30 {m["gain30"]:6.1f}  peak {m["peak_db"]:+6.1f} dB  '
              f'-3dB {m["f_lo"]/1e3:5.1f}-{m["f_hi"]/1e3:6.1f} kHz  en(24/48/85k) {m["en_24k_nV"]:.2f}/{m["en_48k_nV"]:.2f}/{m["en_85k_nV"]:.2f} nV  '
              f'nB {m["nB_24k_fT"]:.2f}/{m["nB_48k_fT"]:.2f}/{m["nB_85k_fT"]:.2f} fT', flush=True)
    json.dump(res, open(S.HERE / 'vlf_damping_study.json', 'w'), indent=1)
