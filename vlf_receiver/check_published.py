"""Check the LTspice driver against the published version-1 ELF receiver (48, 56 and 100 turns).

Runs the unchanged netlist values (CF, RINT, RLEAK as in core.inc, version-1 output filters) for each published
coil and compares the gain and input-referred EMF noise at 1 kHz with published_coils.json.
"""
import json
import numpy as np
import spice as S

PUB = {m['turns']: m for m in json.load(open(S.HERE / 'published_coils.json'))['models']}
NET = {48: (120e-12, 1065.38540163, 2.4e3), 56: (150e-12, 1351.50817849, 3e3), 100: (270e-12, 2584.31315529, 6.8e3)}

if __name__ == '__main__':
    for n, (cf, rint, rleak) in NET.items():
        p = PUB[n]
        core = S.core_text(p['R'], p['L'], p['C'], cf, rint, rleak, 'output_filters.inc')
        a = S.run(f'check{n}', core, 'ac', fstart=10, fstop=1e5); z = S.run(f'check{n}', core, 'noise', fstart=10, fstop=1e5)
        g = float(np.interp(1e3, a['frequency'].real, np.abs(a['v(filtered)'])))
        en = float(np.interp(1e3, z['frequency'].real, z['v(inoise)'].real)) * 1e9
        print(f'N={n:3d}: gain at 1 kHz {g:.3f} V/V (published {p["gain_1k"]:.3f}); '
              f'EMF noise {en:.4f} nV/rtHz (published {p["noise_emf_nV"]:.4f})', flush=True)
