"""Site model with background cancellation: witness sensors at local sources + remote reference for the distant part.

Background split (site_model.noise): natural (distant, coherent) + man-made excess of the semi-urban model, of
which a fraction f_loc is local (per node, incoherent) and the rest distant; optional extra local VLF emitter
floor x_loc.  Witnesses cancel a fraction cov of the local power to depth D_w; a remote reference (self-noise
n_ref, coherence gamma2) cancels the distant part.  The narrowband clutter penalty (5.1 dB without witnesses)
falls with witness coverage per the measured tail mixture (impulsive_penalty.json 'clutter_vs_coverage').
Layouts: perimeter poles+roofs and street canyon mixed, B = 8 and 16 triaxial nodes per 100 m.
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import site_model as SM

J = json.load(open(os.path.join(HERE, 'impulsive_penalty.json')))
CV = {float(k): v for k, v in J['clutter_vs_coverage'].items()}
def clutter(cov):
    ks = sorted(CV); return float(np.interp(cov, ks, [CV[k] for k in ks]))

LOW, GOOD = 100e-15, 3e-15           # remote-reference self-noise: cheap/low-sensitivity vs node-grade
CONFIGS = [  # label, canc dict, witness coverage (for the clutter penalty)
    ('none', None, 0.0),
    ('witnesses, 80% of local power', dict(f_loc=0.9, cov=0.8, D_w=30), 0.8),
    ('witnesses, 97% of local power', dict(f_loc=0.9, cov=0.97, D_w=30), 0.97),
    ('remote ref, low-sens (100 fT)', dict(f_loc=0.9, n_ref=LOW, gamma2=0.99), 0.0),
    ('remote ref, node-grade (3 fT)', dict(f_loc=0.9, n_ref=GOOD, gamma2=0.99), 0.0),
    ('wit 97% + low-sens ref', dict(f_loc=0.9, cov=0.97, D_w=30, n_ref=LOW, gamma2=0.99), 0.97),
    ('wit 97% + node-grade ref', dict(f_loc=0.9, cov=0.97, D_w=30, n_ref=GOOD, gamma2=0.99), 0.97),
    ('  same, f_loc 0.5', dict(f_loc=0.5, cov=0.97, D_w=30, n_ref=GOOD, gamma2=0.99), 0.97),
    ('ideal (to receiver noise, no clutter)', dict(f_loc=1.0, cov=1.0, D_w=80, n_ref=1e-18, gamma2=1.0), 1.0),
    ('VLF emitters 100 fT, none', dict(f_loc=0.9, x_loc=100e-15), 0.0),
    ('VLF emitters 100 fT, wit 97%', dict(f_loc=0.9, x_loc=100e-15, cov=0.97, D_w=30), 0.97),
]
PZ = ((2, 10), (10, 20), (20, 30))
CZ = ((3, 9), (26, 40))
CLS = (('5-inch FPV', 'P3'), ('5-inch FPV', 'PWM'), ('DJI-like', 'P3'), ('DJI-like', 'PWM'), ('heavy-lift', 'P3'), ('heavy-lift', 'PWM'))

def job(a):
    kind, lab, rest = a
    return kind, lab, (SM.run_perimeter if kind == 'perimeter' else SM.run_canyon)(rest)

if __name__ == '__main__':
    jobs = []
    for case, proc in CLS:
        for lab, canc, cov in CONFIGS:
            L = clutter(cov)
            for B in (8, 16):
                jobs.append(('perimeter', lab, (case, proc, 'poles+roofs', B, 100.0, True, L, 120, 700 + B, PZ, canc)))
                jobs.append(('canyon', lab, (case, proc, 'mixed', B, 100.0, True, L, 100, 900 + B, CZ, canc)))
    with ProcessPoolExecutor(max_workers=20) as ex:
        R = list(ex.map(job, jobs))
    out = []; P = lambda s: (print(s, flush=True), out.append(s))
    P('Node noise (fT/rtHz) per configuration at each class line frequency (receiver noise included):')
    fl = sorted({SM.FUND_F[c] if p == 'P3' else SM.line(c, p)[1] for c, p in CLS})
    P(f'{"config":40s}' + ''.join(f'{f/1e3:>9.2f}k' for f in fl))
    for lab, canc, cov in CONFIGS:
        P(f'{lab:40s}' + ''.join(f'{1e15*SM.noise(f, canc=canc):10.1f}' for f in fl) + f'   clutter penalty {clutter(cov):.1f} dB')
    for case, proc in CLS:
        P(f'\n=== {case}, {"ELF fundamental joint-tracked" if proc == "P3" else "PWM carrier coherent"} ===')
        P(f'{"config":40s}{"B":>3s} | perimeter poles+roofs: Pd>=0.9 at 2-10/10-20/20-30 m | canyon mixed: P(det 50 m) in-canyon / above-roof')
        for lab, _, _ in CONFIGS:
            for B in (8, 16):
                rp = next(o for k, l, (a, o) in R if k == 'perimeter' and l == lab and a[0] == case and a[1] == proc and a[3] == B)
                rc = next(o for k, l, (a, o) in R if k == 'canyon' and l == lab and a[0] == case and a[1] == proc and a[3] == B)
                P(f'{lab:40s}{B:3d} |        ' + ' / '.join(f'{rp[z][1]:4.2f}' for z in PZ) + '                     |      '
                  + ' / '.join(f'{rc[z][1]:4.2f}' for z in CZ))
    open(os.path.join(HERE, 'site_witness_out.txt'), 'w').write('\n'.join(out) + '\n')
    json.dump([dict(kind=k, config=l, case=a[0], proc=a[1], B=a[3], res={f'{z[0]}-{z[1]}': v for z, v in o.items()}) for k, l, (a, o) in R],
              open(os.path.join(HERE, 'site_witness.json'), 'w'), indent=1)
