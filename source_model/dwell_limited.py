"""Dwell-limited detection: a drone crossing a sensor curtain at speed v is within range for T ~ 2 r1 / v.
Solve r1 = R(T = 2 r1 / v) self-consistently (tracked coherent integration), then the curtain pitch."""
import json, os, sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'range_budget'))
from range_budget import max_range, wire_loop_noise
import range_by_class as RB          # scenarios and source lines (re-running it is cheap)
from array_coverage_3d import plane_ok, smax

R = json.load(open(os.path.join(HERE, 'drone_source_results.json')))
CASES = [('indoor facility, 5-inch', '5-inch FPV (6S)', 'indoor lab', (100, 10e3)),
         ('indoor facility, 5-inch, wideband', '5-inch FPV (6S)', 'indoor lab', (20, 1e6)),
         ('urban perimeter, photo 11"', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', (100, 10e3)),
         ('urban perimeter, photo 11", wideband', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', (20, 1e6)),
         ('urban perimeter, heavy-lift', 'heavy-lift (30" props, 12S)', 'semi-urban outdoor', (100, 10e3)),
         ('remote perimeter, photo 11"', 'photo/mid (11" props, 4S)', 'quiet outdoor', (100, 10e3))]
print('Self-consistent dwell-limited range (tracked coherent, T = 2 r1 / v, T >= 0.2 s) and curtain pitch')
print(f'{"case":38s} ' + ' '.join(f'{"v="+str(v)+" m/s: r1, T, pitch":>28s}' for v in (2, 5, 15)))
for lab, cls, bg, (flo, fhi) in CASES:
    nf = RB.SCEN[bg]; cells = []
    for v in (2, 5, 15):
        best = (0, 0)
        for L in R[cls]['lines']:
            if not (flo <= L['f'] <= fhi):
                continue
            n = np.hypot(nf(L['f']), wire_loop_noise(L['f'])[0])
            g = lambda r: max_range(L['m'][1], n, max(0.2, 2 * r / v), max(0.2, 2 * r / v)) - r
            r = brentq(g, 0.05, 500) if g(0.05) > 0 else 0.0
            if r > best[0]:
                best = (r, max(0.2, 2 * r / v))
        cells.append(f'{best[0]:6.1f} m {best[1]:5.1f} s {smax(plane_ok, best[0]):6.1f} m   ')
    print(f'{lab:38s} ' + ' '.join(cells))
