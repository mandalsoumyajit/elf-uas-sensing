"""Array coverage from single-node detection ranges (optimal SNR combining across nodes).

With an r^-3 field, single-node power SNR ~ (r1/r)^6 where r1 is the single-node detection range.
Optimal (MRC / incoherent-sum GLRT, equal noise) combining: detection when sum_i (r1/r_i)^6 >= 1.
  - square grid, spacing d, drone at height h above the sensor plane: worst point is the cell centre.
  - perimeter line (fence), spacing d, drone crossing the line at height h: worst point is the gap midpoint.
Returns the largest d giving full coverage, and nodes per hectare / per 100 m of perimeter.
"""
import json, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
O = json.load(open(os.path.join(HERE, 'range_by_class.json')))

def grid_ok(d, r1, h):
    k = np.arange(-6, 7)
    X, Y = np.meshgrid((k + 0.5) * d, (k + 0.5) * d)
    r2 = X ** 2 + Y ** 2 + h ** 2
    return np.sum((r1 ** 2 / r2) ** 3) - 1.0

def line_ok(d, r1, h):
    k = np.arange(-10, 11)
    r2 = ((k + 0.5) * d) ** 2 + h ** 2
    return np.sum((r1 ** 2 / r2) ** 3) - 1.0

def max_spacing(f, r1, h):
    if f(1e-3, r1, h) < 0:
        return 0.0
    return brentq(f, 1e-3, 10 * r1 + 10, args=(r1, h))

CASES = [  # (label, class, background, band)  -- realistic use cases only
    ('indoor facility, 5-inch', '5-inch FPV (6S)', 'indoor lab', 'current (<=10 kHz)'),
    ('indoor facility, 5-inch, wideband', '5-inch FPV (6S)', 'indoor lab', 'wideband'),
    ('urban perimeter, photo 11"', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', 'current (<=10 kHz)'),
    ('urban perimeter, photo 11", wideband', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', 'wideband'),
    ('urban perimeter, heavy-lift', 'heavy-lift (30" props, 12S)', 'semi-urban outdoor', 'current (<=10 kHz)'),
    ('remote perimeter, photo 11"', 'photo/mid (11" props, 4S)', 'quiet outdoor', 'current (<=10 kHz)'),
    ('remote perimeter, heavy-lift', 'heavy-lift (30" props, 12S)', 'quiet outdoor', 'current (<=10 kHz)'),
]
print('Max node spacing (m) for full coverage, nominal source; 60 s tracked [0.2 s snapshot]. h = drone height above sensors.')
print(f'{"case":40s} {"r1(60s)":>8s} {"r1(0.2s)":>8s} | ' + ' | '.join(f'grid h={h:>2d} m    line h={h:>2d} m' for h in (2, 10, 25)))
rows = []
for lab, cls, bg, band in CASES:
    r60 = O[f'{cls} | {bg} | {band} | 60 s tracked']['range'][1]
    r02 = O[f'{cls} | {bg} | {band} | 0.2 s']['range'][1]
    cells = []
    for h in (2, 10, 25):
        g60, g02 = max_spacing(grid_ok, r60, h), max_spacing(grid_ok, r02, h)
        l60, l02 = max_spacing(line_ok, r60, h), max_spacing(line_ok, r02, h)
        cells.append(f'{g60:5.1f} [{g02:4.1f}]   {l60:5.1f} [{l02:4.1f}]')
        rows.append(dict(case=lab, h=h, r1_60=r60, r1_02=r02, grid60=g60, grid02=g02, line60=l60, line02=l02))
    print(f'{lab:40s} {r60:8.1f} {r02:8.1f} | ' + ' | '.join(cells))
json.dump(rows, open(os.path.join(HERE, 'array_coverage.json'), 'w'), indent=1)

print('\nArray gain in reach (equidistant N nodes, r^-6 power law): N^(1/6) ->',
      ', '.join(f'N={n}: x{n ** (1 / 6):.2f}' for n in (2, 4, 9, 16, 100)))
print('\nNodes needed (60 s tracked, h = 2 m): per hectare (grid) and per 100 m of perimeter (line):')
for r in rows:
    if r['h'] == 2 and r['grid60'] > 0:
        print(f'  {r["case"]:40s} grid {1e4 / r["grid60"] ** 2:7.0f} /ha   line {100 / r["line60"]:5.0f} /100 m')
