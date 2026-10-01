"""3-D node placement: curtain (vertical plane) and volume (cubic lattice) layouts vs a ground grid.

Detection when sum_i (r1/r_i)^6 >= 1 (optimal combining, r^-3 field), r1 = single-node range.
  curtain : square lattice of pitch s in a vertical plane along the perimeter (poles with stacked nodes,
            facades). A drone crossing the boundary passes through the plane; worst point = cell centre (d=0
            from the plane). Nodes per 100 m of perimeter for a curtain of height H: (100/s) * ceil(H/s).
  volume  : cubic lattice of pitch s filling the protected volume (indoor halls, nodes on walls/ceilings/racks);
            worst point = cube body centre.
  ground  : horizontal grid at the floor, drone altitude h above it (previous analysis).
"""
import json, os
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
O = json.load(open(os.path.join(HERE, 'range_by_class.json')))
K = np.arange(-6, 7) + 0.5

def plane_ok(s, r1, d=0.0):      # square lattice in a plane, point at the cell centre, offset d from the plane
    X, Y = np.meshgrid(K * s, K * s)
    return np.sum((r1 ** 2 / (X ** 2 + Y ** 2 + d ** 2)) ** 3) - 1.0

def cube_ok(s, r1):
    X, Y, Z = np.meshgrid(K * s, K * s, K * s)
    return np.sum((r1 ** 2 / (X ** 2 + Y ** 2 + Z ** 2)) ** 3) - 1.0

def smax(f, r1, *a):
    return brentq(f, 1e-3, 10 * r1 + 10, args=(r1, *a)) if f(1e-3, r1, *a) > 0 else 0.0

CASES = [
    ('indoor facility, 5-inch', '5-inch FPV (6S)', 'indoor lab', 'current (<=10 kHz)'),
    ('indoor facility, 5-inch, wideband', '5-inch FPV (6S)', 'indoor lab', 'wideband'),
    ('urban perimeter, photo 11"', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', 'current (<=10 kHz)'),
    ('urban perimeter, photo 11", wideband', 'photo/mid (11" props, 4S)', 'semi-urban outdoor', 'wideband'),
    ('urban perimeter, heavy-lift', 'heavy-lift (30" props, 12S)', 'semi-urban outdoor', 'current (<=10 kHz)'),
    ('remote perimeter, photo 11"', 'photo/mid (11" props, 4S)', 'quiet outdoor', 'current (<=10 kHz)'),
    ('remote perimeter, heavy-lift', 'heavy-lift (30" props, 12S)', 'quiet outdoor', 'current (<=10 kHz)'),
]
rows = []
print('60 s tracked coherent, nominal source. Curtain pitch covers crossings at ANY altitude up to the curtain height H.')
print(f'{"case":38s} {"r1":>5s} | {"curtain s":>9s} {"nodes/100m H=10":>16s} {"H=30":>6s} | {"cube s":>6s} {"nodes per 1000 m^3":>19s} | {"ground grid nodes/ha for h<=2 m":>32s}')
for lab, cls, bg, band in CASES:
    r1 = O[f'{cls} | {bg} | {band} | 60 s tracked']['range'][1]
    sc = smax(plane_ok, r1)
    sv = smax(cube_ok, r1)
    sg = smax(plane_ok, r1, 2.0)
    n10 = (100 / sc) * np.ceil(10 / sc); n30 = (100 / sc) * np.ceil(30 / sc)
    rows.append(dict(case=lab, r1=r1, curtain=sc, n100_H10=n10, n100_H30=n30, cube=sv, n_per_1000m3=1000 / sv ** 3, ground_per_ha=1e4 / sg ** 2))
    print(f'{lab:38s} {r1:5.1f} | {sc:8.1f}m {n10:16.0f} {n30:6.0f} | {sv:5.1f}m {1000 / sv ** 3:19.2f} | {1e4 / sg ** 2:32.0f}')
json.dump(rows, open(os.path.join(HERE, 'array_coverage_3d.json'), 'w'), indent=1)

# example: 100 m x 100 m site, protect ingress up to H = 30 m (4 curtain walls + a lid at 30 m)
print('\nExample site 100 m x 100 m, ingress protected up to 30 m: curtain on 4 sides + lid (horizontal plane at 30 m)')
for r in rows:
    if r['curtain'] == 0:
        continue
    walls = 4 * r['n100_H30']
    lid = (100 / r['curtain']) ** 2
    ground = r['ground_per_ha'] * 1.0
    print(f'  {r["case"]:38s} walls {walls:6.0f} + lid {lid:6.0f} = {walls + lid:6.0f} nodes'
          f'   (vs ground grid {ground:5.0f} nodes covering only h<=2 m)')
