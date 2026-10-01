"""Mutual inductance and coupling coefficient between ELF and VLF windings if they were co-located (filament model).

Turns are coaxial square filaments (Neumann integral, numerically for unequal sides; validated against the exact
equal-side formula of coil_model.py). Arrangements:
  shared   : VLF turns on the same boards, 5 mm above the top ELF turn (same side length)
  nested   : separate VLF frame of side 0.45 m, coplanar with the middle of the ELF winding
  offset   : separate full-size VLF frame displaced 0.30 m along the axis
"""
import json
import numpy as np
import coil_model as D

def square_pts(a, n=240):
    """segment midpoints and vectors of a square loop of side a in the z = 0 plane."""
    t = (np.arange(n) + 0.5) / n
    c = np.array([[-a / 2, -a / 2], [a / 2, -a / 2], [a / 2, a / 2], [-a / 2, a / 2]])
    P, V = [], []
    for i in range(4):
        p0, p1 = c[i], c[(i + 1) % 4]
        P.append(p0 + np.outer(t, p1 - p0)); V.append(np.tile((p1 - p0) / n, (n, 1)))
    return np.vstack(P), np.vstack(V)

def m_numeric(a1, a2, d, n=240):
    P1, V1 = square_pts(a1, n); P2, V2 = square_pts(a2, n)
    dx = P1[:, None, 0] - P2[None, :, 0]; dy = P1[:, None, 1] - P2[None, :, 1]
    r = np.sqrt(dx * dx + dy * dy + d * d)
    return D.MU0 / (4 * np.pi) * float(np.sum((V1 @ V2.T) / r))

def mutual(nE, nV, aV, z0, pitch=D.PITCH):
    """total mutual inductance: ELF turns at z = 0..(nE-1)*pitch, VLF turns of side aV starting at z0."""
    zE = np.arange(nE) * pitch; zV = z0 + np.arange(nV) * pitch
    d = np.abs(zE[:, None] - zV[None, :]).ravel()
    grid = np.linspace(d.min(), d.max() + 1e-9, 25)
    mg = np.array([m_numeric(D.SIDE, aV, max(g, 1e-4)) for g in grid])
    return float(np.interp(d, grid, mg).sum())

if __name__ == '__main__':
    print('check unequal-side Neumann vs exact equal-side formula (d = 10 mm):',
          m_numeric(D.SIDE, D.SIDE, 0.01), D.m_square(0.01))
    r_eq = json.load(open(D.os.path.join(D.HERE, 'coil_model.json')))['r_eq']
    out = {}
    for nE in (48, 100):
        LE = D.inductance(nE, r_eq)
        for nV in (8, 12, 16):
            for name, aV, z0 in (('shared', D.SIDE, (nE - 1) * D.PITCH + 5e-3), ('nested', 0.45, (nE - 1) * D.PITCH / 2),
                                 ('offset', D.SIDE, (nE - 1) * D.PITCH + 0.30)):
                LV = D.inductance(nV, r_eq) if aV == D.SIDE else None
                if LV is None:                                   # nested frame: scale self term by side (filament model)
                    LV = sum(m_numeric(aV, aV, max(abs(i - j) * D.PITCH, 0.663e-3)) for i in range(nV) for j in range(nV))
                M = mutual(nE, nV, aV, z0)
                k = M / np.sqrt(LE * LV)
                out[f'{nE}|{nV}|{name}'] = dict(LE=LE, LV=LV, M=M, k=k, aV=aV, area_V=nV * aV * aV)
                print(f'ELF {nE:3d} turns, VLF {nV:2d} turns, {name:6s}: L_V {LV*1e6:6.1f} uH, M {M*1e6:7.2f} uH, k = {k:.3f}', flush=True)
    json.dump(out, open(D.os.path.join(D.HERE, 'coupling.json'), 'w'), indent=1)
