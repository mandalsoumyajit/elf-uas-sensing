"""Array model, part 1: trajectory Monte Carlo of drones crossing a perimeter curtain of ELF/VLF nodes.

Geometry : curtain = vertical plane x = 0; nodes on a square lattice of pitch p in (y, z), z in [p/2, H], |y| <= Y.
Drone    : straight, level crossing at random point (y0 in one cell, altitude z0 ~ U[0.5, H]), heading within +-60 deg
           of the plane normal, speed v; line moment from drone_source (nominal), random fixed dipole orientation.
Nodes    : 'scalar' = one loop with normal along x (perpendicular to the curtain); 'triaxial' = full |B|^2.
Noise    : scenario ASD at the line frequency (after per-node periodic cancellation), receiver noise in quadrature.
Windows  : non-overlapping T_w = 2 s along the trajectory (-25 m .. +25 m around the crossing).
Detectors (per window), false-alarm rate 1/hour shared over windows and frequency-search cells:
  P1  incoherent energy, fundamental, all nodes, tau = 0.05 s chunks
  P3  joint-tracked coherent fundamental: tracking possible if sum of the K = 4 strongest node SNRs in 20 Hz
      >= -1.5 dB (measured single-node 50% lock threshold on grid data, applied to the combined array SNR);
      then coherent K-node model-free combining over the window; falls back to P1 if tracking fails
  PWM coherent PWM carrier (no tracking), K-node model-free combining, +-1 kHz carrier search
Output: mean trajectory Pd and fraction of trajectories with Pd >= 0.9 vs pitch.
"""
import copy, json, os, sys
import numpy as np
from scipy.stats import chi2, ncx2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', 'range_budget'))
from range_budget import wire_loop_noise
import drone_source as DS
src = open(os.path.join(HERE, 'range_by_class.py')).read().split("SCEN =")[0]
ns = {'__file__': os.path.join(HERE, 'range_by_class.py')}; exec(src, ns)
SC = {'quiet outdoor': ns['natural'], 'semi-urban outdoor': ns['semi_urban'], 'indoor lab': ns['indoor_lab']}

TW, DT, TAU, BL, K = 2.0, 0.05, 0.05, 20.0, 4
GTH = 10 ** (-1.5 / 10)
FAR_H = 1.0
PFA_WIN = FAR_H / (3600.0 / TW)

def dipole_tensor(d):
    r = np.linalg.norm(d, axis=-1, keepdims=True); u = d / r
    return 1e-7 / r[..., None] ** 3 * (3 * u[..., :, None] * u[..., None, :] - np.eye(3))

def curtain_nodes(p, H, Y=60.0):
    ys = np.arange(-Y, Y + 1e-9, p); zs = np.arange(p / 2, H + 1e-9, p)
    Yg, Zg = np.meshgrid(ys, zs); return np.c_[np.zeros(Yg.size), Yg.ravel(), Zg.ravel()]

def snr_rate(pos, nodes, mvec, n, triax):
    """per-second coherent SNR per node: |B . nhat|^2 / (2 n^2) (scalar, nhat = x) or |B|^2 / (2 n^2)."""
    d = nodes[None, :, :] - pos[:, None, :]
    B = np.einsum('tnij,j->tni', dipole_tensor(d), mvec)
    b2 = (B ** 2).sum(-1) if triax else B[..., 0] ** 2
    return b2 / (2 * n ** 2)

def pd(dof, lam, pfa):
    return ncx2.sf(chi2.isf(pfa, dof), dof, lam)

def window_pd(s_win, proc):
    """s_win: (steps, nodes) per-second SNR samples within one window."""
    m = s_win.mean(0)                                 # mean SNR rate over the window per node
    top = np.sort(m)[::-1][:K]
    if proc == 'PWM':
        return pd(2 * K, 2 * TW * top.sum(), PFA_WIN / (2000 * TW))
    pfa = PFA_WIN / (500 * TAU)
    M = int(TW / TAU)
    near = np.sort(m)[::-1][:16]                       # incoherent energy over the 16 strongest nodes (dof penalty bounded)
    p_inc = pd(2 * len(near) * M, 2 * M * np.sum(near * TAU), pfa)
    if proc == 'P1':
        return p_inc
    if top.sum() / BL < GTH:                           # joint tracker cannot lock -> incoherent fallback
        return p_inc
    return max(p_inc, pd(2 * K, 2 * TW * top.sum(), pfa))

def simulate(p, m_line, n, proc, triax, H=30.0, v=5.0, ntraj=300, seed=0):
    rng = np.random.default_rng(seed)
    nodes = curtain_nodes(p, H)
    steps = int(TW / DT); nwin = int(50.0 / v / TW)
    pds = np.empty(ntraj)
    for k in range(ntraj):
        y0 = rng.uniform(0, p); z0 = rng.uniform(0.5, H)
        ang = rng.uniform(-np.pi / 3, np.pi / 3); hd = np.array([np.cos(ang), np.sin(ang), 0.0])
        u = rng.normal(size=3); mvec = m_line * u / np.linalg.norm(u)
        t = (np.arange(nwin * steps) - nwin * steps / 2) * DT
        pos = np.array([0.0, y0, z0])[None, :] + v * t[:, None] * hd[None, :]
        s = snr_rate(pos, nodes, mvec, n, triax)
        pw = [window_pd(s[w * steps:(w + 1) * steps], proc) for w in range(nwin)]
        pds[k] = 1 - np.prod(1 - np.array(pw))
    return float(pds.mean()), float(np.mean(pds >= 0.9))

C = DS.CLASSES
dji = copy.deepcopy(C['photo/mid (11" props, 4S)']); dji['f_pwm'] = 85e3
CASES = [('5-inch FPV', C['5-inch FPV (6S)'], 'indoor lab'),
         ('5-inch FPV', C['5-inch FPV (6S)'], 'semi-urban outdoor'),
         ('DJI-like photo 11"', dji, 'semi-urban outdoor'),
         ('heavy-lift', C['heavy-lift (30" props, 12S)'], 'semi-urban outdoor')]
PITCHES = [2, 3, 4, 6, 8, 11, 16, 22]

if __name__ == '__main__':
    out = []
    for name, c, bg in CASES:
        op, L = DS.lines_for(c)
        fund = max([x for x in L if x[1] == 'phase leads' and abs(x[0] - op['f_e']) < 1], key=lambda x: x[2][1])
        pw = [x for x in L if abs(x[0] - c['f_pwm']) < 1]
        m_pwm = float(np.sqrt(sum(x[2][1] ** 2 for x in pw)))
        n_f = float(np.hypot(SC[bg](fund[0]), wire_loop_noise(fund[0])[0]))
        n_p = float(np.hypot(SC[bg](c['f_pwm']), wire_loop_noise(c['f_pwm'])[0]))
        print(f'\n=== {name}, {bg}: f_e {op["f_e"]:.0f} Hz (m {fund[2][1]:.1e}, n {n_f*1e15:.0f} fT), '
              f'PWM {c["f_pwm"]/1e3:.0f} kHz (m {m_pwm:.1e}, n {n_p*1e15:.0f} fT); curtain H = 30 m, v = 5 m/s ===', flush=True)
        print(f'{"pitch":>6s} ' + ''.join(f'{lab:>22s}' for lab in ('P1 scalar', 'P3 scalar', 'P3 triaxial', 'PWM scalar', 'PWM triaxial')))
        for p in PITCHES:
            row = []
            for proc, triax, m, nn in (('P1', False, fund[2][1], n_f), ('P3', False, fund[2][1], n_f), ('P3', True, fund[2][1], n_f),
                                       ('PWM', False, m_pwm, n_p), ('PWM', True, m_pwm, n_p)):
                mp, f90 = simulate(p, m, nn, proc, triax, ntraj=200, seed=p)
                row.append((mp, f90)); out.append(dict(case=name, bg=bg, pitch=p, proc=proc, triax=triax, pd_mean=mp, frac90=f90))
            print(f'{p:6d} ' + ''.join(f'   Pd {a:4.2f} / >=0.9: {b:4.2f}' for a, b in row), flush=True)
    json.dump(out, open(os.path.join(HERE, 'array_model_curtain.json'), 'w'), indent=1)
