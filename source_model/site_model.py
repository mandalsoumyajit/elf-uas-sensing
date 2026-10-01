"""Array model, part 3: realistic node heights (ground, poles, facades, rooftops, masts), conductive ground,
measured impulsive-noise penalty.

Green's function : free-space dipole + ground secondary field (homogeneous half-space, empymod tables from
                   ground_tables.py; rho_g = 100 ohm m baseline, 30 / 1000 / none as sensitivity). Buildings are
                   only mounts (their own conductors/rebar are NOT modelled).
Noise            : semi-urban ASD at the line frequency x mount multiplier (default 1), receiver noise in
                   quadrature; impulsive (Class A-like) background enters as an SNR penalty L_imp (dB) measured
                   on the grid data for the same detection statistic (detstat_report.py).
Detection        : array_model.window_pd (P3 = ELF fundamental joint-tracked, PWM = coherent carrier), 2 s windows,
                   1 false alarm per hour.  Nodes triaxial unless stated.
Scenarios
  perimeter : drones cross a straight perimeter (x = 0) at altitude z0, heading within +-60 deg of the normal,
              5 m/s, track -25..+25 m.  Node budget B per 100 m of perimeter, distributed over mounts:
                ground      : 1.5 m (fence/bollard), spacing 100/B
                poles       : 10 m light poles, nodes at 5 and 10 m, pole spacing 200/B
                poles+roofs : half on 10 m poles at the fence, half on rooftops of buildings 15 m inside
                              (roof height U[8, 25] m, node 1 m above roof)
                towers      : 25 m lattice towers / tall facades, nodes at 6, 12, 18, 24 m, tower spacing 400/B
                curtain     : idealised square lattice to 30 m (reference from part 1)
  canyon    : drone flies along a 20 m wide street between building rows (20 m frontage, heights U[9, 24] m),
              track 100 m, 5 m/s, heading within +-5 deg of the street axis, lateral offset U[-7, 7] m,
              altitude in-canyon U[3, 9] m (below every roof = radar/optical NLOS) or above-roof U[26, 40] m.
              Node budget B per 100 m of street:
                ground      : 1.5 m at both kerbs, staggered
                streetlights: 8 m poles at both kerbs, staggered
                facades     : on facades at random height U[3, h_b - 1]
                rooftops    : at the roof edge, h_b + 1
                mixed       : streetlight / facade / rooftop in turn
Output : site_model_out.txt, site_model.json
"""
import copy, json, os, sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import array_model as AM

GT = np.load(os.path.join(HERE, 'ground_tables.npy'), allow_pickle=True).item()
_GREEN = {}

class Green:
    """B at node per unit moment at the source, including the ground secondary field."""
    def __init__(self, f, rho_g=None):
        self.rho_g = rho_g
        if rho_g is not None:
            ft = min(GT['FREQS'], key=lambda x: abs(np.log(x / f)))
            assert abs(ft / f - 1) < 0.02, f'no ground table for {f} Hz'
            pts = (np.log(GT['RHO']), np.log(GT['HS']))
            self.I = {k: (RegularGridInterpolator(pts, v.real, bounds_error=False, fill_value=None),
                          RegularGridInterpolator(pts, v.imag, bounds_error=False, fill_value=None))
                      for k, v in GT['tabs'][(ft, rho_g)].items()}

    def __call__(self, src, nodes):
        d = nodes[None, :, :] - src[:, None, :]
        G = AM.dipole_tensor(d).astype(complex)
        if self.rho_g is None:
            return G
        rho = np.hypot(d[..., 0], d[..., 1]).ravel()
        Hs = (src[:, None, 2] + nodes[None, :, 2]).ravel()
        q = np.c_[np.log(np.clip(rho, GT['RHO'][0], GT['RHO'][-1])), np.log(np.clip(Hs, GT['HS'][0], GT['HS'][-1]))]
        sc = (rho ** 2 + Hs ** 2) ** -1.5
        S = {k: (a(q) + 1j * b(q)) * sc for k, (a, b) in self.I.items()}
        ph = np.arctan2(d[..., 1], d[..., 0]).ravel(); c, s = np.cos(ph), np.sin(ph)
        Gs = np.empty((len(ph), 3, 3), complex)                 # R L R^T, L = [[xx,0,xz],[0,yy,0],[zx,0,zz]]
        Gs[:, 0, 0] = c * c * S['xx'] + s * s * S['yy']; Gs[:, 1, 1] = s * s * S['xx'] + c * c * S['yy']
        Gs[:, 0, 1] = Gs[:, 1, 0] = c * s * (S['xx'] - S['yy'])
        Gs[:, 0, 2] = c * S['xz']; Gs[:, 1, 2] = s * S['xz']
        Gs[:, 2, 0] = c * S['zx']; Gs[:, 2, 1] = s * S['zx']; Gs[:, 2, 2] = S['zz']
        return G + Gs.reshape(G.shape)

def green(f, rho_g):
    k = (f, rho_g)
    if k not in _GREEN:
        _GREEN[k] = Green(f, rho_g)
    return _GREEN[k]

# ---------------------------------------------------------------- node layouts
def buildings(rng, Y, side_x, frontage=20.0, lo=9.0, hi=24.0):
    ys = np.arange(-Y, Y + frontage, frontage)
    return {sx: (ys, rng.uniform(lo, hi, len(ys))) for sx in side_x}

def roof_h(B, sx, y):
    ys, hs = B[sx]; return hs[np.clip(np.searchsorted(ys, y) - 1, 0, len(hs) - 1)]

def perimeter_nodes(cfg, Bn, rng, Y=80.0):
    s = 100.0 / Bn
    if cfg == 'ground':
        y = np.arange(-Y, Y + 1e-9, s); return np.c_[np.zeros_like(y), y, np.full_like(y, 1.5)]
    if cfg == 'poles':
        y = np.arange(-Y, Y + 1e-9, 2 * s)
        return np.r_[np.c_[np.zeros_like(y), y, np.full_like(y, 5.0)], np.c_[np.zeros_like(y), y, np.full_like(y, 10.0)]]
    if cfg == 'poles+roofs':
        y = np.arange(-Y, Y + 1e-9, 2 * s); yr = y + s
        hr = rng.uniform(8.0, 25.0, len(yr)) + 1.0
        return np.r_[np.c_[np.zeros_like(y), y, np.full_like(y, 10.0)], np.c_[np.full_like(yr, -15.0), yr, hr]]
    if cfg == 'towers':
        y = np.arange(-Y, Y + 1e-9, 4 * s)
        return np.concatenate([np.c_[np.zeros_like(y), y, np.full_like(y, z)] for z in (6.0, 12.0, 18.0, 24.0)])
    if cfg == 'curtain':
        p = np.sqrt(3000.0 / Bn); return AM.curtain_nodes(p, 30.0, Y=Y)
    raise ValueError(cfg)

def canyon_nodes(cfg, Bn, Bld, rng, Y=150.0, W=20.0):
    s = 200.0 / Bn                                   # per side; sides staggered by s/2
    out = []
    for k, sx in enumerate((-1, 1)):
        y = np.arange(-Y, Y + 1e-9, s) + k * s / 2
        kinds = {'ground': ['g'], 'streetlights': ['l'], 'facades': ['f'], 'rooftops': ['r'], 'mixed': ['l', 'f', 'r']}[cfg]
        for i, yy in enumerate(y):
            kd = kinds[(i + k) % len(kinds)]; hb = roof_h(Bld, sx, yy)
            if kd == 'g':
                out.append((sx * (W / 2 - 1), yy, 1.5))
            elif kd == 'l':
                out.append((sx * (W / 2 - 1), yy, 8.0))
            elif kd == 'f':
                out.append((sx * W / 2, yy, rng.uniform(3.0, hb - 1.0)))
            else:
                out.append((sx * (W / 2 + 0.5), yy, hb + 1.0))
    return np.array(out)

# ---------------------------------------------------------------- detection along a trajectory
def snr_track(pos, nodes, mvec, n, f, rho_g, triax):
    G = green(f, rho_g)(pos, nodes)
    B = np.einsum('tnij,j->tni', G, mvec)
    b2 = (np.abs(B) ** 2).sum(-1) if triax else np.abs(B[..., 2]) ** 2
    return b2 / (2 * n ** 2)

def track_pd(pos, nodes, mvec, n, f, rho_g, triax, proc, Limp, steps):
    near = np.min(np.linalg.norm(nodes[None, :, :] - pos[::steps, None, :], axis=-1), axis=0) < 60.0
    s = snr_track(pos, nodes[near], mvec, n[near] if np.ndim(n) else n, f, rho_g, triax) * 10 ** (-Limp / 10)
    nwin = len(pos) // steps
    pw = np.array([AM.window_pd(s[w * steps:(w + 1) * steps], proc) for w in range(nwin)])
    return pw

LINES = {}
def line(case, proc):
    c = CASES[case]
    if (case, proc) not in LINES:
        op, L = AM.DS.lines_for(c)
        if proc == 'PWM':
            m = float(np.sqrt(sum(x[2][1] ** 2 for x in L if abs(x[0] - c['f_pwm']) < 1))); f = c['f_pwm']
        else:
            fund = max([x for x in L if x[1] == 'phase leads' and abs(x[0] - op['f_e']) < 1], key=lambda x: x[2][1])
            m, f = fund[2][1], fund[0]
        LINES[(case, proc)] = (m, f)
    return LINES[(case, proc)]

def noise(f, bg='semi-urban outdoor', mult=1.0, canc=None):
    """Per-node noise ASD (T/rtHz) at f.  canc = None: no cancellation.  Otherwise a dict:
         f_loc   fraction of the man-made excess (scenario minus natural) produced by LOCAL sources (per node,
                 spatially incoherent between nodes); the rest of the man-made part and all natural noise are
                 'distant' (spatially coherent over the site)
         x_loc   optional extra local man-made ASD (T/rtHz, white) e.g. switching emitters in the VLF band
         cov     fraction of local power whose sources carry a witness (current clamp / coil at the source)
         D_w     witness cancellation depth (dB) for covered sources (set by witness SNR and source rank)
         gamma2  coherence of the distant background between the remote reference and the node
         n_ref   remote-reference self-noise ASD (T/rtHz); None = no remote reference
    Wiener residual of the distant part: 1 - gamma2 * S / (S + n_ref^2), S = distant background power.
    Drone leakage into witnesses (at the sources) and the remote reference (>= 50 m away) is neglected."""
    rx2 = AM.wire_loop_noise(f)[0] ** 2
    tot2 = (mult * AM.SC[bg](f)) ** 2
    if canc is None:
        return float(np.sqrt(tot2 + rx2))
    nat2 = min(AM.ns['natural'](f) ** 2, tot2)
    mm2 = tot2 - nat2
    loc2 = canc.get('f_loc', 0.0) * mm2 + canc.get('x_loc', 0.0) ** 2
    dist2 = nat2 + (1 - canc.get('f_loc', 0.0)) * mm2
    cov = canc.get('cov', 0.0); rw = 10 ** (-canc.get('D_w', 0.0) / 10)
    loc_res = loc2 * ((1 - cov) + cov * rw)
    if canc.get('n_ref') is not None:
        dist_res = dist2 * (1 - canc.get('gamma2', 0.0) * dist2 / (dist2 + canc['n_ref'] ** 2))
    else:
        dist_res = dist2
    return float(np.sqrt(loc_res + dist_res + rx2))

C = AM.DS.CLASSES
_dji = copy.deepcopy(C['photo/mid (11" props, 4S)']); _dji['f_pwm'] = 85e3
CASES = {'5-inch FPV': C['5-inch FPV (6S)'], 'DJI-like': _dji, 'heavy-lift': C['heavy-lift (30" props, 12S)']}
FUND_F = {'5-inch FPV': 1074.0, 'DJI-like': 802.0, 'heavy-lift': 366.0}

def run_perimeter(args):
    case, proc, cfg, Bn, rho_g, triax, Limp, ntraj, seed, zbins = args[:10]; canc = args[10] if len(args) > 10 else None
    m, f = line(case, proc); f = FUND_F[case] if proc != 'PWM' else f
    n = noise(f, canc=canc); rng = np.random.default_rng(seed)
    v, DT = 5.0, 0.1; steps = int(AM.TW / DT); nwin = int(50.0 / v / AM.TW)
    res = {zb: [] for zb in zbins}
    for zb in zbins:
        for k in range(ntraj):
            nodes = perimeter_nodes(cfg, Bn, rng)
            y0 = rng.uniform(-10, 10); z0 = rng.uniform(*zb)
            ang = rng.uniform(-np.pi / 3, np.pi / 3); hd = np.array([np.cos(ang), np.sin(ang), 0.0])
            u = rng.normal(size=3); mvec = m * u / np.linalg.norm(u)
            t = (np.arange(nwin * steps) - nwin * steps / 2) * DT
            pos = np.array([0.0, y0, z0])[None, :] + v * t[:, None] * hd[None, :]
            pw = track_pd(pos, nodes, mvec, n, f, rho_g, triax, proc, Limp, steps)
            res[zb].append(1 - np.prod(1 - pw))
    return args, {zb: (float(np.mean(r)), float(np.mean(np.array(r) >= 0.9))) for zb, r in res.items()}

def run_canyon(args):
    case, proc, cfg, Bn, rho_g, triax, Limp, ntraj, seed, zbins = args[:10]; canc = args[10] if len(args) > 10 else None
    m, f = line(case, proc); f = FUND_F[case] if proc != 'PWM' else f
    n = noise(f, canc=canc); rng = np.random.default_rng(seed)
    v, DT = 5.0, 0.1; steps = int(AM.TW / DT); nwin = int(100.0 / v / AM.TW)
    res = {zb: [] for zb in zbins}
    for zb in zbins:
        for k in range(ntraj):
            Bld = buildings(rng, 160.0, (-1, 1))
            nodes = canyon_nodes(cfg, Bn, Bld, rng)
            x0 = rng.uniform(-7, 7); z0 = rng.uniform(*zb); y0 = rng.uniform(-60, -40)
            ang = rng.uniform(-np.radians(5), np.radians(5)); hd = np.array([np.sin(ang), np.cos(ang), 0.0])
            u = rng.normal(size=3); mvec = m * u / np.linalg.norm(u)
            t = np.arange(nwin * steps) * DT
            pos = np.array([x0, y0, z0])[None, :] + v * t[:, None] * hd[None, :]
            pw = track_pd(pos, nodes, mvec, n, f, rho_g, triax, proc, Limp, steps)
            res[zb].append((pw.mean(), 1 - np.prod(1 - pw[:5])))     # window coverage; detected within first 50 m
    return args, {zb: (float(np.mean([a for a, _ in r])), float(np.mean([b for _, b in r]))) for zb, r in res.items()}

if __name__ == '__main__':
    LIMP = json.load(open(os.path.join(HERE, 'impulsive_penalty.json')))
    L_mit = LIMP['measured']; L_raw = LIMP['gaussian']
    PZ = ((2, 10), (10, 20), (20, 30), (30, 45), (45, 60))
    CZ = ((3, 9), (26, 40))
    jobs_p, jobs_c = [], []
    for case in CASES:
        for proc in ('P3', 'PWM'):
            for Bn in (4, 8, 16, 32):
                for cfg in ('ground', 'poles', 'poles+roofs', 'towers', 'curtain'):
                    jobs_p.append((case, proc, cfg, Bn, 100.0, True, L_mit[proc], 150, Bn * 7 + len(cfg), PZ))
                for cfg in ('ground', 'streetlights', 'facades', 'rooftops', 'mixed'):
                    jobs_c.append((case, proc, cfg, Bn, 100.0, True, L_mit[proc], 120, Bn * 11 + len(cfg), CZ))
    # sensitivity: ground resistivity, impulsive penalty, scalar vs triaxial (DJI-like PWM and 5-inch P3, mixed layouts)
    sens = []
    for case, proc in (('DJI-like', 'PWM'), ('5-inch FPV', 'P3'), ('heavy-lift', 'PWM')):
        for Bn in (8, 16):
            for lab, rho_g, triax, L in (('free space', None, True, L_mit[proc]), ('rho 1000', 1000.0, True, L_mit[proc]),
                                         ('rho 30', 30.0, True, L_mit[proc]), ('Gaussian bg (no line clutter)', 100.0, True, 0.0),
                                         ('scalar (vertical axis)', 100.0, False, L_mit[proc])):
                sens.append((lab, ('canyon', (case, proc, 'mixed', Bn, rho_g, triax, L, 120, 900 + Bn, CZ))))
                sens.append((lab, ('perimeter', (case, proc, 'poles+roofs', Bn, rho_g, triax, L, 150, 700 + Bn, PZ))))
    with ProcessPoolExecutor(max_workers=20) as ex:
        RP = list(ex.map(run_perimeter, jobs_p)); RC = list(ex.map(run_canyon, jobs_c))
        fs =[(lab, kind, ex.submit(run_perimeter if kind == 'perimeter' else run_canyon, a)) for lab, (kind, a) in sens]
        RS = [(lab, kind, fu.result()) for lab, kind, fu in fs]
    lines = []
    P = lambda *a: (print(*a, flush=True), lines.append(' '.join(str(x) for x in a)))
    P(f'Background-clutter penalty used (dB, measured on grid data): {L_mit}; sensitivity: Gaussian background (0 dB)')
    P('\n=== PERIMETER crossing: fraction of crossings detected with Pd >= 0.9, by altitude band (triaxial, rho_g 100 ohm m) ===')
    for case in CASES:
        for proc in ('P3', 'PWM'):
            P(f'\n{case}, {"ELF fundamental joint-tracked" if proc == "P3" else "PWM carrier coherent"}  [nodes per 100 m of perimeter]')
            P(f'{"layout":14s}{"B":>4s} ' + ''.join(f'{f"{a}-{b} m":>10s}' for a, b in PZ))
            for cfg in ('ground', 'poles', 'poles+roofs', 'towers', 'curtain'):
                for Bn in (4, 8, 16, 32):
                    r = next(o for a, o in RP if a[0] == case and a[1] == proc and a[2] == cfg and a[3] == Bn)
                    P(f'{cfg:14s}{Bn:4d} ' + ''.join(f'{r[zb][1]:10.2f}' for zb in PZ))
    P('\n=== STREET CANYON flight: track coverage (mean window Pd) | P(detected within first 50 m), triaxial, rho_g 100 ===')
    for case in CASES:
        for proc in ('P3', 'PWM'):
            P(f'\n{case}, {"ELF fundamental joint-tracked" if proc == "P3" else "PWM carrier coherent"}  [nodes per 100 m of street]')
            P(f'{"layout":14s}{"B":>4s} ' + ''.join(f'{f"alt {a}-{b} m":>22s}' for a, b in CZ))
            for cfg in ('ground', 'streetlights', 'facades', 'rooftops', 'mixed'):
                for Bn in (4, 8, 16, 32):
                    r = next(o for a, o in RC if a[0] == case and a[1] == proc and a[2] == cfg and a[3] == Bn)
                    P(f'{cfg:14s}{Bn:4d} ' + ''.join(f'{r[zb][0]:10.2f} | {r[zb][1]:5.2f}    ' for zb in CZ))
    P('\n=== SENSITIVITY (poles+roofs perimeter: frac Pd>=0.9 per altitude band; mixed canyon: P(det within 50 m)) ===')
    for lab, kind, (a, o) in RS:
        zb = PZ if kind == 'perimeter' else CZ
        vals = ''.join(f'{o[z][1]:7.2f}' for z in zb)
        P(f'{kind:9s} {a[0]:11s} {a[1]:4s} B={a[3]:2d} {lab:24s}: {vals}')
    open(os.path.join(HERE, 'site_model_out.txt'), 'w').write('\n'.join(lines) + '\n')
    js = dict(perimeter=[dict(case=a[0], proc=a[1], cfg=a[2], B=a[3], res={f'{k[0]}-{k[1]}': v for k, v in o.items()}) for a, o in RP],
              canyon=[dict(case=a[0], proc=a[1], cfg=a[2], B=a[3], res={f'{k[0]}-{k[1]}': v for k, v in o.items()}) for a, o in RC],
              sens=[dict(label=lab, kind=kind, case=a[0], proc=a[1], B=a[3], res={f'{k[0]}-{k[1]}': v for k, v in o.items()}) for lab, kind, (a, o) in RS])
    json.dump(js, open(os.path.join(HERE, 'site_model.json'), 'w'), indent=1)
