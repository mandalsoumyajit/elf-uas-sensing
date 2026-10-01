"""Multiple drones: when do their lines collide, and can the array separate them spatially?

(a) Spectral collisions.  A target line 'collides' if a line of ANOTHER drone falls within +-delta of it
    (lines of the same drone are co-located and merge harmlessly into one dipole).
    ELF fundamental: 4 motor lines per drone spread over W = 0.3 f_e (manoeuvring), delta = 10 Hz (tracker
    half-bandwidth).  PWM carrier: one carrier per ESC (4 per drone), offset by oscillator tolerance:
    RC oscillator +-1.5 % or crystal +-50 ppm (uniform); delta = 1 Hz (two 0.5 Hz bins of a 2 s window).
(b) Spatial separability when lines collide: fraction of drone-1 SNR retained after projecting out drone 2's
    3-D response subspace (drone-2 position known, moment unknown) = what a null-steering / joint fit leaves.
(c) Two-source CRLB: drone-1 position CRLB when both drones share one frequency (14 unknowns) vs alone (7).
Geometry: street canyon 'mixed' layout, triaxial, B = 8 and 16 nodes per 100 m; DJI-like PWM 85 kHz (and
5-inch FPV PWM 24 kHz), rho_g = 100 ohm m; drones in the canyon (alt 3-20 m), separation s in random direction.
"""
import os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import site_model as SM

def collisions():
    rng = np.random.default_rng(0); rows = []
    for name, fe in SM.FUND_F.items():
        W, dl = 0.3 * fe, 10.0
        rows.append((f'{name} fundamental ({fe:.0f} Hz, W {W:.0f} Hz)', [1 - (1 - 2 * dl / W) ** (4 * (N - 1)) for N in (1, 2, 3, 5, 10)]))
    for fp in (24e3, 85e3):
        for osc, tol in (('RC +-1.5%', 0.015), ('crystal +-50 ppm', 50e-6)):
            W, dl = 2 * tol * fp, 1.0
            p = [1 - max(0.0, 1 - 2 * dl / W) ** (4 * (N - 1)) for N in (1, 2, 3, 5, 10)]
            rows.append((f'PWM {fp/1e3:.0f} kHz, {osc} (W {W:.1f} Hz)', p))
    return rows

def resp(r, nodes, G):
    return G(r[None], nodes)[0].transpose(0, 1, 2).reshape(-1, 3)        # (3 Nn, 3): columns = unit moments x,y,z

def amps(th, nodes, G):
    return np.exp(1j * th[6]) * resp(th[:3], nodes, G) @ th[3:6]

def fim(ths, nodes, G, sig):
    k = 7 * len(ths); J = np.empty((3 * len(nodes), k), complex)
    for s, th in enumerate(ths):
        for i in range(7):
            h = 1e-4 if i < 3 or i == 6 else 1e-4 * np.linalg.norm(th[3:6]); e = np.zeros(7); e[i] = h
            J[:, 7 * s + i] = (amps(th + e, nodes, G) - amps(th - e, nodes, G)) / (2 * h)
    Jw = J / sig; return 2 * np.real(Jw.conj().T @ Jw)

def pos_crlb(I):
    if np.linalg.cond(I) > 1e15:
        return np.inf
    return float(np.sqrt(np.trace(np.linalg.inv(I)[:3, :3])))

def sep_job(args):
    case, B, s, seed = args
    m0, f = SM.line(case, 'PWM'); n = SM.noise(f); sig = np.sqrt(2 * n ** 2 / 2.0)
    G = SM.Green(f, 100.0); rng = np.random.default_rng(seed); out = []
    while len(out) < 60:
        Bld = SM.buildings(rng, 160.0, (-1, 1)); nodes = SM.canyon_nodes('mixed', B, Bld, rng)
        r1 = np.array([rng.uniform(-7, 7), rng.uniform(-10, 10), rng.uniform(3, 20)])
        for _ in range(50):
            u = rng.normal(size=3); u /= np.linalg.norm(u); r2 = r1 + s * u
            if abs(r2[0]) < 9 and 2 < r2[2] < 22:
                break
        else:
            continue
        nd = nodes[np.linalg.norm(nodes - 0.5 * (r1 + r2), axis=1) < 40]
        u1, u2 = rng.normal(size=3), rng.normal(size=3)
        th1 = np.r_[r1, m0 * u1 / np.linalg.norm(u1), 0.3]; th2 = np.r_[r2, m0 * u2 / np.linalg.norm(u2), 1.9]
        a1 = amps(th1, nd, G) / sig
        snr1 = np.sum(np.abs(a1) ** 2)
        if snr1 < 30:
            continue
        A2 = resp(r2, nd, G) / sig
        Q, _ = np.linalg.qr(A2)
        keep = np.sum(np.abs(a1 - Q @ (Q.conj().T @ a1)) ** 2) / snr1
        c1 = pos_crlb(fim([th1], nd, G, sig)); c12 = pos_crlb(fim([th1, th2], nd, G, sig))
        out.append((keep, c1, c12, 10 * np.log10(snr1)))
    return args, np.array(out)

if __name__ == '__main__':
    L = []
    P = lambda s: (print(s, flush=True), L.append(s))
    P('(a) Probability that a given line of one drone has a line of ANOTHER drone within +-delta (instantaneous)')
    P(f'{"line":52s}' + ''.join(f'{f"N={N}":>8s}' for N in (1, 2, 3, 5, 10)))
    for lab, p in collisions():
        P(f'{lab:52s}' + ''.join(f'{v:8.2f}' for v in p))
    seps = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    jobs = [(case, B, s, 31 * B + int(10 * s)) for case in ('DJI-like', '5-inch FPV') for B in (8, 16) for s in seps]
    with ProcessPoolExecutor(max_workers=20) as ex:
        R = list(ex.map(sep_job, jobs))
    P('\n(b,c) Two drones on the SAME frequency, street canyon mixed layout, triaxial, PWM carrier, 2 s window, '
      'windows with drone-1 array SNR >= 15 dB')
    P(f'{"case":11s}{"B":>4s}{"sep (m)":>9s} {"SNR kept after nulling drone 2":>32s} {"CRLB alone":>12s} {"CRLB joint":>12s} {"joint/alone":>12s} {"joint finite":>13s}')
    for (case, B, s, _), o in R:
        fin = np.isfinite(o[:, 2])
        ratio = np.where(fin, o[:, 2] / o[:, 1], np.inf)
        P(f'{case:11s}{B:4d}{s:9.0f} {np.median(o[:, 0]):17.2f} [10%: {np.percentile(o[:, 0], 10):4.2f}]      '
          f'{np.median(o[:, 1]):10.2f} m {np.median(o[:, 2]):10.2f} m {np.median(ratio):11.2f} {fin.mean():12.2f}')
    open(os.path.join(HERE, 'multi_drone_out.txt'), 'w').write('\n'.join(L) + '\n')
