"""Study 10 (v6): inclination from one vector node with the measurement-informed source and noise models.

Replaces the SNR-prescribed study 2: here the noise is an absolute background level (ASD at 1.1 kHz) and
accuracy is reported against range.  One triaxial point node at the origin; drone at horizontal range
U[1, 4] m, azimuth +-15 deg, height 0.3 m; roll/pitch U[+-45 deg], yaw U[+-180 deg]; T = 0.2 s and 1 s.
Features (all computed from the same noisy record):
  compact  : six normalised band-covariance features over the operating band (no tracking needed)
  resolved : 36 tone-resolved features from phase-referenced phasors of the four motors (tracking needed;
             if any motor fails the measured tracking threshold the record falls back to the training mean)
  z loop   : single-loop log power in the band;  total : three-axis log power
Training is matched to each condition (noisy records, same level and T, ranges U[1, 4] m); MLP from
simulation_utils, 3 training repeats.  Output: simulation_results/v7_calibrated/inclination.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits
import realistic_model as RM
from features import bandpass, covariance_features, log_power
from simulation_utils import make_model, save_json, provenance

OUT = Path('simulation_results/v7_calibrated'); OUT.mkdir(parents=True, exist_ok=True)
LEVELS = {'quiet outdoor (0.02 pT)': 0.02e-12, 'measured indoor (0.2 pT)': 0.2e-12,
          'semi-urban (1 pT)': 1e-12, 'near electronics (10 pT)': 10e-12}
TS = (0.2, 1.0)
NODE = np.zeros((1, 3))
SRC = RM.Src()

def resolved(Y):
    out = []
    for g in Y:                                   # motor k: complex 3-vector
        h = np.outer(g, g.conj()); h = h / max(np.trace(h).real, 1e-300)
        out.append(np.r_[np.diag(h).real, [h[i, j].real for i, j in ((0, 1), (0, 2), (1, 2))],
                         [h[i, j].imag for i, j in ((0, 1), (0, 2), (1, 2))]])
    return np.concatenate(out)

def one(args):
    seed, T, n = args
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        R = rng.uniform(1, 4); az = rng.uniform(-np.pi / 12, np.pi / 12)
        pos = np.array([R * np.cos(az), R * np.sin(az), 0.3])
        att = np.r_[rng.uniform(-np.pi / 4, np.pi / 4, 2), rng.uniform(-np.pi, np.pi)]
        B, psi, f = RM.simulate(pos, att, NODE, T, rng, SRC)
        B = B[0]; fm = f.mean(1)
        lo, hi = fm.min() - 60, fm.max() + 60
        Y0 = RM.demod(B, psi)                                         # k, 3 (clean)
        feats = {}
        for name, lv in LEVELS.items():
            x = B + RM.noise(B.shape, lv, rng)
            Bf = bandpass(x, RM.FS, lo, hi)
            Y = RM.demod(x, psi)
            ok = bool(np.all(RM.tracking_ok(Y0[:, None, :], fm, lv)))
            feats[name] = dict(compact=covariance_features(Bf), resolved=resolved(Y), ok=ok,
                               z=np.array([log_power(Bf[2:3])]), total=np.array([log_power(Bf)]))
        rows.append((RM.tilt_magnitude(*att[:2]), R, feats))
    return rows

def build(T, n, seed):
    chunks = [(seed * 1000 + i, T, n // 48) for i in range(48)]
    with ProcessPoolExecutor(max_workers=22) as ex:
        return [r for part in ex.map(one, chunks) for r in part]

def main():
    res = {}
    for T in TS:
        tr = build(T, 3072, 1 + int(10 * T)); te = build(T, 1536, 7 + int(10 * T))
        ytr = np.array([r[0] for r in tr]); yte = np.array([r[0] for r in te]); Rte = np.array([r[1] for r in te])
        bins = [(1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5), (3.5, 4)]
        for name in LEVELS:
            okte = np.array([r[2][name]['ok'] for r in te]); oktr = np.array([r[2][name]['ok'] for r in tr])
            for feat in ('compact', 'resolved', 'z', 'total'):
                Xtr = np.array([r[2][name][feat] for r in tr]); Xte = np.array([r[2][name][feat] for r in te])
                preds = []
                for rep in range(3):
                    use = oktr if feat == 'resolved' else np.ones(len(tr), bool)
                    if use.sum() < 200:
                        preds.append(np.full(len(te), ytr.mean())); continue
                    m = make_model('mlp', rep).fit(Xtr[use], ytr[use])
                    p = m.predict(Xte)
                    if feat == 'resolved':
                        p = np.where(okte, p, ytr[use].mean())
                    preds.append(p)
                P = np.array(preds)
                e = np.rad2deg(P - yte[None, :])
                rb = [float(np.sqrt(np.mean(e[:, (Rte >= a) & (Rte < b)] ** 2))) for a, b in bins]
                key = f'T={T}|{name}|{feat}'
                res[key] = dict(rmse_all=float(np.sqrt(np.mean(e ** 2))), rmse_by_range=rb,
                                tracked_fraction=float(okte.mean()) if feat == 'resolved' else 1.0,
                                tracked_by_range=[float(okte[(Rte >= a) & (Rte < b)].mean()) for a, b in bins] if feat == 'resolved' else None)
                print(key, f'{res[key]["rmse_all"]:.2f} deg', [round(v, 1) for v in rb],
                      f'tracked {res[key]["tracked_fraction"]:.2f}' if feat == 'resolved' else '', flush=True)
        res[f'T={T}|mean_predictor'] = float(np.rad2deg(np.sqrt(np.mean((yte - ytr.mean()) ** 2))))
        print(f'T={T} mean predictor {res[f"T={T}|mean_predictor"]:.2f} deg', flush=True)
    res['range_bins_m'] = [[1, 1.5], [1.5, 2], [2, 2.5], [2.5, 3], [3, 3.5], [3.5, 4]]
    save_json(OUT / 'inclination.json', res)
    provenance(OUT, 'study10', dict(levels_T_per_rtHz_at_1p1kHz={k: v for k, v in LEVELS.items()}, T_s=TS, train=3072, test=1536,
                                    repeats=3, source='realistic_model.Src (5-inch class, measured ratio/wander)', node='one triaxial point node'))

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
