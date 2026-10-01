"""Detector penalty vs the fraction c of narrowband background clutter removed/vetoed by witness sensors.
Mixture model on the measured CFAR-normalised 2 s coherent statistic (one node):
    CCDF_c(z) = G(z) + (1 - c) * (CCDF_meas(z) - G(z)),   G = Gaussian through the same chain
then K = 4 node sum (independent nodes), threshold at the array-model Pfa, SNR loss at Pd 0.9 vs Gaussian.
Writes 'clutter_vs_coverage' into source_model/impulsive_penalty.json."""
import json, os
import numpy as np
from scipy.stats import ncx2
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'env', 'detstat_tails.npy'), allow_pickle=True).item()
EC = D['E_COH']; PFA = (1 / 1800) / (2000 * 2.0)

def ksum_ccdf(p, K=4):
    dz = 0.02; z = np.arange(0, 600, dz)
    cdf = np.interp(z, EC[1:], np.cumsum(p) / p.sum(), left=0, right=1)
    pmf = np.diff(np.r_[0, cdf]); n = len(pmf) * K
    s = np.maximum(np.fft.irfft(np.fft.rfft(pmf, 2 * n) ** K, 2 * n)[:n], 0)
    return np.arange(n) * dz, np.cumsum(s[::-1])[::-1] / s.sum()

def thr(e, c):
    i = np.flatnonzero(c <= PFA)[0]
    return float(np.interp(np.log(PFA), [np.log(c[i]), np.log(c[i - 1])], [e[i], e[i - 1]]))

lam = lambda eta: brentq(lambda L: ncx2.sf(eta, 8, L) - 0.9, 1e-6, 1e6)
hm = sum(d[('raw', 'cfar')]['hc'] for d in D['R'].values()).astype(float)
hg = sum(d[('gauss', 'cfar')]['hc'] for d in D['R'].values()).astype(float)
pm, pg = hm / hm.sum(), hg / hg.sum()
tg = thr(*ksum_ccdf(pg)); lg = lam(2 * tg)
out = {}
for c in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0):
    p = np.maximum(pg + (1 - c) * (pm - pg), 0)
    t = thr(*ksum_ccdf(p)); out[f'{c:.2f}'] = float(10 * np.log10(lam(2 * t) / lg))
    print(f'clutter removed {c:4.2f}: K=4 threshold {t:6.1f} (Gaussian {tg:.1f}), penalty {out[f"{c:.2f}"]:+.2f} dB')
fn = os.path.join(HERE, '..', '..', 'source_model', 'impulsive_penalty.json')
J = json.load(open(fn)); J['clutter_vs_coverage'] = out
json.dump(J, open(fn, 'w'), indent=1)
