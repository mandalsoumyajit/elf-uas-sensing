"""Summarise detstat_tails.npy: empirical CCDF of the line-detection statistics, threshold at the array-model
per-cell Pfa, and the equivalent SNR loss at Pd = 0.9 relative to Gaussian noise passed through the identical
chain ('gauss' variant).  K = 4 node sums by convolving the per-node empirical distribution (independent nodes).
Normalisations: 'record' (per-bin median over the whole record) and 'cfar' (per window, running median of
neighbouring bins).  Writes ../../source_model/impulsive_penalty.json for the site model."""
import json, os
import numpy as np
from scipy.stats import chi2, ncx2, gamma
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'env', 'detstat_tails.npy'), allow_pickle=True).item()
R, EC, EI = D['R'], D['E_COH'], D['E_INC']
VARS = ('raw', 'blank', 'clip', 'gauss'); NORMS = ('record', 'cfar')
PFA_COH = (1 / 1800) / (2000 * 2.0)          # array model: 1 FA/h, 2 s windows, 2000 carrier-search cells
PFA_INC = (1 / 1800) / (500 * 0.05)

def ccdf(h, e):
    c = np.cumsum(h[::-1])[::-1] / h.sum(); return e[:-1], c

def thr(e, c, pfa):
    ok = c > 0
    if c[ok][-1] <= pfa:
        i = np.flatnonzero(c <= pfa)[0]
        return float(np.interp(np.log(pfa), [np.log(c[i]), np.log(c[i - 1])], [e[i], e[i - 1]])), False
    sel = ok & (c < 30 * c[ok][-1])
    p = np.polyfit(e[sel], np.log(c[sel]), 1)
    return float((np.log(pfa) - p[1]) / p[0]), True

def need_lambda(eta, dof):
    return brentq(lambda L: ncx2.sf(eta, dof, L) - 0.9, 1e-6, 1e6)

def ksum_ccdf(h, e, K):
    dz = 0.02; z = np.arange(0, 600, dz)
    cdf = np.interp(z, e[1:], np.cumsum(h) / h.sum(), left=0, right=1)
    pmf = np.diff(np.r_[0, cdf]); n = len(pmf) * K
    s = np.maximum(np.fft.irfft(np.fft.rfft(pmf, 2 * n) ** K, 2 * n)[:n], 0)
    return np.arange(n) * dz, np.cumsum(s[::-1])[::-1] / s.sum()

def summarise(hc, hi):
    e, cc = ccdf(hc, EC); t1, x1 = thr(e, cc, PFA_COH)
    zz, c4 = ksum_ccdf(hc, EC, 4); t4, x4 = thr(zz, c4, PFA_COH)
    ei, ci = ccdf(hi, EI); ti, xi = thr(ei, ci, PFA_INC)
    return dict(t1=t1, t4=t4, ti=ti, ext=x1 or x4 or xi, p10=float(cc[np.searchsorted(e, 10.0)]),
                l1=need_lambda(2 * t1, 2), l4=need_lambda(2 * t4, 8), li=need_lambda(2 * ti, 80))

groups = {}
for (c, ch), d in R.items():
    for key, v in d.items():
        for g in (('all',) + key, (ch,) + key):
            a = groups.setdefault(g, dict(hc=0, hi=0, frac=[], wmed=[]))
            a['hc'] = a['hc'] + v['hc']; a['hi'] = a['hi'] + v['hi']; a['frac'].append(v['frac']); a['wmed'].append(v['wmed'])
S = {g: summarise(a['hc'], a['hi']) for g, a in groups.items()}
dB = lambda a, b: 10 * np.log10(a / b)
print(f'Per-cell Pfa: coherent {PFA_COH:.1e}, incoherent {PFA_INC:.1e}. Theory (Gaussian, exact normalisation): '
      f'thr K=1 {chi2.isf(PFA_COH, 2)/2:.1f}, K=4 {chi2.isf(PFA_COH, 8)/2:.1f}, incoherent {chi2.isf(PFA_INC, 80)/2:.1f}')
print('Loss = extra SNR for Pd 0.9 vs Gaussian noise through the same chain (same normalisation).')
pen = {}
for norm in NORMS:
    print(f'\n--- normalisation: {norm} ---')
    for grp in ('all', 'A', 'B', 'C', 'D'):
        G = S[(grp, 'gauss', norm)]
        for v in VARS:
            s = S[(grp, v, norm)]; a = groups[(grp, v, norm)]
            wm = np.concatenate(a['wmed']); spread = np.percentile(wm, 99) / np.median(wm)
            print(f'{grp:3s} {v:6s} P(z>10) {s["p10"]:.1e} | thr K=1 {s["t1"]:6.1f} K=4 {s["t4"]:6.1f} inc {s["ti"]:6.1f}{"*" if s["ext"] else " "}| '
                  f'loss K=1 {dB(s["l1"], G["l1"]):+5.1f} dB, K=4 {dB(s["l4"], G["l4"]):+5.1f} dB, incoherent {dB(s["li"], G["li"]):+5.1f} dB'
                  + (f' | samples blanked/clipped {100*np.mean(a["frac"]):.1f}%' if grp == 'all' and v in ('blank', 'clip') else '')
                  + f' | window-level p99/median {spread:.2f}')
            if grp == 'all':
                pen[(v, norm)] = dict(K4=dB(s['l4'], G['l4']), inc=dB(s['li'], G['li']),
                                      frac=float(np.mean(a['frac'])))
print('* = some threshold extrapolated beyond the empirical tail')
best_mit = min(('raw', 'blank', 'clip'), key=lambda v: pen[(v, 'cfar')]['K4'] - 10 * np.log10(1 - pen[(v, 'cfar')]['frac']))
out = dict(raw={'P3': pen[('raw', 'record')]['K4'], 'PWM': pen[('raw', 'record')]['K4'], 'P1': pen[('raw', 'record')]['inc']},
           mitigated={'P3': pen[(best_mit, 'cfar')]['K4'] - 10 * np.log10(1 - pen[(best_mit, 'cfar')]['frac']),
                      'PWM': pen[(best_mit, 'cfar')]['K4'] - 10 * np.log10(1 - pen[(best_mit, 'cfar')]['frac']),
                      'P1': pen[(best_mit, 'cfar')]['inc'] - 10 * np.log10(1 - pen[(best_mit, 'cfar')]['frac'])},
           note=f'raw = record-normalised, no blanking; mitigated = CFAR + {best_mit} (incl. its signal loss); coherent K=4 statistic')
json.dump(out, open(os.path.join(HERE, '..', '..', 'source_model', 'impulsive_penalty.json'), 'w'), indent=1)
print('\nimpulsive_penalty.json:', out)

fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
cols = {'raw': '0.25', 'blank': '#d1603d', 'clip': '#2ca02c', 'gauss': '#1f6fb4'}
for norm, ls in (('record', '--'), ('cfar', '-')):
    for v in VARS:
        e, cc = ccdf(groups[('all', v, norm)]['hc'], EC); axs[0].semilogy(e, cc, ls, color=cols[v], lw=1.2, label=f'{v}, {norm}')
        ei, ci = ccdf(groups[('all', v, norm)]['hi'], EI); axs[1].semilogy(ei, ci, ls, color=cols[v], lw=1.2)
axs[0].axhline(PFA_COH, color='0.6', lw=0.7, ls=':'); axs[1].axhline(PFA_INC, color='0.6', lw=0.7, ls=':')
axs[0].set_xlim(0, 60); axs[0].set_ylim(1e-9, 1); axs[1].set_xlim(20, 250); axs[1].set_ylim(1e-9, 1)
axs[0].set_xlabel('coherent statistic (2 s window, one node)'); axs[1].set_xlabel('incoherent statistic (40 × 0.05 s)')
axs[0].set_ylabel('P(statistic > x)'); axs[0].legend(fontsize=5.5, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'env', 'detstat_tails.png'), dpi=170)
