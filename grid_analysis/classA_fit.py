"""Middleton Class A fit to narrowband envelope statistics of the background.

Envelope samples: STFT (0.1 s Hann, 10 Hz bins) inter-harmonic bins in three bands, deglitched data,
channel/cell combinations with the drone >= 4 m from the antenna. Each bin normalized by its whole-record
rms ('global') or per 10 s ('local'), so E[r^2] = 1.

Class A envelope model (narrowband receiver output), normalized power:
    p(r) = sum_m exp(-A) A^m/m! * (2 r / W_m) exp(-r^2 / W_m),   W_m = (m/A + G) / (1 + G)
    APD(x) = sum_m exp(-A) A^m/m! * exp(-x^2 / W_m)
A: impulsive index, G (Gamma): Gaussian-to-impulsive power ratio. MLE over (log A, log G).
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import stft
from scipy.optimize import minimize
from scipy.special import gammaln
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'env')
EXP = sys.argv[1]
FS = 20000.0
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
BANDS = {'150-950 Hz': (150, 950), '1.5-4.5 kHz': (1500, 4500), '7-9.5 kHz': (7000, 9500)}
MMAX = 60

def deglitch(x):
    x = x.copy()
    for _ in range(2):
        s = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2)
        nb = 0.5 * (x[:-2] + x[2:])
        sp = np.flatnonzero((np.abs(x[1:-1] - nb) > 10 * s) & (np.abs(x[:-2] - x[2:]) < 3 * s)) + 1
        x[sp] = 0.5 * (x[sp - 1] + x[sp + 1])
    return x

def keep_bins(f, lo, hi):
    k = (f >= lo) & (f <= hi) & (np.abs(f - 5930) > 40)
    for h in range(1, 200):
        k &= np.abs(f - 50.05 * h) >= 15
    return k

def envelopes(args):
    c, ch = args
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)
    x = deglitch(x); x -= x.mean()
    f, t, Z = stft(x, fs=FS, nperseg=2000, noverlap=0, boundary=None, padded=False)
    out = {}
    rng = np.random.default_rng(c * 7 + ord(ch))
    for name, (lo, hi) in BANDS.items():
        Zb = Z[keep_bins(f, lo, hi)]
        g = np.abs(Zb) / np.sqrt(np.mean(np.abs(Zb) ** 2, axis=1, keepdims=True))
        nb = Zb.shape[1] // 100
        Zl = np.abs(Zb[:, : nb * 100]).reshape(Zb.shape[0], nb, 100)
        l = (Zl / np.sqrt(np.mean(Zl ** 2, axis=2, keepdims=True))).reshape(Zb.shape[0], -1)
        # subsample to limit size but keep all large values (tails matter)
        def sub(r):
            r = r.ravel(); big = r[r > 2.5]; small = r[r <= 2.5]
            keep = rng.choice(small, size=min(len(small), 60000), replace=False)
            return keep, big, len(small)
        out[name] = {'global': sub(g), 'local': sub(l)}
    return (c, ch), out

def logw(A):
    m = np.arange(MMAX)
    return -A + m * np.log(A) - gammaln(m + 1)

def negll(p, r_small, r_big, n_small, n_big=None):
    A, G = np.exp(p)
    if not (1e-4 < A < 50 and 1e-6 < G < 1e3):
        return 1e30
    m = np.arange(MMAX); W = (m / A + G) / (1 + G); lw = logw(A)
    def ll(r):
        lp = lw[None, :] + np.log(2 * r[:, None] / W[None, :]) - r[:, None] ** 2 / W[None, :]
        mx = lp.max(1, keepdims=True); return (mx[:, 0] + np.log(np.exp(lp - mx).sum(1)))
    nb = len(r_big) if n_big is None else n_big
    # weighted so that each subsample represents its full population
    return -(ll(r_small).sum() * (n_small / len(r_small)) + ll(r_big).sum() * (nb / len(r_big)))

def fit_combo(args):
    key, rs, rb, ns, nb = args
    best = None
    for p0 in ([np.log(0.1), np.log(1.0)], [np.log(0.01), np.log(0.1)], [np.log(1.0), np.log(10.0)]):
        r = minimize(negll, p0, args=(rs, rb, ns, nb), method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-2, 'maxiter': 800})
        if best is None or r.fun < best.fun:
            best = r
    A, G = np.exp(best.x)
    llr = -((np.log(2 * rs) - rs ** 2).sum() * ns / len(rs) + (np.log(2 * rb) - rb ** 2).sum() * nb / len(rb))
    return key, A, G, llr - best.fun

def apd(x, A, G):
    m = np.arange(MMAX); W = (m / A + G) / (1 + G)
    return (np.exp(logw(A))[None, :] * np.exp(-x[:, None] ** 2 / W[None, :])).sum(1)

if __name__ == '__main__':
    cache = os.path.join(OUT, 'classA_env.npy')
    if os.path.exists(cache):
        E = np.load(cache, allow_pickle=True).item()
    else:
        jobs = [(c, ch) for c in range(1, 26) for ch in 'ABCD'
                if np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]) >= 4.0]
        with ProcessPoolExecutor(max_workers=16) as ex:
            E = dict(ex.map(envelopes, jobs))
        np.save(cache, E, allow_pickle=True)
    rng = np.random.default_rng(5)
    combos, data = [], {}
    for name in BANDS:
        for mode in ('global', 'local'):
            for ch in 'ABCD':
                keys = [k for k in E if k[1] == ch]
                rs_all = np.concatenate([E[k][name][mode][0] for k in keys]); rb_all = np.concatenate([E[k][name][mode][1] for k in keys])
                ns = sum(E[k][name][mode][2] for k in keys); nb = len(rb_all)
                rs = rng.choice(rs_all, min(len(rs_all), 60000), replace=False)
                rb = rng.choice(rb_all, min(nb, 40000), replace=False)
                data[(ch, name, mode)] = (rs_all, rb_all, ns)
                combos.append(((ch, name, mode), rs, rb, ns, nb))
    with ProcessPoolExecutor(max_workers=20) as ex:
        fits = {k: (A, G, d) for k, A, G, d in ex.map(fit_combo, combos)}
    res = {}
    xs = np.linspace(0.01, 7, 300)
    cols = {'A': '#1f6fb4', 'B': '#2ca02c', 'C': '#d1603d', 'D': '#7b4ea3'}
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.0), sharex=True, sharey=True)
    for bi, name in enumerate(BANDS):
        for mi, mode in enumerate(('global', 'local')):
            ax = axs[mi, bi]
            for ch in 'ABCD':
                rs_all, rb_all, ns = data[(ch, name, mode)]; A, G, dll = fits[(ch, name, mode)]
                ntot = ns + len(rb_all)
                emp = np.array([(np.sum(rs_all > x) * ns / len(rs_all) + np.sum(rb_all > x)) / ntot for x in xs])
                e3, e4 = [(np.sum(rs_all > t) * ns / len(rs_all) + np.sum(rb_all > t)) / ntot for t in (3.0, 4.0)]
                f3, f4 = apd(np.array([3.0, 4.0]), A, G)
                res[(ch, name, mode)] = dict(A=A, G=G, dLL=dll, n=ntot, emp3=e3, fit3=f3, emp4=e4, fit4=f4)
                ax.semilogy(20 * np.log10(xs), emp, color=cols[ch], lw=1.3)
                ax.semilogy(20 * np.log10(xs), apd(xs, A, G), color=cols[ch], lw=0.9, ls='--')
            ax.semilogy(20 * np.log10(xs), np.exp(-xs ** 2), 'k:', lw=1)
            ax.set_ylim(1e-6, 1.2); ax.set_xlim(-10, 17); ax.set_title(f'{name}, {mode} norm.', fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel('envelope / rms (dB)')
    for ax in axs[:, 0]:
        ax.set_ylabel('P(envelope > x)')
    axs[0, 0].plot([], [], 'k-', label='measured'); axs[0, 0].plot([], [], 'k--', label='Class A fit'); axs[0, 0].plot([], [], 'k:', label='Rayleigh')
    axs[0, 0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'classA_fit.png'), dpi=170)
    np.save(os.path.join(OUT, 'classA_fit.npy'), res, allow_pickle=True)
    print('Middleton Class A MLE (pooled over far cells). dLL = log-likelihood gain over Rayleigh (full-sample weighted)')
    for (ch, name, mode), v in sorted(res.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        print(f'{ch} {name:12s} {mode:6s}: A = {v["A"]:.3g}, Gamma = {v["G"]:.3g} | dLL {v["dLL"]:10.1f} (n={v["n"]}) | '
              f'P>3rms emp {v["emp3"]:.2e} fit {v["fit3"]:.2e} | P>4rms emp {v["emp4"]:.2e} fit {v["fit4"]:.2e}')
    print(f'Rayleigh: P>3rms {np.exp(-9):.2e}, P>4rms {np.exp(-16):.2e}')

