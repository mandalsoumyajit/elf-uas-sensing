"""Amplitude statistics of the background and impulsive events.

(1) Narrowband statistics: STFT (0.1 s Hann, non-overlapping, 10 Hz bins); keep inter-harmonic bins
    (>= 15 Hz from 50.05*k, away from 5.93 kHz and drone bands) in three bands:
      B1 150-950 Hz, B2 1.5-4.5 kHz, B3 7-9.5 kHz.
    Per bin, the complex series is normalized either by its whole-record rms ('global': includes slow level
    changes) or by its rms in each 10 s block ('local': short-term shape). Pooled statistics per band:
    kurtosis of Re/Im (Gaussian 3), V_d = 20log10(rms env / mean env) (Rayleigh 1.049 dB), and the APD
    P(|z|/rms > x) for comparison with exp(-x^2).
    Only channel/cell combinations with the drone >= 4 m from the antenna are used.
(2) Impulses: 7-9.5 kHz band-pass on all four channels of selected cells; events = excursions > 8 sigma
    (sigma from the median absolute deviation), merged within 5 ms; coincidence across channels within 2 ms
    compared with the chance rate.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import stft, butter, sosfiltfilt
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]
OUT = os.path.join(HERE, 'env')
FS = 20000.0
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
BANDS = {'B1 150-950 Hz': (150, 950), 'B2 1.5-4.5 kHz': (1500, 4500), 'B3 7-9.5 kHz': (7000, 9500)}
XG = np.linspace(0, 6, 61)

DEGLITCH = '--raw' not in sys.argv
def deglitch(x):
    """Replace single-sample spikes (> 10 sigma from the neighbour mean while the neighbours agree)."""
    x = x.copy()
    for _ in range(2):
        s = 1.4826 * np.median(np.abs(np.diff(x))) / np.sqrt(2)
        nb = 0.5 * (x[:-2] + x[2:])
        sp = np.flatnonzero((np.abs(x[1:-1] - nb) > 10 * s) & (np.abs(x[:-2] - x[2:]) < 3 * s)) + 1
        x[sp] = 0.5 * (x[sp - 1] + x[sp + 1])
    return x

def load(c, ch):
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)
    if DEGLITCH:
        x = deglitch(x)
    return x - x.mean()

def keep_bins(f, lo, hi):
    k = (f >= lo) & (f <= hi) & (np.abs(f - 5930) > 40)
    for h in range(1, 200):
        k &= np.abs(f - 50.05 * h) >= 15
    return k

def band_stats(Z):
    """Z: bins x time complex (already normalized). Returns kurtosis(Re,Im), Vd, APD exceedance on XG."""
    re = np.concatenate([Z.real.ravel(), Z.imag.ravel()])
    k = np.mean(re ** 4) / np.mean(re ** 2) ** 2
    env = np.abs(Z).ravel()
    vd = 20 * np.log10(np.sqrt(np.mean(env ** 2)) / np.mean(env))
    r = env / np.sqrt(np.mean(env ** 2))
    apd = np.array([np.mean(r > x) for x in XG])
    return k, vd, apd

def per_file(args):
    c, ch = args
    x = load(c, ch)
    f, t, Z = stft(x, fs=FS, nperseg=2000, noverlap=0, boundary=None, padded=False)
    out = {}
    for name, (lo, hi) in BANDS.items():
        Zb = Z[keep_bins(f, lo, hi)]
        g = Zb / np.sqrt(np.mean(np.abs(Zb) ** 2, axis=1, keepdims=True))
        nb = Zb.shape[1] // 100
        Zl = Zb[:, : nb * 100].reshape(Zb.shape[0], nb, 100)
        loc = (Zl / np.sqrt(np.mean(np.abs(Zl) ** 2, axis=2, keepdims=True))).reshape(Zb.shape[0], -1)
        out[name] = {'global': band_stats(g), 'local': band_stats(loc)}
    return (c, ch), out

SOS = butter(6, [7000, 9500], btype='band', fs=FS, output='sos')
def events(y):
    s = 1.4826 * np.median(np.abs(y))
    idx = np.flatnonzero(np.abs(y) > 8 * s)
    if len(idx) == 0:
        return np.array([]), s
    starts = idx[np.r_[True, np.diff(idx) > int(0.005 * FS)]]
    return starts / FS, s

def impulse_cell(c):
    ys = {ch: sosfiltfilt(SOS, load(c, ch)) for ch in 'ABCD'}
    n = min(len(v) for v in ys.values()); T = n / FS
    ev = {ch: events(ys[ch][:n])[0] for ch in 'ABCD'}
    win = 0.002
    res = {'T': T, 'rates': {ch: len(ev[ch]) / T for ch in 'ABCD'}}
    # fraction of events on channel ch that coincide with >=1 event on each other channel
    co = {}
    for a in 'ABCD':
        for b in 'ABCD':
            if a >= b or len(ev[a]) == 0 or len(ev[b]) == 0:
                continue
            j = np.searchsorted(ev[b], ev[a]); hit = np.zeros(len(ev[a]), bool)
            for off in (-1, 0):
                jj = np.clip(j + off, 0, len(ev[b]) - 1); hit |= np.abs(ev[b][jj] - ev[a]) <= win
            chance = 1 - np.exp(-len(ev[b]) / T * 2 * win)
            co[a + b] = (hit.mean(), chance)
    # events seen on all four channels
    if all(len(ev[ch]) for ch in 'ABCD'):
        base = ev['A']; allfour = np.ones(len(base), bool)
        for b in 'BCD':
            j = np.searchsorted(ev[b], base); ok = np.zeros(len(base), bool)
            for off in (-1, 0):
                jj = np.clip(j + off, 0, len(ev[b]) - 1); ok |= np.abs(ev[b][jj] - base) <= win
            allfour &= ok
        res['all4_rate'] = allfour.sum() / T
    res['coinc'] = co
    return c, res

if __name__ == '__main__':
    jobs = [(c, ch) for c in range(1, 26) for ch in 'ABCD'
            if np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]) >= 4.0]
    with ProcessPoolExecutor(max_workers=16) as ex:
        dist = dict(ex.map(per_file, jobs))
        imp = dict(ex.map(impulse_cell, [1, 5, 13, 21, 25, 7, 19]))
    np.save(os.path.join(OUT, 'env_dist.npy'), {'dist': dist, 'imp': imp, 'XG': XG}, allow_pickle=True)
    print(f'{len(jobs)} channel/cell combinations (drone >= 4 m)')
    for name in BANDS:
        for mode in ('global', 'local'):
            K = np.array([dist[j][name][mode][0] for j in dist]); V = np.array([dist[j][name][mode][1] for j in dist])
            print(f'{name:16s} {mode:6s}: kurtosis median {np.median(K):.2f} (range {K.min():.2f}-{K.max():.2f}); '
                  f'Vd median {np.median(V):.2f} dB (range {V.min():.2f}-{V.max():.2f}); Gaussian 3.00 / 1.05 dB')
    for c, r in imp.items():
        co = ', '.join(f'{k}:{v[0]:.2f}(chance {v[1]:.3f})' for k, v in r['coinc'].items())
        print(f'cell {c:2d} impulses/s ' + ' '.join(f'{ch}:{v:.2f}' for ch, v in r['rates'].items())
              + f' | on all four: {r.get("all4_rate", 0):.3f}/s | pairwise coincidence {co}')
