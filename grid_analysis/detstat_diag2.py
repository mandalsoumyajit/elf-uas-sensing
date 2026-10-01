"""Are the residual narrowband exceedances (CFAR, 2 s coherent) drone harmonics or background interferers?
For each far channel-record: exceedances z > 40 (Gaussian P ~ 4e-18) -> frequency, window;
tests: rate vs drone distance; f / f_ref clustering near integers (drone harmonics; f_ref = tracked reference
motor fundamental of that cell); persistence across adjacent windows; same exceedance on the other channels."""
import os, sys
import numpy as np
from scipy.signal import sosfiltfilt
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor
sys.argv = sys.argv[:2]
import periodic_cancel as PC
import sync_comb as SCB
from classA_fit import deglitch, ANT, CXY
import detstat_tails as DT

FS = PC.FS

def zmap(c, ch):
    x = deglitch(PC.load(c, ch)); th, _, _ = PC.mains_phase(x)
    y = sosfiltfilt(DT.SOS, SCB.sync_comb(x, th)[int(5 * FS):])
    n = (len(y) // DT.NW) * DT.NW
    Pf = np.abs(np.fft.rfft(y[:n].reshape(-1, DT.NW), axis=1)) ** 2
    return Pf / (median_filter(Pf, size=(1, 101), mode='nearest') / np.log(2))

def run(c):
    f = np.fft.rfftfreq(DT.NW, 1 / FS); k = DT.keep(f)
    Z = {ch: zmap(c, ch) for ch in 'ABCD'}
    nw = min(z.shape[0] for z in Z.values())
    out = []
    for ch in 'ABCD':
        d = np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1])
        z = Z[ch][:nw]; kk = k & (z.mean(0) < 1.5)
        w, b = np.nonzero((z > 40) & kk[None, :])
        for wi, bi in zip(w, b):
            pers = max(z[max(wi - 1, 0), bi - 2:bi + 3].max() if wi > 0 else 0, z[min(wi + 1, nw - 1), bi - 2:bi + 3].max() if wi < nw - 1 else 0)
            other = max(Z[o][wi, bi - 2:bi + 3].max() for o in 'ABCD' if o != ch)
            out.append((c, ch, d, f[bi], wi, z[wi, bi], pers, other))
    return c, out, {ch: (np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]), int((k & (Z[ch][:nw].mean(0) < 1.5)).sum() * nw)) for ch in 'ABCD'}

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=13) as ex:
        R = list(ex.map(run, range(1, 26)))
    ev = np.array([e[2:] for _, o, _ in R for e in o], float)          # d, f, w, z, pers, other
    cells = np.array([e[0] for _, o, _ in R for e in o]); chs = np.array([e[1] for _, o, _ in R for e in o])
    tot = {(c, ch): v for c, _, n in R for ch, v in n.items()}
    fref = {c: float(PC.A[c]['fc']) for c in range(1, 26)}
    print(f'{len(ev)} exceedances z > 40 (2 s CFAR coherent, bands 1.5-4.5 & 7-9.5 kHz) over all 100 channel-records')
    for lo, hi in ((0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8)):
        sel = [(c, ch) for (c, ch), (d, n) in tot.items() if lo <= d < hi]
        ncell = sum(tot[s][1] for s in sel); ne = sum(1 for c, ch in zip(cells, chs) if (c, ch) in sel)
        print(f'  drone-antenna distance {lo}-{hi} m: {len(sel):3d} records, exceedance rate {ne / max(ncell, 1):.2e} per bin-window')
    ratio = ev[:, 1] / np.array([fref[c] for c in cells])
    frac = np.abs(ratio - np.round(ratio))
    rnd = np.random.default_rng(0).uniform(1.2, 9, 200000); rfr = np.abs(rnd - np.round(rnd))
    print(f'  |f/f_ref - nearest integer| < 0.05: {np.mean(frac < 0.05):.2f} of exceedances (uniform expectation {np.mean(rfr < 0.05):.2f})')
    print(f'  persistent (same +-1 Hz in an adjacent window with z > 10): {np.mean(ev[:, 4] > 10):.2f}')
    print(f'  seen on another channel in the same window (+-1 Hz, z > 10): {np.mean(ev[:, 5] > 10):.2f}')
    far = ev[:, 0] >= 4
    print(f'  far (>= 4 m) only: n={far.sum()}, near-integer {np.mean(frac[far] < 0.05):.2f}, persistent {np.mean(ev[far, 4] > 10):.2f}, '
          f'other-channel {np.mean(ev[far, 5] > 10):.2f}')
    h, e = np.histogram(ev[far, 1], bins=np.arange(1500, 9600, 500))
    print('  far exceedances by frequency (500 Hz bins):', dict(zip(e[:-1].astype(int), h)))
    np.save(os.path.join(DT.__file__.rsplit(os.sep, 1)[0] if os.sep in DT.__file__ else '.', 'env', 'detstat_diag2.npy'),
            dict(ev=ev, cells=cells, chs=chs), allow_pickle=True)
