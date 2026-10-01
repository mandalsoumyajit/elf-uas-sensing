"""Where do the large coherent-statistic values come from: impulses (all bins at once), wandering lines
(frequency-clustered, time-persistent) or nonstationary floor?  One far channel per record for a few records."""
import os, sys
import numpy as np
from scipy.signal import sosfiltfilt
sys.argv = sys.argv[:2]
import periodic_cancel as PC
import sync_comb as SCB
from classA_fit import deglitch
import detstat_tails as DT

FS = PC.FS
for c, ch in ((1, 'C'), (5, 'D'), (25, 'A'), (21, 'B'), (13, 'C')):
    x = deglitch(PC.load(c, ch)); th, _, _ = PC.mains_phase(x)
    y = sosfiltfilt(DT.SOS, SCB.sync_comb(x, th)[int(5 * FS):])
    n = (len(y) // DT.NW) * DT.NW
    X = np.fft.rfft(y[:n].reshape(-1, DT.NW), axis=1); f = np.fft.rfftfreq(DT.NW, 1 / FS)
    k = DT.keep(f); P = np.abs(X[:, k]) ** 2; fk = f[k]
    z = P / (np.median(P, 0, keepdims=True) / np.log(2)); good = z.mean(0) < 1.5; z = z[:, good]; fk = fk[good]
    big = z > 30
    nb_t = big.sum(1)                                  # exceedances per window
    nb_f = big.sum(0)                                  # exceedances per bin
    # cluster: fraction of exceedances whose bin has a neighbour (+-2 bins) exceeding 10 in the same window
    w, b = np.nonzero(big)
    nbr = np.array([(z[wi, max(0, bi - 3):bi + 4] > 10).sum() - 1 for wi, bi in zip(w, b)])
    # frequency histogram of exceedances (100 Hz)
    hist, ed = np.histogram(fk[b], bins=np.arange(1500, 9600, 250))
    top = np.argsort(hist)[::-1][:4]
    # window-level power fluctuation
    pw = np.median(z, 1)
    print(f'cell {c:2d} {ch}: exceed(z>30) {big.mean():.1e} (Gauss {np.exp(-30):.0e}); per-window counts: max {nb_t.max()}, '
          f'windows with >0: {np.mean(nb_t > 0):.2f}; with neighbours>10: {np.mean(nbr > 0):.2f}; '
          f'window median-z spread {pw.min():.2f}-{pw.max():.2f}; busiest 250 Hz bands: '
          + ', '.join(f'{ed[i]:.0f}({hist[i]})' for i in top), flush=True)
    # time-domain kurtosis of the band-passed residual and of 10 ms rms
    r = y[: (len(y) // 200) * 200].reshape(-1, 200).std(1)
    print(f'          residual kurtosis {np.mean(y**4)/np.mean(y**2)**2:.1f}; 10 ms rms: p99/p50 {np.percentile(r,99)/np.median(r):.2f}, '
          f'max/p50 {r.max()/np.median(r):.1f}', flush=True)
