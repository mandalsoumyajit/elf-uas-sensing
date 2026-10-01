"""Environmental background characterization from the raw grid data (drone always present, so we use
channel/cell combinations where the drone is far from the antenna(s) concerned).

Outputs (in ./env/):
  env_stats.npz  per-file time series: 1 s broadband floor (2-4.5 and 7-9.5 kHz inter-harmonic medians),
                 10 s grid frequency, 50/150/250 Hz amplitudes, 10 s kurtosis / outlier rate (7-9.5 kHz band),
                 1 s 50 Hz phase (for cross-channel sync check)
  env_coh.npz    coherence spectra for the six antenna pairs (cells far from both antennas)
Figures and a printed summary are produced by env_report.py.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import welch, csd, butter, sosfiltfilt, stft
from scipy.stats import kurtosis
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]
OUT = os.path.join(HERE, 'env'); os.makedirs(OUT, exist_ok=True)
FS = 20000.0
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
def dist(c, ch):
    return float(np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]))
def load(c, ch):
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float); return x - x.mean()

def harm_mask(f, width):
    m = np.zeros_like(f, bool)
    for k in range(1, 200):
        m |= np.abs(f - 50.05 * k) <= width
    return m

SOS_HI = butter(6, [7000, 9500], btype='band', fs=FS, output='sos')

def per_file(args):
    c, ch = args
    x = load(c, ch)
    # 1 s floor levels (inter-harmonic medians)
    f1, t1, Z = stft(x, fs=FS, nperseg=20000, noverlap=0, boundary=None, padded=False)
    P = np.abs(Z) ** 2 / (FS * 0.375 * 20000 / 20000)          # rough density scaling (Hann), counts^2/Hz
    ih = ~harm_mask(f1, 3.0) & (np.abs(f1 - 5930) > 30)
    b1 = ih & (f1 > 2000) & (f1 < 4500) & ~((f1 > 950) & (f1 < 1450))
    b2 = ih & (f1 > 7000) & (f1 < 9500)
    floor_mid = np.median(P[b1], axis=0); floor_hi = np.median(P[b2], axis=0)
    # 50 Hz phase per 1 s (bin exactly at 50 Hz on 1 s windows is 50 Hz; use complex demod at measured mains freq)
    # 10 s grid frequency + harmonic amplitudes via zero-padded FFT peak interpolation
    L = int(10 * FS); nw = len(x) // L
    ff, a50, a150, a250, kurt, outl = [], [], [], [], [], []
    xh = sosfiltfilt(SOS_HI, x)
    win = np.hanning(L)
    for k in range(nw):
        seg = x[k * L:(k + 1) * L] * win
        F = np.fft.rfft(seg, n=8 * L); fq = np.fft.rfftfreq(8 * L, 1 / FS)
        def peak(f0, w=0.5):
            m = (fq > f0 - w) & (fq < f0 + w); i = np.argmax(np.abs(F[m])); return fq[m][i], 2 * np.abs(F[m][i]) / win.sum()
        f50, amp50 = peak(50.05)
        ff.append(f50); a50.append(amp50); a150.append(peak(3 * f50)[1]); a250.append(peak(5 * f50)[1])
        s = xh[k * L:(k + 1) * L]
        kurt.append(kurtosis(s, fisher=False)); outl.append(np.mean(np.abs(s) > 6 * np.std(s)))
    # 1 s 50 Hz phase using the record-mean mains frequency
    fm = np.median(ff)
    n1 = int(FS); m1 = len(x) // n1
    ph = np.angle((x[: m1 * n1] * np.exp(-2j * np.pi * fm * np.arange(m1 * n1) / FS)).reshape(m1, n1).mean(1))
    return (c, ch), dict(floor_mid=floor_mid, floor_hi=floor_hi, fgrid=np.array(ff), a50=np.array(a50), a150=np.array(a150),
                         a250=np.array(a250), kurt=np.array(kurt), outl=np.array(outl), ph50=ph, fm=fm)

PAIRS = [('A', 'B'), ('C', 'D'), ('A', 'D'), ('B', 'C'), ('A', 'C'), ('B', 'D')]
def per_pair(args):
    a, b, c = args
    x, y = load(c, a), load(c, b); n = min(len(x), len(y)); x, y = x[:n], y[:n]
    f, pxx = welch(x, fs=FS, nperseg=2 ** 15); _, pyy = welch(y, fs=FS, nperseg=2 ** 15); _, pxy = csd(x, y, fs=FS, nperseg=2 ** 15)
    return (a, b, c), f, pxx, pyy, pxy

if __name__ == '__main__':
    jobs = [(c, ch) for c in range(1, 26) for ch in 'ABCD']
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = dict(ex.map(per_file, jobs))
    np.save(os.path.join(OUT, 'env_stats.npy'), res, allow_pickle=True)
    print('per-file stats done', flush=True)
    pj = []
    for a, b in PAIRS:
        far = [c for c in range(1, 26) if min(dist(c, a), dist(c, b)) >= 3.5]
        pj += [(a, b, c) for c in far]
    with ProcessPoolExecutor(max_workers=16) as ex:
        out = list(ex.map(per_pair, pj))
    f = out[0][1]
    coh = {}
    for (a, b, c), _, pxx, pyy, pxy in out:
        d = coh.setdefault((a, b), {'pxx': 0, 'pyy': 0, 'pxy': 0, 'cells': []})
        d['pxx'] = d['pxx'] + pxx; d['pyy'] = d['pyy'] + pyy; d['pxy'] = d['pxy'] + pxy; d['cells'].append(c)
        d.setdefault('g2_each', []).append(np.abs(pxy) ** 2 / (pxx * pyy))
    np.save(os.path.join(OUT, 'env_coh.npy'), {'f': f, 'coh': coh}, allow_pickle=True)
    print('coherence done; pairs/cells:', {k: v['cells'] for k, v in coh.items()})
