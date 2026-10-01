"""How much does the measured impulsive (Class A-like) background cost a narrowband line detector?

Empirical tails of the actual detection statistics used in the array model, on the measured grid background:
  coherent    z = |X(f)|^2 / sigma^2(f), 2 s rectangular window (0.5 Hz bins)      -> Exp(1) if Gaussian
  incoherent  z = sum over 40 x 0.05 s frames of |X|^2 / sigma^2 (20 Hz bins)      -> Gamma(40, 1) if Gaussian
sigma^2(f): per-bin median over the record / ln 2 (robust).  Channel/cell pairs with the drone >= 4 m away;
bands 1.5-4.5 kHz and 7-9.5 kHz, bins near mains harmonics, the 5.93 kHz line and any bin whose record-mean z
exceeds 1.5 (residual stable lines) excluded.
Pre-processing variants (all after the order-tracked synchronous comb, which is part of the baseline chain):
  comb        comb only
  comb+dg     single-sample glitch repair first
  comb+dg+bl  + blanking: band-passed residual samples with |y| > 4.5 sigma_MAD zeroed (+-0.5 ms guard)
  comb+dg+cl  + soft clipping at 3 sigma_MAD
Output: tail counts -> threshold inflation at the per-cell Pfa used by the array model and the equivalent SNR loss.
"""
import os, sys
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter
from concurrent.futures import ProcessPoolExecutor
sys.argv = sys.argv[:2]
import periodic_cancel as PC
import sync_comb as SCB
from classA_fit import deglitch, ANT, CXY

FS = PC.FS
NW, NF = 40000, 1000
E_COH = np.concatenate([np.linspace(0, 5, 51)[:-1], np.logspace(np.log10(5), 4, 400)])
E_INC = np.concatenate([np.linspace(0, 80, 801)[:-1], np.logspace(np.log10(80), 5, 400)])
VARIANTS = ('raw', 'blank', 'clip', 'gauss')     # all after glitch repair + synchronous comb + 300-9800 Hz band-pass
SOS = butter(4, [300, 9800], btype='band', fs=FS, output='sos')

def keep(f):
    k = ((f >= 1500) & (f <= 4500)) | ((f >= 7000) & (f <= 9500))
    k &= np.abs(f - 5930) > 40
    k &= np.abs(f - 50.05 * np.round(f / 50.05)) >= 3.0
    return k

def stats(y, norm):
    """norm = 'record': per-bin median over the record; 'cfar': per window, running median over +-25 Hz (coherent)
    or +-200 Hz (incoherent frames) of neighbouring bins in the same window (cell-averaging CFAR, median form)."""
    n = (len(y) // NW) * NW; y = y[:n]
    X = np.fft.rfft(y.reshape(-1, NW), axis=1); f = np.fft.rfftfreq(NW, 1 / FS)
    Pf = np.abs(X) ** 2; k = keep(f)
    if norm == 'cfar':
        zc = (Pf / (median_filter(Pf, size=(1, 101), mode='nearest') / np.log(2)))[:, k]
    else:
        P = Pf[:, k]; zc = P / (np.median(P, axis=0, keepdims=True) / np.log(2))
    zc = zc[:, zc.mean(0) < 1.5]
    Y = np.fft.rfft(y.reshape(-1, NF), axis=1); g = np.fft.rfftfreq(NF, 1 / FS)
    Qf = np.abs(Y) ** 2; kg = keep(g)
    if norm == 'cfar':
        Q = (Qf / (median_filter(Qf, size=(1, 21), mode='nearest') / np.log(2)))[:, kg]
    else:
        Q = Qf[:, kg]; Q = Q / (np.median(Q, axis=0, keepdims=True) / np.log(2))
    gq = Q.mean(0) < 1.5
    zi = Q[:, gq][: (Q.shape[0] // 40) * 40].reshape(-1, 40, gq.sum()).sum(1)
    return np.histogram(zc, E_COH)[0], np.histogram(zi, E_INC)[0], np.median(zc, 1)

def run(args):
    c, ch = args
    x = deglitch(PC.load(c, ch))
    theta, _, _ = PC.mains_phase(x)
    y0 = sosfiltfilt(SOS, SCB.sync_comb(x, theta)[int(5 * FS):])
    s = 1.4826 * np.median(np.abs(y0))
    rng = np.random.default_rng(c * 13 + ord(ch))
    out = {}
    for v in VARIANTS:
        frac = 0.0
        if v == 'gauss':
            y = sosfiltfilt(SOS, rng.normal(size=len(y0)))
        elif v == 'blank':
            m = np.abs(y0) > 4.5 * s
            m = np.convolve(m.astype(float), np.ones(21), 'same') > 0
            frac = m.mean(); y = np.where(m, 0.0, y0)
        elif v == 'clip':
            frac = np.mean(np.abs(y0) > 3 * s); y = np.clip(y0, -3 * s, 3 * s)
        else:
            y = y0
        for norm in ('record', 'cfar'):
            hc, hi, wmed = stats(y, norm)
            out[(v, norm)] = dict(hc=hc, hi=hi, frac=frac, wmed=wmed)
    return (c, ch), out

if __name__ == '__main__':
    jobs = [(c, ch) for c in range(1, 26) for ch in 'ABCD'
            if np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]) >= 4.0]
    with ProcessPoolExecutor(max_workers=16) as ex:
        R = dict(ex.map(run, jobs))
    np.save(os.path.join(PC.HERE if hasattr(PC, 'HERE') else '.', 'env', 'detstat_tails.npy'),
            dict(R=R, E_COH=E_COH, E_INC=E_INC), allow_pickle=True)
    print(f'{len(jobs)} channel-records with the drone >= 4 m away')
