"""Attribute 7-9.5 kHz impulsive events to acquisition artefacts (held-sample runs, spikes) or to the environment,
and recompute narrowband statistics after excluding artefact-affected time.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, stft
from concurrent.futures import ProcessPoolExecutor

EXP = sys.argv[1]
FS = 20000.0
SOS = butter(6, [7000, 9500], btype='band', fs=FS, output='sos')
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
BANDS = {'B1 150-950 Hz': (150, 950), 'B2 1.5-4.5 kHz': (1500, 4500), 'B3 7-9.5 kHz': (7000, 9500)}

def load(c, ch):
    return sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)

def artefacts(x, runlen=3):
    d = np.diff(x)
    runs = np.flatnonzero(np.convolve((d == 0).astype(int), np.ones(runlen - 1, int), 'valid') == runlen - 1)
    s = 1.4826 * np.median(np.abs(d)) / np.sqrt(2)
    nb = 0.5 * (x[:-2] + x[2:])
    sp = np.flatnonzero((np.abs(x[1:-1] - nb) > 10 * s) & (np.abs(x[:-2] - x[2:]) < 3 * s)) + 1
    return np.unique(np.r_[runs, sp])

def events(y):
    s = 1.4826 * np.median(np.abs(y))
    idx = np.flatnonzero(np.abs(y) > 8 * s)
    if len(idx) == 0:
        return np.array([], int)
    return idx[np.r_[True, np.diff(idx) > int(0.005 * FS)]]

def keep_bins(f, lo, hi):
    k = (f >= lo) & (f <= hi) & (np.abs(f - 5930) > 40)
    for h in range(1, 200):
        k &= np.abs(f - 50.05 * h) >= 15
    return k

def one(args):
    c, ch = args
    x = load(c, ch); T = len(x) / FS
    art = artefacts(x)
    y = sosfiltfilt(SOS, x - x.mean())
    ev = events(y)
    if len(ev) and len(art):
        j = np.searchsorted(art, ev); near = np.zeros(len(ev), bool)
        for off in (-1, 0):
            jj = np.clip(j + off, 0, len(art) - 1); near |= np.abs(art[jj] - ev) <= int(0.002 * FS)
    else:
        near = np.zeros(len(ev), bool)
    clean_ev = ev[~near]
    # event shape for clean events: duration above 3 sigma of envelope
    s = 1.4826 * np.median(np.abs(y)); durs = []; peaks = []
    for e in clean_ev[:200]:
        seg = np.abs(y[max(0, e - 200): e + 400]); durs.append(np.sum(seg > 3 * s) / FS * 1e3); peaks.append(seg.max() / s)
    # narrowband stats with 0.1 s frames containing artefacts removed
    f, t, Z = stft(x - x.mean(), fs=FS, nperseg=2000, noverlap=0, boundary=None, padded=False)
    bad = np.zeros(Z.shape[1], bool); bad[np.clip(art // 2000, 0, Z.shape[1] - 1)] = True
    stats = {}
    for name, (lo, hi) in BANDS.items():
        Zb = Z[keep_bins(f, lo, hi)][:, ~bad]
        nb = Zb.shape[1] // 100
        Zl = Zb[:, : nb * 100].reshape(Zb.shape[0], nb, 100)
        loc = (Zl / np.sqrt(np.mean(np.abs(Zl) ** 2, axis=2, keepdims=True))).reshape(Zb.shape[0], -1)
        g = Zb / np.sqrt(np.mean(np.abs(Zb) ** 2, axis=1, keepdims=True))
        def st(Zn):
            re = np.concatenate([Zn.real.ravel(), Zn.imag.ravel()]); env = np.abs(Zn).ravel()
            return np.mean(re ** 4) / np.mean(re ** 2) ** 2, 20 * np.log10(np.sqrt(np.mean(env ** 2)) / np.mean(env))
        stats[name] = {'global': st(g), 'local': st(loc), 'frac_bad': bad.mean()}
    return (c, ch), dict(art_rate=len(art) / T, ev_rate=len(ev) / T, ev_art_frac=near.mean() if len(ev) else np.nan,
                         clean_rate=len(clean_ev) / T, clean_dur=np.median(durs) if durs else np.nan,
                         clean_peak=np.median(peaks) if peaks else np.nan, stats=stats)

if __name__ == '__main__':
    jobs = [(c, ch) for c in range(1, 26) for ch in 'ABCD']
    with ProcessPoolExecutor(max_workers=16) as ex:
        R = dict(ex.map(one, jobs))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env', 'glitch_events.npy'), R, allow_pickle=True)
    for ch in 'ABCD':
        v = [R[(c, ch)] for c in range(1, 26)]
        print(f'ch {ch}: artefact samples/s median {np.median([r["art_rate"] for r in v]):6.1f} | 7-9.5 kHz events/s median '
              f'{np.median([r["ev_rate"] for r in v]):5.2f}, of which near an artefact {np.nanmedian([r["ev_art_frac"] for r in v]):.2f} | '
              f'clean events/s median {np.median([r["clean_rate"] for r in v]):.2f} (range {min(r["clean_rate"] for r in v):.2f}-{max(r["clean_rate"] for r in v):.2f}), '
              f'duration {np.nanmedian([r["clean_dur"] for r in v]):.2f} ms, peak {np.nanmedian([r["clean_peak"] for r in v]):.1f} sigma')
    print('\nNarrowband statistics, artefact frames removed, drone >= 4 m from antenna:')
    far = [(c, ch) for (c, ch) in R if np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]) >= 4.0]
    for chs, lab in (('ABD', 'channels A,B,D'), ('C', 'channel C')):
        sel = [k for k in far if k[1] in chs]
        for name in BANDS:
            for mode in ('local', 'global'):
                K = np.array([R[k]['stats'][name][mode][0] for k in sel]); V = np.array([R[k]['stats'][name][mode][1] for k in sel])
                print(f'  {lab:14s} {name:15s} {mode:6s}: kurtosis median {np.median(K):.2f} (max {K.max():.2f}), Vd median {np.median(V):.2f} dB (max {V.max():.2f})')
    print('\nClean channel-A event rate by acquisition order (cells 1..25):',
          ' '.join(f'{R[(c, "A")]["clean_rate"]:.2f}' for c in range(1, 26)))
