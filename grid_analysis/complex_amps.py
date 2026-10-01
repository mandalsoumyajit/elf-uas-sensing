"""Step A: complex coherent amplitudes of the tracked motor line on all four antennas, per cell.

Reference choice uses only the data (no ground truth): channel/line with the largest 1.0-1.4 kHz excess;
a track with sd < 3 Hz (stationary non-drone line) is rejected in favour of the next candidate.
Outputs per cell: record-level amplitudes, 10 s and 1 s window amplitudes, and null noise variances
(from reference phase shifted by 20-150 s) for each window length.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, stft, find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]
FS, FSB = 20000.0, 500.0
DEC = int(FS / FSB)
psdz = np.load(os.path.join(HERE, 'raw_psd.npz'))
PN, PP, PF = list(psdz['names']), psdz['psd'], psdz['f']
SOS_BB = butter(8, 150, fs=FS, output='sos'); SOS_FINE = butter(4, 15, fs=FSB, output='sos')
rng = np.random.default_rng(3)
CH = 'ABCD'

def load(c, ch):
    return sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)
def baseband(x, fc):
    n = np.arange(len(x)); y = (x - x.mean()) * np.exp(-2j * np.pi * fc * n / FS)
    return (sosfiltfilt(SOS_BB, y.real) + 1j * sosfiltfilt(SOS_BB, y.imag))[::DEC]
def lpf(z):
    return sosfiltfilt(SOS_FINE, z.real) + 1j * sosfiltfilt(SOS_FINE, z.imag)
def phase_track(b):
    f, t, Z = stft(b, fs=FSB, nperseg=50, noverlap=45, nfft=1024, return_onesided=False, boundary=None)
    f = np.fft.fftshift(f); P = np.fft.fftshift(np.abs(Z) ** 2, axes=0); ok = np.abs(f) < 120; f, P = f[ok], P[ok]
    k = np.argmax(P.mean(1)); tr = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        lo, hi = max(0, k - 40), min(len(f), k + 41); k = lo + np.argmax(P[lo:hi, j]); tr[j] = f[k]
    tr = np.convolve(tr, np.ones(5) / 5, mode='same')
    fi = np.interp(np.arange(len(b)) / FSB, t, tr)
    phi0 = 2 * np.pi * np.cumsum(fi) / FSB
    return phi0 + np.unwrap(np.angle(lpf(b * np.exp(-1j * phi0)))), fi

def candidates(c):
    """(excess dB, channel, frequency) of line peaks in 1.0-1.4 kHz, best first."""
    out = []
    b = (PF > 1000) & (PF < 1400)
    for ch in CH:
        p = PP[PN.index(f'c{c:02d}_{ch}.mat')][b]
        ref = np.median(np.stack([PP[PN.index(f'c{k:02d}_{ch}.mat')][b] for k in range(1, 26)]), 0)
        ex = 10 * np.log10(p / ref)
        pk, _ = find_peaks(ex, distance=int(20 / (PF[1] - PF[0])))
        for i in pk[np.argsort(ex[pk])[::-1][:3]]:
            out.append((ex[i], ch, PF[b][i]))
    return sorted(out, reverse=True)

def window_means(z, T):
    m = int(T * FSB); k = len(z) // m
    return z[: k * m].reshape(k, m).mean(1)

F0 = 1200.0
SOS_WIDE = butter(8, 230, fs=FS, output='sos')
def baseband_wide(x):
    n = np.arange(len(x)); y = (x - x.mean()) * np.exp(-2j * np.pi * F0 * n / FS)
    return (sosfiltfilt(SOS_WIDE, y.real) + 1j * sosfiltfilt(SOS_WIDE, y.imag))[::DEC]

SOS_BB2 = butter(8, 120, fs=FSB, output='sos')
def lpf_bb(z):
    return sosfiltfilt(SOS_BB2, z.real) + 1j * sosfiltfilt(SOS_BB2, z.imag)

def shift(b, df):
    return b * np.exp(-2j * np.pi * df * np.arange(len(b)) / FSB)

def track_ok(fi):
    sd = np.std(fi)
    jumps = np.mean(np.abs(np.diff(fi[:: int(0.01 * FSB)])) > 5)   # >5 Hz change per 10 ms
    return 3.0 <= sd <= 45.0 and jumps < 0.02, sd, jumps

res = {}
for c in range(1, 26):
    raw = {ch: load(c, ch) for ch in CH}
    wide = {ch: baseband_wide(raw[ch]) for ch in CH}
    del raw
    best = None
    for ex, ref, fc in candidates(c)[:8]:
        bbc = {ch: lpf_bb(shift(wide[ch], fc - F0)) for ch in CH}
        phi, fi = phase_track(bbc[ref])
        ok, sd, jumps = track_ok(fi)
        if not ok:
            continue
        z = {ch: lpf(bbc[ch] * np.exp(-1j * phi)) for ch in CH}
        # score: coherent power on the other antennas relative to a shifted-phase null
        score = 0.0
        ph = np.roll(phi, int(60 * FSB))
        for ch in CH:
            if ch == ref:
                continue
            s = np.abs(z[ch].mean()) ** 2
            n = np.abs(lpf(bbc[ch] * np.exp(-1j * ph)).mean()) ** 2
            score += 10 * np.log10(s / n)
        if best is None or score > best[0]:
            best = (score, ref, fc, phi, fi, bbc)
    if best is None:
        print(f'cell {c:2d}: no acceptable reference line'); continue
    score, ref, fc, phi, fi, bb = best
    zc = np.stack([lpf(bb[ch] * np.exp(-1j * phi)) for ch in CH])          # 4 x n
    rec = zc.mean(1)
    w10 = np.stack([window_means(z, 10) for z in zc], 1)                   # nw x 4
    w1 = np.stack([window_means(z, 1) for z in zc], 1)
    nv = {'rec': np.zeros(4), '10': np.zeros(4), '1': np.zeros(4)}
    NS = 8
    for _ in range(NS):
        ph = np.roll(phi, int(rng.uniform(20, 150) * FSB))
        zn = np.stack([lpf(bb[ch] * np.exp(-1j * ph)) for ch in CH])
        nv['rec'] += np.abs(zn.mean(1)) ** 2 / NS
        nv['10'] += np.mean(np.abs(np.stack([window_means(z, 10) for z in zn], 1)) ** 2, 0) / NS
        nv['1'] += np.mean(np.abs(np.stack([window_means(z, 1) for z in zn], 1)) ** 2, 0) / NS
    res[c] = dict(ref=ref, fc=fc, track_sd=np.std(fi), rec=rec, w10=w10, w1=w1, nv_rec=nv['rec'], nv10=nv['10'], nv1=nv['1'])
    snr = 10 * np.log10(np.abs(rec) ** 2 / nv['rec'])
    print(f'cell {c:2d}: ref {ref} line {fc:7.1f} Hz track sd {np.std(fi):5.1f} Hz | record SNR per antenna (dB) '
          + ' '.join(f'{ch}:{s:5.1f}' for ch, s in zip(CH, snr)), flush=True)
np.save(os.path.join(HERE, 'complex_amps.npy'), res, allow_pickle=True)
