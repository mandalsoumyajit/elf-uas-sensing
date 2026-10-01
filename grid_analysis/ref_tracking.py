"""Nearest-antenna phase tracking + coherent integration on the far antennas (raw grid data).

For a cell next to antenna R (reference), track the strongest motor line on R, then demodulate each far
antenna with R's phase. Methods, all evaluated on the far antenna over integration time T:
  fixed : coherent average at a fixed frequency (median of the reference track), no tracking
  incoh : incoherent average along the reference frequency track (0.1 s coherent chunks, power-averaged)
  coh   : coherent average along the full reference phase (frequency track + fine phase)
Null: identical processing with the reference phase circularly shifted by 20-150 s (wrong rotor phase,
same data, same spectra). SNR = mean statistic / mean null statistic; also the empirical exceedance.
Also reported: stability of the far/reference relative phase (the quasi-static "same phase at all nodes").
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, stft

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'raw_export')
FS, FSB = 20000.0, 500.0
DEC = int(FS / FSB)
psdz = np.load(os.path.join(HERE, 'raw_psd.npz'))
PN, PP, PF = list(psdz['names']), psdz['psd'], psdz['f']
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
rng = np.random.default_rng(1)

def load(c, ch):
    return sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float)

SOS_BB = butter(8, 150, fs=FS, output='sos')
def baseband(x, fc):
    n = np.arange(len(x))
    y = (x - x.mean()) * np.exp(-2j * np.pi * fc * n / FS)
    y = sosfiltfilt(SOS_BB, y.real) + 1j * sosfiltfilt(SOS_BB, y.imag)
    return y[::DEC]

SOS_FINE = butter(4, 15, fs=FSB, output='sos')
def lpf(z):
    return sosfiltfilt(SOS_FINE, z.real) + 1j * sosfiltfilt(SOS_FINE, z.imag)

def ridge(b):
    """Frequency track (Hz, relative to fc) of the strongest line in complex baseband b."""
    f, t, Z = stft(b, fs=FSB, nperseg=50, noverlap=45, nfft=1024, return_onesided=False, boundary=None)
    f = np.fft.fftshift(f); P = np.fft.fftshift(np.abs(Z) ** 2, axes=0)
    ok = np.abs(f) < 120
    f, P = f[ok], P[ok]
    k = np.argmax(P.mean(1)); tr = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        lo, hi = max(0, k - 40), min(len(f), k + 41)          # continuity: +-~20 Hz per 10 ms step
        k = lo + np.argmax(P[lo:hi, j]); tr[j] = f[k]
    tr = np.convolve(tr, np.ones(5) / 5, mode='same')
    return t, tr

def phase_track(bref):
    t, tr = ridge(bref)
    tt = np.arange(len(bref)) / FSB
    fi = np.interp(tt, t, tr)
    phi0 = 2 * np.pi * np.cumsum(fi) / FSB
    zr = lpf(bref * np.exp(-1j * phi0))
    phi = phi0 + np.unwrap(np.angle(zr))
    return phi, fi, np.abs(zr)

def stats(btgt, phi, fmed, Ts):
    """Per-T statistics (lists over non-overlapping segments) for the three methods."""
    n = len(btgt)
    zc = lpf(btgt * np.exp(-1j * phi))                          # coherent along reference phase
    zf = btgt * np.exp(-2j * np.pi * fmed * np.arange(n) / FSB)  # fixed frequency
    L = int(0.1 * FSB)
    chunks = zc[: (n // L) * L].reshape(-1, L).mean(1)          # 0.1 s coherent chunks
    out = {}
    for T in Ts:
        m = int(T * FSB); k = n // m
        if k < 1:
            continue
        coh = np.abs(zc[: k * m].reshape(k, m).mean(1)) ** 2
        fix = np.abs(zf[: k * m].reshape(k, m).mean(1)) ** 2
        mc = int(round(T / 0.1))
        kk = len(chunks) // mc
        inc = (np.abs(chunks[: kk * mc].reshape(kk, mc)) ** 2).mean(1)
        out[T] = {'coh': coh, 'fixed': fix, 'incoh': inc}
    return out, zc

TS = [0.1, 1, 10, 100, 300]
CASES = {1: 'A', 5: 'B', 25: 'C', 21: 'D', 13: None}
NSHIFT = 40
lines_out = []
for c, ref in CASES.items():
    cx = ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5)
    raw = {ch: load(c, ch) for ch in 'ABCD'}
    if ref is None:   # choose the channel with the strongest line excess
        best = None
        for ch in 'ABCD':
            p = PP[PN.index(f'c{c:02d}_{ch}.mat')]; b = (PF > 1000) & (PF < 1400)
            ref_med = np.median(np.stack([PP[PN.index(f'c{k:02d}_{ch}.mat')] for k in range(1, 26)]), 0)
            s = np.max(10 * np.log10(p[b] / ref_med[b]))
            if best is None or s > best[1]:
                best = (ch, s)
        ref = best[0]
    p = PP[PN.index(f'c{c:02d}_{ref}.mat')]; b = (PF > 1000) & (PF < 1400)
    fc = PF[b][np.argmax(p[b])]
    bb = {ch: baseband(raw[ch], fc) for ch in 'ABCD'}
    phi, fi, amp = phase_track(bb[ref])
    fmed = np.median(fi)
    print(f'\n=== cell {c} (x,y={cx}), reference {ref}, line {fc:.1f} Hz; track sd {np.std(fi):.1f} Hz, '
          f'5-95% span {np.ptp(np.percentile(fi, [5, 95])):.1f} Hz ===')
    for tgt in 'ABCD':
        if tgt == ref:
            continue
        dist = np.hypot(cx[0] - ANT[tgt][0], cx[1] - ANT[tgt][1])
        real, zc = stats(bb[tgt], phi, fmed, TS)
        # relative-phase stability: phase of 10 s coherent sums
        m10 = int(10 * FSB); k10 = len(zc) // m10
        ph = np.angle(zc[: k10 * m10].reshape(k10, m10).mean(1))
        R = np.abs(np.mean(np.exp(1j * ph)))
        nulls = {T: {'coh': [], 'fixed': [], 'incoh': []} for T in real}
        for _ in range(NSHIFT):
            s = int(rng.uniform(20, 150) * FSB)
            nr, _ = stats(bb[tgt], np.roll(phi, s), np.median(np.roll(fi, s)), TS)
            for T in nr:
                for kname in nr[T]:
                    nulls[T][kname].append(nr[T][kname])
        row = f'  target {tgt} at {dist:4.1f} m | relative-phase consistency over 10 s blocks R={R:.2f}\n'
        for T in real:
            parts = []
            for kname in ('fixed', 'incoh', 'coh'):
                nul = np.concatenate(nulls[T][kname]); rv = real[T][kname]
                snr = 10 * np.log10(rv.mean() / nul.mean())
                pfa = np.mean(nul[None, :] >= rv[:, None])       # mean exceedance per real segment
                parts.append(f'{kname} {snr:6.1f} dB (Pfa~{pfa:.3f})')
            row += f'     T={T:>5}s: ' + ' | '.join(parts) + '\n'
        print(row, end='')
        lines_out.append((c, ref, tgt, dist, R, {T: {k: 10 * np.log10(real[T][k].mean() / np.concatenate(nulls[T][k]).mean())
                                                  for k in real[T]} for T in real}))
np.save(os.path.join(HERE, 'ref_tracking_results.npy'), np.array(lines_out, dtype=object), allow_pickle=True)
