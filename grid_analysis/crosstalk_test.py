"""Is the reference-coherent component on far antennas magnetic field or electrical crosstalk?

For each reference antenna R and each cell near R, track R's strongest motor line over the whole record,
then estimate the complex transfer ratio H = S_target / S_ref (coherent line amplitudes over ~316 s).
Crosstalk: |H| independent of drone position.  Field: |H| follows geometry, log|H| ~ -k log(r_tgt/r_ref).
Fit log10|H| = a_pair - k * log10(r_tgt/r_ref) across cells (one intercept per R->target pair, common k).
Noise bias: |S_target|^2 is corrected by the shifted-reference null mean.
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, stft

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]
FS, FSB = 20000.0, 500.0
DEC = int(FS / FSB)
psdz = np.load(os.path.join(HERE, 'raw_psd.npz'))
PN, PP, PF = list(psdz['names']), psdz['psd'], psdz['f']
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
NEAR = {'A': [1, 2, 6, 7, 3, 11], 'B': [5, 4, 10, 9, 3, 15], 'C': [25, 24, 20, 19, 23, 15], 'D': [21, 22, 16, 17, 23, 11]}
SOS_BB = butter(8, 150, fs=FS, output='sos'); SOS_FINE = butter(4, 15, fs=FSB, output='sos')
rng = np.random.default_rng(2)

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

rows = []
for R, cells in NEAR.items():
    for c in cells:
        cx = np.array([(c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5])
        p = PP[PN.index(f'c{c:02d}_{R}.mat')]; b = (PF > 1000) & (PF < 1400)
        fc = PF[b][np.argmax(p[b])]
        bb = {ch: baseband(load(c, ch), fc) for ch in 'ABCD'}
        phi, fi = phase_track(bb[R])
        Sref = np.mean(lpf(bb[R] * np.exp(-1j * phi)))
        rR = np.linalg.norm(cx - ANT[R])
        for tg in 'ABCD':
            if tg == R:
                continue
            zt = lpf(bb[tg] * np.exp(-1j * phi)); St = np.mean(zt)
            nul = np.array([np.abs(np.mean(lpf(bb[tg] * np.exp(-1j * np.roll(phi, int(rng.uniform(20, 150) * FSB)))))) ** 2
                            for _ in range(12)])
            p_sig = np.abs(St) ** 2 - nul.mean()
            snr = 10 * np.log10(np.abs(St) ** 2 / nul.mean())
            H = np.sqrt(max(p_sig, 1e-30)) / np.abs(Sref)
            rT = np.linalg.norm(cx - ANT[tg])
            rows.append((R, tg, c, rR, rT, 20 * np.log10(H), snr, np.degrees(np.angle(St / Sref)), np.std(fi)))
            print(f'ref {R} cell {c:2d} tgt {tg}: r_ref {rR:.2f} m r_tgt {rT:.2f} m | 20log|H| {20*np.log10(H):7.1f} dB, '
                  f'target SNR {snr:5.1f} dB, rel. phase {np.degrees(np.angle(St/Sref)):7.1f} deg, track sd {np.std(fi):5.1f} Hz', flush=True)

# pooled fit with per-pair intercepts, only well-detected targets
R_ = np.array([r[0] for r in rows]); T_ = np.array([r[1] for r in rows])
lr = np.log10(np.array([r[4] / r[3] for r in rows])); h = np.array([r[5] for r in rows]) / 20; snr = np.array([r[6] for r in rows])
ok = snr > 10
pairs = sorted(set(zip(R_[ok], T_[ok])))
A = np.zeros((ok.sum(), len(pairs) + 1)); A[:, 0] = -lr[ok]
for i, (rr, tt) in enumerate(zip(R_[ok], T_[ok])):
    A[i, 1 + pairs.index((rr, tt))] = 1
coef, res, *_ = np.linalg.lstsq(A, h[ok], rcond=None)
pred = A @ coef; resid = h[ok] - pred
Ac = A.copy(); Ac[:, 0] = 0
c0, *_ = np.linalg.lstsq(Ac[:, 1:], h[ok], rcond=None)
resid0 = h[ok] - Ac[:, 1:] @ c0
cov = np.linalg.inv(A.T @ A) * (resid @ resid) / max(1, ok.sum() - A.shape[1])
print(f'\nPooled fit over {ok.sum()} ref/cell/target combinations (target SNR > 10 dB), {len(pairs)} pair intercepts:')
print(f'  amplitude decay exponent k = {coef[0]:.2f} +/- {np.sqrt(cov[0,0]):.2f}  (crosstalk -> 0; field -> ~2.2 from the power-map fit)')
print(f'  rms residual: with geometry {20*np.sqrt(np.mean(resid**2)):.1f} dB; constant ratio per pair (crosstalk model) {20*np.sqrt(np.mean(resid0**2)):.1f} dB')
np.save(os.path.join(HERE, 'crosstalk_rows.npy'), np.array(rows, dtype=object), allow_pickle=True)
