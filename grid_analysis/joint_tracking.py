"""Joint (multi-node) tracking of a wandering motor line on the raw grid data.

For each cell: the antenna nearest the drone (reference, ~40 dB) provides the ground-truth rotor phase.
Tracking is attempted on the three OTHER antennas with extra noise added so that the strongest of them has a
target SNR in a 20 Hz band (each channel's noise scaled by the same factor, preserving natural SNR ratios).
Trackers:
  greedy1 : greedy STFT ridge on the strongest test channel (the method used so far)
  vit1    : Viterbi ridge on the strongest test channel
  vitJ    : Viterbi ridge on the summed normalised power of all 3 test channels; phase from the coherent
            (principal-eigenvector) combination of the 3 channels  [model-free joint tracking]
  vitJ5   : as vitJ, ridge score also includes the 5th harmonic (power at 5f in a 5 f_c baseband)
  fixed   : no tracking (fixed median frequency) -- baseline
Score: coherent amplitude retained on the reference antenna in 10 s blocks, |sum z_ref e^{-j phi_test}| /
|sum z_ref e^{-j phi_true}|; success if >= 0.7 (-3 dB).
"""
import os, sys
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, stft
from scipy.ndimage import maximum_filter1d
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]
FS, FSB, FSB5 = 20000.0, 500.0, 2500.0
CH = 'ABCD'
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()
CELLS = [1, 2, 4, 5, 16, 21, 24, 25]
EQUAL = os.environ.get("EQUAL", "0") == "1"
TARGETS = [-3, -6, -9, -12, -15] if EQUAL else [3, 0, -3, -6, -9]
SOS15 = butter(4, 15, fs=FSB, output='sos'); SOS10 = butter(4, 10, fs=FSB, output='sos')

def lp(z, sos):
    return sosfiltfilt(sos, z.real) + 1j * sosfiltfilt(sos, z.imag)

def baseband(x, fc, fsb, bw):
    n = np.arange(len(x)); y = (x - x.mean()) * np.exp(-2j * np.pi * fc * n / FS)
    sos = butter(8, bw, fs=FS, output='sos')
    y = sosfiltfilt(sos, y.real) + 1j * sosfiltfilt(sos, y.imag)
    return y[:: int(FS / fsb)]

def mains_mask(f_abs, half=4.0):
    d = np.abs(((f_abs + 25.0) % 50.0) - 25.0)
    return d <= half

def spec(b, fsb, nper, hop, fmax, scale=1.0, fc0=None):
    f, t, Z = stft(b, fs=fsb, nperseg=nper, noverlap=nper - hop, nfft=4 * nper, return_onesided=False, boundary=None)
    f = np.fft.fftshift(f); P = np.fft.fftshift(np.abs(Z) ** 2, axes=0)
    keep = np.abs(f) <= fmax; f, P = f[keep], P[keep]
    if fc0 is not None:                               # notch mains harmonics (coherent across nodes, stationary)
        m = mains_mask(f + fc0)
        P[m] = np.median(P[~m], axis=0, keepdims=True)
    return f / scale, t, P

def norm(P):
    return P / np.median(P, axis=0, keepdims=True)

def greedy(f, S):
    k = np.argmax(S.mean(1)); tr = np.empty(S.shape[1])
    for j in range(S.shape[1]):
        lo, hi = max(0, k - 20), min(len(f), k + 21); k = lo + np.argmax(S[lo:hi, j]); tr[j] = f[k]
    return tr

def viterbi(f, S, J=4):
    D = S[:, 0].copy(); back = np.zeros(S.shape, np.int32); idx = np.arange(len(f))
    for j in range(1, S.shape[1]):
        m = maximum_filter1d(D, size=2 * J + 1, mode='nearest')
        # argmax within window (approximate: pick neighbour achieving the max)
        cand = np.stack([np.roll(D, s) for s in range(-J, J + 1)]); arg = np.argmax(cand, axis=0) - J
        back[:, j] = np.clip(idx - arg, 0, len(f) - 1); D = m + S[:, j]
    k = int(np.argmax(D)); path = np.empty(S.shape[1], int)
    for j in range(S.shape[1] - 1, -1, -1):
        path[j] = k; k = back[k, j]
    return f[path]

def phase_from_track(t, tr, n):
    fi = np.interp(np.arange(n) / FSB, t, np.convolve(tr, np.ones(5) / 5, mode='same'))
    return 2 * np.pi * np.cumsum(fi) / FSB, fi

def refine(phi0, bbs, joint):
    z = np.stack([lp(b * np.exp(-1j * phi0), SOS15) for b in bbs])
    if joint:
        Rm = z @ z.conj().T / z.shape[1]; w = np.linalg.eigh(Rm)[1][:, -1]
        zc = w.conj() @ z
    else:
        zc = z[0]
    return phi0 + np.unwrap(np.angle(zc))

def score(phi, phi_true, zref_raw):
    """Did the test track lock onto a REAL rotor line (any motor)? Coherent power on the high-SNR reference
    antenna along the test track, per 10 s block, relative to the same track time-shifted by 60 s (null)."""
    m = int(10 * FSB); k = len(phi) // m
    z = np.abs((zref_raw[: k * m] * np.exp(-1j * phi[: k * m])).reshape(k, m).mean(1)) ** 2
    ps = np.roll(phi, int(60 * FSB))
    nul = np.abs((zref_raw[: k * m] * np.exp(-1j * ps[: k * m])).reshape(k, m).mean(1)) ** 2
    return z / np.median(nul)

def run_cell(c):
    rng = np.random.default_rng(c)
    fc, ref = float(A[c]['fc']), A[c]['ref']
    raw = {ch: sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float) for ch in CH}
    bb = {ch: baseband(raw[ch], fc, FSB, 150) for ch in CH}
    bb5 = {ch: baseband(raw[ch], 5 * fc, FSB5, 700) for ch in CH if ch != ref}
    n = min(len(v) for v in bb.values()); bb = {k: v[:n] for k, v in bb.items()}
    # ground truth from the reference antenna (high SNR)
    f, t, P = spec(bb[ref], FSB, 50, 5, 120)
    phi_true = refine(phase_from_track(t, greedy(f, norm(P)), n)[0], [bb[ref]], False)
    zref = bb[ref]                                    # raw reference baseband (scored against truth)
    tests = [ch for ch in CH if ch != ref]
    # natural line amplitude and 20 Hz noise per test channel
    amp, noi = {}, {}
    for ch in tests:
        amp[ch] = np.abs(lp(bb[ch] * np.exp(-1j * phi_true), SOS15).mean()) ** 2
        noi[ch] = np.mean(np.abs(lp(bb[ch] * np.exp(-1j * np.roll(phi_true, int(60 * FSB))), SOS10)) ** 2)
    snr0 = {ch: 10 * np.log10(amp[ch] / noi[ch]) for ch in tests}
    order = sorted(tests, key=lambda ch: -snr0[ch]); best = order[0]
    rows = []
    for tgt in TARGETS:
        k = 10 ** ((snr0[best] - tgt) / 10)
        kc = ({ch: 10 ** ((snr0[ch] - tgt) / 10) for ch in tests} if EQUAL else {ch: k for ch in tests})
        if min(kc.values()) < 1:
            continue
        noisy, noisy5 = {}, {}
        for ch in tests:
            k = kc[ch]
            sig2 = (k - 1) * noi[ch] / (20.0 / FSB)             # complex white noise variance per sample
            noisy[ch] = bb[ch] + np.sqrt(sig2 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            p5 = np.mean(np.abs(bb5[ch]) ** 2)                    # scale 5th-harmonic noise by the same factor on its own floor
            fl5 = np.median(np.abs(np.fft.fft(bb5[ch][: 2 ** 16])) ** 2) / 2 ** 16
            noisy5[ch] = bb5[ch] + np.sqrt((k - 1) * fl5 / 2) * (rng.standard_normal(len(bb5[ch])) + 1j * rng.standard_normal(len(bb5[ch])))
        res = {}
        # single strongest test channel
        f, t, P = spec(noisy[best], FSB, 50, 5, 120, fc0=fc); S = norm(P)
        for name, tr in (('greedy1', greedy(f, S)), ('vit1', viterbi(f, S))):
            res[name] = score(refine(phase_from_track(t, tr, n)[0], [noisy[best]], False), phi_true, zref)
        # joint
        Sn = {ch: norm(spec(noisy[ch], FSB, 50, 5, 120, fc0=fc)[2]) for ch in tests}
        Ssum = sum(Sn.values())
        trJ = viterbi(f, Ssum)
        # joint, COHERENT blind beamformer: whiten each channel by its noise level, principal eigenvector of the
        # band cross-spectral matrix (constant relative gains of a stationary drone), combine, then track (no truth used)
        Xs, sig = [], {}
        for ch in order:
            fz, _, Z = stft(noisy[ch], fs=FSB, nperseg=50, noverlap=45, nfft=200, return_onesided=False, boundary=None)
            band = (np.abs(fz) <= 120) & ~mains_mask(fz + fc, 6.0)        # line band, mains harmonics excluded
            sig[ch] = np.sqrt(np.median(np.abs(Z[band]) ** 2))
            Xs.append(Z[band] / sig[ch])
        X = np.stack([x.reshape(-1) for x in Xs])
        Rm = X @ X.conj().T / X.shape[1]; w = np.linalg.eigh(Rm)[1][:, -1]
        y = sum(np.conj(w[i]) * noisy[ch] / sig[ch] for i, ch in enumerate(order))
        fy, ty, Py = spec(y, FSB, 50, 5, 120, fc0=fc)
        trB = viterbi(fy, norm(Py))
        res['vitJw'] = score(refine(phase_from_track(ty, trB, n)[0], [y], False), phi_true, zref)
        res['vitJ'] = score(refine(phase_from_track(t, trJ, n)[0], [noisy[ch] for ch in order], True), phi_true, zref)
        # joint + 5th harmonic: power at 5f sampled on the fundamental grid
        S5 = 0
        for ch in tests:
            f5, t5, P5 = spec(noisy5[ch], FSB5, 250, 25, 650, fc0=5 * fc)
            P5i = np.stack([np.interp(5 * f, f5, P5[:, j]) for j in range(min(P5.shape[1], Ssum.shape[1]))], 1)
            S5 = S5 + norm(P5i)
        m = min(Ssum.shape[1], S5.shape[1])
        trJ5 = viterbi(f, Ssum[:, :m] + S5[:, :m])
        res['vitJ5'] = score(refine(phase_from_track(t[:m], trJ5, n)[0], [noisy[ch] for ch in order], True), phi_true, zref)
        res['fixed'] = score(2 * np.pi * np.median(trJ) * np.arange(n) / FSB, phi_true, zref)
        snrs = {ch: snr0[ch] - 10 * np.log10(kc[ch]) for ch in tests}
        rows.append((c, tgt, snrs, {kk: (float(np.median(v)), float(np.mean(v >= 100.0))) for kk, v in res.items()}))
    return c, snr0, rows

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(run_cell, CELLS))
    import pickle
    pickle.dump(out, open(os.path.join(HERE, 'joint_tracking_eq.pkl' if EQUAL else 'joint_tracking.pkl'), 'wb'))
    methods = ['fixed', 'greedy1', 'vit1', 'vitJ', 'vitJ5', 'vitJw']
    for c, snr0, rows in out:
        print(f'cell {c:2d}: natural 20 Hz SNR of test antennas ' + ', '.join(f'{k} {v:5.1f} dB' for k, v in snr0.items()))
    print('\nSuccess fraction of 10 s blocks (test track locked to a real rotor line: >=20 dB on reference vs shifted null), pooled over cells; strongest-test-channel SNR in 20 Hz:')
    print(f'{"SNR":>5s} ' + ''.join(f'{m:>9s}' for m in methods) + '   (n cells)   | joint-sum SNR (dB, mean)')
    for tgt in TARGETS:
        rr = [r for _, _, rows in out for r in rows if r[1] == tgt]
        if not rr:
            continue
        js = np.mean([10 * np.log10(sum(10 ** (v / 10) for v in r[2].values())) for r in rr])
        print(f'{tgt:5d} ' + ''.join(f'{np.mean([r[3][m][1] for r in rr]):9.2f}' for m in methods) + f'   ({len(rr)})        | {js:5.1f}')
    print('\nMedian lock contrast on reference (dB):')
    for tgt in TARGETS:
        rr = [r for _, _, rows in out for r in rows if r[1] == tgt]
        if rr:
            print(f'{tgt:5d} ' + ''.join(f'{10 * np.log10(max(1e-3, np.median([r[3][m][0] for r in rr]))):9.1f}' for m in methods))
