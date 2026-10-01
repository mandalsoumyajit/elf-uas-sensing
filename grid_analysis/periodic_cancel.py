"""Per-node cancellation of the LOCAL PERIODIC background class (no spatial reference) -- O(N*K) version.

1. Mains phase theta(t): heterodyne the strongest low odd mains harmonic H, low-pass, differentiate phase
   -> instantaneous f0(t); theta = 2*pi*cumsum(f0)/fs  (order tracking; absorbs grid-frequency drift).
2. Harmonic amplitudes per 1 s Hann block (50% overlap) by projection onto exp(-j k theta(t)); harmonics
   are orthogonal over ~50 cycles so no least-squares solve is needed. Powers of exp(-j theta) by recursion.
3. Crossfaded (overlap-add) reconstruction of all harmonics k = 1..K and of other frequency-stable lines;
   subtract. Cost: ~K complex multiply-adds per sample per channel; no matrix solves.
Drone motor lines wander by tens of Hz, so they are removed only while within ~+-1 Hz of a stable line.
"""
import os, sys, time
import numpy as np
import scipy.io as sio
from scipy.signal import welch, butter, sosfiltfilt, stft, find_peaks, coherence
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = sys.argv[1]; FS = 20000.0; L = 20000; HOP = L // 2
A = np.load(os.path.join(HERE, 'complex_amps.npy'), allow_pickle=True).item()

def load(c, ch):
    x = sio.loadmat(os.path.join(EXP, f'c{c:02d}_{ch}.mat'))['x'].ravel().astype(float); return x - x.mean()

def mains_phase(x):
    f, p = welch(x, fs=FS, nperseg=2 ** 15)
    H = max((1, 3, 5, 7, 9, 11, 13), key=lambda h: p[np.argmin(np.abs(f - 50.0 * h))])
    t = np.arange(len(x)) / FS
    z = x * np.exp(-2j * np.pi * 50.0 * H * t)
    sos = butter(4, 3.0, fs=FS, output='sos')
    z = sosfiltfilt(sos, z.real) + 1j * sosfiltfilt(sos, z.imag)
    ph = np.unwrap(np.angle(z))
    df = np.gradient(ph) * FS / (2 * np.pi)                  # Hz offset of harmonic H
    k = int(0.5 * FS); df = np.convolve(df, np.ones(k) / k, mode='same')
    f0 = 50.0 + df / H
    return 2 * np.pi * np.cumsum(f0) / FS, f0, H

def blend_weights(n):
    """Hann (periodic) windows at 50% overlap: analysis windows and crossfade synthesis weights summing to 1."""
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
    starts = np.arange(0, n - L + 1, HOP)
    return w, starts

def project_and_rebuild(x, phase, kmax, w, starts):
    """Subtract sum_k 2 Re(c_k(t) e^{j k phase}) with block-wise c_k, crossfaded."""
    n = len(x); model = np.zeros(n)
    E = np.exp(-1j * phase); P = np.ones(n, complex)
    norm = w.sum(); synth = np.zeros(n);
    for s in starts:
        synth[s:s + L] += w
    synth[synth == 0] = 1.0
    for k in range(1, kmax + 1):
        P *= E
        ck = np.array([np.dot(w * x[s:s + L], P[s:s + L]) / norm for s in starts])
        ct = np.zeros(n, complex)
        for s, c in zip(starts, ck):
            ct[s:s + L] += w * c
        model += 2 * np.real(ct / synth * np.conj(P))
    return model

def stable_lines(x):
    f, p = welch(x, fs=FS, nperseg=2 ** 16)
    lp = 10 * np.log10(p); base = np.convolve(lp, np.ones(301) / 301, mode='same')
    pk, _ = find_peaks(lp - base, height=10, distance=20)
    fz, tz, Z = stft(x, fs=FS, nperseg=int(2 * FS), noverlap=0, boundary=None); P = np.abs(Z) ** 2
    out = []
    for i in pk:
        fr = f[i]
        if fr < 30 or fr > 9800 or abs(((fr + 25) % 50) - 25) < 1.0:
            continue
        sel = (fz > fr - 3) & (fz < fr + 3)
        tr = fz[sel][np.argmax(P[sel], axis=0)]
        if np.std(tr) < 0.3:
            out.append(float(np.median(tr)))
    return out

def cancel_channel(args):
    c, ch = args
    x = load(c, ch); n = (len(x) // HOP) * HOP; x = x[:n]
    t0 = time.perf_counter()
    theta, f0, H = mains_phase(x)
    w, starts = blend_weights(n)
    kmax = int(9900 / 50.2)
    model = project_and_rebuild(x, theta, kmax, w, starts)
    extra = stable_lines(x - model)
    tt = np.arange(n) / FS
    for fr in extra:
        model += project_and_rebuild(x - model, 2 * np.pi * fr * tt, 1, w, starts)
    dt = time.perf_counter() - t0
    return (c, ch), x, x - model, dict(f0_med=float(np.median(f0)), f0_sd=float(np.std(f0)), H=H, n_extra=len(extra),
                                       extra=extra, rt_factor=(n / FS) / dt)

SOS_BB = butter(8, 150, fs=FS, output='sos'); SOS15 = butter(4, 15, fs=500.0, output='sos')
def bbase(x, fc):
    z = x * np.exp(-2j * np.pi * fc * np.arange(len(x)) / FS)
    return (sosfiltfilt(SOS_BB, z.real) + 1j * sosfiltfilt(SOS_BB, z.imag))[::40]
def lp(z):
    return sosfiltfilt(SOS15, z.real) + 1j * sosfiltfilt(SOS15, z.imag)
def truth_phase(b):
    f, t, Z = stft(b, fs=500.0, nperseg=50, noverlap=45, nfft=1024, return_onesided=False, boundary=None)
    f = np.fft.fftshift(f); P = np.fft.fftshift(np.abs(Z) ** 2, axes=0); ok = np.abs(f) < 120; f, P = f[ok], P[ok]
    k = np.argmax(P.mean(1)); tr = np.empty(P.shape[1])
    for j in range(P.shape[1]):
        lo, hi = max(0, k - 40), min(len(f), k + 41); k = lo + np.argmax(P[lo:hi, j]); tr[j] = f[k]
    fi = np.interp(np.arange(len(b)) / 500.0, t, np.convolve(tr, np.ones(5) / 5, mode='same'))
    phi0 = 2 * np.pi * np.cumsum(fi) / 500.0
    return phi0 + np.unwrap(np.angle(lp(b * np.exp(-1j * phi0))))

if __name__ == '__main__':
    cells = (1, 13, 25)
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = {k: (x, y, info) for k, x, y, info in ex.map(cancel_channel, [(c, ch) for c in cells for ch in 'ABCD'])}
    for c in cells:
        fc, ref = float(A[c]['fc']), A[c]['ref']
        n = min(len(res[(c, ch)][0]) for ch in 'ABCD')
        phi = truth_phase(bbase(res[(c, ref)][0][:n], fc))
        print(f'\n=== cell {c}: drone line {fc:.0f} Hz, reference {ref} ===', flush=True)
        for ch in 'ABCD':
            x, y, info = res[(c, ch)]; x, y = x[:n], y[:n]
            f, px = welch(x, fs=FS, nperseg=2 ** 15); _, py = welch(y, fs=FS, nperseg=2 ** 15)
            band = (f > 20) & (f < 9800); df = f[1] - f[0]
            harm = np.zeros_like(f, bool)
            for k in range(1, 200):
                harm |= np.abs(f - 50.0 * k) <= 1.5 * df
            ih = band & ~harm & ~((f > 950) & (f < 1450)) & ~((f > 5100) & (f < 7000))
            tot = 10 * np.log10(px[band].sum() / py[band].sum())
            hrm = 10 * np.log10(np.median(px[band & harm]) / np.median(py[band & harm]))
            flo = 10 * np.log10(np.median(py[ih]) / np.median(px[ih]))
            a0 = np.abs(lp(bbase(x, fc) * np.exp(-1j * phi)).mean()); a1 = np.abs(lp(bbase(y, fc) * np.exp(-1j * phi)).mean())
            print(f'  {ch}: f0 {info["f0_med"]:.3f} Hz (sd {info["f0_sd"]*1e3:.0f} mHz, tracked on H{info["H"]}), {info["n_extra"]} other stable lines | '
                  f'total power removed {tot:5.1f} dB, harmonic bins -{hrm:4.1f} dB, inter-harmonic floor {flo:+.2f} dB | '
                  f'drone line retained {20*np.log10(a1/a0):+.2f} dB | {info["rt_factor"]:.0f}x real time', flush=True)
        pairs = [(p, q) for p in 'ABCD' for q in 'ABCD' if p < q and ref not in (p, q)]
        for a, b in pairs:
            xa, xb = res[(c, a)][0][:n], res[(c, b)][0][:n]; ya, yb = res[(c, a)][1][:n], res[(c, b)][1][:n]
            f, g0 = coherence(xa, xb, fs=FS, nperseg=2 ** 15); _, g1 = coherence(ya, yb, fs=FS, nperseg=2 ** 15)
            harm = np.zeros_like(f, bool)
            for k in range(1, 200):
                harm |= np.abs(f - 50.0 * k) <= 0.6 * (f[1] - f[0])
            sel = harm & (f > 40) & (f < 9800)
            print(f'  coherence {a}-{b} at harmonic bins: median {np.median(g0[sel]):.3f} -> {np.median(g1[sel]):.3f}, '
                  f'90th pct {np.percentile(g0[sel], 90):.2f} -> {np.percentile(g1[sel], 90):.2f}', flush=True)
