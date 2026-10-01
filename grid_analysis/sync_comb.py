"""Order-tracked recursive synchronous comb: K-independent per-node cancellation of mains-periodic interference.

For sample n, the sample one mains cycle earlier is n' with theta(n') = theta(n) - 2*pi (theta: tracked mains phase).
  prediction p[n] = I(e, n')                       (8-tap windowed-sinc fractional read of the template buffer)
  output     r[n] = x[n] - p[n]
  template   e[n] = p[n] + alpha_k (x[n] - p[n])   (exponential synchronous average; alpha_k = max(1/(k+1), 1/M))
Notch width per harmonic ~ alpha*f0/pi. Cost ~ 8 MAC (interpolation) + ~6 ops per sample, independent of the
number of harmonics. Vectorised per block shorter than one cycle (all reads fall in earlier blocks).
Benchmarked against the O(N*K) projection canceller in periodic_cancel.py.
"""
import os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
sys.argv = sys.argv[:2]
import periodic_cancel as PC

TAPS = np.arange(-3, 5)

def interp8(buf, pos):
    i0 = np.floor(pos).astype(np.int64); mu = pos - i0
    u = TAPS[None, :] - mu[:, None]
    w = np.sinc(u) * (0.42 + 0.5 * np.cos(np.pi * u / 4.5) + 0.08 * np.cos(2 * np.pi * u / 4.5))   # Blackman-windowed sinc
    w /= w.sum(1, keepdims=True)
    return np.einsum('ij,ij->i', buf[i0[:, None] + TAPS[None, :]], w)

def sync_comb(x, theta, M=50):
    n = len(x); idx = np.arange(n, dtype=float)
    prev = np.interp(theta - 2 * np.pi, theta, idx)            # fractional index one cycle earlier
    e = np.zeros(n); r = x.copy()
    T0 = float(np.median(idx[1:] - prev[1:]))
    start = int(np.ceil(T0)) + 8
    e[:start] = x[:start]                                        # first cycle seeds the template
    B = int(T0) - 8
    s = start
    while s < n:
        b = slice(s, min(s + B, n))
        p = interp8(e, prev[b])
        k = (s - start) / T0 + 1.0
        a = max(1.0 / (k + 1.0), 1.0 / M)
        r[b] = x[b] - p
        e[b] = p + a * (x[b] - p)
        s += B
    r[:start] = 0.0                                              # warm-up output discarded
    return r

def run(args):
    c, ch = args
    x = PC.load(c, ch); n = (len(x) // PC.HOP) * PC.HOP; x = x[:n]
    t0 = time.perf_counter(); theta, f0, H = PC.mains_phase(x); t_phase = time.perf_counter() - t0
    t1 = time.perf_counter(); y_comb = sync_comb(x, theta); t_comb = time.perf_counter() - t1
    t2 = time.perf_counter()
    w, starts = PC.blend_weights(n)
    y_proj = x - PC.project_and_rebuild(x, theta, int(9900 / 50.2), w, starts)
    t_proj = time.perf_counter() - t2
    dur = n / PC.FS
    return (c, ch), x, y_comb, y_proj, dict(rt_phase=dur / t_phase, rt_comb=dur / t_comb, rt_proj=dur / t_proj)

if __name__ == '__main__':
    import numpy as np
    from scipy.signal import welch, coherence
    cells = (1, 25)
    with ProcessPoolExecutor(max_workers=8) as ex:
        R = {k: (x, yc, yp, i) for k, x, yc, yp, i in ex.map(run, [(c, ch) for c in cells for ch in 'ABCD'])}
    FS = PC.FS; skip = int(5 * FS)                                # exclude comb warm-up from all metrics
    for c in cells:
        fc, ref = float(PC.A[c]['fc']), PC.A[c]['ref']
        n = min(len(R[(c, ch)][0]) for ch in 'ABCD')
        phi = PC.truth_phase(PC.bbase(R[(c, ref)][0][:n], fc))
        print(f'\n=== cell {c}: drone line {fc:.0f} Hz ===', flush=True)
        for ch in 'ABCD':
            x, yc, yp, info = R[(c, ch)]
            x, yc, yp = x[skip:n], yc[skip:n], yp[skip:n]
            f, px = welch(x, fs=FS, nperseg=2 ** 15)
            band = (f > 20) & (f < 9800); df = f[1] - f[0]
            harm = np.zeros_like(f, bool)
            for k in range(1, 200):
                harm |= np.abs(f - 50.0 * k) <= 1.5 * df
            ih = band & ~harm & ~((f > 950) & (f < 1450)) & ~((f > 5100) & (f < 7000))
            a0 = np.abs(PC.lp(PC.bbase(x, fc) * np.exp(-1j * phi[skip // 40: skip // 40 + len(PC.bbase(x, fc))])).mean())
            out = []
            for name, y in (('comb', yc), ('proj', yp)):
                _, py = welch(y, fs=FS, nperseg=2 ** 15)
                bb = PC.bbase(y, fc)
                a1 = np.abs(PC.lp(bb * np.exp(-1j * phi[skip // 40: skip // 40 + len(bb)])).mean())
                out.append(f'{name}: total -{10*np.log10(px[band].sum()/py[band].sum()):4.1f} dB, '
                           f'harm -{10*np.log10(np.median(px[band&harm])/np.median(py[band&harm])):4.1f} dB, '
                           f'floor {10*np.log10(np.median(py[ih])/np.median(px[ih])):+.2f} dB, drone {20*np.log10(a1/a0):+.2f} dB')
            print(f'  {ch}: ' + ' | '.join(out) + f' || speed x real time: phase {info["rt_phase"]:.0f}, comb {info["rt_comb"]:.0f}, '
                  f'projection {info["rt_proj"]:.1f}', flush=True)
        pairs = [(p, q) for p in 'ABCD' for q in 'ABCD' if p < q and ref not in (p, q)]
        for a, b in pairs:
            f, g0 = coherence(R[(c, a)][0][skip:n], R[(c, b)][0][skip:n], fs=FS, nperseg=2 ** 15)
            _, gc = coherence(R[(c, a)][1][skip:n], R[(c, b)][1][skip:n], fs=FS, nperseg=2 ** 15)
            _, gp = coherence(R[(c, a)][2][skip:n], R[(c, b)][2][skip:n], fs=FS, nperseg=2 ** 15)
            harm = np.zeros_like(f, bool)
            for k in range(1, 200):
                harm |= np.abs(f - 50.0 * k) <= 0.6 * (f[1] - f[0])
            sel = harm & (f > 40) & (f < 9800)
            print(f'  harmonic coherence {a}-{b} (90th pct): raw {np.percentile(g0[sel], 90):.2f}, comb {np.percentile(gc[sel], 90):.2f}, '
                  f'projection {np.percentile(gp[sel], 90):.2f}', flush=True)
