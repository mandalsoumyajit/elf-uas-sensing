"""Environmental background report: spectra, spatial coherence, temporal behaviour, sync check, statistics."""
import os, csv, datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ENV = os.path.join(HERE, 'env')
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
dist = lambda c, ch: float(np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]))
COL = {'A': '#1f6fb4', 'B': '#2ca02c', 'C': '#d1603d', 'D': '#7b4ea3'}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
# clock time per cell from MAT headers
T0 = {}
for r in csv.DictReader(open(os.path.join(HERE, 'metadata.csv'))):
    if r['channel'] == 'A':
        T0[int(r['cell'])] = dt.datetime.strptime(' '.join(r['header_created'].split()), '%a %b %d %H:%M:%S %Y')
hours = lambda c: T0[c].hour + T0[c].minute / 60

# ---------------- spectra ----------------
P = np.load(os.path.join(HERE, 'raw_psd.npz')); PN, PP, f = list(P['names']), P['psd'], P['f']
def harm(fr, w):
    m = np.zeros_like(fr, bool)
    for k in range(1, 200):
        m |= np.abs(fr - 50.05 * k) <= w
    return m
H = harm(f, 0.8)
IH = ~harm(f, 5) & (np.abs(f - 5930) > 30)
bg = {}
for ch in 'ABCD':
    far = [c for c in range(1, 26) if dist(c, ch) >= 4.0]
    bg[ch] = np.median(np.stack([PP[PN.index(f'c{c:02d}_{ch}.mat')] for c in far]), 0)
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=False)
for ch in 'ABCD':
    axs[0].semilogy(f, bg[ch], lw=0.5, color=COL[ch], label=f'antenna {ch}')
    fl = np.array([np.median(bg[ch][IH & (f > lo) & (f < lo + 250)]) for lo in np.arange(0, 10000, 250)])
    axs[0].semilogy(np.arange(125, 10000, 250), fl, 'o-', ms=2, lw=1.2, color=COL[ch])
    axs[1].semilogy(f, bg[ch], lw=0.7, color=COL[ch])
axs[0].set_xlim(0, 10000); axs[0].set_ylabel('PSD (ADC counts²/Hz)'); axs[0].set_xlabel('Hz')
axs[0].set_title('(a) Background PSD (median over cells ≥4 m from the antenna); dots: inter-harmonic floor', loc='left', fontsize=8.5)
axs[0].axvspan(950, 1450, color='0.9', zorder=0); axs[0].axvspan(5100, 7000, color='0.9', zorder=0)
axs[0].legend(fontsize=7, frameon=False, ncol=4)
axs[1].set_xlim(0, 600); axs[1].set_xlabel('Hz'); axs[1].set_ylabel('PSD (counts²/Hz)')
axs[1].set_title('(b) 0-600 Hz: mains harmonic comb (50.05 Hz)', loc='left', fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(ENV, 'env_spectra.png'), dpi=170)

print('=== SPECTRA (ADC counts; median over cells >= 4 m from each antenna) ===')
for ch in 'ABCD':
    df = f[1] - f[0]
    tot = bg[ch][(f > 20) & (f < 9800)].sum() * df
    hp = bg[ch][H & (f > 20) & (f < 9800)].sum() * df
    fl = lambda lo, hi: np.median(bg[ch][IH & (f > lo) & (f < hi)])
    k50 = [bg[ch][np.argmin(np.abs(f - 50.05 * k))] for k in (1, 3, 5, 7)]
    print(f'{ch}: mains-harmonic share of 20 Hz-9.8 kHz power {100*hp/tot:.0f}% | inter-harmonic floor (counts²/Hz): '
          f'0.2-0.9 kHz {fl(200,900):.3f}, 1.5-4.5 kHz {fl(1500,4500):.3f}, 7-9.5 kHz {fl(7000,9500):.4f} | '
          f'50/150/250/350 Hz line PSD {np.round(k50,0)}')
# conditional field conversion (assumed: 300 turns, 1 m loop, AFE gain G, x1/3 attenuation, 3.3 V / 4096)
NA = 300 * np.pi * 0.5 ** 2
for G in (1000.0,):
    for ch in 'ABCD':
        for fr, (lo, hi) in ((600, (200, 900)), (3000, (1500, 4500)), (8000, (7000, 9500))):
            v = np.sqrt(np.median(bg[ch][IH & (f > lo) & (f < hi)])) * 3.3 / 4096 * 3 / G
            print(f'  [if G={G:.0f}] {ch} floor at ~{fr} Hz: {v*1e9:7.1f} nV/rtHz at loop -> {v/(2*np.pi*fr*NA)*1e15:8.1f} fT/rtHz', end=';')
        print()

# ---------------- spatial coherence ----------------
C = np.load(os.path.join(ENV, 'env_coh.npy'), allow_pickle=True).item(); fc, coh = C['f'], C['coh']
Hc = harm(fc, 0.35); IHc = ~harm(fc, 5) & (np.abs(fc - 5930) > 30) & ~((fc > 950) & (fc < 1450)) & ~((fc > 5100) & (fc < 7000))
fig, axs = plt.subplots(3, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
print('\n=== SPATIAL COHERENCE gamma^2 (cells far from both antennas; per-cell spectra, median over cells) ===')
for ax, ((a, b), d) in zip(axs.ravel(), coh.items()):
    g2 = np.median(np.stack(d['g2_each']), 0)
    ax.plot(fc[IHc], g2[IHc], ',', color='0.5', alpha=0.5)
    ax.plot(fc[Hc], g2[Hc], '.', ms=2.5, color=COL[a])
    sep = np.hypot(ANT[a][0] - ANT[b][0], ANT[a][1] - ANT[b][1])
    ax.set_title(f'{a}–{b} ({sep:.1f} m)', fontsize=8); ax.set_xlim(0, 10000); ax.set_ylim(0, 1)
    s = lambda m, lo, hi: np.median(g2[m & (fc > lo) & (fc < hi)])
    i5930 = np.argmin(np.abs(fc - 5930)); i50 = np.argmin(np.abs(fc - 50.05))
    print(f'{a}-{b} ({sep:.1f} m, {len(d["cells"])} cells): 50 Hz {g2[i50]:.2f}; harmonics <1 kHz {s(Hc,60,1000):.2f}, 1-3 kHz {s(Hc,1000,3000):.2f}, '
          f'3-10 kHz {s(Hc,3000,9800):.2f} (90th pct all harmonics {np.percentile(g2[Hc & (fc<9800)], 90):.2f}); broadband between harmonics '
          f'<1 kHz {s(IHc,60,1000):.3f}, 1.5-4.5 kHz {s(IHc,1500,4500):.3f}, 7-9.5 kHz {s(IHc,7000,9500):.3f}; 5.93 kHz spike {g2[i5930]:.2f}')
for ax in axs[-1]:
    ax.set_xlabel('Hz')
for ax in axs[:, 0]:
    ax.set_ylabel('γ²')
fig.suptitle('Background coherence between antennas (coloured: mains harmonics; grey: between harmonics)', fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(ENV, 'env_coherence.png'), dpi=170)

# ---------------- temporal ----------------
S = np.load(os.path.join(ENV, 'env_stats.npy'), allow_pickle=True).item()
print('\n=== TEMPORAL ===')
fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
for ch in 'ABCD':
    far = [c for c in range(1, 26) if dist(c, ch) >= 4.0]
    x = [hours(c) for c in far]
    axs[0].plot(x, [10 * np.log10(np.median(S[(c, ch)]['floor_hi'])) for c in far], 'o-', ms=3, color=COL[ch], label=ch)
    axs[1].plot(x, [20 * np.log10(np.median(S[(c, ch)]['a50'])) for c in far], 'o-', ms=3, color=COL[ch])
    fl = np.array([10 * np.log10(np.median(S[(c, ch)]['floor_hi'])) for c in far])
    wr = np.median([np.std(10 * np.log10(S[(c, ch)]['floor_hi'])) for c in far])
    a50 = np.array([20 * np.log10(np.median(S[(c, ch)]['a50'])) for c in far])
    print(f'{ch}: 7-9.5 kHz floor over the evening: range {fl.max()-fl.min():.1f} dB across sessions; within-record 1 s scatter sd {wr:.2f} dB | '
          f'50 Hz amplitude range {a50.max()-a50.min():.1f} dB across sessions')
for c in range(1, 26):
    tt = hours(c) + np.arange(len(S[(c, 'C')]['fgrid'])) * 10 / 3600
    axs[2].plot(tt, S[(c, 'C')]['fgrid'], '.', ms=1.5, color='k')
fg = np.concatenate([S[(c, 'C')]['fgrid'] for c in range(1, 26)])
print(f'grid frequency (10 s estimates, channel C): median {np.median(fg):.3f} Hz, range {fg.min():.3f}-{fg.max():.3f} Hz, sd {np.std(fg)*1e3:.0f} mHz')
axs[0].set_ylabel('7-9.5 kHz floor (dB re counts²/Hz)'); axs[0].legend(fontsize=7, frameon=False, ncol=4)
axs[1].set_ylabel('50 Hz amplitude (dB re count)'); axs[2].set_ylabel('grid frequency (Hz)'); axs[2].set_xlabel('local time on 8 Dec 2025 (h)')
axs[0].set_title('Sessions with the drone ≥4 m from each antenna', loc='left', fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(ENV, 'env_temporal.png'), dpi=170)

# ---------------- sampling-sync check ----------------
print('\n=== SYNC CHECK: 50 Hz phase difference between channels within each record (1 s steps) ===')
worst = []
for c in range(1, 26):
    ph = {ch: S[(c, ch)]['ph50'] for ch in 'ABCD'}
    n = min(len(v) for v in ph.values())
    for b in 'BCD':
        d = np.unwrap(ph[b][:n] - ph['A'][:n])
        jumps = np.abs(np.diff(d))
        worst.append((c, b, np.degrees(np.std(d - np.polyval(np.polyfit(np.arange(n), d, 1), np.arange(n)))),
                      np.degrees(np.ptp(d)), np.degrees(jumps.max())))
w = np.array([x[2:] for x in worst])
print(f'detrended sd of A-x phase difference: median {np.median(w[:,0]):.2f} deg, max {w[:,0].max():.2f} deg; '
      f'total drift over record median {np.median(w[:,1]):.2f} deg, max {w[:,1].max():.1f} deg; largest 1 s jump {w[:,2].max():.1f} deg '
      f'(one 50 us sample slip = 0.9 deg at 50 Hz)')

# ---------------- amplitude statistics ----------------
D = np.load(os.path.join(ENV, 'env_dist.npy'), allow_pickle=True).item(); XG = D['XG']; dist_ = D['dist']
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
for ax, name in zip(axs, ['B1 150-950 Hz', 'B2 1.5-4.5 kHz', 'B3 7-9.5 kHz']):
    for ch in 'ABCD':
        apds = [dist_[k][name]['global'][2] for k in dist_ if k[1] == ch]
        if apds:
            ax.semilogy(20 * np.log10(np.maximum(XG, 1e-3)), np.median(np.stack(apds), 0), color=COL[ch], lw=1.2, label=ch)
    ax.semilogy(20 * np.log10(np.maximum(XG, 1e-3)), np.exp(-XG ** 2), 'k--', lw=1, label='Rayleigh')
    ax.set_xlim(-10, 16); ax.set_ylim(1e-6, 1.2); ax.set_title(name, fontsize=8); ax.set_xlabel('envelope / rms (dB)')
axs[0].set_ylabel('P(envelope > x)'); axs[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(ENV, 'env_apd.png'), dpi=170)
print('\n=== AMPLITUDE STATISTICS by channel (deglitched; drone >= 4 m; global normalization = whole record) ===')
for name in ['B1 150-950 Hz', 'B2 1.5-4.5 kHz', 'B3 7-9.5 kHz']:
    row = f'{name:15s}'
    for ch in 'ABCD':
        ks = [dist_[k][name]['local'][0] for k in dist_ if k[1] == ch]; vs = [dist_[k][name]['global'][1] for k in dist_ if k[1] == ch]
        kg = [dist_[k][name]['global'][0] for k in dist_ if k[1] == ch]
        p4 = [dist_[k][name]['global'][2][np.argmin(np.abs(XG - 3.0))] for k in dist_ if k[1] == ch]
        row += f' | {ch}: kurt local {np.median(ks):.2f} global {np.median(kg):.2f}, Vd {np.median(vs):.2f} dB, P(env>3 rms) {np.median(p4):.1e}'
    print(row)
print('Rayleigh reference: kurtosis 3, Vd 1.05 dB, P(env > 3 rms) = exp(-9) = 1.2e-04')
