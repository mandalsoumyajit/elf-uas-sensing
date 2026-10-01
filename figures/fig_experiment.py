"""Data figures for the grid experiment (Section IV): signature, background, coherent integration, localization."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.signal import welch
import blocks as B
import plotstyle

G = os.path.join(B.ROOT, 'grid_analysis'); ENV = os.path.join(G, 'env')
EXP = os.environ.get('ELF_RAW_EXPORT', os.path.join(B.ROOT, 'raw_export'))   # exported grid records (export_raw.m)
ANT = {'A': (-0.1, 5.1), 'B': (5.1, 5.1), 'C': (5.1, -0.1), 'D': (-0.1, -0.1)}
CXY = {c: ((c - 1) % 5 + 0.5, 4.5 - (c - 1) // 5) for c in range(1, 26)}
dist = lambda c, ch: float(np.hypot(CXY[c][0] - ANT[ch][0], CXY[c][1] - ANT[ch][1]))
COL = {'A': '#2874a6', 'B': '#1e8449', 'C': '#c0392b', 'D': '#7d3c98'}
MK = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}
plotstyle.apply()

def tag(ax, s):
    ax.set_title(s, loc='left', fontsize=8)

# ------------------------------------------------------------------ Fig: grid and signature
def signature():
    P = np.load(os.path.join(G, 'raw_psd.npz')); names, psd, f = list(P['names']), P['psd'], P['f']
    A = np.load(os.path.join(G, 'complex_amps.npy'), allow_pickle=True).item()
    fig = plt.figure(figsize=(7.16, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.35, 1.05], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    for c in range(1, 26):
        x, y = CXY[c]
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc='none', ec='0.7', lw=0.5))
        ax.text(x, y, str(c), ha='center', va='center', fontsize=5.8, color='0.35')
    for ch, (x, y) in ANT.items():
        u = np.array([2.5 - x, 2.5 - y]); u /= np.linalg.norm(u); v = np.array([-u[1], u[0]]) * 0.42
        ax.plot([x - v[0], x + v[0]], [y - v[1], y + v[1]], color=COL[ch], lw=2.2, solid_capstyle='butt')
        ax.annotate('', (x + 0.55 * u[0], y + 0.55 * u[1]), (x, y), arrowprops=dict(arrowstyle='-|>', color=COL[ch], lw=0.7))
        ax.text(x - 0.42 * u[0], y - 0.42 * u[1], ch, color=COL[ch], ha='center', va='center', fontsize=7.5, fontweight='bold')
    ax.set_xlim(-0.8, 5.8); ax.set_ylim(-0.8, 5.8); ax.set_aspect('equal'); ax.set_xticks([0, 5]); ax.set_yticks([0, 5])
    ax.set_xlabel('x (m)', labelpad=0); ax.set_ylabel('y (m)', labelpad=0); tag(ax, '(a) grid and loop antennas')
    ax = fig.add_subplot(gs[1])
    for c, lab, col in ((1, 'cell 1 (0.6 m from A)', COL['A']), (25, 'cell 25 (6.5 m from A)', '0.45')):
        p = psd[names.index(f'c{c:02d}_A.mat')]
        k = 8; pp = p[: len(p) // k * k].reshape(-1, k).mean(1); ff = f[: len(f) // k * k].reshape(-1, k).mean(1)
        ax.semilogy(ff / 1e3, pp, color=col, lw=0.6, label=lab)
    ax.axvspan(1.0, 1.4, color='#fdf0e6', zorder=0); ax.axvspan(5.2, 6.9, color='#fdf0e6', zorder=0); ax.axvspan(7.4, 9.7, color='#fdf0e6', zorder=0)
    ax.text(1.2, 3e4, 'f\u2091', ha='center', fontsize=7); ax.text(6.05, 3e4, '5f\u2091', ha='center', fontsize=7); ax.text(8.5, 3e4, '7f\u2091', ha='center', fontsize=7)
    ax.set_xlim(0, 10); ax.set_ylim(1e-3, 1e5); ax.set_xlabel('frequency (kHz)'); ax.set_ylabel('PSD (ADC counts\u00b2/Hz)')
    ax.legend(loc='lower left', fontsize=6.3); tag(ax, '(b) antenna A spectra')
    ax = fig.add_subplot(gs[2])
    r = np.array([dist(c, ch) for ch in 'ABCD' for c in range(1, 26)])
    s = np.array([20 * np.log10(np.abs(A[c]['rec'][i])) for i in range(4) for c in range(1, 26)])
    M = np.zeros((100, 5)); M[:, 0] = np.log10(r)                      # common slope, per-antenna offsets (unequal gains)
    for i in range(4):
        M[25 * i:25 * (i + 1), 1 + i] = 1
    coef, *_ = np.linalg.lstsq(M, s, rcond=None); k = coef[0]
    for i, ch in enumerate('ABCD'):
        ax.semilogx(r[25 * i:25 * (i + 1)], s[25 * i:25 * (i + 1)] - coef[1 + i], MK[ch], ms=3, color=COL[ch], mec='none',
                    label=f'antenna {ch}', alpha=0.85)
    xx = np.array([0.5, 7.5]); ax.plot(xx, k * np.log10(xx), 'k--', lw=0.8)
    ax.text(1.6, k * np.log10(1.6) + 6, f'|B| $\\propto$ r$^{{{k/20:.1f}}}$', fontsize=6.5)
    ax.set_xlabel('drone\u2013antenna distance (m)'); ax.set_ylabel('motor-line amplitude (dB)')
    ax.set_xticks([0.5, 1, 2, 4, 7]); ax.set_xticklabels(['0.5', '1', '2', '4', '7']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=6, loc='lower left', handletextpad=0.2); tag(ax, '(c) motor line vs distance')
    B.save(fig, 'exp_signature'); print('fitted amplitude exponent', k / 20)

# ------------------------------------------------------------------ Fig: background characterisation
def comb_example():
    cache = os.path.join(B.HERE, 'comb_example.npz')
    if not os.path.exists(cache):
        sys.argv = [sys.argv[0], EXP]; sys.path.insert(0, G)
        import periodic_cancel as PC, sync_comb as SCB
        x = PC.load(1, 'C'); th, _, _ = PC.mains_phase(x); y = SCB.sync_comb(x, th)
        s = int(5 * 20000); f, p0 = welch(x[s:], fs=20000, nperseg=2 ** 15); _, p1 = welch(y[s:], fs=20000, nperseg=2 ** 15)
        np.savez(cache, f=f, p0=p0, p1=p1)
    return np.load(cache)

def background():
    P = np.load(os.path.join(G, 'raw_psd.npz')); names, psd, f = list(P['names']), P['psd'], P['f']
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.25), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]
    harm = np.zeros_like(f, bool)
    for k in range(1, 200):
        harm |= np.abs(f - 50.05 * k) <= 5
    ih = ~harm & (np.abs(f - 5930) > 30)
    for ch in 'ABCD':
        far = [c for c in range(1, 26) if dist(c, ch) >= 4]
        bg = np.median(np.stack([psd[names.index(f'c{c:02d}_{ch}.mat')] for c in far]), 0)
        cen = np.arange(125, 9800, 250); fl = [np.median(bg[ih & (f > c - 125) & (f < c + 125)]) for c in cen]
        ax.semilogy(cen / 1e3, fl, '-', marker=MK[ch], ms=2, color=COL[ch], lw=0.9, label=ch)
    ax.set_xlabel('frequency (kHz)'); ax.set_ylabel('floor PSD (counts\u00b2/Hz)'); ax.legend(fontsize=6, ncol=2, loc='upper right')
    tag(ax, '(a) inter-harmonic floor')
    ax = axs[1]
    E = comb_example()
    m = (E['f'] > 20) & (E['f'] < 2000)
    ax.semilogy(E['f'][m], E['p0'][m], color='0.6', lw=0.5, label='raw')
    ax.semilogy(E['f'][m], E['p1'][m], color=COL['C'], lw=0.5, label='after comb')
    ax.set_xlabel('frequency (Hz)'); ax.set_ylabel('PSD (counts\u00b2/Hz)'); ax.legend(fontsize=6, loc='upper right')
    tag(ax, '(b) mains comb, ant. C')
    ax = axs[2]
    C = np.load(os.path.join(ENV, 'env_coh.npy'), allow_pickle=True).item(); fc, coh = C['f'], C['coh']
    hc = np.zeros_like(fc, bool)
    for k in range(1, 200):
        hc |= np.abs(fc - 50.05 * k) <= 0.35
    ihc = np.ones_like(fc, bool)
    for k in range(1, 200):
        ihc &= np.abs(fc - 50.05 * k) > 5
    ihc &= (np.abs(fc - 5930) > 30) & ~((fc > 950) & (fc < 1450)) & ~((fc > 5100) & (fc < 7000)) & (fc > 60) & (fc < 9800)
    labs, h_med, b_med = [], [], []
    for (a, b), d in coh.items():
        g2 = np.median(np.stack(d['g2_each']), 0)
        labs.append(f'{a}{b}'); h_med.append(np.median(g2[hc & (fc > 60) & (fc < 9800)])); b_med.append(np.median(g2[ihc]))
    x = np.arange(len(labs))
    ax.bar(x - 0.2, h_med, 0.4, color='#c0662b', label='mains harmonics'); ax.bar(x + 0.2, b_med, 0.4, color='0.55', label='between harmonics')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6); ax.set_yscale('log'); ax.set_ylim(1e-3, 1)
    ax.set_ylabel('median coherence \u03b3\u00b2'); ax.legend(fontsize=5.8, loc='upper left'); tag(ax, '(c) between antennas')
    ax = axs[3]
    D = np.load(os.path.join(ENV, 'detstat_tails.npy'), allow_pickle=True).item(); R, EC = D['R'], D['E_COH']
    for v, lab, col, ls in (('raw', 'measured background', 'k', '-'), ('gauss', 'Gaussian, same chain', '#2874a6', '--')):
        h = sum(d[(v, 'cfar')]['hc'] for d in R.values()).astype(float)
        c = np.cumsum(h[::-1])[::-1] / h.sum()
        ax.semilogy(EC[:-1], c, ls, color=col, lw=1.0, label=lab)
    ax.axhline((1 / 1800) / 4000, color='0.6', lw=0.6, ls=':'); ax.text(32, 2.5e-7, 'design P$_{fa}$', fontsize=6, color='0.4')
    ax.set_xlim(0, 60); ax.set_ylim(1e-9, 1); ax.set_xlabel('2 s coherent statistic'); ax.set_ylabel('exceedance probability')
    ax.legend(fontsize=5.8, loc='upper right'); tag(ax, '(d) detector tails (CFAR)')
    B.save(fig, 'exp_background')

# ------------------------------------------------------------------ Fig: coherent integration
def coherent():
    R = np.load(os.path.join(G, 'ref_tracking_results.npy'), allow_pickle=True)
    X = np.load(os.path.join(G, 'crosstalk_rows.npy'), allow_pickle=True)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    for c, ref, tgt, dd, Rc, d in R:
        if c == 13:
            continue
        T = sorted(d)
        ax.plot(T, [d[t]['coh'] for t in T], '-o', ms=2.2, lw=0.9, color=COL['A'] if dd < 5 else COL['C'], alpha=0.85)
    tt = np.array([1, 300]); ax.plot(tt, 7 + 10 * np.log10(tt), 'k--', lw=0.7); ax.text(15, 21.5, '10 dB/decade', rotation=24, fontsize=6.3)
    ax.plot([], [], '-o', color=COL['A'], ms=2.2, label='far node at 4.6 m'); ax.plot([], [], '-o', color=COL['C'], ms=2.2, label='far node at 6.5 m')
    ax.axhline(0, color='0.7', lw=0.6); ax.set_xscale('log'); ax.set_xlabel('integration time T (s)'); ax.set_ylabel('line SNR at far node (dB)')
    ax.legend(fontsize=6.3, loc='upper left'); tag(ax, '(a) phase-referenced coherent integration')
    ax = axs[1]
    Rr = np.array([x[0] for x in X]); Tt = np.array([x[1] for x in X])
    lr = np.log10(np.array([x[4] / x[3] for x in X])); h = np.array([x[5] for x in X]); snr = np.array([x[6] for x in X])
    ok = snr > 10; pairs = sorted(set(zip(Rr[ok], Tt[ok])))
    Am = np.zeros((ok.sum(), len(pairs) + 1)); Am[:, 0] = -lr[ok]
    for i, (a, b) in enumerate(zip(Rr[ok], Tt[ok])):
        Am[i, 1 + pairs.index((a, b))] = 1
    coef, *_ = np.linalg.lstsq(Am, h[ok] / 20, rcond=None)
    hc = h[ok] - 20 * (Am[:, 1:] @ coef[1:])
    ax.plot(lr[ok], hc, 'o', ms=2.6, color=COL['A'], alpha=0.7, mec='none')
    xx = np.linspace(lr[ok].min(), lr[ok].max(), 2)
    ax.plot(xx, -20 * coef[0] * xx, 'k-', lw=0.9, label=f'field: |H| $\\propto$ (r$_t$/r$_r$)$^{{-{coef[0]:.2f}}}$')
    ax.plot(xx, 0 * xx, color='0.5', ls='--', lw=0.9, label='electrical crosstalk (flat)')
    ax.set_xlabel('log$_{10}$(r$_{target}$ / r$_{reference}$)'); ax.set_ylabel('20 log|H| \u2212 pair gain (dB)')
    ax.legend(fontsize=6.3); tag(ax, '(b) coherent component follows geometry')
    B.save(fig, 'exp_coherent'); print('crosstalk exponent', coef[0])

# ------------------------------------------------------------------ Fig: localization
def localization():
    R = np.load(os.path.join(G, 'fit_position_results.npy'), allow_pickle=True).item()['results']
    d = np.load(os.path.join(G, 'raw_localize.npz'))
    e4 = np.linalg.norm(d['pred_phys'] - d['xy'][d['y'] - 1], axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.95], wspace=0.38))
    curves = [('band power, empirical decay', None, '0.5', '-'), ('|line amplitude|, power law', 'V1', '#9e9ac8', '--'),
              ('|line amplitude|, dipole + loops', 'V2m', '#6baed6', '-.'), ('complex amplitude, dipole + loops', 'V2', COL['C'], '-')]
    inner = [7, 8, 9, 12, 13, 14, 17, 18, 19]
    for ax, sel in zip(axs[:2], ('all', 'interior')):
        for lab, k, col, ls in curves:
            if k is None:
                e, ci = e4, np.isin(d['y'], inner)
            else:
                arr = np.array(R[k]['w1']); e = arr[:, 1]; ci = np.isin(arr[:, 0], inner)
            x = np.sort(e if sel == 'all' else e[ci])
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), ls, color=col, lw=1.2, label=lab)
        ax.axvline(0.71, color='0.7', lw=0.6, ls=':'); ax.set_xlim(0, 2.0); ax.set_ylim(0, 1)
        ax.set_xlabel('position error per 1 s window (m)'); ax.set_ylabel('fraction of windows')
        tag(ax, '(a) all 25 held-out cells' if sel == 'all' else '(b) interior 3\u00d73 cells')
    axs[0].legend(fontsize=5.6, loc='lower right')
    ax = axs[2]
    arr = np.array(R['V2']['w1']); med = np.array([np.median(arr[arr[:, 0] == c, 1]) for c in range(1, 26)])
    im = ax.imshow(med.reshape(5, 5), cmap='Oranges', vmin=0, vmax=1.0, extent=(0, 5, 0, 5))
    for c in range(1, 26):
        x, y = CXY[c]; ax.text(x, y, f'{med[c-1]:.2f}', ha='center', va='center', fontsize=5.6, color='k' if med[c - 1] < 0.7 else 'w')
    for ch, (x, y) in ANT.items():
        ax.plot(x, y, MK[ch], color=COL[ch], ms=4, clip_on=False)
    ax.set_xticks([0, 5]); ax.set_yticks([0, 5]); ax.set_xlabel('x (m)', labelpad=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('median error (m)', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    tag(ax, '(c) complex model, per cell')
    B.save(fig, 'exp_localization')

if __name__ == '__main__':
    which = sys.argv[1:] or ['signature', 'background', 'coherent', 'localization']
    for w in which:
        globals()[w]()
