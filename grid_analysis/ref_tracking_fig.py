"""Figure: (a) far-antenna coherent SNR vs integration time with nearest-antenna phase reference;
(b) transfer ratio vs distance ratio (field vs crosstalk test)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = np.load(os.path.join(HERE, 'ref_tracking_results.npy'), allow_pickle=True)
X = np.load(os.path.join(HERE, 'crosstalk_rows.npy'), allow_pickle=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))

ax = axs[0]
for c, ref, tgt, dist, Rc, d in R:
    if c == 13:
        continue
    T = sorted(d); y = [d[t]['coh'] for t in T]
    ax.plot(T, y, '-o', ms=2.5, lw=1, color='#1f6fb4' if dist < 5 else '#d1603d', alpha=0.8)
tt = np.array([1, 300]); ax.plot(tt, 7 + 10 * np.log10(tt), 'k--', lw=0.8)
ax.text(20, 22, '10 dB/decade', rotation=28, fontsize=7)
ax.plot([], [], '-o', color='#1f6fb4', ms=2.5, label='far antenna at 4.6 m')
ax.plot([], [], '-o', color='#d1603d', ms=2.5, label='far antenna at 6.5 m')
ax.axhline(0, color='0.6', lw=0.8)
ax.set_xscale('log'); ax.set_xlabel('integration time T (s)'); ax.set_ylabel('coherent line SNR (dB)')
ax.set_title('(a) Nearest-antenna phase reference', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=7, loc='upper left')

ax = axs[1]
Rr = np.array([x[0] for x in X]); Tt = np.array([x[1] for x in X])
lr = np.log10(np.array([x[4] / x[3] for x in X])); h = np.array([x[5] for x in X]); snr = np.array([x[6] for x in X])
ok = snr > 10
pairs = sorted(set(zip(Rr[ok], Tt[ok])))
A = np.zeros((ok.sum(), len(pairs) + 1)); A[:, 0] = -lr[ok]
for i, (a, b) in enumerate(zip(Rr[ok], Tt[ok])):
    A[i, 1 + pairs.index((a, b))] = 1
coef, *_ = np.linalg.lstsq(A, h[ok] / 20, rcond=None)
hc = h[ok] - 20 * (A[:, 1:] @ coef[1:])          # remove per-pair gain ratios
ax.plot(lr[ok], hc, 'o', ms=3, color='#1f6fb4', alpha=0.7)
xx = np.linspace(lr[ok].min(), lr[ok].max(), 2)
ax.plot(xx, -20 * coef[0] * xx, 'k-', lw=1, label=f'field: slope k = {coef[0]:.2f}')
ax.plot(xx, 0 * xx, color='0.5', ls='--', lw=1, label='crosstalk: k = 0')
ax.set_xlabel('log10(r_target / r_reference)'); ax.set_ylabel('20log|H| − pair gain (dB)')
ax.set_title('(b) Coherent component follows geometry', loc='left', fontsize=9)
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'ref_tracking.png'), dpi=180)
print('saved')
