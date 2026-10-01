"""Figure: LOCO localization error CDFs (1-s windows) for band power vs coherent-amplitude models."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = np.load(os.path.join(HERE, 'fit_position_results.npy'), allow_pickle=True).item()['results']
d = np.load(os.path.join(HERE, 'raw_localize.npz'))
e4 = np.linalg.norm(d['pred_phys'] - d['xy'][d['y'] - 1], axis=1)
inner = [7, 8, 9, 12, 13, 14, 17, 18, 19]

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
curves = [('band power, empirical model', e4, np.isin(d['y'], inner), '0.5'),
          ('line |amplitude|, power law (V1)', None, None, '#9e9ac8'),
          ('line |amplitude|, dipole+loop (V2m)', None, None, '#6baed6'),
          ('complex amplitude, dipole+loop (V2)', None, None, '#d1603d')]
keys = [None, 'V1', 'V2m', 'V2']
for ax, sel in zip(axs, ('all', 'interior')):
    for (lab, e, ci, col), k in zip(curves, keys):
        if k is not None:
            arr = np.array(R[k]['w1']); e = arr[:, 1]; ci = np.isin(arr[:, 0], inner)
        x = np.sort(e if sel == 'all' else e[ci])
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=col, lw=1.6, label=lab)
    ax.axvline(0.71, color='0.7', lw=0.8, ls=':')
    ax.set_xlim(0, 2.5); ax.set_xlabel('position error, 1 s window (m)'); ax.set_ylabel('fraction of windows')
    ax.set_title('(a) all 25 cells' if sel == 'all' else '(b) interior 3×3 cells', loc='left', fontsize=9)
axs[0].legend(frameon=False, fontsize=6.5, loc='lower right')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fit_position_cdf.png'), dpi=180)
print('saved')
