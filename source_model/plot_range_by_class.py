"""Figure: detection range by drone class and background, current band vs wideband, 0.2 s vs 60 s tracked."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
O = json.load(open(os.path.join(HERE, 'range_by_class.json')))
classes = ['micro (sub-250 g, 3", 4S)', '5-inch FPV (6S)', 'photo/mid (11" props, 4S)', 'heavy-lift (30" props, 12S)']
short = ['micro\n<250 g', '5-inch\nFPV', 'photo\n11"', 'heavy-lift\n30"']
scen = ['quiet outdoor', 'semi-urban outdoor', 'indoor lab', 'indoor near electronics']
cols = ['#1f6fb4', '#6baed6', '#d1603d', '#f2a07b']
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, band, title in zip(axs, ['current (<=10 kHz)', 'wideband'], ['(a) Receiver band ≤10 kHz (present hardware)', '(b) Wideband receiver (incl. 16–48 kHz PWM lines)']):
    x = np.arange(len(classes))
    for k, (s, c) in enumerate(zip(scen, cols)):
        xs = x + (k - 1.5) * 0.19
        # realistic use cases only: indoor detection applies to small drones (micro, 5-inch), not photo/heavy-lift
        keep = np.array([not (s.startswith('indoor') and i >= 2) for i in range(len(classes))])
        for mode, mk, alpha in (('0.2 s', 'o', 0.45), ('60 s tracked', 's', 1.0)):
            r = np.array([O[f'{cl} | {s} | {band} | {mode}']['range'] for cl in classes])[keep]
            ax.errorbar(xs[keep] + (0.05 if mode != '0.2 s' else -0.05), r[:, 1], yerr=[r[:, 1] - r[:, 0], r[:, 2] - r[:, 1]],
                        fmt=mk, ms=3.5, color=c, alpha=alpha, lw=0.9, capsize=1.5,
                        label=s if mode == '60 s tracked' else None)
    ax.set_yscale('log'); ax.set_ylim(0.2, 80); ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title(title, fontsize=8.5, loc='left'); ax.grid(alpha=0.3, which='both', axis='y')
axs[0].set_ylabel('detection range (m), Pd 0.9')
axs[0].legend(fontsize=6.5, frameon=False, loc='upper left')
axs[1].text(0.98, 0.03, 'faint circles: 0.2 s snapshot\nsquares: 60 s tracked coherent', transform=axs[1].transAxes,
            ha='right', va='bottom', fontsize=6.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'range_by_class.png'), dpi=180)
print('saved')
