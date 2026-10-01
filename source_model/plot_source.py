"""Figure: per-motor line spectra (field at 1 m) for the four drone classes, with uncertainty bars."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'drone_source_results.json')))
COL = {'rotor residual': '#7b4ea3', 'phase leads': '#1f6fb4', 'phase leads (PWM ripple)': '#6baed6',
       'ESC switching loop': '#d1603d', 'DC bus': '#2ca02c', 'DC bus (PWM)': '#98df8a', 'winding (placeholder)': '0.6'}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), sharey=True)
for ax, (name, d) in zip(axs.ravel(), R.items()):
    for L in d['lines']:
        lo, nom, hi = L['B1m_nT']
        jitter = {'ESC switching loop': 1.04, 'DC bus (PWM)': 0.96}.get(L['source'], 1.0)
        f = L['f'] * jitter
        ax.vlines(f, max(lo, 1e-5), hi, color=COL[L['source']], lw=3, alpha=0.35)
        ax.plot(f, nom, 'o', ms=4, color=COL[L['source']])
    op = d['op']
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-4, 30); ax.set_xlim(20, 6e4)
    ax.set_title(f"{name}\nhover {op['rpm']:.0f} rpm, f_e {op['f_e']:.0f} Hz", fontsize=8, loc='left')
    ax.grid(alpha=0.3, which='both')
for ax in axs[-1]:
    ax.set_xlabel('frequency (Hz)')
for ax in axs[:, 0]:
    ax.set_ylabel('|B| per axis at 1 m (nT, per motor)')
for s, c in COL.items():
    axs[0, 0].plot([], [], 'o', color=c, label=s)
axs[0, 0].legend(fontsize=6, frameon=False, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'drone_source_lines.png'), dpi=180)
print('saved')
