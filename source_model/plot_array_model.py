"""Figure: fraction of curtain crossings detected (Pd >= 0.9) vs node pitch, by drone class and processor."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'array_model_curtain.json')))
cases = [('5-inch FPV', 'semi-urban outdoor'), ('DJI-like photo 11"', 'semi-urban outdoor'), ('heavy-lift', 'semi-urban outdoor')]
sty = {('P1', False): ('ELF fundamental, incoherent, scalar', '0.6', ':'),
       ('P3', False): ('ELF fundamental, joint-tracked, scalar', '#6baed6', '--'),
       ('P3', True): ('ELF fundamental, joint-tracked, triaxial', '#1f6fb4', '-'),
       ('PWM', False): ('PWM carrier, coherent, scalar', '#f2a07b', '--'),
       ('PWM', True): ('PWM carrier, coherent, triaxial', '#d1603d', '-')}
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
for ax, (name, bg) in zip(axs, cases):
    for (proc, tri), (lab, col, ls) in sty.items():
        rr = sorted([r for r in R if r['case'] == name and r['bg'] == bg and r['proc'] == proc and r['triax'] == tri], key=lambda r: r['pitch'])
        ax.plot([r['pitch'] for r in rr], [r['frac90'] for r in rr], ls, color=col, lw=1.5, marker='o', ms=2.5, label=lab)
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 4, 8, 16]))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(['2', '4', '8', '16']))
    ax.set_xlabel('curtain node pitch (m)'); ax.set_title(f'{name}, {bg.split()[0]}', fontsize=8.5, loc='left'); ax.grid(alpha=0.3)
axs[0].set_ylabel('crossings detected with Pd ≥ 0.9')
axs[0].legend(fontsize=6, frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'array_model_curtain.png'), dpi=180)
print('saved')
