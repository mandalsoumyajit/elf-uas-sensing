"""Figure: site model. Top: perimeter crossings detected (Pd >= 0.9) vs altitude band per mount layout (B = 16
nodes / 100 m, PWM coherent solid, ELF joint-tracked dashed). Bottom: street canyon, P(detected within 50 m)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'site_model.json')))
B = 16
CASES = ('5-inch FPV', 'DJI-like', 'heavy-lift')
PL = {'ground': ('ground 1.5 m', '0.55'), 'poles': ('poles 5+10 m', '#6baed6'), 'poles+roofs': ('poles + rooftops', '#1f6fb4'),
      'towers': ('25 m towers', '#d1603d'), 'curtain': ('ideal curtain to 30 m', 'k')}
CL = {'ground': ('ground', '0.55'), 'streetlights': ('streetlights 8 m', '#6baed6'), 'facades': ('facades', '#2ca02c'),
      'rooftops': ('rooftops', '#d1603d'), 'mixed': ('mixed', '#1f6fb4')}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.0), sharey=True)
for j, case in enumerate(CASES):
    ax = axs[0, j]
    for cfg, (lab, col) in PL.items():
        for proc, ls in (('PWM', '-'), ('P3', '--')):
            r = next(x for x in R['perimeter'] if x['case'] == case and x['proc'] == proc and x['cfg'] == cfg and x['B'] == B)
            zs = list(r['res'].keys()); mid = [np.mean([float(v) for v in k.split('-')]) for k in zs]
            ax.plot(mid, [r['res'][k][1] for k in zs], ls, color=col, lw=1.4 if proc == 'PWM' else 1.0, marker='o', ms=2.5,
                    label=lab if proc == 'PWM' else None)
    ax.set_title(f'perimeter, {case}', fontsize=8, loc='left'); ax.set_xlabel('crossing altitude (m)'); ax.grid(alpha=0.3)
    ax = axs[1, j]; w = 0.15
    for i, (cfg, (lab, col)) in enumerate(CL.items()):
        for k, (proc, hatch) in enumerate((('PWM', None), ('P3', '///'))):
            r = next(x for x in R['canyon'] if x['case'] == case and x['proc'] == proc and x['cfg'] == cfg and x['B'] == B)
            vals = [r['res'][z][1] for z in r['res']]
            xs = np.arange(len(vals)) * 1.0 + (i - 2) * w + (k - 0.5) * w / 2
            ax.bar(xs, vals, w / 2, color=col, hatch=hatch, edgecolor='white' if hatch is None else col, alpha=1 if hatch is None else 0.45,
                   label=lab if k == 0 and j == 0 else None, lw=0.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['in canyon\n3–9 m', 'above roofs\n26–40 m'])
    ax.set_title(f'street canyon, {case}', fontsize=8, loc='left'); ax.grid(alpha=0.3, axis='y')
axs[0, 0].set_ylabel('crossings with Pd ≥ 0.9'); axs[1, 0].set_ylabel('P(detected within 50 m of track)')
axs[0, 0].legend(fontsize=6, frameon=False, loc='upper right'); axs[1, 0].legend(fontsize=6, frameon=False, loc='upper right')
fig.text(0.5, 0.005, f'{B} triaxial nodes per 100 m; solid/filled: PWM carrier coherent; dashed/hatched: ELF fundamental joint-tracked; '
         '\nground 100 Ω·m; measured background-clutter penalty 5 dB', ha='center', fontsize=6.5)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(os.path.join(HERE, 'site_model.png'), dpi=180)
print('saved')
