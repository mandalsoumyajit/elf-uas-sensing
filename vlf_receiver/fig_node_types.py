"""Figure for the two node types (paper Fig. dual_channel): (a) field-equivalent noise of the ELF node (published
100-turn design) and VLF node options vs frequency, with background models; (b) damping of the 12-turn VLF winding:
undamped, passive shunt, retuned active damping, and the response after the VLF output filters.

Run coil_model.py and vlf_damping_study.py first; the figure reads their LTspice results from this folder.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spice as S

ROOT = S.HERE.parent
sys.path.insert(0, str(ROOT / 'vector_sim'))
import plotstyle
plotstyle.apply()

C = json.load(open(S.HERE / 'coil_model.json'))
VLF = {c['turns']: c for c in C['vlf']}; ELF = {m['turns']: m for m in C['elf']}
OUT = ROOT / 'figures' / 'output'
OUT.mkdir(parents=True, exist_ok=True)

# Background models from source_model/range_by_class.py (definitions only; the module body needs other results)
RBC = ROOT / 'source_model' / 'range_by_class.py'
ns = {'__file__': str(RBC)}; exec(RBC.read_text().split('SCEN =')[0], ns)

# ELF node noise (published 100-turn design, unchanged netlist values)
e = ELF[100]
core = S.core_text(e['R'], e['L'], e['C'], 270e-12, 2584.31315529, 6.8e3, 'output_filters.inc')
zE = S.run('elf100_wide', core, 'noise', fstart=50, fstop=3e5)
curves = [('ELF node, 100 turns (v1)', zE, e['area'], '#2874a6', '-', (100, 1e4))]
for n, kind, rint, col, ls in ((12, 'active', 100, '#c0392b', '-'), (24, 'active', 399, '#c0662b', '--'), (8, 'passive', None, '#7d3c98', ':')):
    name = f'vlf{n}_{kind}_{int(rint) if rint else 0}_noise.raw'
    z = S.raw(S.HERE / name)
    lab = f'VLF node, {n} turns, {"active" if kind == "active" else "passive"} damping'
    curves.append((lab, z, VLF[n]['area'], col, ls, (1e4, 2e5)))

fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw=dict(wspace=0.3))
ax = axs[0]
for lab, z, area, col, ls, (lo, hi) in curves:
    f = z['frequency'].real; en = z['v(inoise)'].real; m = (f >= lo) & (f <= hi)
    ax.loglog(f[m] / 1e3, en[m] / (2 * np.pi * f[m] * area) * 1e15, ls, color=col, lw=1.2, label=lab)
ff = np.logspace(2, np.log10(2e5), 200)
ax.loglog(ff / 1e3, ns['natural'](ff) * 1e15, color='0.55', lw=0.9, ls='-.', label='natural background (model)')
ax.loglog(ff / 1e3, ns['semi_urban'](ff) * 1e15, color='0.3', lw=0.9, ls=(0, (1, 1)), label='semi-urban background (model)')
for fx in (24, 48, 85):
    ax.axvline(fx, color='0.85', lw=0.6, zorder=0)
ax.set_xlabel('frequency (kHz)'); ax.set_ylabel('field noise (fT/$\\sqrt{\\mathrm{Hz}}$)')
ax.set_ylim(0.1, 3e4); ax.legend(fontsize=5.4, loc='upper right', frameon=True, framealpha=0.95, edgecolor='none', ncol=2, columnspacing=0.8, handlelength=1.6); ax.set_title('(a) receiver noise vs background', loc='left', fontsize=8)
ax = axs[1]
for fn, lab, col, ls in (('vlf12_undamped_0_ac.raw', 'undamped', '0.5', ':'), ('vlf12_passive_0_ac.raw', 'passive shunt', '#7d3c98', '--'),
                         ('vlf12_active_100_ac.raw', 'active (retuned)', '#c0392b', '-')):
    a = S.raw(S.HERE / fn); f = a['frequency'].real; m = (f >= 3e3) & (f <= 2e6)
    ax.semilogx(f[m] / 1e3, 20 * np.log10(np.abs(a['v(out)'][m])), ls, color=col, lw=1.1, label=f'{lab}, front-end output')
    if 'active' in fn:
        ax.semilogx(f[m] / 1e3, 20 * np.log10(np.abs(a['v(filtered)'][m])), '-', color='#2874a6', lw=1.1, label='active, after VLF output filters')
ax.axvline(470, color='0.8', lw=0.6, zorder=0); ax.text(480, 62, 'SRF', fontsize=6, color='0.4')
ax.set_xlabel('frequency (kHz)'); ax.set_ylabel('gain (dB re 1 V/V EMF)'); ax.set_ylim(10, 70)
ax.legend(fontsize=5.8, loc='lower left'); ax.set_title('(b) VLF node, 12-turn winding: damping', loc='left', fontsize=8)
for x in (axs[0], axs[1]):
    x.spines[['top', 'right']].set_visible(False)
plotstyle.save(fig, 'dual_channel', str(OUT)); print('saved', OUT / 'dual_channel')
for lab, z, area, col, ls, (lo, hi) in curves:
    f = z['frequency'].real; en = z['v(inoise)'].real
    print(lab, {fx: round(float(np.interp(fx * 1e3, f, en) / (2 * np.pi * fx * 1e3 * area) * 1e15), 2) for fx in (0.5, 1, 5, 24, 48, 85) if lo <= fx * 1e3 <= hi})
