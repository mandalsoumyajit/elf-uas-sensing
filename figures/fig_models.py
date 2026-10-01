"""Model figures: detection range by class/environment (Section V), array design (Section VII), vector simulations (VI)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import blocks as B
import plotstyle

SM = os.path.join(B.ROOT, 'source_model'); SIM = os.path.join(B.ROOT, 'vector_sim', 'simulation_results', 'v6_realistic')
plotstyle.apply()
C4 = ['#2874a6', '#1e8449', '#c0392b', '#c0662b', '#7d3c98', '#555555']

def tag(ax, s):
    ax.set_title(s, loc='left', fontsize=8)

def ranges():
    O = json.load(open(os.path.join(SM, 'range_by_class.json')))
    classes = ['micro (sub-250 g, 3", 4S)', '5-inch FPV (6S)', 'photo/mid (11" props, 4S)', 'heavy-lift (30" props, 12S)']
    short = ['micro\n<250 g', '5-inch\nFPV', 'photo\n11"', 'heavy-lift\n30"']
    scen = [('quiet outdoor', 'quiet outdoor'), ('semi-urban outdoor', 'semi-urban'), ('indoor lab', 'indoor (measured)'), ('indoor near electronics', 'indoor, near electronics')]
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.55), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, band, title in zip(axs, ['current (<=10 kHz)', 'wideband'], ['(a) receiver band ≤ 10 kHz (present hardware)', '(b) wideband receiver (incl. 16–85 kHz PWM lines)']):
        x = np.arange(4)
        for k, (s, lab) in enumerate(scen):
            xs = x + (k - 1.5) * 0.19
            keep = np.array([not (s.startswith('indoor') and i >= 2) for i in range(4)])
            for mode, mk, fill in (('0.2 s', 'o', 'none'), ('60 s tracked', 's', C4[k])):
                r = np.array([O[f'{cl} | {s} | {band} | {mode}']['range'] for cl in classes])[keep]
                ax.errorbar(xs[keep] + (0.045 if mode != '0.2 s' else -0.045), r[:, 1], yerr=[r[:, 1] - r[:, 0], r[:, 2] - r[:, 1]],
                            fmt=mk, ms=3.3, color=C4[k], mfc=fill, lw=0.8, capsize=1.3, label=lab if mode != '0.2 s' else None)
        ax.set_yscale('log'); ax.set_ylim(0.2, 80); ax.set_xticks(x); ax.set_xticklabels(short)
        ax.grid(alpha=0.3, which='major', axis='y'); tag(ax, title); ax.spines[['top', 'right']].set_visible(False)
    axs[0].set_ylabel('detection range (m), P$_d$ = 0.9'); axs[0].legend(fontsize=6.2, loc='upper left', ncol=2)
    axs[1].text(0.98, 0.03, 'open circles: 0.2 s snapshot\nfilled squares: 60 s tracked, coherent', transform=axs[1].transAxes, ha='right', va='bottom', fontsize=6.2)
    B.save(fig, 'model_range')

def array():
    cur = json.load(open(os.path.join(SM, 'array_model_curtain.json')))
    site = json.load(open(os.path.join(SM, 'site_model.json')))
    wit = json.load(open(os.path.join(SM, 'site_witness.json')))
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), gridspec_kw=dict(wspace=0.38, width_ratios=[1, 1, 1.15]))
    ax = axs[0]
    sty = {('P3', False): ('ELF, single-axis', C4[0], '--'), ('P3', True): ('ELF, triaxial', C4[0], '-'),
           ('PWM', False): ('PWM, single-axis', C4[2], '--'), ('PWM', True): ('PWM, triaxial', C4[2], '-')}
    for (proc, tri), (lab, col, ls) in sty.items():
        rr = sorted([r for r in cur if r['case'] == 'DJI-like photo 11"' and r['bg'] == 'semi-urban outdoor' and r['proc'] == proc and r['triax'] == tri], key=lambda r: r['pitch'])
        ax.plot([r['pitch'] for r in rr], [r['frac90'] for r in rr], ls, color=col, marker='o', ms=2.2, lw=1.0, label=lab)
    ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(['2', '4', '8', '16']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('curtain node pitch (m)'); ax.set_ylabel('crossings detected (P$_d$ ≥ 0.9)'); ax.legend(fontsize=5.6, loc='lower left', handlelength=1.8)
    tag(ax, '(a) 30 m curtain, DJI-like')
    ax = axs[1]
    lay = [('ground', 'ground 1.5 m', C4[5], 'v'), ('poles', 'poles 5+10 m', C4[0], 'o'), ('poles+roofs', 'poles + rooftops', C4[1], 's'),
           ('towers', '25 m towers', C4[3], '^'), ('curtain', 'ideal curtain', 'k', 'x')]
    for cfg, lab, col, mk in lay:
        r = next(x for x in site['perimeter'] if x['case'] == 'DJI-like' and x['proc'] == 'PWM' and x['cfg'] == cfg and x['B'] == 16)
        zs = list(r['res'].keys()); mid = [np.mean([float(v) for v in k.split('-')]) for k in zs]
        ax.plot(mid, [r['res'][k][1] for k in zs], '-', marker=mk, ms=2.8, color=col, lw=1.0, label=lab)
    ax.set_xlabel('crossing altitude (m)'); ax.set_ylabel('crossings detected (P$_d$ ≥ 0.9)'); ax.legend(fontsize=5.6, loc='upper right')
    tag(ax, '(b) perimeter, 16 nodes/100 m')
    ax = axs[2]
    cfgs = [('none', 'none'), ('witnesses, 97% of local power', 'witnesses'), ('remote ref, node-grade (3 fT)', 'remote ref.'),
            ('wit 97% + node-grade ref', 'both'), ('ideal (to receiver noise, no clutter)', 'ideal')]
    cases = [('DJI-like', 'PWM', 'DJI-like, PWM', C4[2]), ('heavy-lift', 'P3', 'heavy-lift, ELF line', C4[0]), ('5-inch FPV', 'PWM', '5-inch, PWM', C4[3])]
    x = np.arange(len(cfgs)); w = 0.26
    for j, (case, proc, lab, col) in enumerate(cases):
        v = [next(r for r in wit if r['kind'] == 'canyon' and r['config'] == c and r['case'] == case and r['proc'] == proc and r['B'] == 8)['res']['3-9'][1] for c, _ in cfgs]
        ax.bar(x + (j - 1) * w, v, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in cfgs], rotation=30, ha='right', fontsize=6.2)
    ax.set_ylim(0, 1.05); ax.set_ylabel('P(detected within 50 m)'); ax.legend(fontsize=5.6, loc='upper left')
    tag(ax, '(c) street canyon, 8 nodes/100 m')
    B.save(fig, 'model_array')

def simulation():
    inc = json.load(open(os.path.join(SIM, 'inclination.json'))); pos = json.load(open(os.path.join(SIM, 'position.json')))
    lv = [('quiet outdoor (0.02 pT)', '0.02 pT'), ('after cancellation (0.3 pT)', '0.3 pT'), ('indoor / semi-urban (1 pT)', '1 pT'), ('noisy indoor (10 pT)', '10 pT')]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6), gridspec_kw=dict(wspace=0.36))
    rb = np.array(inc['range_bins_m']).mean(1)
    for ax, T in zip(axs[:2], (0.2, 1.0)):
        for k, (key, lab) in enumerate(lv[:3]):
            ax.plot(rb, inc[f'T={T}|{key}|resolved']['rmse_by_range'], '-o', ms=2.5, color=C4[k], lw=1.0, label=f'vector, tone-resolved, {lab}')
            ax.plot(rb, inc[f'T={T}|{key}|compact']['rmse_by_range'], '--', color=C4[k], lw=0.8)
        ax.plot(rb, inc[f'T={T}|indoor / semi-urban (1 pT)|z']['rmse_by_range'], ':', color='k', lw=1.0, label='single loop (any level)')
        ax.axhline(inc[f'T={T}|mean_predictor'], color='0.6', lw=0.6)
        ax.set_ylim(0, 14); ax.set_xlabel('horizontal range (m)'); ax.set_ylabel('inclination RMSE (deg)')
        tag(ax, f'({"a" if T == 0.2 else "b"}) one vector node, T = {T:g} s')
    h, l = axs[0].get_legend_handles_labels(); h.append(plt.Line2D([], [], ls='--', color='0.4')); l.append('six compact features (dashed)')
    fig.legend(h, l, loc='lower left', ncol=3, fontsize=6, bbox_to_anchor=(0.04, -0.01))
    ax = axs[2]
    x = np.arange(4); w = 0.2
    series = [('triax', 'known', 'triaxial, attitude known', C4[0], None), ('triax', 'free', 'triaxial, attitude-free', C4[2], None),
              ('scalar', 'known', 'single-axis, attitude known', C4[0], '///'), ('scalar', 'free', 'single-axis, attitude-free', C4[2], '///')]
    for j, (n, e, lab, col, h) in enumerate(series):
        v = [pos[f'T=1.0|{key}|{n}|{e}']['median_m'] for key, _ in lv]
        ax.bar(x + (j - 1.5) * w, v, w, color=col if h is None else 'white', edgecolor=col, hatch=h, lw=0.6, label=lab)
    cr = [pos[f'T=1.0|{key}|triax|crlb_free']['median_m'] for key, _ in lv]
    ax.plot(x - 0.1, cr, 'k_', ms=9, mew=1.2, label='CRLB, triaxial attitude-free')
    ax.axhline(pos['centre_baseline_median_m'], color='0.6', lw=0.6, ls=':')
    ax.set_yscale('log'); ax.set_ylim(1e-3, 3e4); ax.set_xticks(x); ax.set_xticklabels([l for _, l in lv], fontsize=6.3)
    ax.set_xlabel('background ASD at 1.1 kHz'); ax.set_ylabel('median position error, T = 1 s (m)'); ax.legend(fontsize=5.2, loc='upper left', ncol=1, handlelength=1.2)
    tag(ax, '(c) four nodes, 5 m square')
    fig.subplots_adjust(bottom=0.27)
    B.save(fig, 'sim_results')

if __name__ == '__main__':
    for w in (sys.argv[1:] or ['ranges', 'array']):
        globals()[w]()
