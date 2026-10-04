"""Explanatory block diagrams for the revised manuscript (vector PDF in images/v2)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import blocks as B
from blocks import box, arrow, right, left, top, bottom

# ------------------------------------------------------------------ 1. experimental processing pipeline
def pipeline():
    fig, ax = B.canvas(7.16, 1.95)
    w, h = 1.62, 0.56; y1 = 1.33
    b1 = box(ax, 0.04, y1, w, h, 'Loop array\nfour 1 m, 300-turn loops at the\ncorners of a 5 m square', 'acq')
    b2 = box(ax, 1.82, y1, w, h, 'Analog front ends\nLNA, filtering, bias and scaling\n(gain not characterized)', 'acq')
    b3 = box(ax, 3.60, y1, w, h, 'Synchronous ADC\n4 channels, 20 kS/s, 12 bit;\n~316 s per cell, 25 cells', 'acq')
    b4 = box(ax, 5.38, y1, 1.74, h, 'Per-node mains cancellation\norder-tracked synchronous comb\n(Section IV-C)', 'proc')
    for p, q in ((b1, b2), (b2, b3), (b3, b4)):
        arrow(ax, right(p), left(q))
    b5 = box(ax, 5.38, 0.30, 1.74, 0.66, 'Reference-node line tracking\nstrongest motor line on the\nnearest node (10 ms ridge);\nmotor phase ψ(t)', 'proc')
    b6 = box(ax, 3.60, 0.30, w, 0.66, 'Phase-referenced demodulation\nall nodes demodulated with ψ(t)\n→ complex line amplitudes', 'proc')
    b7 = box(ax, 1.82, 0.68, w, 0.44, 'Coherent detection\nSNR grows ~10 dB/decade', 'out')
    b8 = box(ax, 1.82, 0.10, w, 0.44, 'Field-model inversion\ndipole source + finite loops', 'out')
    b9 = box(ax, 0.04, 0.10, w, 0.44, 'Position of held-out cell\nleave-one-cell-out test', 'out')
    b10 = box(ax, 0.04, 0.68, w, 0.44, 'Far-node detection\n6.5 m in 10–100 s', 'out')
    arrow(ax, bottom(b4), top(b5), 'ψ from reference node', toff=(0.62, -0.06))
    arrow(ax, (5.38, y1 + 0.05), (4.85, 0.96), 'all nodes', toff=(-0.12, 0.0))
    arrow(ax, left(b5), right(b6))
    arrow(ax, (3.60, 0.80), right(b7)); arrow(ax, (3.60, 0.42), right(b8))
    arrow(ax, left(b8), right(b9)); arrow(ax, left(b7), right(b10))
    B.save(fig, 'diagram_pipeline')

# ------------------------------------------------------------------ 2. equivalent-source model
def source():
    fig, ax = B.canvas(7.16, 3.25)
    b1 = box(ax, 0.04, 2.12, 1.36, 1.00, 'Operating point\nhover thrust per motor\n→ rpm (thrust table)\n→ fₘ, fₑ = P fₘ\n→ phase current, duty', 'src')
    srcs = ['Rotor residual dipole\nrotating at fₘ (magnet imbalance)',
            'Phase-lead loop\nlinear, body axis; fₑ and 6k±1 harmonics',
            'Winding net dipole\nrotating at fₑ (small)',
            'ESC switching loop, DC bus\nPWM carrier f_pwm; 6fₑ ripple']
    bs = [box(ax, 1.62, 2.80 - i * 0.44, 2.10, 0.38, s_, 'src', fs=6.5) for i, s_ in enumerate(srcs)]
    b3 = box(ax, 3.96, 2.12, 1.44, 1.00, 'Four-motor sum\nattitude R(θ, φ, γ)\nrotor offsets qₖ\nindependent motor phases\nfrequency wander', 'src')
    b4 = box(ax, 5.62, 2.12, 1.50, 1.00, 'Propagation, reception\nquasi-static dipole tensor\n+ ground secondary field\nloop flux, finite aperture\nfront-end response', 'acq')
    for b in bs:
        arrow(ax, right(b1), left(b), lw=0.6)
        arrow(ax, right(b), left(b3), lw=0.6)
    arrow(ax, right(b3), left(b4))
    sx = fig.add_axes([0.10, 0.09, 0.86, 0.24])
    lines = [(153, 2.0e-4, 'rotor residual, fₘ'), (1074, 1.06e-3, 'phase leads, fₑ'), (5372, 4.0e-4, '5fₑ'), (6446, 1.1e-4, '6fₑ'),
             (7520, 3.3e-4, '7fₑ'), (24000, 4.6e-4, 'PWM carrier, f_pwm')]
    for f, m, lab in lines:
        sx.vlines(f, 1e-5, m, color=B.EDGE['src'], lw=1.2); sx.plot(f, m, 'o', ms=3, color=B.EDGE['src'])
        dx = {'5fₑ': 0.84, '7fₑ': 1.16}.get(lab, 1.0)
        sx.text(f * dx, m * 1.7, lab, ha='center', va='bottom', fontsize=5.8)
    sx.set_xscale('log'); sx.set_yscale('log'); sx.set_ylim(1e-5, 4e-2); sx.set_xlim(80, 6e4)
    sx.set_xlabel('frequency (Hz)', fontsize=6.5, labelpad=1); sx.set_ylabel('moment (A m²)', fontsize=6.5, labelpad=1)
    sx.tick_params(labelsize=6); sx.spines[['top', 'right']].set_visible(False)
    sx.set_title('Spectral lines of one 5-inch-class motor at hover (nominal per-motor moments; each spans ~×5 either way)',
                 fontsize=6.5, loc='left')
    B.save(fig, 'diagram_source')

# ------------------------------------------------------------------ 3. noise classes and mitigation
def noise():
    H = 1.82
    fig, ax = B.canvas(7.16, H)
    rows = [('Local periodic interference\nmains comb from wiring; up to 94% of in-band power', 'Per-node synchronous comb\norder-tracked on the mains phase; O(1) per sample', 'measured: −17 dB total,\ndrone line ±0.1 dB'),
            ('Local broadband noise\nnearby electronics; incoherent between nodes (γ² < 0.03)', 'Witness sensors at the sources\ncoil or current clamp; low sensitivity suffices', 'modelled; needs coverage\nof most local power'),
            ('Distant broadband noise\nsferics, distant grid; coherent over the site', 'Remote reference (Wiener)\nnode-grade sensor needed in the VLF band', 'modelled; site coherence\nnot yet measured'),
            ('Intermittent narrowband lines\nsingle-node, persist > 2 s; set the false-alarm level', 'CFAR + spatial consistency\ndipole fit across nodes; site line catalogue', 'measured penalty ≈ 5 dB\nat Pfa 10⁻⁷ per cell'),
            ('Impulsive bursts\nheavy envelope tails (Class A, A ≈ 0.003–0.05)', 'Blanking / clipping\nshort-window tracking stage', 'measured: helps short\nwindows only')]
    cols = ((0.04, 2.45, 'Background class'), (2.82, 2.45, 'Mitigation'), (5.60, 1.52, 'Evidence'))
    for x, w, t in cols:
        ax.text(x + w / 2, H - 0.03, t, ha='center', va='top', fontsize=6.6, fontweight='bold', color='0.3')
    h = 0.29
    for i, (a, b, c) in enumerate(rows):
        y = H - 0.18 - h - i * 0.32
        ba = box(ax, cols[0][0], y, cols[0][1], h, a, 'noise', fs=6.2)
        bb = box(ax, cols[1][0], y, cols[1][1], h, b, 'proc', fs=6.2)
        bc = box(ax, cols[2][0], y, cols[2][1], h, c, 'neutral', fs=5.9, bold_first=False)
        arrow(ax, right(ba), left(bb)); arrow(ax, right(bb), left(bc), style='-', lw=0.5)
    B.save(fig, 'diagram_noise')

# ------------------------------------------------------------------ 4. deployment geometries
def node(ax, x, z, kind):
    st = {'g': ('v', '#555555'), 'p': ('o', '#2874a6'), 'r': ('s', '#c0392b'), 'f': ('D', '#1e8449'), 't': ('^', '#c0662b')}[kind]
    ax.plot(x, z, st[0], ms=4.2, color=st[1], mec='k', mew=0.4, zorder=5)

def deploy():
    import plotstyle
    plotstyle.apply()
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.05), gridspec_kw=dict(width_ratios=[1.3, 0.9, 1.05]))
    ax = axs[0]
    ax.axhline(0, color='k', lw=0.8)
    ax.add_patch(Rectangle((-0.3, 0), 0.6, 4, fc='0.75', ec='none'))                  # fence / wall
    for x in (6, 30):
        ax.plot([x, x], [0, 10], color='0.35', lw=1.2); node(ax, x, 5, 'p'); node(ax, x, 10, 'p')
    for x in (-4, 2, 12, 18, 24):
        node(ax, x, 1.5, 'g')
    ax.add_patch(Rectangle((-30, 0), 14, 16, fc='#e5e5e5', ec='0.5', lw=0.6)); node(ax, -23, 17, 'r')
    ax.add_patch(Polygon([[40, 0], [42, 0], [41.2, 25], [40.8, 25]], fc='none', ec='0.35', lw=0.8))
    for z in (6, 12, 18, 24):
        node(ax, 41, z, 't')
    xs = np.linspace(-34, 46, 50); ax.plot(xs, 20 + 0 * xs, '--', color='#7d3c98', lw=0.9)
    ax.annotate('', (44, 20), (38, 20), arrowprops=dict(arrowstyle='-|>', color='#7d3c98', lw=0.9))
    ax.text(-33, 21, 'drone crossing (alt. 2–60 m)', fontsize=6.3, color='#7d3c98')
    ax.text(-28.5, 7.5, 'building\ninside\nperimeter', fontsize=6.0, ha='left')
    ax.text(1, 4.6, 'fence', fontsize=6, rotation=90, va='bottom')
    ax.set_xlim(-35, 47); ax.set_ylim(-1, 30); ax.set_xlabel('distance along the approach (m)'); ax.set_ylabel('height (m)')
    ax.set_title('(a) Perimeter: mounts at ground, poles, rooftops, towers', loc='left')
    ax.spines[['top', 'right']].set_visible(False)
    ax = axs[1]
    ax.axhline(0, color='k', lw=0.8)
    ax.add_patch(Rectangle((-22, 0), 12, 18, fc='#e5e5e5', ec='0.5', lw=0.6)); ax.add_patch(Rectangle((10, 0), 12, 12, fc='#e5e5e5', ec='0.5', lw=0.6))
    node(ax, -9, 1.5, 'g'); node(ax, 9, 1.5, 'g')
    for x in (-9, 9):
        ax.plot([x, x], [0, 8], color='0.35', lw=1.1)
    node(ax, -9, 8, 'p'); node(ax, 9, 8, 'p')
    node(ax, -10, 11, 'f'); node(ax, 10, 6, 'f'); node(ax, -10.5, 19, 'r'); node(ax, 10.5, 13, 'r')
    ax.add_patch(Rectangle((-8, 3), 16, 6, fc='#f3ecf7', ec='#7d3c98', lw=0.6, ls='--'))
    ax.add_patch(Rectangle((-8, 26), 16, 14, fc='#f3ecf7', ec='#7d3c98', lw=0.6, ls='--'))
    ax.text(0, 6, 'in canyon\n3–9 m', ha='center', va='center', fontsize=6.2, color='#7d3c98')
    ax.text(0, 33, 'above roofs\n26–40 m', ha='center', va='center', fontsize=6.2, color='#7d3c98')
    ax.text(-16, 9, 'h = 9–24 m', ha='center', fontsize=6, rotation=90)
    ax.set_xlim(-23, 23); ax.set_ylim(-1, 42); ax.set_xlabel('across the street (m)'); ax.set_ylabel('height (m)')
    ax.set_title('(b) Street canyon (cross-section)', loc='left'); ax.spines[['top', 'right']].set_visible(False)
    # (c) plan view: the two node types scattered independently over a site
    ax = axs[2]
    ax.add_patch(Rectangle((0, 0), 120, 80, fc='none', ec='0.35', lw=0.9))                          # perimeter fence
    ax.add_patch(Rectangle((62, 36), 36, 30, fc='#e5e5e5', ec='0.5', lw=0.6)); ax.text(80, 51, 'building', ha='center', va='center', fontsize=5.8)
    ax.add_patch(Rectangle((14, 44), 22, 18, fc='#e5e5e5', ec='0.5', lw=0.6))
    ax.plot([52, 64], [0, 0], color='white', lw=2.2, solid_capstyle='butt'); ax.text(58, 4, 'gate', ha='center', fontsize=5.6)
    per = [(x, -4) for x in np.arange(0, 121, 15)] + [(x, 84) for x in np.arange(0, 121, 15)] + \
          [(-4, y) for y in np.arange(15, 80, 15)] + [(124, y) for y in np.arange(15, 80, 15)]
    vlf = per + [(25, 63), (80, 67), (40, 30), (85, 22)]                                              # perimeter + rooftops/poles inside
    elf = [(44, 7), (72, 7), (70, 42), (90, 60), (25, 40)]                                            # gate, indoors, loading area
    ax.plot(*zip(*vlf), 'o', ms=3.2, mfc='white', mec='#b03a2e', mew=0.9, ls='', label='VLF node (PWM band)')
    ax.plot(*zip(*elf), 's', ms=3.2, color='k', ls='', label='ELF node (motor band)')
    xs = np.linspace(-14, 105, 40); ax.plot(xs, 98 - 0.9 * (xs + 14), '--', color='#7d3c98', lw=0.9, label='drone track')
    ax.annotate('', (100, 4), (93, 11), arrowprops=dict(arrowstyle='-|>', color='#7d3c98', lw=0.9))
    ax.text(60, -14, '~15 m VLF pitch along the fence', fontsize=5.6, ha='center', color='#b03a2e')
    ax.set_xlim(-16, 136); ax.set_ylim(-20, 106); ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    hc, lc = ax.get_legend_handles_labels()
    fig.legend(hc, ['VLF node', 'ELF node', 'drone track'], loc='lower center', ncol=3, fontsize=6.2, bbox_to_anchor=(0.86, -0.02),
               handletextpad=0.3, columnspacing=0.8)
    ax.set_title('(c) Plan view: two node types', loc='left'); ax.spines[['top', 'right']].set_visible(False)
    hs = [plt.Line2D([], [], ls='', marker=m, color=c, mec='k', mew=0.4, ms=4.2, label=l) for m, c, l in
          (('v', '#555555', 'ground 1.5 m'), ('o', '#2874a6', 'pole / streetlight'), ('D', '#1e8449', 'facade'),
           ('s', '#c0392b', 'rooftop'), ('^', '#c0662b', 'tower'))]
    fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=6.2, bbox_to_anchor=(0.34, -0.02), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); B.save(fig, 'diagram_deployment')

# ------------------------------------------------------------------ 5. simulation chain (Section VI)
def simchain():
    fig, ax = B.canvas(7.16, 0.86)
    w, h, y = 1.30, 0.66, 0.10
    labels = [('Source model\n5-inch class; measured\naxial/rotating ratio\nand frequency wander', 'src'),
              ('Fields at the nodes\nfour rotors, attitude,\nquasi-static tensor;\npoint triaxial/loop nodes', 'src'),
              ('Measured-statistics\nbackground\ncoloured ASD, residual\nmains, impulsive bursts', 'noise'),
              ('Tracking gate\nline SNR ≥ −1.5 dB in\n20 Hz on the strongest\nnode (measured)', 'proc'),
              ('Estimators\ninclination: one node\nposition: four nodes;\nCRLB for reference', 'out')]
    xs = [0.04 + i * 1.44 for i in range(5)]
    bb = [box(ax, x, y, w, h, t, k, fs=6.4) for x, (t, k) in zip(xs, labels)]
    for p, q in zip(bb[:-1], bb[1:]):
        arrow(ax, right(p), left(q))
    B.save(fig, 'diagram_simulation')

if __name__ == '__main__':
    pipeline(); source(); noise(); deploy(); simchain()
