"""Small helpers for block diagrams drawn with matplotlib (vector PDF, same fonts as the data figures)."""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'vector_sim'))
import plotstyle
OUTDIR = os.path.join(HERE, 'output')

FILL = {'acq': '#e8eef7', 'proc': '#fdf0e6', 'out': '#e9f4ea', 'src': '#f3ecf7', 'noise': '#fbeaea', 'neutral': '#f2f2f2'}
EDGE = {'acq': '#2874a6', 'proc': '#c0662b', 'out': '#1e8449', 'src': '#7d3c98', 'noise': '#b03a2e', 'neutral': '#555555'}


def canvas(w, h):
    plotstyle.apply()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, text, kind='proc', fs=6.8, bold_first=True, ha='center'):
    """(x, y) = lower-left corner, inches.  First line of text is bold if bold_first."""
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.05', lw=0.7,
                       fc=FILL[kind], ec=EDGE[kind])
    ax.add_patch(p)
    lines = text.split('\n')
    if bold_first and len(lines) > 1:
        th = fs / 72 * 1.35; lh = (fs - 0.6) / 72 * 1.18
        ytop = y + h / 2 + (th + (len(lines) - 1) * lh) / 2          # vertically centred text block
        xx = x + w / 2 if ha == 'center' else x + 0.05
        ax.text(xx, ytop, lines[0], ha=ha, va='top', fontsize=fs, fontweight='bold')
        ax.text(xx, ytop - th, '\n'.join(lines[1:]), ha=ha, va='top', fontsize=fs - 0.6, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold_first else 'normal')
    return (x, y, w, h)


def arrow(ax, p, q, text=None, fs=6, style='-|>', color='#333333', rad=0.0, lw=0.8, toff=(0, 0.05)):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, lw=lw, color=color,
                        connectionstyle=f'arc3,rad={rad}', shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if text:
        ax.text((p[0] + q[0]) / 2 + toff[0], (p[1] + q[1]) / 2 + toff[1], text, ha='center', va='bottom', fontsize=fs, color='#333333')


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


def label(ax, x, y, text, fs=7, **kw):
    ax.text(x, y, text, fontsize=fs, **kw)


def save(fig, stem):
    plotstyle.save(fig, stem, OUTDIR); plt.close(fig); print('saved', stem)
