"""Shared matplotlib style for publication-quality IEEE two-column PDFs.

Import this module at the top of each study script:
    import plotstyle
    plotstyle.apply()

Key choices:
- Type 42 (TrueType) font embedding so IEEE Xplore PDF checker accepts the file
- Serif fonts matching IEEE conference body text
- Figure widths: 3.5" single-column, 7.16" double-column
- Tick/label sizes tuned for 8pt body text reading at print resolution
- No color-only cues: markers + linestyles provide redundancy for B/W printing
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


# IEEE column widths in inches
WIDTH_SINGLE = 3.5
WIDTH_DOUBLE = 7.16

# Consistent color palette across all figures. Chosen for CVD safety
# (deuteranopia) and B/W print legibility.
COLOR_VECTOR = "#c0392b"   # deep red
COLOR_SCALAR = "#2874a6"   # steel blue
COLOR_Z      = "#1e8449"   # forest green
COLOR_BASELINE = "#555555" # medium gray
COLOR_REF    = "#888888"   # reference lines
COLOR_BLACK  = "#000000"


def apply():
    """Apply rcParams for publication figures."""
    mpl.rcParams.update({
        # Embed fonts as Type 42 (TrueType) so IEEE PDF checker accepts
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,

        # Line weights
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # Layout
        "axes.grid": False,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,

        # Output
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 130,
    })


def save(fig, stem, outdir="."):
    """Save figure as both PDF (for the paper) and PNG (for quick preview).

    Parameters
    ----------
    fig : matplotlib Figure
    stem : str, filename without extension
    outdir : str, output directory
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    png_path = os.path.join(outdir, f"{stem}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    return pdf_path
