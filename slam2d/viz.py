"""Shared plotting style: one colour per estimator, thin marks, recessive axes.

Colours follow the entity everywhere (paper figures and animation). The three
series colours pass the colour-vision checks pairwise; identity is also carried
by direct labels and legends.
"""
from __future__ import annotations

import matplotlib as mpl

INK = "#0b0b0b"          # ground truth, primary text
INK2 = "#52514e"         # secondary text, reference curves
GRID = "#e4e3df"         # hairline grid, one step off the surface
BAND = "#eceae4"         # 95 % region
SURFACE = "#ffffff"

COLORS = {
    "ekf": "#2a78d6",        # standard EKF   (blue)
    "standard": "#2a78d6",
    "fej": "#eb6834",        # FEJ-EKF        (orange)
    "graph": "#1baf7a",      # own optimiser  (aqua)
    "gtsam": "#eda100",      # GTSAM          (yellow)
    "ideal": INK2,           # ideal EKF, a reference
    "truth": INK,
}
LABELS = {"ekf": "Standard EKF", "standard": "Standard EKF", "fej": "FEJ-EKF",
          "graph": "Graph (own)", "gtsam": "GTSAM", "ideal": "Ideal EKF", "truth": "Ground truth"}

COL_W = 3.5      # IEEE single column [in]
PAGE_W = 7.16    # IEEE double column [in]


def use_paper_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
