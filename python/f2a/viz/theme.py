"""Light theme configuration for matplotlib/seaborn charts.

Designed to match the f2a HTML report's light-theme aesthetic.
"""

import matplotlib.pyplot as plt

F2A_PALETTE = [
    "#3498db", "#27ae60", "#f39c12", "#e74c3c",
    "#9b59b6", "#1abc9c", "#2ecc71", "#e67e22",
    "#2980b9", "#8e44ad", "#16a085", "#d35400",
]

BG_COLOR = "#ffffff"
SURFACE_COLOR = "#f8f9fa"
TEXT_COLOR = "#2c3e50"
GRID_COLOR = "#e0e0e0"
ACCENT_COLOR = "#3498db"


def apply_light_theme() -> None:
    """Apply the f2a light theme to matplotlib globally."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": SURFACE_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.facecolor": BG_COLOR,
        "legend.edgecolor": GRID_COLOR,
        "legend.framealpha": 0.8,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "savefig.facecolor": BG_COLOR,
        "savefig.bbox": "tight",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT_COLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Backward compatibility alias
apply_dark_theme = apply_light_theme
