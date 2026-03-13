"""Distribution visualization module."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from f2a.core.schema import DataSchema
from f2a.viz.theme import DEFAULT_THEME, F2ATheme


class DistributionPlotter:
    """Generate distribution-related visualizations."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        theme: F2ATheme | None = None,
    ) -> None:
        self._df = df
        self._schema = schema
        self._theme = theme or DEFAULT_THEME

    def violin_plots(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """Generate violin plots for numeric columns."""
        cols = columns or self._schema.numeric_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.violinplot(data=self._df, y=col, ax=ax, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("Violin Plots", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig

    def kde_plots(self, columns: list[str] | None = None, **kwargs: Any) -> plt.Figure:
        """Generate KDE (Kernel Density Estimation) plots for numeric columns."""
        cols = columns or self._schema.numeric_columns
        if not cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            return fig

        n = len(cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flat if n > 1 else [axes]

        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.kdeplot(data=self._df, x=col, ax=ax, fill=True, **kwargs)
            ax.set_title(col)

        for idx in range(n, len(list(axes))):
            axes[idx].set_visible(False)

        fig.suptitle("Kernel Density Estimation", fontsize=self._theme.title_size + 2, y=1.02)
        fig.tight_layout()
        return fig
