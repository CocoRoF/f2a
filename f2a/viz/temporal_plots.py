"""Temporal and sequential visualizations.

New in v1.1.1: Autocorrelation plots, rolling statistics,
trend overlays, and lag scatter plots.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from f2a.viz.theme import F2ATheme


class TemporalPlotter:
    """Generates temporal and sequential analysis charts."""

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or F2ATheme()
        self._theme.apply()

    def autocorrelation_plot(
        self,
        acf_data: dict[str, list[float]],
        title: str = "Autocorrelation (ACF)",
        **kwargs: Any,
    ) -> plt.Figure:
        """Bar plot of autocorrelation coefficients per lag."""
        cols = list(acf_data.keys())[:8]
        if not cols:
            return self._empty("No autocorrelation data")

        n = len(cols)
        n_cols = min(4, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        palette = sns.color_palette("husl", n)
        for i, col in enumerate(cols):
            ax = axes_flat[i]
            vals = acf_data[col]
            lags = list(range(1, len(vals) + 1))

            colors = ["#e74c3c" if abs(v) > 0.5 else "#3498db" for v in vals]
            ax.bar(lags, vals, color=colors, width=0.7, alpha=0.8)

            # Confidence bounds (approximate)
            n_obs = 100  # estimated
            ci = 1.96 / np.sqrt(n_obs)
            ax.axhline(ci, color="#aaa", linestyle="--", linewidth=0.8)
            ax.axhline(-ci, color="#aaa", linestyle="--", linewidth=0.8)
            ax.axhline(0, color="#333", linewidth=0.5)

            ax.set_title(col, fontsize=10, fontweight="600")
            ax.set_xlabel("Lag", fontsize=8)
            ax.set_ylim(-1, 1)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    def rolling_stats_plot(
        self,
        df: pd.DataFrame,
        rolling_data: dict[str, dict[str, Any]],
        title: str = "Rolling Statistics",
        **kwargs: Any,
    ) -> plt.Figure:
        """Rolling mean and std deviation overlay."""
        cols = list(rolling_data.keys())[:6]
        if not cols:
            return self._empty("No rolling stats data")

        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(12, n * 2.5), sharex=False)
        if n == 1:
            axes = [axes]

        palette = sns.color_palette("husl", n)
        for i, col in enumerate(cols):
            ax = axes[i]
            info = rolling_data[col]
            window = info.get("window", 10)

            series = df[col].dropna().values
            x = np.arange(len(series))

            ax.plot(x, series, alpha=0.3, linewidth=0.5, color="#bbb", label="Raw")

            rm = pd.Series(series).rolling(window=window, min_periods=1).mean()
            rs = pd.Series(series).rolling(window=window, min_periods=1).std()

            ax.plot(x, rm, color=palette[i], linewidth=2, label=f"Rolling Mean (w={window})")
            ax.fill_between(x, rm - rs, rm + rs, alpha=0.15, color=palette[i])

            ax.set_ylabel(col, fontsize=9, fontweight="600")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    def trend_plot(
        self,
        df: pd.DataFrame,
        trend_data: dict[str, dict[str, Any]],
        title: str = "Trend Detection",
        **kwargs: Any,
    ) -> plt.Figure:
        """Scatter with trend line overlay for each column."""
        cols = list(trend_data.keys())[:6]
        if not cols:
            return self._empty("No trend data")

        n = len(cols)
        n_cols = min(3, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        palette = sns.color_palette("husl", n)
        for i, col in enumerate(cols):
            ax = axes_flat[i]
            info = trend_data[col]
            series = df[col].dropna().values
            x = np.arange(len(series))

            ax.scatter(x, series, s=3, alpha=0.3, color=palette[i])

            slope = info["slope"]
            intercept = series.mean() - slope * x.mean()
            trend_line = slope * x + intercept

            trend = info["trend"]
            line_color = "#e74c3c" if trend == "decreasing" else "#27ae60" if trend == "increasing" else "#95a5a6"
            ax.plot(x, trend_line, color=line_color, linewidth=2.5, linestyle="--",
                    label=f"Trend: {trend}")

            ax.set_title(col, fontsize=10, fontweight="600")
            ax.legend(fontsize=7, loc="best")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    def lag_scatter(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        lag: int = 1,
        max_sample: int = 1000,
        title: str | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """Lag scatter plots (value at t vs value at t-lag)."""
        cols = numeric_cols[:6]
        if not cols:
            return self._empty("No columns for lag scatter")

        n = len(cols)
        n_cols = min(3, n)
        n_rows = math.ceil(n / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        palette = sns.color_palette("husl", n)
        for i, col in enumerate(cols):
            ax = axes_flat[i]
            series = df[col].dropna().values
            if len(series) <= lag:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", fontsize=9)
                ax.set_title(col, fontsize=10)
                continue

            x = series[:-lag]
            y = series[lag:]

            if len(x) > max_sample:
                idx = np.random.default_rng(42).choice(len(x), max_sample, replace=False)
                x, y = x[idx], y[idx]

            ax.scatter(x, y, s=5, alpha=0.4, color=palette[i])

            # Diagonal reference line
            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, "--", color="#ccc", linewidth=1)

            ax.set_xlabel(f"t", fontsize=8)
            ax.set_ylabel(f"t + {lag}", fontsize=8)
            ax.set_title(col, fontsize=10, fontweight="600")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(title or f"Lag-{lag} Scatter Plots", fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    def _empty(self, msg: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#999")
        ax.axis("off")
        return fig
