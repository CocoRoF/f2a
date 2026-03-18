"""Enhanced visualization module — visually striking and diverse chart types.

New in v1.1.1: Radar charts, parallel coordinates, ridgeline plots,
hexbin density, swarm plots, clustermaps, pair scatter matrices,
and waterfall/lollipop charts.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from f2a.viz.theme import F2ATheme


class EnhancedPlotter:
    """Generates visually enhanced, diverse chart types."""

    def __init__(self, theme: F2ATheme | None = None) -> None:
        self._theme = theme or F2ATheme()
        self._theme.apply()

    # ------------------------------------------------------------------
    # 1. Radar / Spider Chart
    # ------------------------------------------------------------------
    def radar_chart(
        self,
        scores: dict[str, float],
        title: str = "Quality Radar",
        **kwargs: Any,
    ) -> plt.Figure:
        """Create a radar/spider chart for multi-dimensional scores."""
        labels = list(scores.keys())
        values = [min(max(v, 0), 1) for v in scores.values()]
        n = len(labels)
        if n < 3:
            return self._empty_figure("Need >= 3 dimensions for radar chart")

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values_closed = values + values[:1]
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Draw the chart
        ax.plot(angles_closed, values_closed, "o-", linewidth=2.5, color="#3498db", markersize=8)
        ax.fill(angles_closed, values_closed, alpha=0.20, color="#3498db")

        # Add reference ring at 0.5
        ref_vals = [0.5] * (n + 1)
        ax.plot(angles_closed, ref_vals, "--", color="#bdc3c7", linewidth=1, alpha=0.7)

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, size=10, fontweight="600")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=7, color="#999")
        ax.set_title(title, size=14, fontweight="bold", pad=20, color="#2c3e50")

        # Value annotations
        for angle, val, label in zip(angles, values, labels):
            ax.annotate(
                f"{val:.0%}",
                xy=(angle, val),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#e74c3c" if val < 0.5 else "#27ae60",
            )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 2. Parallel Coordinates Plot
    # ------------------------------------------------------------------
    def parallel_coordinates(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        color_col: str | None = None,
        max_sample: int = 500,
        title: str = "Parallel Coordinates",
        **kwargs: Any,
    ) -> plt.Figure:
        """Parallel coordinates plot for multi-dimensional comparison."""
        if len(numeric_cols) < 2:
            return self._empty_figure("Need >= 2 numeric columns")

        cols = numeric_cols[:12]
        sample = df.sample(n=min(len(df), max_sample), random_state=42) if len(df) > max_sample else df

        # Normalize to [0, 1]
        norm_df = pd.DataFrame()
        for c in cols:
            series = pd.to_numeric(sample[c], errors="coerce")
            mn, mx = series.min(), series.max()
            norm_df[c] = (series - mn) / (mx - mn + 1e-12)

        fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.2), 6))

        if color_col and color_col in sample.columns:
            categories = sample[color_col].astype(str)
            unique_cats = sorted(categories.unique())[:10]
            palette = sns.color_palette("husl", len(unique_cats))
            cat_colors = {cat: palette[i] for i, cat in enumerate(unique_cats)}

            for i, (_, row) in enumerate(norm_df.iterrows()):
                cat = categories.iloc[i]
                if cat in cat_colors:
                    ax.plot(range(len(cols)), row.values, alpha=0.25, linewidth=0.8, color=cat_colors[cat])
            handles = [mpatches.Patch(color=c, label=k) for k, c in cat_colors.items()]
            ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, len(norm_df)))
            for i, (_, row) in enumerate(norm_df.iterrows()):
                ax.plot(range(len(cols)), row.values, alpha=0.15, linewidth=0.6, color=colors[i])

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Normalized Value", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50")
        ax.grid(axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 3. Ridgeline / Joy Plot
    # ------------------------------------------------------------------
    def ridgeline_plot(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        title: str = "Ridgeline Distribution",
        **kwargs: Any,
    ) -> plt.Figure:
        """Ridgeline (joy) plot showing stacked KDE distributions."""
        cols = [c for c in numeric_cols[:15] if df[c].notna().sum() > 5]
        if len(cols) < 1:
            return self._empty_figure("No suitable columns for ridgeline")

        n = len(cols)
        fig, axes = plt.subplots(n, 1, figsize=(10, max(4, n * 1.1)), sharex=False)
        if n == 1:
            axes = [axes]

        palette = sns.color_palette("husl", n)
        overlap = 0.6

        for i, (col, ax) in enumerate(zip(cols, axes)):
            data = df[col].dropna().values
            try:
                sns.kdeplot(data, ax=ax, fill=True, color=palette[i], alpha=0.7, linewidth=1.5)
            except Exception:
                ax.hist(data, bins=30, color=palette[i], alpha=0.7, density=True)

            ax.set_yticks([])
            ax.set_ylabel("")
            ax.text(-0.01, 0.5, col, transform=ax.transAxes, fontsize=9,
                    fontweight="600", ha="right", va="center", color="#2c3e50")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            if i < n - 1:
                ax.set_xticklabels([])
                ax.spines["bottom"].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. Hexbin Density Plot
    # ------------------------------------------------------------------
    def hexbin_density(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        gridsize: int = 30,
        title: str | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """2D hexagonal binning density plot."""
        x = pd.to_numeric(df[x_col], errors="coerce").dropna()
        y = pd.to_numeric(df[y_col], errors="coerce").dropna()
        common = x.index.intersection(y.index)
        x, y = x.loc[common], y.loc[common]

        if len(x) < 5:
            return self._empty_figure(f"Not enough data for hexbin ({x_col} vs {y_col})")

        fig, ax = plt.subplots(figsize=(9, 7))
        hb = ax.hexbin(x, y, gridsize=gridsize, cmap="YlOrRd", mincnt=1, edgecolors="white", linewidths=0.2)
        cb = fig.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label("Count", fontsize=10)

        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(title or f"Hexbin Density: {x_col} vs {y_col}", fontsize=14, fontweight="bold", color="#2c3e50")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 5. Swarm Plot
    # ------------------------------------------------------------------
    def swarm_plot(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        max_sample: int = 300,
        title: str = "Swarm Plot (Data Points)",
        **kwargs: Any,
    ) -> plt.Figure:
        """Bee swarm plot showing individual data points."""
        cols = numeric_cols[:8]
        if not cols:
            return self._empty_figure("No numeric columns for swarm plot")

        sample = df.sample(n=min(len(df), max_sample), random_state=42) if len(df) > max_sample else df
        n_cols = len(cols)
        n_rows = math.ceil(n_cols / 4)
        fig, axes = plt.subplots(n_rows, min(4, n_cols), figsize=(min(4, n_cols) * 3.5, n_rows * 4))
        axes_flat = np.array(axes).flatten() if n_cols > 1 else [axes]

        palette = sns.color_palette("husl", n_cols)
        for i, col in enumerate(cols):
            ax = axes_flat[i]
            data = sample[col].dropna()
            try:
                sns.swarmplot(y=data, ax=ax, color=palette[i], size=3, alpha=0.7)
            except Exception:
                sns.stripplot(y=data, ax=ax, color=palette[i], size=3, alpha=0.5, jitter=True)
            ax.set_title(col, fontsize=10, fontweight="600")
            ax.set_ylabel("")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 6. Clustermap (Correlation with Dendrograms)
    # ------------------------------------------------------------------
    def clustermap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Clustered Correlation Heatmap",
        **kwargs: Any,
    ) -> plt.Figure:
        """Correlation heatmap with hierarchical clustering dendrograms."""
        if corr_matrix.empty or corr_matrix.shape[0] < 2:
            return self._empty_figure("Not enough columns for clustermap")

        try:
            g = sns.clustermap(
                corr_matrix,
                cmap="RdBu_r",
                center=0,
                vmin=-1,
                vmax=1,
                annot=corr_matrix.shape[0] <= 12,
                fmt=".2f",
                linewidths=0.5,
                figsize=(max(8, corr_matrix.shape[0] * 0.7), max(8, corr_matrix.shape[0] * 0.7)),
                dendrogram_ratio=(0.12, 0.12),
                cbar_pos=(0.02, 0.8, 0.03, 0.15),
            )
            g.fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
            return g.fig
        except Exception:
            return self._empty_figure("Failed to create clustermap")

    # ------------------------------------------------------------------
    # 7. Pairwise Scatter Matrix (Enhanced)
    # ------------------------------------------------------------------
    def pair_scatter_matrix(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        max_cols: int = 5,
        max_sample: int = 500,
        title: str = "Pair Scatter Matrix",
        **kwargs: Any,
    ) -> plt.Figure:
        """Enhanced pair scatter matrix with regression lines and KDE diagonals."""
        cols = numeric_cols[:max_cols]
        if len(cols) < 2:
            return self._empty_figure("Need >= 2 columns for pair scatter")

        sample = df[cols].sample(n=min(len(df), max_sample), random_state=42) if len(df) > max_sample else df[cols]

        g = sns.PairGrid(sample.dropna(), diag_sharey=False)
        g.map_upper(sns.scatterplot, alpha=0.3, s=15, color="#3498db")
        g.map_lower(sns.kdeplot, cmap="Blues", fill=True, alpha=0.6)
        g.map_diag(sns.histplot, kde=True, color="#3498db", alpha=0.5)
        g.fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        g.fig.tight_layout()
        return g.fig

    # ------------------------------------------------------------------
    # 8. Lollipop Chart (Feature Importance)
    # ------------------------------------------------------------------
    def lollipop_chart(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance (Lollipop)",
        top_n: int = 20,
        **kwargs: Any,
    ) -> plt.Figure:
        """Lollipop chart for feature importance ranking."""
        if importance_df.empty:
            return self._empty_figure("No feature importance data")

        # Get top N features by first numeric column
        val_col = importance_df.select_dtypes(include=[np.number]).columns[0]
        sorted_df = importance_df.nlargest(top_n, val_col)

        fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_df) * 0.35)))

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_df)))
        y_pos = range(len(sorted_df))

        ax.hlines(y=y_pos, xmin=0, xmax=sorted_df[val_col].values, colors=colors, linewidth=2.5, alpha=0.8)
        ax.scatter(sorted_df[val_col].values, y_pos, color=colors, s=80, zorder=5, edgecolors="white", linewidths=1.5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(sorted_df.index.tolist() if sorted_df.index.name or isinstance(sorted_df.index[0], str)
                          else [f"Feature {i}" for i in range(len(sorted_df))], fontsize=9)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 9. Heatmap with Value Annotations (Enhanced)
    # ------------------------------------------------------------------
    def annotated_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Annotated Heatmap",
        cmap: str = "YlOrRd",
        fmt: str = ".2f",
        **kwargs: Any,
    ) -> plt.Figure:
        """Enhanced heatmap with clear value annotations and styling."""
        if data.empty:
            return self._empty_figure("No data for heatmap")

        fig, ax = plt.subplots(figsize=(max(8, data.shape[1] * 0.8), max(6, data.shape[0] * 0.6)))
        sns.heatmap(
            data, annot=data.shape[0] <= 15 and data.shape[1] <= 15,
            fmt=fmt, cmap=cmap, ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8},
            square=data.shape[0] == data.shape[1],
        )
        ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50", pad=15)
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 10. Distribution Comparison (Grouped)
    # ------------------------------------------------------------------
    def distribution_comparison(
        self,
        df: pd.DataFrame,
        numeric_col: str,
        group_col: str,
        title: str | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """Compare distributions across groups with overlaid KDE + rug plots."""
        if numeric_col not in df.columns or group_col not in df.columns:
            return self._empty_figure(f"Missing columns: {numeric_col} or {group_col}")

        groups = df[group_col].astype(str).unique()[:8]
        palette = sns.color_palette("husl", len(groups))

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, grp in enumerate(sorted(groups)):
            data = df.loc[df[group_col].astype(str) == grp, numeric_col].dropna()
            if len(data) > 2:
                try:
                    sns.kdeplot(data, ax=ax, fill=True, alpha=0.25, color=palette[i], label=f"{group_col}={grp}", linewidth=2)
                except Exception:
                    ax.hist(data, bins=30, alpha=0.3, color=palette[i], label=f"{group_col}={grp}", density=True)

        ax.set_xlabel(numeric_col, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(title or f"Distribution of {numeric_col} by {group_col}", fontsize=14, fontweight="bold", color="#2c3e50")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 11. Correlation Scatter with Marginals
    # ------------------------------------------------------------------
    def scatter_with_marginals(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        max_sample: int = 1000,
        title: str | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """Scatter plot with marginal histograms (JointPlot style)."""
        sample = df[[x_col, y_col]].dropna()
        if len(sample) > max_sample:
            sample = sample.sample(max_sample, random_state=42)

        if len(sample) < 3:
            return self._empty_figure(f"Not enough data for {x_col} vs {y_col}")

        g = sns.JointGrid(data=sample, x=x_col, y=y_col, height=8)
        g.plot_joint(sns.scatterplot, alpha=0.4, s=20, color="#3498db")
        g.plot_joint(sns.kdeplot, levels=5, color="#e74c3c", linewidths=1, alpha=0.5)
        g.plot_marginals(sns.histplot, kde=True, color="#3498db", alpha=0.5)
        g.fig.suptitle(
            title or f"{x_col} vs {y_col}",
            fontsize=14, fontweight="bold", color="#2c3e50", y=1.02,
        )
        return g.fig

    # ------------------------------------------------------------------
    # 12. Data Completeness Heatmap (by column groups)
    # ------------------------------------------------------------------
    def completeness_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Data Completeness Overview",
        **kwargs: Any,
    ) -> plt.Figure:
        """Heatmap showing non-null percentage across all columns."""
        completeness = (df.notna().sum() / len(df) * 100).to_frame("Completeness %")
        completeness = completeness.sort_values("Completeness %", ascending=True)

        fig, ax = plt.subplots(figsize=(6, max(4, len(completeness) * 0.35)))

        colors = ["#e74c3c" if v < 50 else "#f39c12" if v < 90 else "#27ae60"
                  for v in completeness["Completeness %"]]

        bars = ax.barh(range(len(completeness)), completeness["Completeness %"].values, color=colors, height=0.7)

        for i, (bar, val) in enumerate(zip(bars, completeness["Completeness %"].values)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8, fontweight="bold",
                    color="#e74c3c" if val < 50 else "#2c3e50")

        ax.set_yticks(range(len(completeness)))
        ax.set_yticklabels(completeness.index, fontsize=9)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Completeness %", fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50")
        ax.axvline(x=100, color="#27ae60", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 13. Outlier Proportion Gauge Chart
    # ------------------------------------------------------------------
    def outlier_gauge(
        self,
        outlier_summary: pd.DataFrame,
        title: str = "Outlier Proportion by Column",
        **kwargs: Any,
    ) -> plt.Figure:
        """Donut/gauge chart showing outlier proportions per column."""
        if outlier_summary.empty or "outlier_%" not in outlier_summary.columns:
            return self._empty_figure("No outlier data for gauge")

        data = outlier_summary["outlier_%"].sort_values(ascending=False)
        n = len(data)
        n_cols = min(4, n)
        n_rows = math.ceil(n / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, (col_name, pct) in enumerate(data.items()):
            ax = axes_flat[i]
            pct_val = min(pct, 100)
            remaining = 100 - pct_val

            color = "#27ae60" if pct_val < 5 else "#f39c12" if pct_val < 15 else "#e74c3c"
            wedges, _ = ax.pie(
                [pct_val, remaining],
                colors=[color, "#ecf0f1"],
                startangle=90,
                wedgeprops={"width": 0.35, "edgecolor": "white", "linewidth": 2},
            )
            ax.text(0, 0, f"{pct_val:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
            ax.set_title(str(col_name), fontsize=9, fontweight="600", color="#2c3e50")

        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.02)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 14. Feature Correlation Bar (Top Correlated Pairs)
    # ------------------------------------------------------------------
    def top_correlations_bar(
        self,
        corr_matrix: pd.DataFrame,
        top_n: int = 15,
        title: str = "Top Correlated Feature Pairs",
        **kwargs: Any,
    ) -> plt.Figure:
        """Horizontal bar chart of top correlated feature pairs."""
        if corr_matrix.empty:
            return self._empty_figure("No correlation data")

        # Extract upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        pairs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i + 1, corr_matrix.shape[1]):
                pairs.append((
                    f"{corr_matrix.index[i]} ↔ {corr_matrix.columns[j]}",
                    corr_matrix.iloc[i, j],
                ))

        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        pairs = pairs[:top_n]

        if not pairs:
            return self._empty_figure("No correlation pairs found")

        labels, values = zip(*pairs)

        fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.35)))
        colors = ["#e74c3c" if v < 0 else "#3498db" for v in values]
        bars = ax.barh(range(len(labels)), values, color=colors, height=0.6, alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(
                val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, fontweight="bold",
                ha="left" if val >= 0 else "right",
            )

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Correlation Coefficient", fontsize=10)
        ax.set_title(title, fontsize=14, fontweight="bold", color="#2c3e50")
        ax.axvline(x=0, color="#333", linewidth=0.8)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 15. Statistics Summary Dashboard
    # ------------------------------------------------------------------
    def stats_dashboard(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        title: str = "Statistical Summary Dashboard",
        **kwargs: Any,
    ) -> plt.Figure:
        """Combined dashboard: histogram + boxplot + stats text for each column."""
        cols = numeric_cols[:6]
        if not cols:
            return self._empty_figure("No numeric columns for dashboard")

        n = len(cols)
        fig, axes = plt.subplots(n, 3, figsize=(14, n * 2.8),
                                 gridspec_kw={"width_ratios": [3, 2, 2]})
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(cols):
            data = df[col].dropna()
            color = sns.color_palette("husl", n)[i]

            # Histogram
            axes[i, 0].hist(data, bins=30, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
            axes[i, 0].set_title(col, fontsize=10, fontweight="600", loc="left")
            axes[i, 0].spines["top"].set_visible(False)
            axes[i, 0].spines["right"].set_visible(False)

            # Boxplot
            bp = axes[i, 1].boxplot(data, vert=False, widths=0.6,
                                     patch_artist=True,
                                     boxprops={"facecolor": color, "alpha": 0.6},
                                     medianprops={"color": "#e74c3c", "linewidth": 2})
            axes[i, 1].set_yticklabels([])
            axes[i, 1].spines["top"].set_visible(False)
            axes[i, 1].spines["right"].set_visible(False)

            # Stats text
            ax_text = axes[i, 2]
            ax_text.axis("off")
            stats_text = (
                f"Mean: {data.mean():.4f}\n"
                f"Median: {data.median():.4f}\n"
                f"Std: {data.std():.4f}\n"
                f"Skew: {data.skew():.4f}\n"
                f"Kurt: {data.kurtosis():.4f}\n"
                f"Range: [{data.min():.2f}, {data.max():.2f}]"
            )
            ax_text.text(0.1, 0.5, stats_text, transform=ax_text.transAxes,
                        fontsize=8, fontfamily="monospace", va="center",
                        bbox={"facecolor": "#f8f9fa", "edgecolor": "#ddd", "boxstyle": "round,pad=0.5"})

        fig.suptitle(title, fontsize=14, fontweight="bold", color="#2c3e50", y=1.01)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _empty_figure(self, message: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#999")
        ax.axis("off")
        return fig
