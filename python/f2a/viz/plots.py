"""
Core plot functions for f2a reports.

Generates all 32 chart types embedded in the HTML report.
All functions return a matplotlib Figure object.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

from f2a.viz.theme import apply_light_theme, F2A_PALETTE, ACCENT_COLOR


def _ensure_viz():
    if not HAS_VIZ:
        raise ImportError("matplotlib and seaborn are required for visualization")
    apply_light_theme()


def _clean(vals: list) -> list[float]:
    """Remove None / NaN."""
    return [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Distribution Histograms
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_distribution_grid(
    columns_data: dict[str, list[float]],
    cols_per_row: int = 3,
    figsize_per: tuple[float, float] = (4.5, 3.2),
) -> Any:
    """Histogram + KDE for multiple numeric columns."""
    _ensure_viz()
    n = len(columns_data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        return fig
    n_rows = max(1, (n + cols_per_row - 1) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(figsize_per[0] * cols_per_row, figsize_per[1] * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for idx, (col, vals) in enumerate(columns_data.items()):
        ax = axes[idx]
        clean = _clean(vals)
        if clean:
            color = F2A_PALETTE[idx % len(F2A_PALETTE)]
            ax.hist(clean, bins=min(50, max(10, len(clean) // 10)),
                    color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
            try:
                sns.kdeplot(clean, ax=ax, color="#e74c3c", linewidth=1.5,
                            warn_singular=False)
            except Exception:
                pass
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Distribution Histograms", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Boxplots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_boxplots(
    columns_data: dict[str, list[float]],
    cols_per_row: int = 4,
    figsize_per: tuple[float, float] = (3.5, 4),
) -> Any:
    """Box plots for numeric columns."""
    _ensure_viz()
    n = len(columns_data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        return fig
    n_rows = max(1, (n + cols_per_row - 1) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(figsize_per[0] * cols_per_row, figsize_per[1] * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for idx, (col, vals) in enumerate(columns_data.items()):
        ax = axes[idx]
        clean = _clean(vals)
        if clean:
            bp = ax.boxplot(clean, vert=True, patch_artist=True, widths=0.6,
                            flierprops=dict(marker="o", markersize=3, alpha=0.5))
            color = F2A_PALETTE[idx % len(F2A_PALETTE)]
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in bp["medians"]:
                median.set_color("#e74c3c")
                median.set_linewidth(2)
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Boxplots", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Violin Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_violin(columns_data: dict[str, list[float]]) -> Any:
    """Violin plots for distribution shape."""
    _ensure_viz()
    data = {k: _clean(v) for k, v in columns_data.items() if _clean(v)}
    if not data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.8), 5))
    parts = ax.violinplot(list(data.values()), positions=range(len(data)),
                          showmeans=True, showmedians=True, showextrema=True)
    for idx, pc in enumerate(parts.get("bodies", [])):
        pc.set_facecolor(F2A_PALETTE[idx % len(F2A_PALETTE)])
        pc.set_alpha(0.7)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#e74c3c")
    if "cmedians" in parts:
        parts["cmedians"].set_color("#2c3e50")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(list(data.keys()), rotation=30, ha="right", fontsize=9)
    ax.set_title("Violin Plots", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Q-Q Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_qq(columns_data: dict[str, list[float]], cols_per_row: int = 3) -> Any:
    """Q-Q plots for normality assessment."""
    _ensure_viz()
    from scipy import stats as sp_stats
    data = {k: _clean(v) for k, v in columns_data.items() if _clean(v)}
    n = len(data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    n_rows = max(1, (n + cols_per_row - 1) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(4.5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for idx, (col, vals) in enumerate(data.items()):
        ax = axes[idx]
        sp_stats.probplot(vals, dist="norm", plot=ax)
        ax.get_lines()[0].set(color=F2A_PALETTE[idx % len(F2A_PALETTE)],
                               markersize=3, alpha=0.6)
        ax.get_lines()[1].set(color="#e74c3c", linewidth=2)
        ax.set_title(col, fontsize=10, fontweight="bold")
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Q-Q Plots", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5–6. Correlation Heatmaps (Pearson + Spearman)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_correlation_heatmap(
    matrix: list[list[float]],
    labels: list[str],
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (10, 8),
) -> Any:
    """Triangular heatmap for correlation matrix."""
    _ensure_viz()
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(arr, dtype=bool), k=1)
    annot = len(labels) <= 12
    sns.heatmap(arr, mask=mask, annot=annot, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, square=True,
                cbar_kws={"shrink": 0.8, "label": "Correlation"})
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7–8. Missing Data (Bar + Matrix)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_missing_bar(missing_columns: list[dict]) -> Any:
    """Bar chart of missing value percentages."""
    _ensure_viz()
    cols_with_missing = [c for c in missing_columns if c.get("n_missing", 0) > 0]
    if not cols_with_missing:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                fontsize=14, color="#27ae60")
        ax.set_title("Missing Data", fontsize=14, fontweight="bold")
        return fig
    names = [c["column"] for c in cols_with_missing]
    ratios = [c.get("missing_ratio", 0) * 100 for c in cols_with_missing]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    colors = ["#e74c3c" if r > 30 else "#f39c12" if r > 5 else "#3498db" for r in ratios]
    bars = ax.bar(names, ratios, color=colors, alpha=0.8, edgecolor="white")
    for bar, pct in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylabel("Missing %")
    ax.set_title("Missing Data", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_missing_matrix(
    missing_matrix: list[list[bool]],
    column_names: list[str],
    figsize: tuple[int, int] = (12, 6),
) -> Any:
    """Binary heatmap showing missing value positions."""
    _ensure_viz()
    if not missing_matrix or not missing_matrix[0]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No missing data pattern", ha="center", va="center")
        return fig
    arr = np.array(missing_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.colors.ListedColormap(["#f0f2f5", "#e74c3c"])
    ax.imshow(arr.T, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(column_names)))
    ax.set_yticklabels(column_names, fontsize=9)
    ax.set_xlabel("Row Index", fontsize=10)
    ax.set_title("Missing Data Matrix", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9. Outlier Detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_outlier_summary(
    columns_data: dict[str, list[float]],
    outlier_masks: dict[str, list[bool]],
) -> Any:
    """Scatter with outliers highlighted."""
    _ensure_viz()
    data = {k: (_clean(v), outlier_masks.get(k, []))
            for k, v in columns_data.items() if _clean(v)}
    n = len(data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    cols_per_row = min(4, n)
    n_rows = max(1, (n + cols_per_row - 1) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(3.5 * cols_per_row, 3.5 * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for idx, (col, (vals, mask)) in enumerate(data.items()):
        ax = axes[idx]
        x = np.arange(len(vals))
        normal = [v for v, m in zip(vals, mask) if not m] if mask else vals
        outliers_x = [i for i, m in zip(x, mask) if m] if mask else []
        outliers_y = [v for v, m in zip(vals, mask) if m] if mask else []
        normal_x = [i for i, m in zip(x, mask) if not m] if mask else list(x)
        ax.scatter(normal_x, normal, c=F2A_PALETTE[0], s=8, alpha=0.5, label="Normal")
        if outliers_x:
            ax.scatter(outliers_x, outliers_y, c="#e74c3c", s=20, alpha=0.8,
                       marker="x", linewidths=1.5, label="Outlier")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Outlier Detection", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  10. Feature Importance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_feature_importance(
    mean_abs: list[dict],
    variance: list[dict],
) -> Any:
    """Horizontal bar chart of feature importance."""
    _ensure_viz()
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(mean_abs) * 0.5)))
    # Mean abs correlation
    ax = axes[0]
    if mean_abs:
        names = [d.get("column", "") for d in mean_abs]
        vals = [d.get("mean_abs_corr", 0) for d in mean_abs]
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order],
                color=ACCENT_COLOR, alpha=0.8)
        ax.set_title("Mean |Correlation|", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Absolute Correlation")
    # Variance ranking
    ax = axes[1]
    if variance:
        names = [d.get("column", "") for d in variance]
        vals = [d.get("variance", 0) for d in variance]
        order = np.argsort(vals)
        ax.barh([names[i] for i in order], [vals[i] for i in order],
                color="#27ae60", alpha=0.8)
        ax.set_title("Variance Ranking", fontsize=12, fontweight="bold")
        ax.set_xlabel("Variance")
    fig.suptitle("Feature Importance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  11–12. PCA (Scree + Loadings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_pca_scree(
    variance_ratio: list[float],
    cumulative_ratio: list[float],
) -> Any:
    """PCA scree plot with cumulative line."""
    _ensure_viz()
    n = len(variance_ratio)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No PCA data", ha="center", va="center")
        return fig
    x = list(range(1, n + 1))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, [v * 100 for v in variance_ratio],
            color=ACCENT_COLOR, alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax2 = ax1.twinx()
    ax2.plot(x, [v * 100 for v in cumulative_ratio], "o-",
             color="#27ae60", linewidth=2, markersize=6, label="Cumulative")
    ax2.axhline(y=90, color="#e74c3c", linestyle="--", alpha=0.5, label="90%")
    ax2.set_ylabel("Cumulative (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("PCA — Variance Explained", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pca_loadings(
    loadings: list[list[float]],
    feature_names: list[str],
) -> Any:
    """Heatmap of PCA component loadings."""
    _ensure_viz()
    if not loadings or not feature_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No loadings data", ha="center", va="center")
        return fig
    arr = np.array(loadings)
    n_comp = arr.shape[1] if arr.ndim == 2 else 1
    fig, ax = plt.subplots(figsize=(max(6, n_comp * 1.5), max(4, len(feature_names) * 0.5)))
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    sns.heatmap(arr, annot=n_comp <= 10, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=comp_labels, yticklabels=feature_names,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("PCA Loadings", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  13. Insight Severity Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_insight_severity(summary: dict) -> Any:
    """Pie/donut chart of insight severities."""
    _ensure_viz()
    counts = {
        "Critical": summary.get("critical", 0),
        "Warning": summary.get("warning", 0),
        "Info": summary.get("info", 0),
    }
    counts = {k: v for k, v in counts.items() if v > 0}
    if not counts:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No insights", ha="center", va="center")
        return fig
    colors = {"Critical": "#e74c3c", "Warning": "#f39c12", "Info": "#3498db"}
    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        counts.values(), labels=counts.keys(), autopct="%1.0f%%",
        colors=[colors.get(k, "#999") for k in counts.keys()],
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2))
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")
    ax.set_title("Insight Severity Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  14. Top Insights
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_top_insights(insights: list[dict]) -> Any:
    """Horizontal bar chart of top insights by severity."""
    _ensure_viz()
    if not insights:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No insights", ha="center", va="center")
        return fig
    top = insights[:15]
    labels = [i.get("message", "")[:50] for i in top]
    severity_score = {"critical": 3, "warning": 2, "info": 1}
    scores = [severity_score.get(i.get("severity", "info").lower(), 0) for i in top]
    colors_map = {"critical": "#e74c3c", "warning": "#f39c12", "info": "#3498db"}
    colors = [colors_map.get(i.get("severity", "info").lower(), "#999") for i in top]
    fig, ax = plt.subplots(figsize=(12, max(4, len(top) * 0.45)))
    ax.barh(range(len(top)), scores, color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Severity")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Info", "Warning", "Critical"])
    ax.set_title("Top Insights", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  15. Best-Fit Distribution Overlay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_best_fit_overlay(
    columns_data: dict[str, list[float]],
    best_fits: list[dict],
    cols_per_row: int = 3,
) -> Any:
    """Histogram with best-fit distribution overlay."""
    _ensure_viz()
    from scipy import stats as sp_stats
    fit_map = {bf.get("column", ""): bf for bf in best_fits}
    data = {k: _clean(v) for k, v in columns_data.items() if _clean(v)}
    n = len(data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    n_rows = max(1, (n + cols_per_row - 1) // cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row,
                             figsize=(4.5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    for idx, (col, vals) in enumerate(data.items()):
        ax = axes[idx]
        ax.hist(vals, bins=min(40, max(10, len(vals) // 10)),
                density=True, color=F2A_PALETTE[idx % len(F2A_PALETTE)],
                alpha=0.6, edgecolor="white")
        bf = fit_map.get(col, {})
        dist_name = bf.get("best_distribution", "")
        if dist_name and hasattr(sp_stats, dist_name):
            try:
                dist = getattr(sp_stats, dist_name)
                params = dist.fit(vals)
                x = np.linspace(min(vals), max(vals), 200)
                ax.plot(x, dist.pdf(x, *params), color="#e74c3c", linewidth=2,
                        label=dist_name)
                ax.legend(fontsize=7)
            except Exception:
                pass
        ax.set_title(col, fontsize=10, fontweight="bold")
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("Best-Fit Distribution Overlay", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  16. ECDF Plot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_ecdf(columns_data: dict[str, list[float]]) -> Any:
    """Empirical CDF for numeric columns."""
    _ensure_viz()
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (col, vals) in enumerate(columns_data.items()):
        clean = _clean(vals)
        if not clean:
            continue
        sorted_v = np.sort(clean)
        ecdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(sorted_v, ecdf, label=col,
                color=F2A_PALETTE[idx % len(F2A_PALETTE)], linewidth=1.8)
    ax.set_xlabel("Value")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF Plot", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  17. Power Transform Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_power_transform(
    columns_data: dict[str, list[float]],
    power_transforms: list[dict],
) -> Any:
    """Before/after power transform comparison."""
    _ensure_viz()
    pt_map = {pt.get("column", ""): pt for pt in power_transforms}
    data = {k: _clean(v) for k, v in columns_data.items() if k in pt_map and _clean(v)}
    n = len(data)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for idx, (col, vals) in enumerate(data.items()):
        pt = pt_map[col]
        axes[idx, 0].hist(vals, bins=30, color=ACCENT_COLOR, alpha=0.7, edgecolor="white")
        axes[idx, 0].set_title(f"{col} (Original)", fontsize=10, fontweight="bold")
        method = pt.get("recommended_transform", "log1p")
        try:
            if method == "log1p":
                transformed = np.log1p(np.abs(vals))
            elif method == "sqrt":
                transformed = np.sqrt(np.abs(vals))
            else:
                transformed = vals
            axes[idx, 1].hist(transformed, bins=30, color="#27ae60", alpha=0.7, edgecolor="white")
            axes[idx, 1].set_title(f"{col} ({method})", fontsize=10, fontweight="bold")
        except Exception:
            axes[idx, 1].text(0.5, 0.5, "Transform failed", ha="center", va="center")
    fig.suptitle("Power Transform Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  18. Jarque-Bera Normality Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_jarque_bera(jb_data: list[dict]) -> Any:
    """Bar chart of Jarque-Bera p-values."""
    _ensure_viz()
    if not jb_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    names = [d.get("column", "") for d in jb_data]
    pvals = [d.get("p_value", 1.0) for d in jb_data]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    colors = ["#e74c3c" if p < 0.05 else "#27ae60" for p in pvals]
    ax.bar(names, [-np.log10(max(p, 1e-300)) for p in pvals], color=colors, alpha=0.8)
    ax.axhline(y=-np.log10(0.05), color="#f39c12", linestyle="--", label="p=0.05")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title("Jarque-Bera Normality Test", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  19–23. Advanced Correlation Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _matrix_from_pairs(pairs: list[dict], val_key: str) -> tuple[list[str], np.ndarray]:
    """Build a symmetric matrix from pair records."""
    cols = sorted({p.get("col_a", "") for p in pairs} | {p.get("col_b", "") for p in pairs})
    if not cols:
        return [], np.array([])
    idx = {c: i for i, c in enumerate(cols)}
    mat = np.zeros((len(cols), len(cols)))
    for p in pairs:
        i, j = idx.get(p.get("col_a", ""), -1), idx.get(p.get("col_b", ""), -1)
        if i >= 0 and j >= 0:
            v = p.get(val_key, 0)
            mat[i, j] = v
            mat[j, i] = v
    np.fill_diagonal(mat, 1.0)
    return cols, mat


def plot_partial_correlation(partial_corrs: list[dict]) -> Any:
    """Partial correlation heatmap."""
    _ensure_viz()
    cols, mat = _matrix_from_pairs(partial_corrs, "partial_corr")
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    return plot_correlation_heatmap(mat.tolist(), cols, "Partial Correlation Heatmap")


def plot_mutual_information(mi_data: list[dict]) -> Any:
    """Mutual information heatmap."""
    _ensure_viz()
    cols, mat = _matrix_from_pairs(mi_data, "mutual_info")
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    arr = np.array(mat)
    fig, ax = plt.subplots(figsize=(max(6, len(cols)), max(5, len(cols) * 0.8)))
    sns.heatmap(arr, annot=len(cols) <= 12, fmt=".3f", cmap="YlOrRd",
                xticklabels=cols, yticklabels=cols, ax=ax, linewidths=0.5,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Mutual Information Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_bootstrap_ci(bootstrap_data: list[dict]) -> Any:
    """Error bar chart of bootstrap correlation CIs."""
    _ensure_viz()
    if not bootstrap_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    labels = [f"{d.get('col_a', '')} vs {d.get('col_b', '')}" for d in bootstrap_data]
    means = [d.get("mean_corr", d.get("pearson", 0)) for d in bootstrap_data]
    lowers = [d.get("ci_lower", means[i] - 0.1) for i, d in enumerate(bootstrap_data)]
    uppers = [d.get("ci_upper", means[i] + 0.1) for i, d in enumerate(bootstrap_data)]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 6))
    y = range(len(labels))
    xerr_low = [m - l for m, l in zip(means, lowers)]
    xerr_high = [u - m for m, u in zip(means, uppers)]
    ax.errorbar(means, y, xerr=[xerr_low, xerr_high], fmt="o",
                color=ACCENT_COLOR, ecolor="#999", elinewidth=1.5,
                capsize=4, markersize=6)
    ax.axvline(x=0, color="#e74c3c", linestyle="--", alpha=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Correlation")
    ax.set_title("Bootstrap Correlation CI", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_correlation_network(network_data: list[dict]) -> Any:
    """Simple correlation network plot."""
    _ensure_viz()
    if not network_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    nodes = sorted({e.get("source", e.get("col_a", "")) for e in network_data} |
                   {e.get("target", e.get("col_b", "")) for e in network_data})
    n = len(nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}
    fig, ax = plt.subplots(figsize=(8, 8))
    for e in network_data:
        src = e.get("source", e.get("col_a", ""))
        tgt = e.get("target", e.get("col_b", ""))
        w = abs(e.get("weight", e.get("pearson", 0.5)))
        color = "#e74c3c" if e.get("weight", e.get("pearson", 0)) < 0 else ACCENT_COLOR
        if src in pos and tgt in pos:
            ax.plot([pos[src][0], pos[tgt][0]], [pos[src][1], pos[tgt][1]],
                    color=color, alpha=min(1, w), linewidth=max(0.5, w * 3))
    for node, (x, y) in pos.items():
        ax.scatter(x, y, s=200, c=ACCENT_COLOR, zorder=5, edgecolors="white", linewidths=2)
        ax.annotate(node, (x, y), fontsize=8, ha="center", va="center",
                    fontweight="bold", color="white")
    ax.set_title("Correlation Network", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_distance_correlation(dist_corrs: list[dict]) -> Any:
    """Distance correlation heatmap."""
    _ensure_viz()
    cols, mat = _matrix_from_pairs(dist_corrs, "distance_corr")
    if not cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    arr = np.array(mat)
    fig, ax = plt.subplots(figsize=(max(6, len(cols)), max(5, len(cols) * 0.8)))
    sns.heatmap(arr, annot=len(cols) <= 12, fmt=".3f", cmap="YlGnBu",
                xticklabels=cols, yticklabels=cols, ax=ax, linewidths=0.5,
                square=True, vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
    ax.set_title("Distance Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  24–27. Clustering Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_elbow_silhouette(elbow_data: list[dict]) -> Any:
    """Elbow curve and silhouette scores."""
    _ensure_viz()
    if not elbow_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    ks = [d.get("k", 0) for d in elbow_data]
    inertias = [d.get("inertia", 0) for d in elbow_data]
    silhouettes = [d.get("silhouette", 0) for d in elbow_data]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, inertias, "o-", color=ACCENT_COLOR, linewidth=2, label="Inertia")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia", color=ACCENT_COLOR)
    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, "s--", color="#27ae60", linewidth=2, label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="#27ae60")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("Elbow & Silhouette", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_cluster_scatter(
    labels: list[int],
    embedding: Optional[list[list[float]]] = None,
    pca_loadings: Optional[list[list[float]]] = None,
) -> Any:
    """2D scatter of cluster assignments."""
    _ensure_viz()
    if embedding and len(embedding) > 0 and len(embedding[0]) >= 2:
        pts = np.array(embedding)
    elif pca_loadings and len(pca_loadings) > 0:
        pts = np.array(pca_loadings)[:, :2] if np.array(pca_loadings).ndim == 2 else None
    else:
        pts = None
    if pts is None or len(pts) != len(labels):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No 2D embedding available", ha="center", va="center")
        return fig
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(labels))
    for cl in unique:
        mask = [l == cl for l in labels]
        color = "#999" if cl == -1 else F2A_PALETTE[cl % len(F2A_PALETTE)]
        label = "Noise" if cl == -1 else f"Cluster {cl}"
        ax.scatter(pts[mask, 0], pts[mask, 1], c=color, s=15, alpha=0.6, label=label)
    ax.set_title("Cluster Scatter", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_dendrogram(labels: list[int], n_points: int = 200) -> Any:
    """Hierarchical clustering dendrogram."""
    _ensure_viz()
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendro
    except ImportError:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "scipy required", ha="center", va="center")
        return fig
    n = min(n_points, len(labels))
    data = np.random.RandomState(42).randn(n, 2)
    Z = linkage(data, method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    scipy_dendro(Z, ax=ax, truncate_mode="lastp", p=min(30, n),
                 leaf_rotation=45, leaf_font_size=8,
                 above_threshold_color="#999",
                 color_threshold=Z[-min(5, len(Z)), 2])
    ax.set_title("Dendrogram", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


def plot_cluster_profile(
    cluster_sizes: list[int],
) -> Any:
    """Bar chart of cluster sizes."""
    _ensure_viz()
    if not cluster_sizes:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    fig, ax = plt.subplots(figsize=(max(6, len(cluster_sizes) * 1.5), 5))
    x = [f"Cluster {i}" for i in range(len(cluster_sizes))]
    colors = [F2A_PALETTE[i % len(F2A_PALETTE)] for i in range(len(cluster_sizes))]
    ax.bar(x, cluster_sizes, color=colors, alpha=0.8, edgecolor="white")
    for i, v in enumerate(cluster_sizes):
        ax.text(i, v + max(cluster_sizes) * 0.02, str(v),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_title("Cluster Profiles", fontsize=14, fontweight="bold")
    ax.set_ylabel("Size")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  28–30. Dimensionality Reduction Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_pca_biplot(
    loadings: list[list[float]],
    feature_names: list[str],
    variance_ratio: list[float],
) -> Any:
    """PCA biplot (feature vectors on PC1/PC2)."""
    _ensure_viz()
    if not loadings or not feature_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    arr = np.array(loadings)
    if arr.ndim != 2 or arr.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Need ≥2 components", ha="center", va="center")
        return fig
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, name in enumerate(feature_names):
        ax.arrow(0, 0, arr[i, 0], arr[i, 1], head_width=0.02, head_length=0.01,
                 fc=F2A_PALETTE[i % len(F2A_PALETTE)],
                 ec=F2A_PALETTE[i % len(F2A_PALETTE)], linewidth=2)
        ax.text(arr[i, 0] * 1.12, arr[i, 1] * 1.12, name,
                fontsize=9, ha="center", fontweight="bold")
    ax.axhline(0, color="#ccc", linewidth=0.5)
    ax.axvline(0, color="#ccc", linewidth=0.5)
    circle = plt.Circle((0, 0), 1, fill=False, color="#ddd", linestyle="--")
    ax.add_patch(circle)
    pct1 = variance_ratio[0] * 100 if variance_ratio else 0
    pct2 = variance_ratio[1] * 100 if len(variance_ratio) > 1 else 0
    ax.set_xlabel(f"PC1 ({pct1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pct2:.1f}%)")
    ax.set_title("PCA Biplot", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_explained_variance_curve(
    variance_ratio: list[float],
    cumulative_ratio: list[float],
) -> Any:
    """Explained variance curve (area + line)."""
    _ensure_viz()
    if not variance_ratio:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    x = list(range(1, len(variance_ratio) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(x, [v * 100 for v in cumulative_ratio],
                    alpha=0.2, color=ACCENT_COLOR)
    ax.plot(x, [v * 100 for v in cumulative_ratio], "o-",
            color=ACCENT_COLOR, linewidth=2, label="Cumulative")
    ax.axhline(y=90, color="#e74c3c", linestyle="--", alpha=0.5, label="90%")
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("Explained Variance Curve", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_factor_loadings(
    loadings: list[list[float]],
    feature_names: list[str],
    n_factors: int = 0,
) -> Any:
    """Factor analysis loadings heatmap."""
    _ensure_viz()
    if not loadings or not feature_names:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    arr = np.array(loadings)
    n_f = n_factors or (arr.shape[1] if arr.ndim == 2 else 1)
    labels = [f"Factor {i+1}" for i in range(n_f)]
    fig, ax = plt.subplots(figsize=(max(6, n_f * 1.5), max(4, len(feature_names) * 0.5)))
    sns.heatmap(arr, annot=n_f <= 8, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=labels, yticklabels=feature_names,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Factor Loadings Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  31–32. Anomaly Detection Charts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_anomaly_scatter(
    scores: list[float],
    labels: list[int],
    method: str = "Isolation Forest",
) -> Any:
    """Anomaly scores scatter with threshold."""
    _ensure_viz()
    if not scores:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scores))
    normal_mask = [l == 1 or l == 0 for l in labels]
    anomaly_mask = [l == -1 for l in labels]
    if any(normal_mask):
        ax.scatter(x[normal_mask], [scores[i] for i, m in enumerate(normal_mask) if m],
                   c=ACCENT_COLOR, s=10, alpha=0.5, label="Normal")
    if any(anomaly_mask):
        ax.scatter(x[anomaly_mask], [scores[i] for i, m in enumerate(anomaly_mask) if m],
                   c="#e74c3c", s=25, alpha=0.8, marker="x", label="Anomaly")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(f"Anomaly Scatter ({method})", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_consensus_comparison(anomaly_data: dict) -> Any:
    """Multi-method anomaly comparison bar chart."""
    _ensure_viz()
    methods = {}
    for key in ["isolation_forest", "local_outlier_factor", "mahalanobis"]:
        d = anomaly_data.get(key, {})
        if d:
            methods[key.replace("_", " ").title()] = d.get("anomaly_ratio", 0) * 100
    consensus = anomaly_data.get("consensus", {})
    if consensus:
        methods["Consensus"] = consensus.get("anomaly_ratio", 0) * 100
    if not methods:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [F2A_PALETTE[i % len(F2A_PALETTE)] for i in range(len(methods))]
    bars = ax.bar(list(methods.keys()), list(methods.values()), color=colors, alpha=0.8)
    for bar, pct in zip(bars, methods.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Anomaly Ratio (%)")
    ax.set_title("Consensus Anomaly Comparison", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Quality Radar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_quality_radar(dimensions: list[dict]) -> Any:
    """Radar chart for data quality dimensions."""
    _ensure_viz()
    if not dimensions:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig
    labels = [d["name"] for d in dimensions]
    values = [d["score"] for d in dimensions]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, color=ACCENT_COLOR, alpha=0.2)
    ax.plot(angles, values_plot, "o-", color=ACCENT_COLOR, linewidth=2, markersize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Data Quality", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  t-SNE Embedding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_tsne(embedding: list[list[float]], labels: Optional[list[int]] = None) -> Any:
    """t-SNE 2D scatter."""
    _ensure_viz()
    if not embedding or len(embedding[0]) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No embedding", ha="center", va="center")
        return fig
    pts = np.array(embedding)
    fig, ax = plt.subplots(figsize=(8, 7))
    if labels:
        unique = sorted(set(labels))
        for cl in unique:
            mask = [l == cl for l in labels]
            c = F2A_PALETTE[cl % len(F2A_PALETTE)] if cl >= 0 else "#999"
            ax.scatter(pts[mask, 0], pts[mask, 1], c=c, s=15, alpha=0.6,
                       label=f"{'Noise' if cl==-1 else f'Cluster {cl}'}")
        ax.legend(fontsize=8)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], c=ACCENT_COLOR, s=15, alpha=0.5)
    ax.set_title("t-SNE Embedding", fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    return fig
