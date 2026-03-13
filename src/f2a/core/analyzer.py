"""Analysis orchestrator — coordinates the entire analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from f2a.core.loader import DataLoader
from f2a.core.schema import DataSchema, infer_schema
from f2a.stats.descriptive import DescriptiveStats
from f2a.stats.distribution import DistributionStats
from f2a.stats.correlation import CorrelationStats
from f2a.stats.missing import MissingStats
from f2a.viz.plots import BasicPlotter
from f2a.viz.corr_plots import CorrelationPlotter
from f2a.viz.missing_plots import MissingPlotter
from f2a.report.generator import ReportGenerator
from f2a.utils.logging import get_logger
from f2a.utils.validators import validate_source

logger = get_logger(__name__)


@dataclass
class StatsResult:
    """Container for statistical analysis results."""

    summary: pd.DataFrame
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    correlation_matrix: pd.DataFrame
    missing_info: pd.DataFrame
    distribution_info: pd.DataFrame

    def get_numeric_summary(self) -> pd.DataFrame:
        """Return numeric column summary."""
        return self.numeric_summary

    def get_categorical_summary(self) -> pd.DataFrame:
        """Return categorical column summary."""
        return self.categorical_summary


@dataclass
class VizResult:
    """Container for visualization results."""

    _df: pd.DataFrame
    _schema: DataSchema
    _figures: dict[str, plt.Figure] = field(default_factory=dict)

    def plot_distributions(self) -> plt.Figure:
        """Return distribution histograms for numeric columns."""
        plotter = BasicPlotter(self._df, self._schema)
        fig = plotter.histograms()
        self._figures["distributions"] = fig
        return fig

    def plot_boxplots(self) -> plt.Figure:
        """Return boxplots for numeric columns."""
        plotter = BasicPlotter(self._df, self._schema)
        fig = plotter.boxplots()
        self._figures["boxplots"] = fig
        return fig

    def plot_correlation(self, method: str = "pearson") -> plt.Figure:
        """Return correlation heatmap."""
        plotter = CorrelationPlotter(self._df, self._schema)
        fig = plotter.heatmap(method=method)
        self._figures["correlation"] = fig
        return fig

    def plot_missing(self) -> plt.Figure:
        """Return missing data bar chart."""
        plotter = MissingPlotter(self._df, self._schema)
        fig = plotter.bar()
        self._figures["missing"] = fig
        return fig


@dataclass
class AnalysisReport:
    """Top-level container for analysis results.

    Attributes:
        dataset_name: Dataset name.
        shape: ``(rows, columns)`` tuple.
        schema: Data schema.
        stats: Statistical analysis results.
        viz: Visualization access object.
        warnings: List of warnings found during analysis.
    """

    dataset_name: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    viz: VizResult
    warnings: list[str] = field(default_factory=list)

    def show(self) -> None:
        """Print analysis summary to console."""
        sep = "=" * 60
        print(sep)
        print(f"  f2a Analysis Report: {self.dataset_name}")
        print(sep)
        print(f"\n  Rows: {self.shape[0]:,}  |  Columns: {self.shape[1]}")
        print(f"  Memory: {self.schema.memory_usage_mb} MB")
        print(f"\n  Numeric: {len(self.schema.numeric_columns)}")
        print(f"  Categorical: {len(self.schema.categorical_columns)}")
        print(f"  Text: {len(self.schema.text_columns)}")
        print(f"  Datetime: {len(self.schema.datetime_columns)}")

        print(f"\n{'─' * 60}")
        print("  Summary Statistics:")
        print(self.stats.summary.to_string())

        if self.warnings:
            print(f"\n{'─' * 60}")
            print("  ⚠ Warnings:")
            for w in self.warnings:
                print(f"    • {w}")

        print(sep)

    def to_html(self, output_dir: str = ".") -> Path:
        """Generate and save an HTML report.

        Args:
            output_dir: Output directory path.

        Returns:
            Path to the saved HTML file.
        """
        # Generate visualizations
        figures: dict[str, plt.Figure] = {}
        try:
            figures["Distribution Histograms"] = self.viz.plot_distributions()
        except Exception:
            pass
        try:
            figures["Boxplots"] = self.viz.plot_boxplots()
        except Exception:
            pass
        try:
            figures["Correlation Heatmap"] = self.viz.plot_correlation()
        except Exception:
            pass
        try:
            figures["Missing Data"] = self.viz.plot_missing()
        except Exception:
            pass

        generator = ReportGenerator()
        output_path = Path(output_dir) / f"{self.dataset_name}_report.html"
        generator.save_html(
            output_path=output_path,
            dataset_name=self.dataset_name,
            schema_summary=self.schema.summary_dict(),
            stats_df=self.stats.summary,
            figures=figures,
            warnings=self.warnings,
        )
        return output_path

    def to_dict(self) -> dict[str, Any]:
        """Return analysis results as a dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "shape": self.shape,
            "schema": self.schema.summary_dict(),
            "stats_summary": self.stats.summary.to_dict(),
            "correlation_matrix": self.stats.correlation_matrix.to_dict()
            if not self.stats.correlation_matrix.empty
            else {},
            "warnings": self.warnings,
        }


class Analyzer:
    """Orchestrate the analysis pipeline.

    Example:
        >>> analyzer = Analyzer()
        >>> report = analyzer.run("data.csv")
        >>> report.show()
    """

    def __init__(self) -> None:
        self._loader = DataLoader()

    def run(self, source: str, **kwargs: Any) -> AnalysisReport:
        """Execute the full analysis pipeline.

        Args:
            source: Data source (file path or HuggingFace address).
            **kwargs: Additional arguments passed to the loader.

        Returns:
            :class:`AnalysisReport` instance.
        """
        source = validate_source(source)
        logger.info("Analysis started: %s", source)

        # 1. Load data
        df = self._loader.load(source, **kwargs)

        # 2. Infer schema
        schema = infer_schema(df)
        logger.info("Schema inference complete: %s", schema.summary_dict())

        # 3. Statistical analysis
        warnings: list[str] = []
        stats = self._compute_stats(df, schema, warnings)

        # 4. Assemble results
        dataset_name = Path(source).stem if "/" not in source or "://" not in source else source
        viz = VizResult(_df=df, _schema=schema)

        report = AnalysisReport(
            dataset_name=dataset_name,
            shape=(len(df), len(df.columns)),
            schema=schema,
            stats=stats,
            viz=viz,
            warnings=warnings,
        )

        logger.info("Analysis complete: %s", source)
        return report

    def _compute_stats(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        warnings: list[str],
    ) -> StatsResult:
        """Perform all statistical analyses."""
        desc = DescriptiveStats(df, schema)
        dist = DistributionStats(df, schema)
        corr = CorrelationStats(df, schema)
        miss = MissingStats(df, schema)

        # Descriptive statistics
        summary = desc.summary()
        numeric_summary = desc.numeric_summary()
        categorical_summary = desc.categorical_summary()

        # Correlation analysis
        correlation_matrix = corr.pearson()
        high_corrs = corr.high_correlations(threshold=0.9)
        for col_a, col_b, val in high_corrs:
            warnings.append(f"High correlation: {col_a} ↔ {col_b} (r={val})")

        # Missing data
        missing_info = miss.column_summary()
        total_missing = miss.total_missing_ratio()
        if total_missing > 0.1:
            warnings.append(f"Overall missing ratio is high: {total_missing * 100:.1f}%")

        # Distribution
        distribution_info = dist.analyze()

        return StatsResult(
            summary=summary,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            correlation_matrix=correlation_matrix,
            missing_info=missing_info,
            distribution_info=distribution_info,
        )


def analyze(source: str, **kwargs: Any) -> AnalysisReport:
    """Analyze a data source and return a report.

    This function is the main entry point for ``f2a``.

    Args:
        source: File path or HuggingFace dataset address.
            - File: ``"data.csv"``, ``"data.json"``, ``"data.parquet"``
            - HuggingFace: ``"hf://imdb"``, ``"hf://squad"``
        **kwargs: Additional arguments passed to the data loader.

    Returns:
        :class:`AnalysisReport` — statistics, visualization, and report access object.

    Example:
        >>> import f2a
        >>> report = f2a.analyze("sales.csv")
        >>> report.show()
        >>> report.to_html("output/")
    """
    analyzer = Analyzer()
    return analyzer.run(source, **kwargs)
