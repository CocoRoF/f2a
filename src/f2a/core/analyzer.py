"""Analysis orchestrator — coordinates the entire analysis pipeline."""

from __future__ import annotations

import re
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
class SubsetReport:
    """Analysis results for a single subset/split partition."""

    subset: str
    split: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    viz: VizResult
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Top-level container for analysis results.

    Attributes:
        dataset_name: Dataset name.
        shape: ``(rows, columns)`` tuple (total across all subsets).
        schema: Data schema (of the first / single partition).
        stats: Statistical analysis results (of the first / single partition).
        viz: Visualization access object (of the first / single partition).
        warnings: List of warnings found during analysis.
        subsets: Per-subset/split reports (empty when only one partition).
    """

    dataset_name: str
    shape: tuple[int, int]
    schema: DataSchema
    stats: StatsResult
    viz: VizResult
    warnings: list[str] = field(default_factory=list)
    subsets: list[SubsetReport] = field(default_factory=list)

    def show(self) -> None:
        """Print analysis summary to console."""
        sep = "=" * 60
        print(sep)
        print(f"  f2a Analysis Report: {self.dataset_name}")
        print(sep)

        if self.subsets:
            # Multi-subset mode
            print(f"\n  Total Rows: {self.shape[0]:,}  |  Subsets: {len(self.subsets)}")
            for sr in self.subsets:
                print(f"\n{'-' * 60}")
                print(f"  [{sr.subset} / {sr.split}]  {sr.shape[0]:,} rows x {sr.shape[1]} cols")
                print(f"  Memory: {sr.schema.memory_usage_mb} MB")
                print(f"  Numeric: {len(sr.schema.numeric_columns)} | "
                      f"Categorical: {len(sr.schema.categorical_columns)} | "
                      f"Text: {len(sr.schema.text_columns)} | "
                      f"Datetime: {len(sr.schema.datetime_columns)}")
                print()
                print(sr.stats.summary.to_string())
                if sr.warnings:
                    print("\n  Warnings:")
                    for w in sr.warnings:
                        print(f"    - {w}")
        else:
            # Single-partition mode
            print(f"\n  Rows: {self.shape[0]:,}  |  Columns: {self.shape[1]}")
            print(f"  Memory: {self.schema.memory_usage_mb} MB")
            print(f"\n  Numeric: {len(self.schema.numeric_columns)}")
            print(f"  Categorical: {len(self.schema.categorical_columns)}")
            print(f"  Text: {len(self.schema.text_columns)}")
            print(f"  Datetime: {len(self.schema.datetime_columns)}")

            print(f"\n{'-' * 60}")
            print("  Summary Statistics:")
            print(self.stats.summary.to_string())

            if self.warnings:
                print(f"\n{'-' * 60}")
                print("  Warnings:")
                for w in self.warnings:
                    print(f"    - {w}")

        print(sep)

    def to_html(self, output_dir: str = ".") -> Path:
        """Generate and save an HTML report.

        Args:
            output_dir: Output directory path.

        Returns:
            Path to the saved HTML file.
        """
        generator = ReportGenerator()
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", self.dataset_name)
        safe_name = safe_name.strip(". ")[:120] or "report"
        output_path = Path(output_dir) / f"{safe_name}_report.html"

        if self.subsets:
            # Multi-subset mode: build per-subset section dicts
            subset_sections: list[dict[str, Any]] = []
            for sr in self.subsets:
                figures: dict[str, plt.Figure] = {}
                try:
                    figures["Distribution Histograms"] = sr.viz.plot_distributions()
                except Exception:
                    pass
                try:
                    figures["Boxplots"] = sr.viz.plot_boxplots()
                except Exception:
                    pass
                try:
                    figures["Correlation Heatmap"] = sr.viz.plot_correlation()
                except Exception:
                    pass
                try:
                    figures["Missing Data"] = sr.viz.plot_missing()
                except Exception:
                    pass
                subset_sections.append({
                    "subset": sr.subset,
                    "split": sr.split,
                    "schema_summary": sr.schema.summary_dict(),
                    "stats_df": sr.stats.summary,
                    "figures": figures,
                    "warnings": sr.warnings,
                })
            generator.save_html_multi(
                output_path=output_path,
                dataset_name=self.dataset_name,
                sections=subset_sections,
            )
        else:
            # Single-partition mode
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
        result: dict[str, Any] = {
            "dataset_name": self.dataset_name,
            "shape": self.shape,
            "schema": self.schema.summary_dict(),
            "stats_summary": self.stats.summary.to_dict(),
            "correlation_matrix": self.stats.correlation_matrix.to_dict()
            if not self.stats.correlation_matrix.empty
            else {},
            "warnings": self.warnings,
        }
        if self.subsets:
            result["subsets"] = [
                {
                    "subset": sr.subset,
                    "split": sr.split,
                    "shape": sr.shape,
                    "schema": sr.schema.summary_dict(),
                    "stats_summary": sr.stats.summary.to_dict(),
                    "warnings": sr.warnings,
                }
                for sr in self.subsets
            ]
        return result


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

        # 2. Check for multi-subset HuggingFace data
        has_partitions = "__subset__" in df.columns and "__split__" in df.columns

        if has_partitions:
            return self._run_multi_subset(source, df)

        # Single-partition analysis
        return self._run_single(source, df)

    def _run_single(
        self, source: str, df: pd.DataFrame
    ) -> AnalysisReport:
        """Run analysis on a single DataFrame."""
        schema = infer_schema(df)
        logger.info("Schema inference complete: %s", schema.summary_dict())

        warnings: list[str] = []
        stats = self._compute_stats(df, schema, warnings)

        dataset_name = (
            Path(source).stem
            if "/" not in source or "://" not in source
            else source
        )
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

    def _run_multi_subset(
        self, source: str, df: pd.DataFrame
    ) -> AnalysisReport:
        """Run analysis on a multi-subset HuggingFace DataFrame."""
        groups = df.groupby(["__subset__", "__split__"], sort=False)

        subset_reports: list[SubsetReport] = []
        all_warnings: list[str] = []

        for (subset_name, split_name), group_df in groups:
            # Drop the metadata columns before analysis
            part_df = group_df.drop(columns=["__subset__", "__split__"]).reset_index(drop=True)

            schema = infer_schema(part_df)
            warnings: list[str] = []
            stats = self._compute_stats(part_df, schema, warnings)
            viz = VizResult(_df=part_df, _schema=schema)

            sr = SubsetReport(
                subset=str(subset_name),
                split=str(split_name),
                shape=(len(part_df), len(part_df.columns)),
                schema=schema,
                stats=stats,
                viz=viz,
                warnings=warnings,
            )
            subset_reports.append(sr)
            all_warnings.extend(
                f"[{subset_name}/{split_name}] {w}" for w in warnings
            )
            logger.info(
                "Subset analysis complete: %s/%s (%d rows × %d cols)",
                subset_name, split_name, len(part_df), len(part_df.columns),
            )

        # Use the first subset for top-level schema/stats/viz
        first = subset_reports[0]
        total_rows = sum(sr.shape[0] for sr in subset_reports)
        total_cols = first.shape[1]

        report = AnalysisReport(
            dataset_name=source,
            shape=(total_rows, total_cols),
            schema=first.schema,
            stats=first.stats,
            viz=first.viz,
            warnings=all_warnings,
            subsets=subset_reports,
        )
        logger.info(
            "Multi-subset analysis complete: %s (%d subsets, %d total rows)",
            source, len(subset_reports), total_rows,
        )
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
