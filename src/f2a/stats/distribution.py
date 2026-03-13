"""Distribution analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from f2a.core.schema import DataSchema


class DistributionStats:
    """Analyze distribution characteristics of numeric columns.

    Args:
        df: Target DataFrame to analyze.
        schema: Data schema.
    """

    def __init__(self, df: pd.DataFrame, schema: DataSchema) -> None:
        self._df = df
        self._schema = schema

    def analyze(self) -> pd.DataFrame:
        """Return distribution information for numeric columns.

        Returns:
            DataFrame containing skewness, kurtosis, and normality test results.
        """
        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        rows: list[dict] = []
        for col in cols:
            series = self._df[col].dropna()
            if len(series) < 3:
                continue
            rows.append(self._analyze_column(col, series))

        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    def quantile_table(self, quantiles: list[float] | None = None) -> pd.DataFrame:
        """Return quantile table for numeric columns.

        Args:
            quantiles: List of quantiles to compute. Defaults to
                ``[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]``.

        Returns:
            Quantile DataFrame.
        """
        if quantiles is None:
            quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

        cols = self._schema.numeric_columns
        if not cols:
            return pd.DataFrame()

        return self._df[cols].quantile(quantiles)

    @staticmethod
    def _analyze_column(col: str, series: pd.Series) -> dict:
        """Analyze the distribution of a single numeric column."""
        skew = float(series.skew())
        kurt = float(series.kurtosis())

        # Normality test
        normality_p: float | None = None
        normality_test: str = "n/a"

        n = len(series)
        if 3 <= n <= 5000:
            _, normality_p = sp_stats.shapiro(series)
            normality_test = "shapiro"
        elif n > 5000:
            _, normality_p = sp_stats.normaltest(series)
            normality_test = "dagostino"

        return {
            "column": col,
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
            "normality_test": normality_test,
            "normality_p": round(normality_p, 6) if normality_p is not None else None,
            "is_normal_0.05": normality_p > 0.05 if normality_p is not None else None,
        }
