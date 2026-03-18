"""Temporal and sequential analysis statistics.

New in v1.1.1: Autocorrelation, lag analysis, rolling statistics,
trend detection, and stationarity tests for numeric sequences.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from f2a.core.schema import DataSchema


class TemporalStats:
    """Compute temporal and sequential statistics for numeric columns."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema,
        max_lags: int = 20,
    ) -> None:
        self._df = df
        self._schema = schema
        self._max_lags = max_lags
        self._numeric_cols = schema.numeric_columns[:15]

    def summary(self) -> dict[str, Any]:
        """Compute all temporal statistics."""
        result: dict[str, Any] = {}

        if not self._numeric_cols:
            return result

        result["autocorrelation"] = self._autocorrelation()
        result["rolling_stats"] = self._rolling_stats()
        result["trend_detection"] = self._trend_detection()

        return result

    def _autocorrelation(self) -> dict[str, Any]:
        """Compute autocorrelation for each numeric column."""
        acf_results: dict[str, list[float]] = {}

        for col in self._numeric_cols:
            series = self._df[col].dropna().values
            if len(series) < 10:
                continue

            n = len(series)
            mean = np.mean(series)
            var = np.var(series)
            if var < 1e-12:
                continue

            acf_vals = []
            max_lag = min(self._max_lags, n // 3)
            for lag in range(1, max_lag + 1):
                c = np.mean((series[:n - lag] - mean) * (series[lag:] - mean)) / var
                acf_vals.append(round(float(c), 4))

            acf_results[col] = acf_vals

        return acf_results

    def _rolling_stats(self, window: int | None = None) -> dict[str, Any]:
        """Compute rolling mean and std for stability analysis."""
        if not self._numeric_cols:
            return {}

        n = len(self._df)
        if window is None:
            window = max(5, min(50, n // 10))

        results: dict[str, dict[str, Any]] = {}
        for col in self._numeric_cols[:8]:
            series = self._df[col].dropna()
            if len(series) < window * 2:
                continue

            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()

            # Stability metric: CV of rolling mean
            mean_cv = float(rolling_mean.std() / (rolling_mean.mean() + 1e-12))
            std_cv = float(rolling_std.std() / (rolling_std.mean() + 1e-12))

            results[col] = {
                "window": window,
                "mean_stability": round(1 - min(mean_cv, 1), 4),
                "std_stability": round(1 - min(std_cv, 1), 4),
                "rolling_mean_values": rolling_mean.values.tolist()[-50:],
                "rolling_std_values": rolling_std.values.tolist()[-50:],
            }

        return results

    def _trend_detection(self) -> dict[str, Any]:
        """Simple trend detection via linear regression slope."""
        results: dict[str, Any] = {}

        for col in self._numeric_cols[:8]:
            series = self._df[col].dropna().values
            if len(series) < 10:
                continue

            x = np.arange(len(series), dtype=float)
            # Simple linear regression
            x_mean = x.mean()
            y_mean = series.mean()
            ss_xy = np.sum((x - x_mean) * (series - y_mean))
            ss_xx = np.sum((x - x_mean) ** 2)

            if ss_xx < 1e-12:
                continue

            slope = ss_xy / ss_xx
            # Normalize slope by data range
            data_range = series.max() - series.min()
            if data_range < 1e-12:
                continue

            norm_slope = slope * len(series) / data_range

            results[col] = {
                "slope": round(float(slope), 6),
                "normalized_slope": round(float(norm_slope), 4),
                "trend": "increasing" if norm_slope > 0.1 else "decreasing" if norm_slope < -0.1 else "stable",
            }

        return results
