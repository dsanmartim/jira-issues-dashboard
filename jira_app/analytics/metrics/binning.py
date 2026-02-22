"""Time binning and histogram utilities.

This module provides functions for creating time-based bins and histograms
for aging and duration analysis.
"""

from __future__ import annotations

import math

import altair as alt
import pandas as pd


def determine_time_bin_step(
    values: pd.Series,
    *,
    target_bins: int = 12,
    min_step: float = 1.0,
) -> float | None:
    """Calculate an appropriate step size for time-based bins.

    Parameters
    ----------
    values : pd.Series
        Numeric values to bin (e.g., days_open, duration_days).
    target_bins : int
        Desired number of bins (default 12).
    min_step : float
        Minimum step size in days (default 1.0).

    Returns
    -------
    float or None
        Calculated step size, or None if values are empty/invalid.
    """
    if values is None:
        return None

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    value_range = float(numeric.max() - numeric.min())
    if math.isnan(value_range) or value_range <= 0:
        return float(min_step)
    bins = max(target_bins, 1)
    raw_step = value_range / bins
    step = max(math.ceil(raw_step), min_step)
    return float(step)


def build_time_bucket_spec(
    values: pd.Series,
    *,
    target_bins: int = 12,
    min_step: float = 1.0,
) -> dict | None:
    """Build bin edges and labels for time-based bucketing.

    Creates a specification for binning time values (days) into
    human-readable buckets like "0–<7d", "7–<14d", "≥30d".

    Parameters
    ----------
    values : pd.Series
        Numeric values to bin (e.g., days_open, duration_days).
    target_bins : int
        Desired number of bins (default 12).
    min_step : float
        Minimum step size in days (default 1.0).

    Returns
    -------
    dict or None
        Dictionary with keys:
        - bins: list of bin edges
        - labels: list of human-readable labels
        - step: calculated step size
        Returns None if values are empty/invalid.
    """
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    step = determine_time_bin_step(numeric, target_bins=target_bins, min_step=min_step)
    if step is None:
        return None
    max_value = float(numeric.max())
    if math.isnan(max_value) or max_value <= 0:
        return None
    max_edge = math.ceil(max_value / step) * step
    edges: list[float] = [0.0]
    current = step
    while current <= max_edge + 1e-9:
        edges.append(round(current, 6))
        current += step
    edges.append(float("inf"))
    labels: list[str] = []
    for idx in range(len(edges) - 2):
        lower = int(edges[idx])
        upper = int(edges[idx + 1])
        labels.append(f"{lower}–<{upper}d")
    labels.append(f"≥{int(edges[-2])}d")
    return {
        "bins": edges,
        "labels": labels,
        "step": step,
    }


def create_histogram_chart(
    data: pd.DataFrame,
    value_col: str,
    *,
    title: str,
    color_col: str | None = None,
    facet_col: str | None = None,
    facet_columns: int = 2,
    bin_step: float | None = None,
    max_bins: int = 20,
) -> alt.Chart | None:
    """Create a histogram chart for time/duration values.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the values to histogram.
    value_col : str
        Column name containing numeric values to bin.
    title : str
        Chart title.
    color_col : str, optional
        Column for color encoding (categorical).
    facet_col : str, optional
        Column for faceting into multiple charts.
    facet_columns : int
        Number of columns in faceted layout (default 2).
    bin_step : float, optional
        Fixed bin step size. If None, uses max_bins.
    max_bins : int
        Maximum number of bins when bin_step is not specified (default 20).

    Returns
    -------
    alt.Chart or None
        Altair chart object, or None if data is empty/invalid.
    """
    if data is None or data.empty or value_col not in data:
        return None

    bin_args: dict[str, float | int] = {}
    if bin_step is not None and bin_step > 0:
        bin_args["step"] = bin_step
    else:
        bin_args["maxbins"] = max_bins

    base = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{value_col}:Q",
                bin=alt.Bin(**bin_args),
                title=value_col.replace("_", " ").title(),
            ),
            y=alt.Y("count()", title="Count"),
        )
        .properties(title=title, height=280)
    )

    if color_col and color_col in data:
        base = base.encode(
            color=alt.Color(
                f"{color_col}:N",
                legend=alt.Legend(title=color_col.replace("_", " ").title()),
            )
        )

    if facet_col and facet_col in data:
        facet_chart = base.facet(
            column=alt.Column(
                f"{facet_col}:N",
                title=facet_col.replace("_", " ").title(),
            ),
            columns=facet_columns,
            spacing=12,
        )
        return facet_chart.resolve_scale(x="independent", y="independent")

    return base
