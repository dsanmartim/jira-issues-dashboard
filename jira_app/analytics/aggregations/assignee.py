"""Assignee-based aggregations."""

from __future__ import annotations

import pandas as pd

from jira_app.analytics.metrics.derived import add_derived_metrics


def aggregate_by_assignee(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """Aggregate issues by assignee.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    limit : int
        Maximum number of rows to return (default 200).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns: assignee, issues, time_lost_sum, activity_sum.
    """
    if df.empty:
        return df

    out = add_derived_metrics(df)
    agg = (
        out.groupby("assignee", dropna=False)
        .agg(
            issues=("key", "count"),
            time_lost_sum=("time_lost_value", "sum"),
            activity_sum=("total_activity_in_range", "sum"),
        )
        .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
        .head(limit)
    )
    return agg.reset_index()
