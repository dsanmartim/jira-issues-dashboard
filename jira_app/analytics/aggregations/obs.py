"""OBS hierarchy aggregations."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from jira_app.analytics.metrics.derived import add_derived_metrics

# Type alias for valid OBS grouping columns
ObsGroupColumn = Literal["obs_system", "obs_subsystem", "obs_component"]


def aggregate_by_obs_hierarchy(
    df: pd.DataFrame,
    group_by: ObsGroupColumn,
    limit: int = 200,
) -> pd.DataFrame:
    """Aggregate issues by an OBS hierarchy level.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    group_by : {"obs_system", "obs_subsystem", "obs_component"}
        The OBS hierarchy column to group by.
    limit : int
        Maximum number of rows to return (default 200).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns: group_by column, count,
        time_lost_sum, activity_sum.
    """
    if df.empty:
        return df

    out = add_derived_metrics(df)
    agg = (
        out.groupby(group_by, dropna=False)
        .agg(
            count=("key", "count"),
            time_lost_sum=("time_lost_value", "sum"),
            activity_sum=("total_activity_in_range", "sum"),
        )
        .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
        .head(limit)
        .reset_index()
    )
    return agg


# Convenience functions for backward compatibility
def aggregate_by_obs_system(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """Aggregate issues by OBS system level."""
    return aggregate_by_obs_hierarchy(df, group_by="obs_system", limit=limit)


def aggregate_by_obs_subsystem(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """Aggregate issues by OBS subsystem level."""
    return aggregate_by_obs_hierarchy(df, group_by="obs_subsystem", limit=limit)


def aggregate_by_obs_component(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    """Aggregate issues by OBS component level."""
    return aggregate_by_obs_hierarchy(df, group_by="obs_component", limit=limit)
