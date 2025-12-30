"""OBS hierarchy aggregations."""

from __future__ import annotations

import pandas as pd


def _base(df: pd.DataFrame):
    out = df.copy()
    # Ensure we always operate on a Series;
    # out.get("time_lost") would return scalar 0 if missing.
    if "time_lost" in out.columns:
        out["time_lost_value"] = pd.to_numeric(out["time_lost"], errors="coerce").fillna(0)
    else:
        out["time_lost_value"] = 0
    out["total_activity_in_range"] = (
        out.get("comments_in_range", 0) + out.get("status_changes", 0) + out.get("other_changes", 0)
    )
    return out


def aggregate_by_obs_system(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    if df.empty:
        return df
    out = _base(df)
    agg = (
        out.groupby("obs_system", dropna=False)
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


def aggregate_by_obs_subsystem(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    if df.empty:
        return df
    out = _base(df)
    agg = (
        out.groupby("obs_subsystem", dropna=False)
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


def aggregate_by_obs_component(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    if df.empty:
        return df
    out = _base(df)
    agg = (
        out.groupby("obs_component", dropna=False)
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
