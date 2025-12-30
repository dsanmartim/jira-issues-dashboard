"""Assignee-based aggregations."""

from __future__ import annotations

import pandas as pd


def aggregate_by_assignee(df: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["time_lost_value"] = pd.to_numeric(out.get("time_lost"), errors="coerce").fillna(0)
    out["total_activity_in_range"] = (
        out.get("comments_in_range", 0) + out.get("status_changes", 0) + out.get("other_changes", 0)
    )
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
