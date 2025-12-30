"""Shared helpers for persona-based ticket analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytz

from jira_app.core.column_config import get_columns
from jira_app.core.config import TIMEZONE

DONE_STATUSES = {"done", "resolved", "closed", "cancelled", "completed", "duplicate", "duplicated"}
TZ = pytz.timezone(TIMEZONE)


def clean_entity(series: pd.Series, placeholder: str) -> pd.Series:
    filled = series.fillna(placeholder)
    return filled.replace("", placeholder).astype(str)


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_open_workload(
    df: pd.DataFrame,
    *,
    entity_column: str,
    entity_label: str,
    placeholder: str,
    include_priority: bool = False,
) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()

    status_series = df["status"].astype(str).str.lower()
    open_df = df[~status_series.isin(DONE_STATUSES)].copy()
    if open_df.empty:
        return pd.DataFrame()

    open_df[entity_label] = clean_entity(open_df.get(entity_column, pd.Series(dtype=object)), placeholder)
    open_df["days_open"] = ensure_numeric(open_df.get("days_open"))
    open_df["days_since_update"] = ensure_numeric(open_df.get("days_since_update"))

    aggregations: dict[str, tuple[str, str]] = {
        "open_tickets": ("key", "nunique"),
        "avg_days_open": ("days_open", "mean"),
        "median_days_open": ("days_open", "median"),
        "oldest_days_open": ("days_open", "max"),
        "updated_within_7": (
            "days_since_update",
            lambda s: s.le(7).sum() if s.notna().any() else 0,
        ),
    }

    if include_priority:
        open_df["priority"] = open_df.get("priority", "None").fillna("None").astype(str)
        open_df["is_blocker"] = open_df["priority"].str.startswith(("Blocker", "Critical"), na=False)
        open_df["is_high"] = open_df["priority"].str.startswith("High", na=False)
        aggregations["blocker_critical"] = ("is_blocker", "sum")
        aggregations["high_priority"] = ("is_high", "sum")

    grouped = (
        open_df.groupby(entity_label)
        .agg(**aggregations)
        .sort_values("open_tickets", ascending=False)
        .reset_index()
    )

    if grouped.empty:
        return grouped

    grouped["pct_recent_update"] = (
        grouped["updated_within_7"]
        .where(grouped["open_tickets"] > 0)
        .divide(grouped["open_tickets"].where(grouped["open_tickets"] > 0))
        .multiply(100)
        .fillna(0.0)
    )

    columns = [
        entity_label,
        "open_tickets",
    ]
    if include_priority:
        columns.extend(["blocker_critical", "high_priority"])
    columns.extend(
        [
            "avg_days_open",
            "median_days_open",
            "oldest_days_open",
            "pct_recent_update",
        ]
    )

    display = grouped[columns].rename(
        columns={
            "open_tickets": "Open tickets",
            "blocker_critical": "Blocker/Critical",
            "high_priority": "High",
            "avg_days_open": "Avg days open",
            "median_days_open": "Median days open",
            "oldest_days_open": "Oldest open (days)",
            "pct_recent_update": "% updated ≤7d",
        }
    )

    int_cols = {"Open tickets"}
    if include_priority:
        int_cols.update({"Blocker/Critical", "High"})
    for col in int_cols:
        if col in display.columns:
            display[col] = display[col].astype("Int64")

    for col in ["Avg days open", "Median days open", "Oldest open (days)", "% updated ≤7d"]:
        if col in display.columns:
            display[col] = display[col].astype(float).round(1)

    return display


def build_resolution_metrics(
    df: pd.DataFrame,
    *,
    entity_column: str,
    entity_label: str,
    placeholder: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if "resolution_dt" in df.columns:
        resolution_series = pd.to_datetime(df["resolution_dt"], errors="coerce", utc=True)
    elif "resolution_date" in df.columns:
        resolution_series = pd.to_datetime(df["resolution_date"], errors="coerce", utc=True)
    elif "resolutiondate" in df.columns:
        resolution_series = pd.to_datetime(df["resolutiondate"], errors="coerce", utc=True)
    else:
        return pd.DataFrame()

    resolution_series = resolution_series.dt.tz_convert(TZ)
    mask = resolution_series.notna()
    resolved = df[mask].copy()
    if resolved.empty:
        return pd.DataFrame()

    resolved[entity_label] = clean_entity(resolved.get(entity_column, pd.Series(dtype=object)), placeholder)
    resolved["resolution_dt_local"] = resolution_series[mask]

    if "created_dt" in resolved.columns:
        created_series = pd.to_datetime(resolved["created_dt"], errors="coerce", utc=True)
    else:
        created_series = pd.to_datetime(resolved.get("created"), errors="coerce", utc=True)
    created_series = created_series.dt.tz_convert(TZ)
    resolved["created_dt_local"] = created_series

    valid_mask = resolved["created_dt_local"].notna()
    resolved = resolved[valid_mask].copy()
    if resolved.empty:
        return pd.DataFrame()

    resolved["cycle_time_days"] = (
        resolved["resolution_dt_local"] - resolved["created_dt_local"]
    ).dt.total_seconds() / 86400.0

    def _p90(values: pd.Series) -> float:
        arr = values.dropna().to_numpy()
        if arr.size == 0:
            return float("nan")
        return float(np.percentile(arr, 90))

    grouped = (
        resolved.groupby(entity_label)
        .agg(
            resolved_tickets=("key", "nunique"),
            median_days=("cycle_time_days", "median"),
            mean_days=("cycle_time_days", "mean"),
            p90_days=("cycle_time_days", _p90),
            most_recent=("resolution_dt_local", "max"),
        )
        .sort_values("resolved_tickets", ascending=False)
        .reset_index()
    )

    if grouped.empty:
        return grouped

    grouped["Most recent resolution"] = grouped["most_recent"].dt.strftime("%Y-%m-%d")
    grouped = grouped.drop(columns=["most_recent"])

    display = grouped.rename(
        columns={
            "resolved_tickets": "Resolved tickets",
            "median_days": "Median resolution (days)",
            "mean_days": "Mean resolution (days)",
            "p90_days": "90th percentile (days)",
        }
    )

    display["Resolved tickets"] = display["Resolved tickets"].astype("Int64")
    for col in ["Median resolution (days)", "Mean resolution (days)", "90th percentile (days)"]:
        if col in display.columns:
            display[col] = display[col].astype(float).round(1)

    return display


def build_activity_summary(
    df: pd.DataFrame,
    *,
    entity_column: str,
    entity_label: str,
    placeholder: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    working[entity_label] = clean_entity(working.get(entity_column, pd.Series(dtype=object)), placeholder)

    aggregations: dict[str, tuple[str, str]] = {
        "Tickets": ("key", "nunique"),
    }
    if "comments_in_range" in working.columns:
        aggregations["Comments (range)"] = ("comments_in_range", "sum")
    if "status_changes" in working.columns:
        aggregations["Status changes"] = ("status_changes", "sum")
    if "other_changes" in working.columns:
        aggregations["Other changes"] = ("other_changes", "sum")
    if "activity_score_weighted" in working.columns:
        aggregations["Total weighted score"] = ("activity_score_weighted", "sum")
        aggregations["Avg weighted score"] = ("activity_score_weighted", "mean")

    summary = (
        working.groupby(entity_label)
        .agg(**aggregations)
        .sort_values("Tickets", ascending=False)
        .reset_index()
    )

    if summary.empty:
        return summary

    for col in summary.columns:
        if col in {"Tickets", "Comments (range)", "Status changes", "Other changes"}:
            summary[col] = summary[col].astype("Int64")
        elif "score" in col.lower():
            summary[col] = summary[col].astype(float).round(1)

    return summary


def prepare_open_ticket_detail(
    df: pd.DataFrame,
    *,
    entity_column: str,
    placeholder: str,
    include_status_filter: bool = True,
    columns: list[str] | None = None,
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    working[entity_column] = clean_entity(working.get(entity_column, pd.Series(dtype=object)), placeholder)

    if include_status_filter and "status" in working.columns:
        status_series = working["status"].astype(str).str.lower()
        working = working[~status_series.isin(DONE_STATUSES)]

    if working.empty:
        return pd.DataFrame()

    base_columns = list(columns or get_columns("core"))
    if entity_column not in base_columns:
        base_columns.append(entity_column)
    if extra_columns:
        for col in extra_columns:
            if col not in base_columns:
                base_columns.append(col)

    if "days_open" in working.columns:
        working["days_open"] = ensure_numeric(working.get("days_open")).round(1)
    if "days_since_update" in working.columns:
        working["days_since_update"] = ensure_numeric(working.get("days_since_update")).round(1)

    present = [col for col in base_columns if col in working.columns]
    return working[present]
