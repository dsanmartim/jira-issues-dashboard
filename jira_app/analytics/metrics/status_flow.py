"""Status flow and duration analysis utilities.

This module provides functions to analyze how issues transition through
workflow statuses and compute time spent in each status.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd

from jira_app.core.config import TERMINAL_STATUSES
from jira_app.core.status import clean_status_name, normalize_workflow_status


def normalize_to_tz(value, tz):
    """Normalize a timestamp value to a specific timezone.

    Parameters
    ----------
    value : datetime-like or None
        Raw timestamp value (string, datetime, or None).
    tz : timezone
        Target timezone for conversion.

    Returns
    -------
    datetime or None
        Timezone-aware datetime or None if conversion fails.
    """
    if value is None or pd.isna(value):
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.tz_convert(tz)


def extract_issue_status_durations(row: pd.Series, tz, now_ts) -> dict[str, float]:
    """Extract time spent in each status for a single issue.

    Analyzes the issue's history to compute how many days were spent
    in each workflow status.

    Parameters
    ----------
    row : pd.Series
        A row from an issues DataFrame containing 'histories', 'status',
        'created_dt'/'created', and optionally 'resolution_dt'.
    tz : timezone
        Timezone for timestamp normalization.
    now_ts : datetime
        Current timestamp for calculating time in current status.

    Returns
    -------
    dict[str, float]
        Mapping of status name to days spent in that status.
    """
    histories = row.get("histories")
    if not isinstance(histories, list) or not histories:
        return {}

    events: list[tuple[datetime, str | None, str | None]] = []
    for entry in histories:
        created = normalize_to_tz(entry.get("created"), tz)
        if created is None:
            continue
        items = entry.get("items") or []
        for item in items:
            field_name = str(item.get("field") or "").lower()
            if field_name != "status":
                continue
            from_status = clean_status_name(item.get("fromString") or item.get("from"))
            to_status = clean_status_name(item.get("toString") or item.get("to"))
            events.append((created, from_status, to_status))

    if not events:
        return {}

    events.sort(key=lambda tup: tup[0])
    created_dt = row.get("created_dt") or normalize_to_tz(row.get("created"), tz)
    resolution_dt = row.get("resolution_dt") or normalize_to_tz(
        row.get("resolution_date") or row.get("resolutiondate"), tz
    )

    current_status = clean_status_name(events[0][1] or row.get("status"))
    current_start = created_dt or events[0][0]
    if current_start is None:
        current_start = events[0][0]
    if current_start is None:
        return {}

    durations: defaultdict[str, float] = defaultdict(float)

    for change_time, _from_status, to_status in events:
        if change_time is None or current_start is None:
            continue
        delta = (change_time - current_start).total_seconds() / 86400.0
        if delta >= 0:
            durations[current_status] += delta
        current_status = clean_status_name(to_status or current_status)
        current_start = change_time

    end_time = resolution_dt if resolution_dt is not None and not pd.isna(resolution_dt) else now_ts
    if current_start is not None and end_time is not None and end_time > current_start:
        delta = (end_time - current_start).total_seconds() / 86400.0
        if delta >= 0:
            durations[current_status] += delta

    return dict(durations)


def build_status_duration_frame(df: pd.DataFrame, tz, now_ts) -> pd.DataFrame:
    """Build a DataFrame with status duration records for all issues.

    Creates a long-form DataFrame where each row represents time spent
    by an issue in a particular status.

    Parameters
    ----------
    df : pd.DataFrame
        Issues DataFrame with 'histories' and 'status' columns.
    tz : timezone
        Timezone for timestamp normalization.
    now_ts : datetime
        Current timestamp for calculating time in current status.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: key, status, duration_days, is_open.
        Returns empty DataFrame if no duration data can be computed.
    """
    if df.empty or "histories" not in df.columns:
        return pd.DataFrame()

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        durations = extract_issue_status_durations(row, tz, now_ts)
        if not durations:
            continue
        status_value = str(row.get("status") or "")
        is_open = normalize_workflow_status(status_value) not in TERMINAL_STATUSES
        for status_name, days in durations.items():
            if days is None or pd.isna(days):
                continue
            records.append(
                {
                    "key": row.get("key"),
                    "status": normalize_workflow_status(clean_status_name(status_name)),
                    "duration_days": float(days),
                    "is_open": is_open,
                }
            )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
