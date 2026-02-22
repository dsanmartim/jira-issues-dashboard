"""DataFrame segment filters for dashboard top-N lists."""

from __future__ import annotations

from contextlib import suppress

import pandas as pd
import pytz

from jira_app.analytics.metrics.activity import (
    count_comments_in_range,
    count_histories_in_range,
    normalize_timestamp,
)
from jira_app.core.config import API_COMMENT_AUTHORS, TERMINAL_STATUSES


def _normalize_range_timestamps(start, end):
    """Normalize start/end timestamps for range filtering.

    Returns (start_ts, end_ts, target_tz) or (None, None, None) if invalid.
    """
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if start_ts is None or pd.isna(start_ts) or end_ts is None or pd.isna(end_ts):
        return None, None, None
    if getattr(start_ts, "tzinfo", None) is None:
        start_ts = start_ts.tz_localize(pytz.UTC)
    if getattr(end_ts, "tzinfo", None) is None:
        end_ts = end_ts.tz_localize(pytz.UTC)
    target_tz = start_ts.tz
    return start_ts, end_ts, target_tz


def most_active(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Filter and sort by activity (comments + history changes) in date range."""
    if df.empty:
        return df
    out = df.copy()

    # Use pre-computed comments_in_range or fall back to shared utility
    if "comments_in_range" in out.columns:
        out["filtered_comments_count"] = out["comments_in_range"]
    else:
        start_ts, end_ts, target_tz = _normalize_range_timestamps(start, end)
        if start_ts is None:
            out["filtered_comments_count"] = 0
        elif "comments" in out.columns:
            out["filtered_comments_count"] = out["comments"].apply(
                lambda c: count_comments_in_range(c, start_ts, end_ts, target_tz)
            )
        else:
            out["filtered_comments_count"] = 0

    # Use pre-computed history counts or total_activity_in_range
    if "filtered_histories_count" not in out.columns:
        if "status_changes" in out.columns and "other_changes" in out.columns:
            out["filtered_histories_count"] = out["status_changes"] + out["other_changes"]
        elif "total_activity_in_range" in out.columns and "comments_in_range" in out.columns:
            # Derive from total minus comments
            out["filtered_histories_count"] = out["total_activity_in_range"] - out["comments_in_range"]
        else:
            start_ts, end_ts, target_tz = _normalize_range_timestamps(start, end)
            if start_ts is None:
                out["filtered_histories_count"] = 0
            elif "histories" in out.columns:
                hist = out["histories"].apply(
                    lambda h: count_histories_in_range(h, start_ts, end_ts, target_tz)
                )
                out["filtered_histories_count"] = hist.apply(lambda t: t[0] + t[1])
            else:
                out["filtered_histories_count"] = 0

    out["activity_score"] = out["filtered_comments_count"] + out["filtered_histories_count"]
    return out.sort_values("activity_score", ascending=False)


def most_commented(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    out = most_active(df, start, end)  # ensures filtered_comments_count
    return out.sort_values("filtered_comments_count", ascending=False)


def blocker_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for open Blocker/Critical priority tickets."""
    if df.empty:
        return df
    is_high_priority = df["priority"].astype(str).str.startswith(("Blocker", "Critical"))
    is_open = ~df["status"].isin(TERMINAL_STATUSES)
    return df[is_high_priority & is_open].copy()


def testing_tracking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["status"].astype(str).str.startswith("Testing") | df["status"].astype(str).str.startswith(
        "Tracking"
    )
    return df[mask].copy()


def most_time_lost(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Sort by time lost value (uses pre-computed or derives from time_lost column)."""
    if df.empty:
        return df
    out = df.copy()
    # Use pre-computed time_lost_value if available, otherwise derive
    if "time_lost_value" not in out.columns:
        out["time_lost_value"] = pd.to_numeric(out.get("time_lost"), errors="coerce").fillna(0)
    return out.sort_values("time_lost_value", ascending=False)


def weighted_activity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Assume service already calculated activity_score_weighted
    if "activity_score_weighted" in df.columns:
        return df.sort_values("activity_score_weighted", ascending=False)
    return df


def ole_commented(
    df: pd.DataFrame,
    start,
    end,
    authors: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Filter issues with comments from API/system authors in date range.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    start, end : datetime-like
        Date range for filtering comments.
    authors : frozenset[str] | None
        Set of author names to match. Defaults to API_COMMENT_AUTHORS from config.
    """
    if df.empty:
        return df
    if authors is None:
        authors = API_COMMENT_AUTHORS
    start_ts, end_ts, target_tz = _normalize_range_timestamps(start, end)
    if start_ts is None:
        return pd.DataFrame()

    def extract_stats(comments):
        count = 0
        last_seen = pd.NaT
        if not isinstance(comments, list):
            return count, last_seen
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author_name = str(comment.get("author") or "").strip()
            if author_name not in authors:
                continue
            ts = normalize_timestamp(comment.get("created"), target_tz)
            if ts is None:
                continue
            if ts < start_ts or ts > end_ts:
                continue
            count += 1
            if pd.isna(last_seen) or ts > last_seen:
                last_seen = ts
        return count, last_seen

    out = df.copy()
    if "comments" not in out.columns:
        out["ole_comments_count"] = 0
        out["ole_last_comment"] = pd.NaT
    else:
        stats = out["comments"].apply(extract_stats)
        out["ole_comments_count"] = stats.apply(lambda pair: pair[0])
        out["ole_last_comment"] = stats.apply(lambda pair: pair[1])
    out = out[out["ole_comments_count"] > 0].copy()
    if out.empty:
        return out

    if hasattr(start, "tzinfo") and getattr(start, "tzinfo", None) is not None:
        with suppress(Exception):
            out["ole_last_comment"] = out["ole_last_comment"].dt.tz_convert(start.tzinfo)

    out = out.sort_values(["ole_comments_count", "ole_last_comment"], ascending=[False, False])
    return out
