"""Activity & weighted engagement metrics."""

from __future__ import annotations

import pandas as pd
import pytz


def normalize_timestamp(value, target_tz) -> pd.Timestamp | None:
    """Normalize a timestamp-like value into `target_tz`.

    Returns None when the input cannot be parsed.

    This is the public API. `_normalize_timestamp` remains for backward
    compatibility with older imports.
    """

    return _normalize_timestamp(value, target_tz)


def _normalize_timestamp(value, target_tz) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    try:
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(pytz.UTC)
    except (TypeError, ValueError):
        return None
    try:
        return ts.tz_convert(target_tz)
    except (TypeError, ValueError):
        return None


def within_range(ts, start_local, end_local, *, tzinfo):
    normalized = normalize_timestamp(ts, tzinfo)
    if normalized is None:
        return False
    return (normalized >= start_local) and (normalized <= end_local)


def add_weighted_activity(
    df: pd.DataFrame,
    start,
    end,
    w_comment: float = 2.0,
    w_status: float = 0.5,
    w_other: float = 0.5,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    range_start = pd.to_datetime(start, errors="coerce")
    range_end = pd.to_datetime(end, errors="coerce")
    if range_start is None or pd.isna(range_start) or range_end is None or pd.isna(range_end):
        return out
    if getattr(range_start, "tzinfo", None) is None:
        range_start = range_start.tz_localize(pytz.UTC)
    if getattr(range_end, "tzinfo", None) is None:
        range_end = range_end.tz_localize(pytz.UTC)
    target_tz = range_start.tz
    range_start = range_start.tz_convert(target_tz)
    range_end = range_end.tz_convert(target_tz)

    def comment_count(comments):
        if not isinstance(comments, list):
            return 0
        total = 0
        for c in comments:
            if not isinstance(c, dict):
                continue
            if within_range(c.get("created"), range_start, range_end, tzinfo=target_tz):
                total += 1
        return total

    def history_counts(histories):
        status_changes = 0
        other_changes = 0
        if not isinstance(histories, list):
            return status_changes, other_changes
        for h in histories:
            if not isinstance(h, dict):
                continue
            if not within_range(h.get("created"), range_start, range_end, tzinfo=target_tz):
                continue
            items = h.get("items") or []

            def _field_name(it):
                if isinstance(it, dict):
                    return it.get("field")
                return getattr(it, "field", None)

            if any(_field_name(it) == "status" for it in items):
                status_changes += 1
            else:
                other_changes += 1
        return status_changes, other_changes

    out["comments_in_range"] = out["comments"].apply(comment_count)
    hist = out["histories"].apply(lambda hs: history_counts(hs))
    out["status_changes"] = hist.apply(lambda t: t[0])
    out["other_changes"] = hist.apply(lambda t: t[1])
    out["activity_score_weighted"] = (
        out["comments_in_range"] * w_comment
        + out["status_changes"] * w_status
        + out["other_changes"] * w_other
    )
    return out
