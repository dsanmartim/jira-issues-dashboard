"""DataFrame segment filters for dashboard top-N lists."""

from __future__ import annotations

from contextlib import suppress

import pandas as pd
import pytz

from jira_app.analytics.metrics.activity import normalize_timestamp

OLE_API_DISPLAY_NAME = "Rubin Jira API Access"


def most_active(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "comments_in_range" in out.columns:
        out["filtered_comments_count"] = out["comments_in_range"]
    else:
        start_ts = pd.to_datetime(start, errors="coerce")
        end_ts = pd.to_datetime(end, errors="coerce")
        if start_ts is None or pd.isna(start_ts) or end_ts is None or pd.isna(end_ts):
            out["filtered_comments_count"] = 0
        else:
            if getattr(start_ts, "tzinfo", None) is None:
                start_ts = start_ts.tz_localize(pytz.UTC)
            if getattr(end_ts, "tzinfo", None) is None:
                end_ts = end_ts.tz_localize(pytz.UTC)
            target_tz = start_ts.tz

            def comment_count(comments):
                if not isinstance(comments, list):
                    return 0
                total = 0
                for comment in comments:
                    if not isinstance(comment, dict):
                        continue
                    ts = normalize_timestamp(comment.get("created"), target_tz)
                    if ts is None or ts < start_ts or ts > end_ts:
                        continue
                    total += 1
                return total

            if "comments" in out.columns:
                out["filtered_comments_count"] = out["comments"].apply(comment_count)
            else:
                out["filtered_comments_count"] = 0

    if "filtered_histories_count" not in out.columns:
        if "status_changes" in out.columns and "other_changes" in out.columns:
            out["filtered_histories_count"] = out["status_changes"] + out["other_changes"]
        else:
            start_ts = pd.to_datetime(start, errors="coerce")
            end_ts = pd.to_datetime(end, errors="coerce")
            if start_ts is None or pd.isna(start_ts) or end_ts is None or pd.isna(end_ts):
                out["filtered_histories_count"] = 0
            else:
                if getattr(start_ts, "tzinfo", None) is None:
                    start_ts = start_ts.tz_localize(pytz.UTC)
                if getattr(end_ts, "tzinfo", None) is None:
                    end_ts = end_ts.tz_localize(pytz.UTC)
                target_tz = start_ts.tz

                def history_count(histories):
                    if not isinstance(histories, list):
                        return 0
                    total = 0
                    for history in histories:
                        if not isinstance(history, dict):
                            continue
                        ts = normalize_timestamp(history.get("created"), target_tz)
                        if ts is None or ts < start_ts or ts > end_ts:
                            continue
                        total += 1
                    return total

                if "histories" in out.columns:
                    out["filtered_histories_count"] = out["histories"].apply(history_count)
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
    if df.empty:
        return df
    mask = df["priority"].astype(str).str.startswith(("Blocker", "Critical")) & (df["status"] != "Done")
    return df[mask].copy()


def testing_tracking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["status"].astype(str).str.startswith("Testing") | df["status"].astype(str).str.startswith(
        "Tracking"
    )
    return df[mask].copy()


def most_time_lost(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["time_lost_value"] = pd.to_numeric(out.get("time_lost"), errors="coerce").fillna(0)
    return out.sort_values("time_lost_value", ascending=False)


def weighted_activity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Assume service already calculated activity_score_weighted
    if "activity_score_weighted" in df.columns:
        return df.sort_values("activity_score_weighted", ascending=False)
    return df


def ole_commented(df: pd.DataFrame, start, end, author: str = OLE_API_DISPLAY_NAME) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if start_ts is None or pd.isna(start_ts) or end_ts is None or pd.isna(end_ts):
        return pd.DataFrame()
    if getattr(start_ts, "tzinfo", None) is None:
        start_ts = start_ts.tz_localize(pytz.UTC)
    if getattr(end_ts, "tzinfo", None) is None:
        end_ts = end_ts.tz_localize(pytz.UTC)
    target_tz = start_ts.tz

    def extract_stats(comments):
        count = 0
        last_seen = pd.NaT
        if not isinstance(comments, list):
            return count, last_seen
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author_name = str(comment.get("author") or "").strip()
            if author_name != author:
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
