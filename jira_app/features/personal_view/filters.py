"""User-specific filtering functions for Personal View."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import pytz

from jira_app.analytics.metrics.activity import normalize_timestamp
from jira_app.core.config import API_COMMENT_AUTHORS
from jira_app.core.status import DONE_STATUSES_LOWER


def _extract_text_from_adf(body) -> str:
    """Extract plain text from Atlassian Document Format (ADF) content.

    ADF is a JSON structure used by Jira Cloud for rich text content.
    This function recursively extracts all text nodes.

    Parameters
    ----------
    body : str or dict
        The comment body, either as a JSON string or parsed dict.

    Returns
    -------
    str
        Plain text content extracted from the ADF structure.
    """
    if body is None:
        return ""

    # If it's a string, try to parse as JSON (ADF format)
    if isinstance(body, str):
        # Check if it looks like ADF JSON
        stripped = body.strip()
        if stripped.startswith("{") and '"type"' in stripped:
            try:
                body = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, return as plain text
                return stripped[:500]
        else:
            # Plain text, return directly
            return stripped[:500]

    # If it's a dict, recursively extract text
    if isinstance(body, dict):
        texts: list[str] = []

        # Direct text node
        if body.get("type") == "text" and "text" in body:
            texts.append(str(body["text"]))

        # Recurse into content array
        content = body.get("content")
        if isinstance(content, list):
            for item in content:
                extracted = _extract_text_from_adf(item)
                if extracted:
                    texts.append(extracted)

        return " ".join(texts)

    # If it's a list, process each item
    if isinstance(body, list):
        texts = []
        for item in body:
            extracted = _extract_text_from_adf(item)
            if extracted:
                texts.append(extracted)
        return " ".join(texts)

    # Fallback: convert to string
    return str(body)[:500]


def assigned_to(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """Filter issues assigned to the specified user.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    user : str
        User display name to filter by.

    Returns
    -------
    pd.DataFrame
        Issues assigned to the user.
    """
    if df.empty or not user:
        return pd.DataFrame()
    mask = df["assignee"].astype(str).str.lower() == user.lower()
    return df[mask].copy()


def reported_by(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """Filter issues reported by the specified user.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    user : str
        User display name to filter by.

    Returns
    -------
    pd.DataFrame
        Issues reported by the user.
    """
    if df.empty or not user:
        return pd.DataFrame()
    mask = df["reporter"].astype(str).str.lower() == user.lower()
    return df[mask].copy()


def watching(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """Filter issues the user is watching.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    user : str
        User display name to filter by.

    Returns
    -------
    pd.DataFrame
        Issues the user is watching.
    """
    if df.empty or not user:
        return pd.DataFrame()
    if "watchers" not in df.columns:
        return pd.DataFrame()

    user_lower = user.lower()

    def is_watching(watchers):
        if not isinstance(watchers, list):
            return False
        return any(str(w).lower() == user_lower for w in watchers)

    mask = df["watchers"].apply(is_watching)
    return df[mask].copy()


def mentioned_in(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """Filter issues where the user is mentioned in comments.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    user : str
        User display name to search for in comment bodies.

    Returns
    -------
    pd.DataFrame
        Issues where the user is mentioned.
    """
    if df.empty or not user:
        return pd.DataFrame()
    if "comments" not in df.columns:
        return pd.DataFrame()

    user_lower = user.lower()

    def has_mention(comments):
        if not isinstance(comments, list):
            return False
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            body = str(comment.get("body") or "").lower()
            if user_lower in body:
                return True
        return False

    mask = df["comments"].apply(has_mention)
    return df[mask].copy()


def commented_on(df: pd.DataFrame, user: str) -> pd.DataFrame:
    """Filter issues where the user has commented.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    user : str
        User display name to search for in comment authors.

    Returns
    -------
    pd.DataFrame
        Issues where the user has commented.
    """
    if df.empty or not user:
        return pd.DataFrame()
    if "comments" not in df.columns:
        return pd.DataFrame()

    user_lower = user.lower()

    def has_commented(comments):
        if not isinstance(comments, list):
            return False
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author = str(comment.get("author") or "").lower()
            if author == user_lower:
                return True
        return False

    mask = df["comments"].apply(has_commented)
    return df[mask].copy()


def stale_issues(df: pd.DataFrame, stale_days: int = 30) -> pd.DataFrame:
    """Filter issues that haven't been updated recently.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    stale_days : int
        Number of days since last update to consider stale.

    Returns
    -------
    pd.DataFrame
        Stale issues sorted by days since update (descending).
    """
    if df.empty:
        return df
    if "days_since_update" not in df.columns:
        return pd.DataFrame()
    # Only open issues
    is_open = ~df["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)
    is_stale = df["days_since_update"] >= stale_days
    result = df[is_open & is_stale].copy()
    if not result.empty:
        result = result.sort_values("days_since_update", ascending=False)
    return result


def blocked_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Filter issues with 'Blocked' status.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.

    Returns
    -------
    pd.DataFrame
        Issues with Blocked status.
    """
    if df.empty:
        return df
    mask = df["status"].astype(str).str.lower() == "blocked"
    return df[mask].copy()


def needs_attention(
    df: pd.DataFrame,
    user: str,
    stale_days: int = 30,
) -> pd.DataFrame:
    """Filter issues assigned to user that need attention.

    Includes:
    - Blocked issues
    - Stale issues (no updates in stale_days)
    - High priority issues (Blocker/Critical)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    user : str
        User display name to filter by.
    stale_days : int
        Number of days since last update to consider stale.

    Returns
    -------
    pd.DataFrame
        Issues needing attention, deduplicated and sorted by priority.
    """
    if df.empty or not user:
        return pd.DataFrame()

    # Get user's assigned issues
    user_issues = assigned_to(df, user)
    if user_issues.empty:
        return pd.DataFrame()

    # Filter for open issues only
    is_open = ~user_issues["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)
    user_open = user_issues[is_open].copy()
    if user_open.empty:
        return pd.DataFrame()

    # Criteria for needing attention
    is_blocked = user_open["status"].astype(str).str.lower() == "blocked"
    is_stale = (
        user_open["days_since_update"] >= stale_days if "days_since_update" in user_open.columns else False
    )
    is_high_priority = user_open["priority"].astype(str).str.startswith(("Blocker", "Critical"))

    # Combine criteria
    needs_attn = is_blocked | is_stale | is_high_priority
    result = user_open[needs_attn].copy()

    if not result.empty and "priority_value" in result.columns:
        result = result.sort_values("priority_value", ascending=False)

    return result


def with_ole_comments(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    tz: pytz.BaseTzInfo | None = None,
) -> pd.DataFrame:
    """Filter issues that have OLE/LOVE API comments in the date range.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    start, end : datetime
        Date range for filtering comments.
    tz : timezone, optional
        Timezone for timestamp normalization.

    Returns
    -------
    pd.DataFrame
        Issues with OLE comments, with ole_comments_count and ole_last_comment columns.
    """
    if df.empty:
        return pd.DataFrame()
    if "comments" not in df.columns:
        return pd.DataFrame()

    if tz is None:
        tz = pytz.UTC

    # Normalize timestamps
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if start_ts is None or end_ts is None:
        return pd.DataFrame()

    def extract_ole_stats(comments):
        count = 0
        last_seen = pd.NaT
        if not isinstance(comments, list):
            return count, last_seen
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author_name = str(comment.get("author") or "").strip()
            if author_name not in API_COMMENT_AUTHORS:
                continue
            ts = normalize_timestamp(comment.get("created"), tz)
            if ts is None:
                continue
            if ts < start_ts or ts > end_ts:
                continue
            count += 1
            if pd.isna(last_seen) or ts > last_seen:
                last_seen = ts
        return count, last_seen

    out = df.copy()
    stats = out["comments"].apply(extract_ole_stats)
    out["ole_comments_count"] = stats.apply(lambda pair: pair[0])
    out["ole_last_comment"] = stats.apply(lambda pair: pair[1])
    out = out[out["ole_comments_count"] > 0].copy()

    if not out.empty:
        out = out.sort_values(["ole_comments_count", "ole_last_comment"], ascending=[False, False])

    return out


def extract_recent_activity(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    tz: pytz.BaseTzInfo | None = None,
    activity_types: list[str] | None = None,
) -> pd.DataFrame:
    """Extract recent activity (comments, status changes, other changes) from issues.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with issue data.
    start, end : datetime
        Date range for filtering activity.
    tz : timezone, optional
        Timezone for timestamp normalization.
    activity_types : list[str], optional
        Types to include: "comments", "status", "other". Defaults to all.

    Returns
    -------
    pd.DataFrame
        Activity records with columns: key, summary, activity_type, author, created, details.
    """
    if df.empty:
        return pd.DataFrame()

    if tz is None:
        tz = pytz.UTC
    if activity_types is None:
        activity_types = ["comments", "status", "other"]

    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if start_ts is None or end_ts is None:
        return pd.DataFrame()

    records: list[dict] = []

    for _, row in df.iterrows():
        issue_key = row.get("key", "")
        issue_summary = row.get("summary", "")

        # Extract comments
        if "comments" in activity_types and "comments" in row.index:
            comments = row.get("comments") or []
            if isinstance(comments, list):
                for comment in comments:
                    if not isinstance(comment, dict):
                        continue
                    ts = normalize_timestamp(comment.get("created"), tz)
                    if ts is None or ts < start_ts or ts > end_ts:
                        continue
                    author = str(comment.get("author") or "Unknown")
                    body = _extract_text_from_adf(comment.get("body"))[:200]  # Extract text and truncate
                    records.append(
                        {
                            "key": issue_key,
                            "summary": issue_summary,
                            "activity_type": "Comment",
                            "author": author,
                            "created": ts,
                            "details": body,
                        }
                    )

        # Extract history items
        if ("status" in activity_types or "other" in activity_types) and "histories" in row.index:
            histories = row.get("histories") or []
            if isinstance(histories, list):
                for hist in histories:
                    if not isinstance(hist, dict):
                        continue
                    ts = normalize_timestamp(hist.get("created"), tz)
                    if ts is None or ts < start_ts or ts > end_ts:
                        continue
                    author = str(hist.get("author") or "Unknown")
                    field = str(hist.get("field") or "")
                    items = hist.get("items") or []

                    # Determine activity type
                    if field.lower() == "status":
                        if "status" not in activity_types:
                            continue
                        act_type = "Status Change"
                    else:
                        if "other" not in activity_types:
                            continue
                        act_type = "Field Change"

                    # Build details string
                    details_parts = []
                    for item in items[:3]:  # Limit items
                        if isinstance(item, dict):
                            f = item.get("field", "")
                            fr = item.get("fromString", "") or ""
                            to = item.get("toString", "") or ""
                            if f.lower() == "status":
                                details_parts.append(f"{fr} → {to}")
                            else:
                                details_parts.append(f"{f}: {fr} → {to}" if fr else f"{f}: {to}")
                    details = "; ".join(details_parts)

                    records.append(
                        {
                            "key": issue_key,
                            "summary": issue_summary,
                            "activity_type": act_type,
                            "author": author,
                            "created": ts,
                            "details": details,
                        }
                    )

    if not records:
        return pd.DataFrame()

    activity_df = pd.DataFrame(records)
    activity_df = activity_df.sort_values("created", ascending=False)
    return activity_df
