"""Pure helpers to build Personal View context for testing (no Streamlit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import pytz

from jira_app.core.status import DONE_STATUSES_LOWER
from jira_app.features.personal_view import filters as pv


@dataclass(slots=True)
class PersonalViewContext:
    """Context data for the Personal View page."""

    user: str
    assigned: pd.DataFrame
    reported: pd.DataFrame
    watching: pd.DataFrame
    mentioned: pd.DataFrame
    commented: pd.DataFrame
    needs_attention: pd.DataFrame
    stale: pd.DataFrame
    blocked: pd.DataFrame
    # OLE comments by category
    ole_assigned: pd.DataFrame = field(default_factory=pd.DataFrame)
    ole_reported: pd.DataFrame = field(default_factory=pd.DataFrame)
    ole_watching: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Recent activity
    recent_activity: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Summary metrics
    total_assigned_open: int = 0
    total_reported_open: int = 0
    total_watching: int = 0
    total_needs_attention: int = 0
    total_time_lost: float = 0.0
    total_ole_comments: int = 0
    # Distributions for charts
    status_distribution: dict[str, int] = field(default_factory=dict)
    priority_distribution: dict[str, int] = field(default_factory=dict)
    age_buckets: dict[str, int] = field(default_factory=dict)


def build_personal_context(
    df: pd.DataFrame,
    user: str,
    stale_days: int = 30,
    top_n: int = 25,
    start: datetime | None = None,
    end: datetime | None = None,
    tz: pytz.BaseTzInfo | None = None,
) -> PersonalViewContext:
    """Build context for the Personal View page.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with issue data.
    user : str
        User display name to filter by.
    stale_days : int
        Number of days since last update to consider stale.
    top_n : int
        Maximum number of items in each section.
    start, end : datetime, optional
        Date range for OLE comments and recent activity filtering.
    tz : timezone, optional
        Timezone for timestamp normalization.

    Returns
    -------
    PersonalViewContext
        Assembled context data for the page.
    """
    if df.empty or not user:
        return PersonalViewContext(
            user=user or "",
            assigned=pd.DataFrame(),
            reported=pd.DataFrame(),
            watching=pd.DataFrame(),
            mentioned=pd.DataFrame(),
            commented=pd.DataFrame(),
            needs_attention=pd.DataFrame(),
            stale=pd.DataFrame(),
            blocked=pd.DataFrame(),
        )

    if tz is None:
        tz = pytz.UTC

    # Filter by user
    assigned_df = pv.assigned_to(df, user)
    reported_df = pv.reported_by(df, user)
    watching_df = pv.watching(df, user)
    mentioned_df = pv.mentioned_in(df, user)
    commented_df = pv.commented_on(df, user)
    needs_attn_df = pv.needs_attention(df, user, stale_days)

    # Stale and blocked from user's assigned issues
    stale_df = pd.DataFrame()
    blocked_df = pd.DataFrame()
    assigned_open = pd.DataFrame()
    if not assigned_df.empty:
        # Only open assigned issues for stale check
        is_open = ~assigned_df["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)
        assigned_open = assigned_df[is_open]
        stale_df = pv.stale_issues(assigned_open, stale_days)
        blocked_df = pv.blocked_issues(assigned_open)

    # OLE comments by category
    ole_assigned = pd.DataFrame()
    ole_reported = pd.DataFrame()
    ole_watching = pd.DataFrame()
    total_ole_comments = 0
    if start is not None and end is not None:
        if not assigned_df.empty:
            ole_assigned = pv.with_ole_comments(assigned_df, start, end, tz)
        if not reported_df.empty:
            ole_reported = pv.with_ole_comments(reported_df, start, end, tz)
        if not watching_df.empty:
            ole_watching = pv.with_ole_comments(watching_df, start, end, tz)
        # Calculate total OLE comments (deduplicated by key)
        ole_keys: set[str] = set()
        for ole_df in [ole_assigned, ole_reported, ole_watching]:
            if not ole_df.empty and "key" in ole_df.columns and "ole_comments_count" in ole_df.columns:
                for _, row in ole_df.iterrows():
                    key = row.get("key")
                    if key and key not in ole_keys:
                        ole_keys.add(key)
                        total_ole_comments += int(row.get("ole_comments_count", 0))

    # Recent activity (from all user's involved issues)
    recent_activity = pd.DataFrame()
    if start is not None and end is not None:
        # Combine all user's issues (deduplicated)
        all_user_keys: set[str] = set()
        all_user_issues: list[pd.DataFrame] = []
        for user_df in [assigned_df, reported_df, watching_df]:
            if not user_df.empty and "key" in user_df.columns:
                new_issues = user_df[~user_df["key"].isin(all_user_keys)]
                if not new_issues.empty:
                    all_user_keys.update(new_issues["key"].tolist())
                    all_user_issues.append(new_issues)
        if all_user_issues:
            combined_df = pd.concat(all_user_issues, ignore_index=True)
            recent_activity = pv.extract_recent_activity(combined_df, start, end, tz)

    # Calculate summary metrics
    total_assigned_open = 0
    total_reported_open = 0
    if not assigned_df.empty:
        is_open = ~assigned_df["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)
        total_assigned_open = int(is_open.sum())
    if not reported_df.empty:
        is_open = ~reported_df["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)
        total_reported_open = int(is_open.sum())

    total_watching = len(watching_df)
    total_needs_attention = len(needs_attn_df)

    # Calculate total time lost for assigned issues
    total_time_lost = 0.0
    if not assigned_df.empty and "time_lost" in assigned_df.columns:
        total_time_lost = float(pd.to_numeric(assigned_df["time_lost"], errors="coerce").fillna(0).sum())

    # Status distribution for assigned issues (open only)
    status_distribution: dict[str, int] = {}
    if not assigned_open.empty:
        status_counts = assigned_open["status"].value_counts()
        status_distribution = status_counts.to_dict()

    # Priority distribution for assigned issues (open only)
    priority_distribution: dict[str, int] = {}
    if not assigned_open.empty and "priority" in assigned_open.columns:
        priority_counts = assigned_open["priority"].fillna("Undefined").value_counts()
        priority_distribution = priority_counts.to_dict()

    # Age buckets for assigned issues (open only)
    age_buckets: dict[str, int] = {}
    if not assigned_open.empty and "days_open" in assigned_open.columns:
        days = assigned_open["days_open"].fillna(0)
        age_buckets = {
            "< 7 days": int((days < 7).sum()),
            "7-30 days": int(((days >= 7) & (days < 30)).sum()),
            "30-90 days": int(((days >= 30) & (days < 90)).sum()),
            "90-180 days": int(((days >= 90) & (days < 180)).sum()),
            "> 180 days": int((days >= 180).sum()),
        }

    return PersonalViewContext(
        user=user,
        assigned=assigned_df.head(top_n),
        reported=reported_df.head(top_n),
        watching=watching_df.head(top_n),
        mentioned=mentioned_df.head(top_n),
        commented=commented_df.head(top_n),
        needs_attention=needs_attn_df.head(top_n),
        stale=stale_df.head(top_n),
        blocked=blocked_df.head(top_n),
        ole_assigned=ole_assigned.head(top_n),
        ole_reported=ole_reported.head(top_n),
        ole_watching=ole_watching.head(top_n),
        recent_activity=recent_activity.head(top_n * 4),  # More activity records
        total_assigned_open=total_assigned_open,
        total_reported_open=total_reported_open,
        total_watching=total_watching,
        total_needs_attention=total_needs_attention,
        total_time_lost=total_time_lost,
        total_ole_comments=total_ole_comments,
        status_distribution=status_distribution,
        priority_distribution=priority_distribution,
        age_buckets=age_buckets,
    )
