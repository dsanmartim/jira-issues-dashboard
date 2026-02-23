"""Personal View page - user-specific issue triage and tracking.

This page provides a focused view of issues relevant to the current user,
including assigned, reported, watching, and mentioned issues.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from datetime import datetime, timedelta

import altair as alt
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from jira import JIRAError

from jira_app.app import register_page
from jira_app.core.config import (
    DEFAULT_PROJECT_KEYS,
    DEFAULT_TREND_PRIORITIES,
    STATUS_DISPLAY_ORDER,
    TIMEZONE,
)
from jira_app.core.service import ActivityWeights
from jira_app.core.status import DONE_STATUSES_LOWER, normalize_workflow_status
from jira_app.features.personal_view import build_personal_context
from jira_app.visual.charts import created_trend
from jira_app.visual.column_metadata import apply_column_metadata
from jira_app.visual.progress import ProgressReporter
from jira_app.visual.tables import add_ticket_link, prepare_ticket_table

logger = logging.getLogger(__name__)

TZ = pytz.timezone(TIMEZONE)
PAGE_KEY = "personal_view"

# Page-specific defaults
PERSONAL_VIEW_DATE_RANGE_DAYS = 90
PERSONAL_VIEW_STALE_DAYS = 30
PERSONAL_VIEW_RECENT_ACTIVITY_DAYS = 15

# Priority colors for charts
PRIORITY_COLORS = {
    "Blocker": "#d62728",
    "Critical": "#c0392b",
    "Urgent": "#ff7f0e",
    "High": "#f39c12",
    "Medium": "#f1c40f",
    "Low": "#27ae60",
    "Undefined": "#95a5a6",
}


def _get_unique_users(df: pd.DataFrame) -> list[str]:
    """Extract unique user names from assignee and reporter columns."""
    users: set[str] = set()
    if "assignee" in df.columns:
        assignees = df["assignee"].dropna().astype(str).unique()
        users.update(a for a in assignees if a and a.lower() not in {"unassigned", ""})
    if "reporter" in df.columns:
        reporters = df["reporter"].dropna().astype(str).unique()
        users.update(r for r in reporters if r and r.lower() not in {"unknown", ""})
    return sorted(users, key=str.lower)


def _render_issue_table(
    df: pd.DataFrame,
    server: str,
    title: str,
    empty_msg: str,
    max_rows: int = 25,
) -> None:
    """Render a section with an issue table."""
    st.subheader(title)
    if df.empty:
        st.info(empty_msg)
        return

    display_df = df.head(max_rows)
    prepared, display_cols, cfg = prepare_ticket_table(display_df, server)
    column_config = apply_column_metadata(display_cols, cfg)
    st.dataframe(
        prepared[display_cols],
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )
    if len(df) > max_rows:
        st.caption(f"Showing {max_rows} of {len(df)} issues.")


def _render_issue_table_by_status(
    df: pd.DataFrame,
    server: str,
    empty_msg: str,
    max_rows: int = 25,
) -> None:
    """Render an issue table split into status sub-tabs.

    Creates an "All" tab plus one tab per canonical status present in the data,
    following the same pattern as the Tickets by Status section in Activity Overview.
    """
    if df.empty:
        st.info(empty_msg)
        return

    # Normalize statuses to canonical names
    norm = df["status"].fillna("").astype(str).apply(normalize_workflow_status)
    df_norm = df.copy()
    df_norm["__norm_status"] = norm

    # Build tab list: All + present statuses in canonical order
    present = set(df_norm["__norm_status"].unique())
    tab_order = [s for s in STATUS_DISPLAY_ORDER if s in present]
    if "Unknown" in present and "Unknown" not in tab_order:
        tab_order.append("Unknown")

    status_tabs = st.tabs(["All"] + tab_order)

    for tab, status_name in zip(status_tabs, ["All"] + tab_order, strict=False):
        with tab:
            subset = df_norm if status_name == "All" else df_norm[df_norm["__norm_status"] == status_name]

            if subset.empty:
                st.info(f"No issues with status '{status_name}'.")
                continue

            display_df = subset.head(max_rows)
            prepared, display_cols, cfg = prepare_ticket_table(display_df, server)
            column_config = apply_column_metadata(display_cols, cfg)
            st.dataframe(
                prepared[display_cols],
                hide_index=True,
                width="stretch",
                column_config=column_config,
            )
            if len(subset) > max_rows:
                st.caption(f"Showing {max_rows} of {len(subset)} issues.")


# Canonical priority display order (high → low, plus catch-all)
PRIORITY_DISPLAY_ORDER: list[str] = DEFAULT_TREND_PRIORITIES + ["Urgent", "Undefined"]


def _render_table_by_priority(
    df: pd.DataFrame,
    server: str,
    empty_msg: str,
    max_rows: int = 25,
) -> None:
    """Render an issue table split into priority sub-tabs.

    Creates an "All" tab plus one tab per priority present in the data,
    ordered from highest to lowest severity.
    """
    if df.empty:
        st.info(empty_msg)
        return

    df_p = df.copy()
    df_p["__priority"] = df_p["priority"].fillna("Undefined").astype(str)

    present = set(df_p["__priority"].unique())
    tab_order = [p for p in PRIORITY_DISPLAY_ORDER if p in present]
    # Catch any priorities not in the canonical list
    extras = sorted(p for p in present if p not in PRIORITY_DISPLAY_ORDER)
    tab_order += extras

    prio_tabs = st.tabs(["All"] + tab_order)

    for tab, prio_name in zip(prio_tabs, ["All"] + tab_order, strict=False):
        with tab:
            subset = df_p if prio_name == "All" else df_p[df_p["__priority"] == prio_name]
            if subset.empty:
                st.info(f"No issues with priority '{prio_name}'.")
                continue
            display_df = subset.head(max_rows)
            prepared, display_cols, cfg = prepare_ticket_table(display_df, server)
            column_config = apply_column_metadata(display_cols, cfg)
            st.dataframe(
                prepared[display_cols],
                hide_index=True,
                width="stretch",
                column_config=column_config,
            )
            if len(subset) > max_rows:
                st.caption(f"Showing {max_rows} of {len(subset)} issues.")


def _render_issue_table_by_status_and_priority(
    df: pd.DataFrame,
    server: str,
    empty_msg: str,
    max_rows: int = 25,
) -> None:
    """Render an issue table with status sub-tabs, each containing priority sub-tabs.

    First layer: All + canonical statuses present in data.
    Second layer (inside each status tab): All + priorities present in that subset.
    """
    if df.empty:
        st.info(empty_msg)
        return

    # Normalize statuses
    norm = df["status"].fillna("").astype(str).apply(normalize_workflow_status)
    df_norm = df.copy()
    df_norm["__norm_status"] = norm

    present = set(df_norm["__norm_status"].unique())
    tab_order = [s for s in STATUS_DISPLAY_ORDER if s in present]
    if "Unknown" in present and "Unknown" not in tab_order:
        tab_order.append("Unknown")

    status_tabs = st.tabs(["All"] + tab_order)

    for tab, status_name in zip(status_tabs, ["All"] + tab_order, strict=False):
        with tab:
            subset = df_norm if status_name == "All" else df_norm[df_norm["__norm_status"] == status_name]
            if subset.empty:
                st.info(f"No issues with status '{status_name}'.")
                continue
            _render_table_by_priority(subset, server, empty_msg, max_rows)


def _render_summary_metrics(ctx) -> None:
    """Render the summary metrics section."""
    st.subheader("Summary")
    st.caption("Key metrics for your open issues across all involvement types.")
    cols = st.columns(5)

    # Open Assigned - show total assigned count in delta
    total_assigned = len(ctx.assigned) if hasattr(ctx, "assigned") and ctx.assigned is not None else 0
    assigned_delta = f"{total_assigned} total" if total_assigned > ctx.total_assigned_open else None
    cols[0].metric(
        "Open Assigned",
        ctx.total_assigned_open,
        delta=assigned_delta,
        delta_color="off",
        help="Number of open issues currently assigned to you.",
    )

    # Open Reported - show total reported count in delta
    total_reported = len(ctx.reported) if hasattr(ctx, "reported") and ctx.reported is not None else 0
    reported_delta = f"{total_reported} total" if total_reported > ctx.total_reported_open else None
    cols[1].metric(
        "Open Reported",
        ctx.total_reported_open,
        delta=reported_delta,
        delta_color="off",
        help="Number of open issues you reported.",
    )

    # Watching
    cols[2].metric(
        "Watching",
        ctx.total_watching,
        help="Number of issues in your watch list.",
    )

    # Needs Attention - highlight if there are items
    attention_delta = "action required" if ctx.total_needs_attention > 0 else None
    cols[3].metric(
        "Needs Attention",
        ctx.total_needs_attention,
        delta=attention_delta,
        delta_color="inverse" if ctx.total_needs_attention > 0 else "off",
        help="Issues that are blocked, stale, or high priority (Blocker/Critical).",
    )

    # Time Lost (OBS-specific)
    if ctx.total_time_lost > 0:
        cols[4].metric(
            "Total Time Lost",
            f"{ctx.total_time_lost:.1f}h",
            help="Sum of time lost (hours) across all your assigned issues.",
        )
    else:
        cols[4].metric(
            "Total Time Lost",
            "–",
            delta="no time lost recorded",
            delta_color="off",
            help="Sum of time lost (hours) across all your assigned issues.",
        )


def _render_status_distribution(ctx) -> None:
    """Render status distribution for assigned issues."""
    if not ctx.status_distribution:
        return
    st.subheader("Status Distribution (Assigned)")
    st.caption("Breakdown of your open assigned issues by workflow status.")
    # Sort by count descending
    sorted_dist = sorted(ctx.status_distribution.items(), key=lambda x: x[1], reverse=True)
    total = sum(count for _, count in sorted_dist)
    cols = st.columns(min(len(sorted_dist), 6))
    for i, (status, count) in enumerate(sorted_dist[:6]):
        pct = (count / total * 100) if total > 0 else 0
        cols[i].metric(
            status,
            count,
            delta=f"{pct:.0f}%",
            delta_color="off",
            help=f"{count} of {total} open assigned issues are in '{status}' status.",
        )


def _render_workload_charts(ctx, start, end) -> None:
    """Render workload visualization charts."""
    st.subheader("Workload Overview")
    st.caption("Visual breakdown of your open assigned issues by status, priority, age, and creation trend.")

    # Create two columns for charts
    col1, col2 = st.columns(2)

    # Status Distribution Chart
    with col1:
        if ctx.status_distribution:
            st.markdown("**Status Distribution**")
            st.caption("Current workflow status of your open assigned issues.")
            status_df = pd.DataFrame(
                list(ctx.status_distribution.items()),
                columns=["Status", "Count"],
            )
            chart = (
                alt.Chart(status_df)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color("Status:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["Status", "Count"],
                )
                .properties(height=250)
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.info("No status data available.")

    # Priority Distribution Chart
    with col2:
        if ctx.priority_distribution:
            st.markdown("**Priority Distribution**")
            st.caption("Issue count by priority level, from Blocker to Low.")
            priority_df = pd.DataFrame(
                list(ctx.priority_distribution.items()),
                columns=["Priority", "Count"],
            )
            # Sort by priority order
            priority_order = ["Blocker", "Critical", "Urgent", "High", "Medium", "Low", "Undefined"]
            priority_df["sort_order"] = priority_df["Priority"].apply(
                lambda x: priority_order.index(x) if x in priority_order else 99
            )
            priority_df = priority_df.sort_values("sort_order")

            # Build color scale
            colors = [PRIORITY_COLORS.get(p, "#3498db") for p in priority_df["Priority"]]

            chart = (
                alt.Chart(priority_df)
                .mark_bar()
                .encode(
                    x=alt.X("Priority:N", sort=priority_df["Priority"].tolist(), title=None),
                    y=alt.Y("Count:Q", title="Issues"),
                    color=alt.Color(
                        "Priority:N",
                        scale=alt.Scale(domain=priority_df["Priority"].tolist(), range=colors),
                        legend=None,
                    ),
                    tooltip=["Priority", "Count"],
                )
                .properties(height=250)
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.info("No priority data available.")

    # Age Distribution Chart
    col3, col4 = st.columns(2)

    with col3:
        if ctx.age_buckets and any(v > 0 for v in ctx.age_buckets.values()):
            st.markdown("**Age Distribution**")
            st.caption("How long your open assigned issues have been open.")
            age_df = pd.DataFrame(
                list(ctx.age_buckets.items()),
                columns=["Age Bucket", "Count"],
            )
            # Maintain order
            bucket_order = ["< 7 days", "7-30 days", "30-90 days", "90-180 days", "> 180 days"]
            age_df["sort_order"] = age_df["Age Bucket"].apply(
                lambda x: bucket_order.index(x) if x in bucket_order else 99
            )
            age_df = age_df.sort_values("sort_order")

            chart = (
                alt.Chart(age_df)
                .mark_bar(color="#3498db")
                .encode(
                    x=alt.X("Age Bucket:N", sort=age_df["Age Bucket"].tolist(), title=None),
                    y=alt.Y("Count:Q", title="Issues"),
                    tooltip=["Age Bucket", "Count"],
                )
                .properties(height=250)
            )
            st.altair_chart(chart, width="stretch")
        else:
            st.info("No age data available.")

    # Trend Chart (issues created over time from assigned issues)
    with col4:
        if not ctx.assigned.empty and start is not None and end is not None:
            st.markdown("**Created Issues Over Time (Assigned)**")
            st.caption("Issues assigned to you created within the selected date range.")
            trend_chart, _trend_events = created_trend(ctx.assigned, start, end)
            if trend_chart:
                st.altair_chart(trend_chart, width="stretch")
            else:
                st.info("No issues created in this period.")
        else:
            st.info("No trend data available.")


def _render_ole_comments(ctx, server: str) -> None:
    """Render OLE/LOVE API comments section."""
    total_ole = ctx.total_ole_comments
    if total_ole == 0:
        return

    st.subheader("OLE/LOVE Comments")
    st.caption(
        "Comments posted by the OLE/LOVE automated API (Rubin Jira API Access) "
        f"within the selected date range. {total_ole} comment(s) found across your issues."
    )

    ole_tabs = st.tabs(["Assigned", "Reported", "Watching"])

    with ole_tabs[0]:
        if not ctx.ole_assigned.empty:
            _render_ole_table_by_status(ctx.ole_assigned, server)
        else:
            st.info("No OLE comments on your assigned issues.")

    with ole_tabs[1]:
        if not ctx.ole_reported.empty:
            _render_ole_table_by_status(ctx.ole_reported, server)
        else:
            st.info("No OLE comments on issues you reported.")

    with ole_tabs[2]:
        if not ctx.ole_watching.empty:
            _render_ole_table_by_status(ctx.ole_watching, server)
        else:
            st.info("No OLE comments on issues you're watching.")


def _render_ole_table(df: pd.DataFrame, server: str, caption: str) -> None:
    """Render a table of issues with OLE comments."""
    if df.empty:
        st.info("No data to display.")
        return

    display_df = df.head(15).copy()

    # Format timestamp for display
    if "ole_last_comment" in display_df.columns:
        display_df["ole_last_comment"] = pd.to_datetime(
            display_df["ole_last_comment"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M")

    # Add ticket link using the standard helper
    linked, link_cfg = add_ticket_link(display_df, server)

    # Define column order: Ticket link, OLE-specific, then standard fields
    ole_cols = [
        "Ticket",
        "summary",
        "ole_comments_count",
        "ole_last_comment",
        "status",
        "priority",
        "assignee",
    ]
    display_cols = [c for c in ole_cols if c in linked.columns]

    if not display_cols:
        st.info("No data to display.")
        return

    column_config = apply_column_metadata(display_cols, link_cfg)
    st.dataframe(
        linked[display_cols],
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )
    st.caption(caption)


def _render_ole_table_by_status(
    df: pd.DataFrame,
    server: str,
    max_rows: int = 15,
) -> None:
    """Render OLE comment tables split by status sub-tabs."""
    if df.empty:
        st.info("No data to display.")
        return

    # Normalize statuses
    norm = df["status"].fillna("").astype(str).apply(normalize_workflow_status)
    df_norm = df.copy()
    df_norm["__norm_status"] = norm

    present = set(df_norm["__norm_status"].unique())
    tab_order = [s for s in STATUS_DISPLAY_ORDER if s in present]
    if "Unknown" in present and "Unknown" not in tab_order:
        tab_order.append("Unknown")

    status_tabs = st.tabs(["All"] + tab_order)

    for tab, status_name in zip(status_tabs, ["All"] + tab_order, strict=False):
        with tab:
            subset = df_norm if status_name == "All" else df_norm[df_norm["__norm_status"] == status_name]
            if subset.empty:
                st.info(f"No OLE comments on issues with status '{status_name}'.")
                continue
            _render_ole_table(subset, server, f"{len(subset)} issue(s)")


def _render_recent_activity(ctx, server: str, activity_days: int, full_range: bool = False) -> None:
    """Render recent activity section."""
    if ctx.recent_activity.empty:
        return

    st.subheader("Recent Activity")
    st.caption(
        "Activity on issues you're involved with (assigned, reported, or watching). "
        "Includes comments, status changes, and field updates."
    )

    # Filter by activity days unless showing full range
    activity_df = ctx.recent_activity.copy()
    if not full_range and "created" in activity_df.columns:
        cutoff = datetime.now(pytz.UTC) - timedelta(days=activity_days)
        activity_df["created_ts"] = pd.to_datetime(activity_df["created"], errors="coerce", utc=True)
        activity_df = activity_df[activity_df["created_ts"] >= cutoff]

    if activity_df.empty:
        st.info(f"No activity in the last {activity_days} days.")
        return

    st.markdown(
        f"Showing **{'all' if full_range else f'last {activity_days} days of'}** "
        f"activity ({len(activity_df)} events)"
    )

    # Activity type tabs
    activity_tabs = st.tabs(["All", "Comments", "Status Changes", "Field Changes"])

    with activity_tabs[0]:
        _render_activity_table(activity_df, server)

    with activity_tabs[1]:
        comments_df = activity_df[activity_df["activity_type"] == "Comment"]
        if comments_df.empty:
            st.info("No comments in this period.")
        else:
            _render_activity_table(comments_df, server)

    with activity_tabs[2]:
        status_df = activity_df[activity_df["activity_type"] == "Status Change"]
        if status_df.empty:
            st.info("No status changes in this period.")
        else:
            _render_activity_table(status_df, server)

    with activity_tabs[3]:
        field_df = activity_df[activity_df["activity_type"] == "Field Change"]
        if field_df.empty:
            st.info("No field changes in this period.")
        else:
            _render_activity_table(field_df, server)


def _render_activity_table(df: pd.DataFrame, server: str) -> None:
    """Render an activity table with linked ticket keys."""
    if df.empty:
        return

    display_df = df.head(50).copy()

    # Format timestamp for display
    if "created" in display_df.columns:
        display_df["created"] = pd.to_datetime(display_df["created"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    # Add ticket link using the standard helper
    linked, link_cfg = add_ticket_link(display_df, server)

    # Define column order
    activity_cols = ["Ticket", "summary", "activity_type", "author", "created", "details"]
    available_cols = [c for c in activity_cols if c in linked.columns]
    if not available_cols:
        return

    column_config = apply_column_metadata(available_cols, link_cfg)
    st.dataframe(
        linked[available_cols],
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )


_MAX_HOVER_TICKETS = 10
_MAX_SUMMARY_LEN = 60


def _agg_hover(group: pd.DataFrame) -> dict:
    """Aggregate hover info for a group of issues."""
    keys = group["key"].dropna().astype(str).tolist()
    summaries = group["summary"].fillna("").astype(str).tolist()

    # Build ticket list with titles
    lines: list[str] = []
    for key, summary in zip(keys[:_MAX_HOVER_TICKETS], summaries[:_MAX_HOVER_TICKETS], strict=False):
        title = summary[:_MAX_SUMMARY_LEN]
        if len(summary) > _MAX_SUMMARY_LEN:
            title += "…"
        lines.append(f"{key}: {title}")
    tickets_str = "<br>".join(lines)
    if len(keys) > _MAX_HOVER_TICKETS:
        tickets_str += f"<br>(+{len(keys) - _MAX_HOVER_TICKETS} more)"

    # Priority breakdown
    prio_counts = group["priority"].fillna("Undefined").value_counts()
    prio_str = ", ".join(f"{p} ({c})" for p, c in prio_counts.items())

    # Date range
    dates = pd.to_datetime(group["created"], errors="coerce").dropna()
    if not dates.empty:
        date_str = f"{dates.min().strftime('%Y-%m-%d')} → {dates.max().strftime('%Y-%m-%d')}"
    else:
        date_str = "N/A"

    return {
        "count": len(group),
        "tickets": tickets_str,
        "priorities": prio_str,
        "date_range": date_str,
    }


def _build_obs_sunburst(source: pd.DataFrame) -> go.Figure | None:
    """Build a Plotly sunburst figure with enriched hover at every level.

    Manually constructs the hierarchy so that system-level and subsystem-level
    nodes also have aggregated ticket details (no missing hover data).

    Returns None if there is insufficient data.
    """
    obs_cols = ["obs_system", "obs_subsystem", "obs_component"]
    extra = ["key", "summary", "priority", "created"]
    available = [c for c in obs_cols if c in source.columns]
    if not available:
        return None

    cols_needed = available + [c for c in extra if c in source.columns]
    hier = source[cols_needed].copy()
    for col in available:
        hier[col] = hier[col].fillna("").astype(str).str.strip()
        hier.loc[hier[col] == "", col] = "Unspecified"

    # Ensure extra columns exist for aggregation
    for col in extra:
        if col not in hier.columns:
            hier[col] = ""

    if hier.empty:
        return None

    # Build nodes for every level of the hierarchy
    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    hover_texts: list[str] = []

    hover_tmpl = (
        "<b>%s</b><br>"
        "Issues: %d<br>"
        "<b>Priorities:</b> %s<br>"
        "<b>Created:</b> %s<br>"
        "<br><b>Tickets:</b><br>%s"
        "<extra></extra>"
    )

    # Level 1: Systems
    if "obs_system" in available:
        for system, sys_group in hier.groupby("obs_system"):
            node_id = str(system)
            info = _agg_hover(sys_group)
            ids.append(node_id)
            labels.append(str(system))
            parents.append("")
            values.append(info["count"])
            hover_texts.append(
                hover_tmpl % (system, info["count"], info["priorities"], info["date_range"], info["tickets"])
            )

    # Level 2: Subsystems
    if "obs_subsystem" in available:
        for (system, subsystem), sub_group in hier.groupby(["obs_system", "obs_subsystem"]):
            node_id = f"{system}/{subsystem}"
            parent_id = str(system)
            info = _agg_hover(sub_group)
            ids.append(node_id)
            labels.append(str(subsystem))
            parents.append(parent_id)
            values.append(info["count"])
            hover_texts.append(
                hover_tmpl
                % (subsystem, info["count"], info["priorities"], info["date_range"], info["tickets"])
            )

    # Level 3: Components
    if "obs_component" in available:
        for (system, subsystem, component), comp_group in hier.groupby(
            ["obs_system", "obs_subsystem", "obs_component"]
        ):
            node_id = f"{system}/{subsystem}/{component}"
            parent_id = f"{system}/{subsystem}"
            info = _agg_hover(comp_group)
            ids.append(node_id)
            labels.append(str(component))
            parents.append(parent_id)
            values.append(info["count"])
            hover_texts.append(
                hover_tmpl
                % (component, info["count"], info["priorities"], info["date_range"], info["tickets"])
            )

    if not ids:
        return None

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate=hover_texts,
        )
    )
    fig.update_layout(
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        height=500,
        hoverlabel={"align": "left"},
    )
    return fig


def _render_obs_distribution(ctx) -> None:
    """Render OBS hierarchy distribution for the user's assigned issues.

    Only meaningful for the OBS project which has custom OBS hierarchy fields.
    """
    assigned = ctx.assigned
    if assigned.empty:
        return

    obs_cols = ["obs_system", "obs_subsystem", "obs_component"]
    if not any(c in assigned.columns for c in obs_cols):
        return

    st.subheader("OBS Distribution (Assigned)")
    st.caption(
        "Assigned issues by OBS hierarchy (System → Subsystem → Component). " "Hover for ticket details."
    )

    # Toggle between open issues and all issues
    scope = st.radio(
        "Show",
        options=["All issues", "Open issues"],
        horizontal=True,
        key=f"{PAGE_KEY}_obs_scope",
        help="Open issues = excludes Done, Cancelled, Duplicate, Transferred.",
    )

    if scope == "Open issues":
        source = assigned[~assigned["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)]
    else:
        source = assigned

    if source.empty:
        st.info("No issues to display for the selected scope.")
        return

    fig = _build_obs_sunburst(source)
    if fig is None:
        st.info("No issues with OBS hierarchy data for the selected scope.")
        return

    st.plotly_chart(fig, width="stretch")


@register_page("Personal View")
def render():  # noqa: D401 - Streamlit entrypoint
    """Render the Personal View page."""

    st.title("Personal View")
    st.caption(
        "A focused view of your Jira issues. Track what you're working on, "
        "what needs attention, and stay on top of your workload."
    )

    issue_service = st.session_state.get("issue_service")
    if issue_service is None:
        st.warning("Initialize the Jira connection on the Setup page to use this view.")
        return

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Personal View Settings")

        # Project selection
        project_options = list(DEFAULT_PROJECT_KEYS)
        project_key = f"{PAGE_KEY}_project"
        default_project = st.session_state.get(project_key, project_options[0])
        project_idx = project_options.index(default_project) if default_project in project_options else 0

        selected_project = st.selectbox(
            "Project",
            options=project_options,
            index=project_idx,
            key=project_key,
            help="Select the Jira project to view.",
        )

        # Allow custom project input
        custom_project = st.text_input(
            "Or enter custom project key",
            value="",
            key=f"{PAGE_KEY}_custom_project",
            help="Enter a custom project key if not in the list above.",
        )
        if custom_project.strip():
            selected_project = custom_project.strip().upper()

        # Date range
        st.markdown("---")
        st.subheader("Date Range")
        now_tz = datetime.now(TZ)
        start_key = f"{PAGE_KEY}_start_date"
        end_key = f"{PAGE_KEY}_end_date"
        default_start = st.session_state.get(start_key)
        if default_start is None:
            default_start = (now_tz - timedelta(days=PERSONAL_VIEW_DATE_RANGE_DAYS)).date()
        default_end = st.session_state.get(end_key) or now_tz.date()

        start_date = st.date_input("Start date", value=default_start, key=start_key)
        end_date = st.date_input("End date", value=default_end, key=end_key)

        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        # Stale threshold
        st.markdown("---")
        stale_days = st.number_input(
            "Stale threshold (days)",
            min_value=15,
            max_value=180,
            value=PERSONAL_VIEW_STALE_DAYS,
            step=15,
            key=f"{PAGE_KEY}_stale_days",
            help="Issues not updated within this many days are considered stale.",
        )

    # --- Main Content ---
    data_key = f"{PAGE_KEY}_data"
    weights = st.session_state.get("dashboard_weights") or ActivityWeights()

    # Fetch button
    if st.button("Fetch My Issues", type="primary", key=f"{PAGE_KEY}_fetch"):
        reporter = ProgressReporter("Loading your issues")
        try:
            start_dt = TZ.localize(datetime.combine(start_date, datetime.min.time()))
            end_dt = TZ.localize(datetime.combine(end_date, datetime.max.time()))

            reporter.update(f"Requesting issues from {selected_project}")
            df = issue_service.fetch_and_enrich_range(
                selected_project,
                start_dt,
                end_dt,
                weights=weights,
                progress=reporter.callback,
            )

            # Get current user from API
            current_user = None
            if hasattr(issue_service, "api") and issue_service.api is not None:
                with suppress(Exception):
                    current_user = issue_service.api.get_current_user()

            st.session_state[data_key] = {
                "df": df,
                "project": selected_project,
                "start": start_dt,
                "end": end_dt,
                "current_user": current_user,
            }
            reporter.complete(f"Loaded {len(df)} issues from {selected_project}.")
        except JIRAError as exc:
            error_msg = exc.text if hasattr(exc, "text") else str(exc)
            logger.error("Jira API error: %s", error_msg)
            reporter.error(f"Failed to fetch issues: {error_msg}")
            return
        except (ValueError, KeyError) as exc:
            logger.error("Data processing error: %s", exc)
            reporter.error(f"Failed to process issues: {exc}")
            return

    # Check for loaded data
    data_state = st.session_state.get(data_key)
    if not data_state:
        st.info("Click 'Fetch My Issues' to load your issues.")
        return

    df = data_state.get("df")
    if df is None or df.empty:
        st.info("No issues found for the selected criteria.")
        return

    # User selection
    st.markdown("---")
    current_user = data_state.get("current_user")
    available_users = _get_unique_users(df)

    if not available_users:
        st.warning("No users found in the loaded data.")
        return

    # Determine default user - prioritize current API user
    user_key = f"{PAGE_KEY}_selected_user"
    if user_key not in st.session_state:
        # First time: default to current user from API if available
        if current_user and current_user in available_users:
            default_user_idx = available_users.index(current_user)
        else:
            default_user_idx = 0
    else:
        # Preserve user's previous selection if valid
        prev_user = st.session_state.get(user_key)
        if prev_user and prev_user in available_users:
            default_user_idx = available_users.index(prev_user)
        elif current_user and current_user in available_users:
            default_user_idx = available_users.index(current_user)
        else:
            default_user_idx = 0

    selected_user = st.selectbox(
        "Select User",
        options=available_users,
        index=default_user_idx,
        key=user_key,
        help="Select a user to view their personal dashboard. Auto-detected from API credentials.",
    )

    if current_user and selected_user == current_user:
        st.caption(f"Showing issues for: **{selected_user}** (you)")
    else:
        st.caption(f"Showing issues for: **{selected_user}**")

    # Get date range from data state
    start_dt = data_state.get("start")
    end_dt = data_state.get("end")

    # Build context with date range for OLE and activity
    ctx = build_personal_context(
        df,
        selected_user,
        stale_days=int(stale_days),
        top_n=50,
        start=start_dt,
        end=end_dt,
        tz=TZ,
    )

    server = st.session_state.get("jira_server", "")

    # --- Summary Section ---
    st.markdown("---")
    _render_summary_metrics(ctx)
    _render_status_distribution(ctx)

    # --- Needs Attention Section ---
    if not ctx.needs_attention.empty:
        st.markdown("---")
        st.subheader("Needs Attention")
        st.caption("High priority, blocked, or stale issues assigned to you.")
        _render_issue_table_by_status_and_priority(
            ctx.needs_attention,
            server,
            "No issues need immediate attention.",
        )

    # --- Tabs for different views ---
    st.markdown("---")
    st.subheader("My Issues")
    st.caption(
        "Issues where you are involved, organized by your role. "
        "Assigned = you own it; Reported = you created it; "
        "Watching = you subscribed to updates; Mentioned = your name appears in comments; "
        "My Comments = issues you have commented on."
    )

    # Filter control
    hide_done = st.checkbox(
        "Hide Done/Closed issues",
        value=True,
        key=f"{PAGE_KEY}_hide_done",
        help="When checked, only shows open issues (excludes Done, Cancelled, Duplicate, Transferred).",
    )

    def _filter_open(issue_df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only open issues if hide_done is checked."""
        if not hide_done or issue_df.empty:
            return issue_df
        if "status" not in issue_df.columns:
            return issue_df
        is_open = ~issue_df["status"].astype(str).str.lower().isin(DONE_STATUSES_LOWER)
        return issue_df[is_open]

    tabs = st.tabs(["Assigned", "Reported", "Watching", "Mentioned", "My Comments"])

    with tabs[0]:
        _render_issue_table_by_status(
            _filter_open(ctx.assigned),
            server,
            "No open issues assigned to you." if hide_done else "No issues assigned to you.",
        )

    with tabs[1]:
        _render_issue_table_by_status(
            _filter_open(ctx.reported),
            server,
            "No open issues reported by you." if hide_done else "No issues reported by you.",
        )

    with tabs[2]:
        _render_issue_table_by_status(
            _filter_open(ctx.watching),
            server,
            "No open issues in your watch list." if hide_done else "No issues in your watch list.",
        )

    with tabs[3]:
        _render_issue_table_by_status(
            _filter_open(ctx.mentioned),
            server,
            "No open mentions of you." if hide_done else "No issues with mentions of you.",
        )

    with tabs[4]:
        _render_issue_table_by_status(
            _filter_open(ctx.commented),
            server,
            "No open issues with your comments." if hide_done else "No issues with your comments.",
        )

    # --- Stale Issues Section ---
    if not ctx.stale.empty:
        st.markdown("---")
        st.subheader(f"Stale Issues (No updates in {stale_days}+ days)")
        st.caption(
            "Open issues assigned to you with no activity (comments, status changes, or edits) "
            "within the stale threshold configured in the sidebar."
        )
        _render_issue_table_by_status(
            ctx.stale,
            server,
            "No stale issues found.",
        )

    # --- Blocked Issues Section ---
    if not ctx.blocked.empty:
        st.markdown("---")
        st.subheader("Blocked Issues")
        st.caption(
            "Open issues assigned to you that are currently in 'Blocked' status "
            "and may need action to unblock."
        )
        _render_issue_table(
            ctx.blocked,
            server,
            "",
            "No blocked issues.",
            max_rows=10,
        )

    # --- Workload Charts Section ---
    if ctx.status_distribution or ctx.priority_distribution or ctx.age_buckets:
        st.markdown("---")
        _render_workload_charts(ctx, start_dt, end_dt)

    # --- OLE Comments Section ---
    if ctx.total_ole_comments > 0:
        st.markdown("---")
        _render_ole_comments(ctx, server)

    # --- Recent Activity Section ---
    if not ctx.recent_activity.empty:
        st.markdown("---")
        # Activity timeframe control
        activity_col1, activity_col2 = st.columns([3, 1])
        with activity_col1:
            activity_days = st.slider(
                "Activity timeframe (days)",
                min_value=7,
                max_value=60,
                value=PERSONAL_VIEW_RECENT_ACTIVITY_DAYS,
                step=7,
                key=f"{PAGE_KEY}_activity_days",
                help="Show activity from the last N days.",
            )
        with activity_col2:
            show_full_range = st.checkbox(
                "Show full range",
                value=False,
                key=f"{PAGE_KEY}_show_full_range",
                help="Show all activity in the loaded date range.",
            )
        _render_recent_activity(ctx, server, activity_days, full_range=show_full_range)

    # --- OBS Distribution Section (OBS project only) ---
    project = data_state.get("project", "")
    if project.upper() == "OBS" and not ctx.assigned.empty:
        st.markdown("---")
        _render_obs_distribution(ctx)
