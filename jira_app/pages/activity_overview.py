"""Activity Overview page."""

from collections import defaultdict
from datetime import date, datetime, timedelta

import altair as alt
import pandas as pd
import pytz
import streamlit as st

from jira_app.analytics.aggregations.assignee import aggregate_by_assignee
from jira_app.analytics.aggregations.obs import (
    aggregate_by_obs_component,
    aggregate_by_obs_subsystem,
    aggregate_by_obs_system,
)
from jira_app.analytics.metrics.activity import normalize_timestamp
from jira_app.analytics.metrics.status_flow import build_status_duration_frame
from jira_app.app import register_page
from jira_app.core.config import (
    API_COMMENT_AUTHORS,
    DEFAULT_DATE_RANGE_DAYS,
    DEFAULT_TREND_PRIORITIES,
    DEFAULT_WARN_AGE_DAYS,
    STATUS_DISPLAY_ORDER,
    TERMINAL_STATUSES,
    TIMEZONE,
    normalize_priority_name,
)
from jira_app.core.service import ActivityWeights
from jira_app.core.status import (
    normalize_workflow_status,
)
from jira_app.features.activity_overview.context import build_context
from jira_app.visual.charts import create_bar_chart, created_trend, priority_updates_trend
from jira_app.visual.leaderboard import display_leaderboard, display_ticket_list
from jira_app.visual.progress import ProgressReporter
from jira_app.visual.tables import prepare_ticket_table
from jira_app.visual.wordcloud import WORDCLOUD_AVAILABLE, wordcloud_png

CRITICAL_PREFIXES = ("Blocker", "Critical")


def _build_ole_obs_heatmaps(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    *,
    top_n: int = 15,
) -> list[tuple[str, alt.Chart | None, str]]:
    if df is None or df.empty or "comments" not in df.columns:
        return []

    if start_dt is None or end_dt is None:
        return []

    start_ts = pd.to_datetime(start_dt, errors="coerce")
    end_ts = pd.to_datetime(end_dt, errors="coerce")
    if start_ts is None or pd.isna(start_ts) or end_ts is None or pd.isna(end_ts):
        return []

    if getattr(start_ts, "tzinfo", None) is None:
        start_ts = start_ts.tz_localize(pytz.UTC)
    if getattr(end_ts, "tzinfo", None) is None:
        end_ts = end_ts.tz_localize(pytz.UTC)

    target_tz = start_ts.tz

    levels: list[tuple[str, str]] = [
        ("obs_system", "OBS System"),
        ("obs_subsystem", "OBS Subsystem"),
        ("obs_component", "OBS Component"),
    ]

    level_counts: dict[str, defaultdict[tuple[str, datetime], dict[str, object]]] = {
        label: defaultdict(
            lambda: {
                "count": 0,
                "ticket_counts": defaultdict(int),  # label-based (for display)
                "key_counts": defaultdict(int),  # key-based (for breakdown)
            }
        )
        for _, label in levels
    }
    allowed_authors = API_COMMENT_AUTHORS

    def _escape_html(text: str) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for _, row in df.iterrows():
        comments = row.get("comments")
        if not isinstance(comments, list) or not comments:
            continue

        group_values = {}
        for col, label in levels:
            value = str(row.get(col) or "").strip()
            if value:
                group_values[label] = value
            else:
                group_values[label] = "(Unspecified)"
        if not group_values:
            continue

        ticket_key = str(row.get("key") or "").strip()
        ticket_summary = str(row.get("summary") or "").strip()
        ticket_priority = str(row.get("priority") or "").strip()
        # Build per-ticket label with emoji for Blocker/Critical and plain text otherwise
        pr_display = ""
        if ticket_priority:
            if ticket_priority.startswith("Blocker"):
                pr_display = "⛔ Blocker"
            elif ticket_priority.startswith("Critical"):
                pr_display = "⚠️ Critical"
            else:
                pr_display = ticket_priority
        ticket_label = ""
        if ticket_key and ticket_summary:
            ticket_label = f"{ticket_key}: {ticket_summary}{f' [{pr_display}]' if pr_display else ''}"
        elif ticket_key:
            ticket_label = f"{ticket_key}{f' [{pr_display}]' if pr_display else ''}"
        elif ticket_summary:
            ticket_label = f"{ticket_summary}{f' [{pr_display}]' if pr_display else ''}"

        for comment in comments:
            if not isinstance(comment, dict):
                continue
            author_name = str(comment.get("author") or "").strip()
            if author_name not in allowed_authors:
                continue
            ts = normalize_timestamp(comment.get("created"), target_tz)
            if ts is None or ts < start_ts or ts > end_ts:
                continue
            day = ts.date()
            for label, value in group_values.items():
                bucket = level_counts[label][(value, day)]
                bucket["count"] += 1
                if ticket_label:
                    bucket["ticket_counts"][ticket_label] += 1
                if ticket_key:
                    bucket["key_counts"][ticket_key] += 1
                # Flag if any ticket in this day/group is Blocker/Critical
                if ticket_priority:
                    if ticket_priority.startswith("Blocker"):
                        bucket["has_blocker"] = True
                    if ticket_priority.startswith("Critical"):
                        bucket["has_critical"] = True

    results: list[tuple[str, alt.Chart | None, str]] = []
    for _col, label in levels:
        counter = level_counts[label]
        if not counter:
            results.append((label, None, "No OLE comments captured for this level in the window."))
            continue
        # Determine maximum number of ticket lines needed for clean tooltips
        max_lines = 1
        for (_g, _d), _bucket in counter.items():
            _lc = _bucket.get("ticket_counts", {})
            _lines = len(_lc) if _lc else 1
            if _lines > max_lines:
                max_lines = _lines

        records = []
        for (group, day), bucket in counter.items():
            label_counts = bucket.get("ticket_counts", {})
            key_counts = bucket.get("key_counts", {})
            # Build breakdown string for OLE Comments line: total [KEY -> n, ...]
            if key_counts:
                breakdown_items = sorted(
                    ((str(k), int(v or 0)) for k, v in key_counts.items()),
                    key=lambda x: (-x[1], x[0]),
                )
                breakdown_str = ", ".join([f"{k} → {v}" for k, v in breakdown_items])
                count_breakdown = f"{int(bucket['count'])} [{breakdown_str}]"
            else:
                count_breakdown = str(int(bucket["count"]))

            # Build ticket lines (no counts), create t1..tN columns up to max_lines
            t_fields: dict[str, object] = {}
            if label_counts:
                # Order by number of comments (desc), then by label asc
                sorted_labels = [
                    lbl
                    for lbl, _cnt in sorted(
                        ((lbl, int(cnt or 0)) for lbl, cnt in label_counts.items()),
                        key=lambda x: (-x[1], x[0]),
                    )
                ]
                for idx in range(max_lines):
                    field_name = f"t{idx+1}"
                    if idx < len(sorted_labels):
                        t_fields[field_name] = f"\u2022 {_escape_html(sorted_labels[idx])}"
                    else:
                        t_fields[field_name] = ""
            else:
                # No ticket details captured
                for i in range(max_lines):
                    field_name = f"t{i+1}"
                    t_fields[field_name] = "\u2022 No ticket details captured" if i == 0 else ""
            records.append(
                {
                    "group": group,
                    "date": day,
                    "count": bucket["count"],
                    "count_breakdown": count_breakdown,
                    **t_fields,
                    "has_blocker": bool(bucket.get("has_blocker", False)),
                    "has_critical": bool(bucket.get("has_critical", False)),
                }
            )
        level_df = pd.DataFrame(records)
        if level_df.empty:
            results.append((label, None, "No OLE comments captured for this level in the window."))
            continue
        level_df["date"] = pd.to_datetime(level_df["date"])
        group_totals = level_df.groupby("group")[["count"]].sum().sort_values("count", ascending=False)
        top_groups = group_totals.head(max(top_n, 1)).index.tolist()
        level_df = level_df[level_df["group"].isin(top_groups)]
        if level_df.empty:
            results.append((label, None, "No OLE comments captured for this level in the window."))
            continue

        sort_order = group_totals.loc[top_groups].index.tolist()
        height = max(200, 30 * len(sort_order))
        # Base heatmap: no borders at all
        base = (
            alt.Chart(level_df)
            .mark_rect()
            .encode(
                x=alt.X("yearmonthdate(date):O", title="Date", sort="ascending"),
                y=alt.Y("group:N", sort=sort_order, title=label),
                color=alt.Color(
                    "count:Q",
                    title="OLE comments",
                    scale=alt.Scale(scheme="blues"),
                    legend=alt.Legend(
                        gradientLength=int(height * 0.95),
                        orient="right",
                        titleOrient="right",
                    ),
                ),
                tooltip=(
                    [
                        alt.Tooltip("date:T", title="Date:"),
                        alt.Tooltip("group:N", title=f"{label}:"),
                        alt.Tooltip("count_breakdown:N", title="OLE Comments:"),
                    ]
                    + [
                        alt.Tooltip(f"t{idx}:N", title=("Tickets:" if idx == 1 else "\u200B"))
                        for idx in range(1, max_lines + 1)
                    ]
                ),
            )
            .properties(height=height)
        )

        # Overlay borders only for Blocker/Critical cells;
        # use transparent fill so there's no overlay square
        overlay = (
            alt.Chart(level_df)
            .transform_filter("datum.has_blocker || datum.has_critical")
            .mark_rect(fillOpacity=0)
            .encode(
                x=alt.X("yearmonthdate(date):O", title="Date", sort="ascending"),
                y=alt.Y("group:N", sort=sort_order, title=label),
                stroke=alt.condition(
                    "datum.has_blocker",
                    alt.value("red"),
                    alt.value("orange"),
                ),
                strokeWidth=alt.value(1),
                tooltip=(
                    [
                        alt.Tooltip("date:T", title="Date:"),
                        alt.Tooltip("group:N", title=f"{label}:"),
                        alt.Tooltip("count_breakdown:N", title="OLE Comments:"),
                    ]
                    + [
                        alt.Tooltip(f"t{idx}:N", title=("Tickets:" if idx == 1 else "\u200B"))
                        for idx in range(1, max_lines + 1)
                    ]
                ),
            )
        )

        chart = base + overlay
        caption = (
            f"Top {len(sort_order)} {label.lower()} grouped by day of OLE comments "
            "within the selected window."
        )
        results.append((f"OLE Comments by {label}", chart, caption))

    return results


def _prepare_commonly_reported_faults(df: pd.DataFrame, obs_column: str, tz) -> pd.DataFrame:
    if df is None or df.empty or obs_column not in df.columns:
        return pd.DataFrame()

    working = df.copy()
    obs_series = working[obs_column].fillna("(Unspecified)").astype(str)
    obs_series = obs_series.replace("", "(Unspecified)")
    working["__obs_label"] = obs_series

    status_series = working.get("status")
    if status_series is not None:
        normalized_status = status_series.fillna("").astype(str).apply(normalize_workflow_status)
    else:
        normalized_status = pd.Series("Unknown", index=working.index, dtype=str)
    working["__is_open"] = ~normalized_status.isin(TERMINAL_STATUSES)

    priority_series = working.get("priority")
    if priority_series is not None:
        priority_text = priority_series.fillna("").astype(str)
        working["__is_critical"] = priority_text.str.startswith(CRITICAL_PREFIXES, na=False)
    else:
        working["__is_critical"] = False

    days_open_series = pd.to_numeric(working.get("days_open"), errors="coerce")
    working["__open_days"] = days_open_series.where(working["__is_open"])

    last_seen_series = None
    for candidate in ("updated_dt", "updated", "created_dt", "created"):
        if candidate in working.columns:
            try:
                series = pd.to_datetime(working[candidate], errors="coerce", utc=True)
            except Exception:
                series = pd.to_datetime(working[candidate], errors="coerce")
            if series is None:
                continue
            if last_seen_series is None:
                last_seen_series = series
            if series.notna().any():
                last_seen_series = series
                break
    if last_seen_series is None:
        last_seen_series = pd.Series(pd.NaT, index=working.index)
    else:
        if getattr(last_seen_series.dt, "tz", None) is None:
            try:
                last_seen_series = last_seen_series.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
            except (TypeError, ValueError):
                last_seen_series = pd.to_datetime(last_seen_series, errors="coerce", utc=True)
        last_seen_series = last_seen_series.dt.tz_convert(tz)
    working["__last_seen"] = last_seen_series

    grouped = (
        working.groupby("__obs_label", dropna=False)
        .agg(
            reported_occurrences=("key", "nunique"),
            open_tickets=("__is_open", "sum"),
            critical_count=("__is_critical", "sum"),
            last_seen=("__last_seen", "max"),
            avg_open_days=("__open_days", "mean"),
        )
        .reset_index()
    )

    if grouped.empty:
        return grouped

    grouped["critical_pct"] = (
        grouped["critical_count"]
        .divide(grouped["reported_occurrences"].replace(0, pd.NA))
        .multiply(100)
        .fillna(0.0)
    )
    grouped = grouped.drop(columns=["critical_count"])
    grouped = grouped.sort_values(["reported_occurrences", "last_seen"], ascending=[False, False])
    grouped = grouped.rename(columns={"__obs_label": obs_column}).reset_index(drop=True)
    return grouped


def render_wordcloud(text: str) -> None:
    if not WORDCLOUD_AVAILABLE:
        st.warning("Word cloud library not installed. Install 'wordcloud' to enable this chart.")
        return

    image_bytes = wordcloud_png(text)
    if image_bytes:
        st.image(image_bytes, width="stretch")
    else:
        st.warning("Text input was too small to build a word cloud.")


@register_page("Activity Overview")
def render():
    st.title("Activity Overview")
    st.caption("Consolidated activity, aging, trends, and top-N views for the OBS project.")

    if "issue_service" not in st.session_state:
        st.warning("Jira connection not configured. Please go to the 'Setup / Connection' page.")
        return

    issue_service = st.session_state.issue_service
    # Import project key from config to avoid hard-coding
    from jira_app.core.config import DEFAULT_PROJECT_KEY

    project_key = DEFAULT_PROJECT_KEY

    if "start_date_str" not in st.session_state:
        st.session_state.start_date_str = (
            (datetime.now(pytz.utc) - timedelta(days=DEFAULT_DATE_RANGE_DAYS)).date().isoformat()
        )
    if "end_date_str" not in st.session_state:
        st.session_state.end_date_str = datetime.now(pytz.utc).date().isoformat()
    if "top_n" not in st.session_state:
        st.session_state.top_n = 15
    if "warn_age" not in st.session_state:
        st.session_state.warn_age = DEFAULT_WARN_AGE_DAYS
    if "weight_comment" not in st.session_state:
        st.session_state.weight_comment = 2.0
    if "weight_status" not in st.session_state:
        st.session_state.weight_status = 0.50
    if "weight_other" not in st.session_state:
        st.session_state.weight_other = 0.25

    with st.sidebar:
        st.header("Date Range")
        st.session_state.start_date_str = st.date_input(
            "Start Date", date.fromisoformat(st.session_state.start_date_str)
        ).isoformat()
        st.session_state.end_date_str = st.date_input(
            "End Date", date.fromisoformat(st.session_state.end_date_str)
        ).isoformat()
        st.session_state.top_n = st.number_input(
            "Top N rows",
            min_value=5,
            max_value=100,
            value=st.session_state.top_n,
            step=5,
        )

        existing_df = st.session_state.get("enriched_df")
        if isinstance(existing_df, pd.DataFrame) and not existing_df.empty:
            # Normalize priorities (e.g., "Medium (migrated)" -> "Medium")
            raw_priorities = existing_df.get("priority").dropna().astype(str).tolist()
            normalized = sorted({normalize_priority_name(p) for p in raw_priorities})
            # Order by DEFAULT_TREND_PRIORITIES first, then any extras
            available_priorities = [p for p in DEFAULT_TREND_PRIORITIES if p in normalized]
            available_priorities += [p for p in normalized if p not in available_priorities]
        else:
            available_priorities = list(DEFAULT_TREND_PRIORITIES)

        st.session_state.warn_age = st.number_input(
            "Warn Critical/Blocker age (days)",
            min_value=1,
            max_value=90,
            value=st.session_state.warn_age,
        )

        with st.expander("Activity weighting", expanded=False):
            st.caption("Weights used when computing the weighted activity score.")
            st.session_state.weight_comment = st.number_input(
                "Weight: comment",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.weight_comment),
                step=0.1,
            )
            st.session_state.weight_status = st.number_input(
                "Weight: status change",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.weight_status),
                step=0.1,
            )
            st.session_state.weight_other = st.number_input(
                "Weight: other change",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.weight_other),
                step=0.1,
            )

        fetch_button = st.button("Fetch & Analyze", type="primary")
        refresh_button = st.button("Refresh Analysis")

    tz = pytz.timezone(TIMEZONE)
    now_ts = datetime.now(tz)
    start_date = date.fromisoformat(st.session_state.start_date_str)
    end_date = date.fromisoformat(st.session_state.end_date_str)
    # Persist parsed date objects for other components (drilldowns)
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date
    start_dt = tz.localize(datetime.combine(start_date, datetime.min.time()))
    end_dt = tz.localize(datetime.combine(end_date, datetime.max.time()))
    weights = ActivityWeights(
        comment=float(st.session_state.weight_comment),
        status=float(st.session_state.weight_status),
        other=float(st.session_state.weight_other),
    )

    if fetch_button:
        reporter = ProgressReporter(f"Fetching issues for project '{project_key}'")
        try:
            issues_df = issue_service.fetch_updated_between(
                project_key,
                start_dt,
                end_dt,
                progress=reporter.callback,
            )
            reporter.update("Enriching activity metrics")
            enriched_df = issue_service.enrich(issues_df, start_dt, end_dt, weights=weights)
            st.session_state.issues_df = issues_df
            st.session_state.enriched_df = enriched_df
            st.session_state.dashboard_weights = weights
            st.session_state.dashboard_start_dt = start_dt
            st.session_state.dashboard_end_dt = end_dt
            reporter.complete(f"Fetch and analysis complete ({len(issues_df)} issues).")
        except Exception as exc:  # pragma: no cover - surface errors to UI
            reporter.error(f"Fetch failed: {exc}")
            raise

    if refresh_button and "issues_df" in st.session_state:
        reporter = ProgressReporter("Refreshing analysis")
        try:
            reporter.update("Recomputing activity metrics")
            st.session_state.enriched_df = issue_service.enrich(
                st.session_state.issues_df, start_dt, end_dt, weights=weights
            )
            st.session_state.dashboard_weights = weights
            st.session_state.dashboard_start_dt = start_dt
            st.session_state.dashboard_end_dt = end_dt
            reporter.complete("Refresh complete.")
        except Exception as exc:  # pragma: no cover
            reporter.error(f"Refresh failed: {exc}")
            raise

    if (
        "issues_df" in st.session_state
        and isinstance(st.session_state.issues_df, pd.DataFrame)
        and not st.session_state.issues_df.empty
        and "enriched_df" in st.session_state
        and st.session_state.get("dashboard_weights") != weights
    ):
        reporter = ProgressReporter("Updating activity scores for new weights")
        try:
            reporter.update("Recomputing enriched metrics")
            st.session_state.enriched_df = issue_service.enrich(
                st.session_state.issues_df, start_dt, end_dt, weights=weights
            )
            reporter.complete("Activity scores updated.")
        except Exception as exc:  # pragma: no cover
            reporter.error(f"Failed to update activity scores: {exc}")
            raise
        st.session_state.dashboard_weights = weights
        st.session_state.dashboard_start_dt = start_dt
        st.session_state.dashboard_end_dt = end_dt

    if "enriched_df" not in st.session_state or st.session_state.enriched_df.empty:
        st.info("No data yet. Use Fetch & Analyze.")
        return

    df = st.session_state.enriched_df
    if df.empty:
        st.info("No issues returned for the selected period.")
        return
    server_url = getattr(
        getattr(issue_service, "api", None),
        "server",
        st.session_state.get("jira_server", ""),
    )

    context = build_context(
        df,
        start_dt,
        end_dt,
        top_n=st.session_state.top_n,
        aging_warn_days=st.session_state.warn_age,
    )
    st.session_state.dashboard_context = context

    assignee_table = aggregate_by_assignee(df, limit=st.session_state.top_n)
    obs_system_table = aggregate_by_obs_system(df, limit=st.session_state.top_n)
    obs_subsystem_table = aggregate_by_obs_subsystem(df, limit=st.session_state.top_n)
    obs_component_table = aggregate_by_obs_component(df, limit=st.session_state.top_n)
    created_chart, created_events = created_trend(df, start_dt, end_dt)

    st.success(f"Displaying {len(df)} issues.")

    st.markdown("---")
    st.subheader("High-Level Metrics")
    window_start_str = start_dt.astimezone(tz).strftime("%Y-%m-%d")
    window_end_str = end_dt.astimezone(tz).strftime("%Y-%m-%d")

    # Caption summarizing the scope and meaning of High-Level Metrics
    st.caption(f"Metrics for {window_start_str} to {window_end_str} (inclusive, {TIMEZONE}). ")
    created_series = (
        df["created_dt"]
        if "created_dt" in df.columns
        else pd.to_datetime(df["created"], errors="coerce", utc=True).dt.tz_convert(tz)
    )
    created_mask = created_series.between(start_dt, end_dt, inclusive="both")
    created_in_window = df[created_mask]

    updated_series = (
        df["updated_dt"]
        if "updated_dt" in df.columns
        else pd.to_datetime(df["updated"], errors="coerce", utc=True).dt.tz_convert(tz)
    )
    updated_mask = updated_series.between(start_dt, end_dt, inclusive="both")
    updated_in_window = df[updated_mask]

    resolved_in_window = pd.DataFrame()
    if "resolution_dt" in df.columns:
        resolution_series = df["resolution_dt"]
    elif "resolution_date" in df.columns:
        resolution_series = pd.to_datetime(df["resolution_date"], errors="coerce", utc=True).dt.tz_convert(tz)
    elif "resolutiondate" in df.columns:
        resolution_series = pd.to_datetime(df["resolutiondate"], errors="coerce", utc=True).dt.tz_convert(tz)
    else:
        resolution_series = None
    if resolution_series is not None:
        resolved_mask = resolution_series.between(start_dt, end_dt, inclusive="both")
        resolved_in_window = df[resolved_mask]

    # Compute "real" resolved count (exclude Cancelled/Duplicate) for the headline metric
    real_resolved_count = 0
    if not resolved_in_window.empty:
        res_txt = resolved_in_window.get("resolution")
        stat_txt = resolved_in_window.get("status")
        dup_labels = {"duplicate", "duplicated"}
        cancel_labels = {"cancelled", "canceled"}
        dup_mask = pd.Series(False, index=resolved_in_window.index)
        cancel_mask = pd.Series(False, index=resolved_in_window.index)
        if res_txt is not None:
            low = res_txt.fillna("").astype(str).str.lower()
            dup_mask |= low.isin(dup_labels)
            cancel_mask |= low.isin(cancel_labels)
        if stat_txt is not None:
            low = stat_txt.fillna("").astype(str).str.lower()
            dup_mask |= low.isin(dup_labels)
            cancel_mask |= low.isin(cancel_labels)
        real_resolved_count = int((~(dup_mask | cancel_mask)).sum())

    if "status" in df.columns:
        normalized_status = df["status"].astype(str).apply(normalize_workflow_status)
        open_mask = ~normalized_status.isin(TERMINAL_STATUSES)
    else:
        open_mask = pd.Series(True, index=df.index)
    open_df = df[open_mask].copy()
    open_count = len(open_df)

    staleness_series = pd.Series(dtype="float64")
    if open_count:
        if "days_since_update" in open_df.columns:
            staleness_series = pd.to_numeric(open_df["days_since_update"], errors="coerce")
        elif "updated_dt" in open_df.columns:
            updated_dt_series = open_df["updated_dt"]
            if not pd.api.types.is_datetime64_any_dtype(updated_dt_series):
                updated_dt_series = pd.to_datetime(
                    updated_dt_series, errors="coerce", utc=True
                ).dt.tz_convert(tz)
            staleness_series = (now_ts - updated_dt_series).dt.total_seconds() / 86400.0
        elif "updated" in open_df.columns:
            updated_dt_series = pd.to_datetime(open_df["updated"], errors="coerce", utc=True)
            staleness_series = (now_ts - updated_dt_series.dt.tz_convert(tz)).dt.total_seconds() / 86400.0
        staleness_series = staleness_series.dropna()

    recent_updates_count = int((staleness_series <= 7).sum()) if not staleness_series.empty else 0
    recent_updates_pct = (recent_updates_count / open_count * 100) if open_count else 0.0

    net_change = len(created_in_window) - len(resolved_in_window)

    age_series = pd.Series(dtype="float64")
    if open_count:
        if "days_open" in open_df.columns:
            age_series = pd.to_numeric(open_df["days_open"], errors="coerce")
        elif "created_dt" in open_df.columns:
            created_series = pd.to_datetime(open_df["created_dt"], errors="coerce", utc=True)
            age_series = (now_ts - created_series.dt.tz_convert(tz)).dt.total_seconds() / 86400.0
        elif "created" in open_df.columns:
            created_series = pd.to_datetime(open_df["created"], errors="coerce", utc=True).dt.tz_convert(tz)
            age_series = (now_ts - created_series).dt.total_seconds() / 86400.0
        age_series = age_series.dropna()
    average_age_days = float(age_series.mean()) if not age_series.empty else None
    average_age_display = f"{average_age_days:.1f}" if average_age_days is not None else "N/A"

    done_status_list = ", ".join(sorted(TERMINAL_STATUSES))
    created_help = "Tickets created with timestamps falling within the inclusive analysis window."
    updated_help = "Tickets updated within the same inclusive window derived from Jira's updated timestamp. "
    resolved_help = "Tickets resolved. It excludes 'Cancelled' and 'Duplicate' outcomes. "
    net_change_help = (
        "Created count minus resolved count for the window. Positive values indicate backlog growth."
    )
    recent_help = (
        "Share of open tickets (excluding statuses: "
        f"{done_status_list}) that were updated in the last 7 days."
    )
    average_age_help = (
        "Mean number of days that currently open tickets have been active, using 'days_open' when provided "
        "or derived from creation timestamps."
    )

    # High-Level Metrics unified layout: 3 rows x 4 metrics
    metric_row1 = st.columns(4)
    metric_row1[0].metric(
        "Total Issues Fetched",
        len(df),
        help="Total number of enriched issues currently loaded for this analysis run.",
    )
    metric_row1[1].metric("Created in Window", len(created_in_window), help=created_help)
    metric_row1[2].metric("Updated in Window", len(updated_in_window), help=updated_help)
    metric_row1[3].metric("Resolved in Window", real_resolved_count, help=resolved_help)

    open_help = (
        "Count of tickets currently not in a done/terminal status (statuses excluded: "
        f"{done_status_list})."
    )
    metric_row2 = st.columns(4)
    metric_row2[0].metric("Open Tickets", open_count, help=open_help)
    metric_row2[1].metric(
        "Net Change (Created - Resolved)",
        f"{net_change:+d}",
        delta=f"{len(created_in_window)} created / {len(resolved_in_window)} resolved",
        help=net_change_help,
    )
    recent_delta = f"{recent_updates_count} of {open_count} open" if open_count else None
    if recent_delta:
        metric_row2[2].metric(
            "% Open Updated ≤ 7d",
            f"{recent_updates_pct:.0f}%",
            delta=recent_delta,
            help=recent_help,
        )
    else:
        metric_row2[2].metric(
            "% Open Updated ≤ 7d",
            "0%",
            delta="No open tickets",
            help=recent_help,
        )
    metric_row2[3].metric("Average Age (Days)", average_age_display, help=average_age_help)

    # Duplicate and Cancelled ticket metrics (window)
    duplicate_labels = {"duplicate", "duplicated"}
    cancel_labels = {"cancelled", "canceled"}
    dup_df = pd.DataFrame()
    cancel_df = pd.DataFrame()
    if not df.empty:
        _res = df.get("resolution")
        _stat = df.get("status")
        # Duplicates
        dup_mask = pd.Series(False, index=df.index)
        if _res is not None:
            dup_mask |= _res.fillna("").astype(str).str.lower().isin(duplicate_labels)
        if _stat is not None:
            dup_mask |= _stat.fillna("").astype(str).str.lower().isin(duplicate_labels)
        dup_df = df.loc[dup_mask].copy()
        # Cancelled
        cancel_mask = pd.Series(False, index=df.index)
        if _res is not None:
            cancel_mask |= _res.fillna("").astype(str).str.lower().isin(cancel_labels)
        if _stat is not None:
            cancel_mask |= _stat.fillna("").astype(str).str.lower().isin(cancel_labels)
        cancel_df = df.loc[cancel_mask].copy()
    duplicates_in_window = len(dup_df)
    created_dup = 0
    updated_dup = 0
    resolved_dup = 0
    resolved_cancel = 0
    created_cancel = 0
    updated_cancel = 0
    if duplicates_in_window:
        if not created_in_window.empty:
            created_dup = dup_df["key"].isin(created_in_window["key"]).sum()
        if not updated_in_window.empty:
            updated_dup = dup_df["key"].isin(updated_in_window["key"]).sum()
        if not resolved_in_window.empty:
            resolved_dup = dup_df["key"].isin(resolved_in_window["key"]).sum()
    if not cancel_df.empty and not resolved_in_window.empty:
        resolved_cancel = cancel_df["key"].isin(resolved_in_window["key"]).sum()
    if not cancel_df.empty:
        if not created_in_window.empty:
            created_cancel = cancel_df["key"].isin(created_in_window["key"]).sum()
        if not updated_in_window.empty:
            updated_cancel = cancel_df["key"].isin(updated_in_window["key"]).sum()
    metric_row3 = st.columns(4)
    metric_row3[0].metric(
        "Duplicated (Window)",
        duplicates_in_window,
        help="Tickets whose resolution or status is 'Duplicate' / 'Duplicated' in the loaded dataset.",
    )
    if len(created_in_window) > 0:
        pct_created_dup = (created_dup / len(created_in_window)) * 100 if len(created_in_window) else 0
        metric_row3[1].metric(
            "% Created Duplicate",
            f"{pct_created_dup:.1f}%",
            help=f"{created_dup} of {len(created_in_window)} created tickets marked duplicate.",
        )
    else:
        metric_row3[1].metric("% Created Duplicate", "0%", help="No tickets created in window.")
    if len(updated_in_window) > 0:
        pct_updated_dup = (updated_dup / len(updated_in_window)) * 100
        metric_row3[2].metric(
            "% Updated Duplicate",
            f"{pct_updated_dup:.1f}%",
            help=f"{updated_dup} of {len(updated_in_window)} updated tickets marked duplicate.",
        )
    else:
        metric_row3[2].metric("% Updated Duplicate", "0%", help="No tickets updated in window.")
    if len(resolved_in_window) > 0:
        pct_resolved_dup = (resolved_dup / len(resolved_in_window)) * 100
        metric_row3[3].metric(
            "% Resolved Duplicate",
            f"{pct_resolved_dup:.1f}%",
            help=f"{resolved_dup} of {len(resolved_in_window)} resolved tickets marked duplicate.",
        )
    else:
        metric_row3[3].metric("% Resolved Duplicate", "0%", help="No tickets resolved in window.")

    # Additional Cancelled metrics row
    metric_row4 = st.columns(4)
    metric_row4[0].metric(
        "Cancelled (Window)",
        len(cancel_df),
        help="Tickets whose resolution or status is 'Cancelled'/'Canceled' in the loaded dataset.",
    )
    if len(created_in_window) > 0:
        pct_created_cancel = (created_cancel / len(created_in_window)) * 100
        metric_row4[1].metric(
            "% Created Cancelled",
            f"{pct_created_cancel:.1f}%",
            help=f"{created_cancel} of {len(created_in_window)} created tickets marked cancelled.",
        )
    else:
        metric_row4[1].metric("% Created Cancelled", "0%", help="No tickets created in window.")
    if len(updated_in_window) > 0:
        pct_updated_cancel = (updated_cancel / len(updated_in_window)) * 100
        metric_row4[2].metric(
            "% Updated Cancelled",
            f"{pct_updated_cancel:.1f}%",
            help=f"{updated_cancel} of {len(updated_in_window)} updated tickets marked cancelled.",
        )
    else:
        metric_row4[2].metric("% Updated Cancelled", "0%", help="No tickets updated in window.")
    if len(resolved_in_window) > 0:
        pct_resolved_cancel = (resolved_cancel / len(resolved_in_window)) * 100
        metric_row4[3].metric(
            "% Resolved Cancelled",
            f"{pct_resolved_cancel:.1f}%",
            help=f"{resolved_cancel} of {len(resolved_in_window)} resolved tickets marked cancelled.",
        )
    else:
        metric_row4[3].metric("% Resolved Cancelled", "0%", help="No tickets resolved in window.")

    # Aging warning shown at end of High-Level Metrics section
    if context.aging_warning_count:
        st.warning(
            f"{context.aging_warning_count} blocker/critical tickets are older than "
            f"{st.session_state.warn_age} days."
        )

    st.markdown("---")
    st.subheader("Created, Updated and Resolved")
    # Full-width tabs with tables always visible
    # Pre-initialize toggle state to avoid first-use focus jump
    if "resolved_exclude_cancel_duplicate" not in st.session_state:
        st.session_state.resolved_exclude_cancel_duplicate = False
    tab_labels = [
        f"Created ({len(created_in_window)})",
        f"Updated ({len(updated_in_window)})",
        f"Resolved ({len(resolved_in_window)})",
    ]
    created_tab, updated_tab, resolved_tab = st.tabs(tab_labels)

    # Created tab
    with created_tab:
        created_date_col = None
        for candidate in ("created_dt", "created"):
            if candidate in df.columns:
                created_date_col = candidate
                break
        display_ticket_list(
            created_in_window,
            server=server_url,
            tz=tz,
            date_col=created_date_col,
            date_label="created",
            empty_message="No tickets created in this window.",
            sort_by="created" if created_date_col else None,
            ascending=False,
        )

    # Updated tab
    with updated_tab:
        updated_date_col = None
        for candidate in ("updated_dt", "updated"):
            if candidate in df.columns:
                updated_date_col = candidate
                break
        display_ticket_list(
            updated_in_window,
            server=server_url,
            tz=tz,
            date_col=updated_date_col,
            date_label="updated",
            empty_message="No tickets updated in this window.",
            sort_by="updated" if updated_date_col else None,
            ascending=False,
        )

    # Resolved tab
    with resolved_tab:
        resolved_date_col = None
        for candidate in ("resolution_dt", "resolution_date", "resolutiondate"):
            if candidate in df.columns:
                resolved_date_col = candidate
                break
        # Toggle to exclude Cancelled & Duplicate from the resolved table view
        exclude_cancel_dupe = st.checkbox(
            "Exclude Cancelled & Duplicate",
            key="resolved_exclude_cancel_duplicate",
            help="When enabled, hides tickets marked Cancelled/Canceled or Duplicate/Duplicated.",
        )
        to_show = resolved_in_window
        if exclude_cancel_dupe and not resolved_in_window.empty:
            r_res = resolved_in_window.get("resolution")
            r_stat = resolved_in_window.get("status")
            dup_labels = {"duplicate", "duplicated"}
            cancel_labels = {"cancelled", "canceled"}
            dup_mask = pd.Series(False, index=resolved_in_window.index)
            cancel_mask = pd.Series(False, index=resolved_in_window.index)
            if r_res is not None:
                low = r_res.fillna("").astype(str).str.lower()
                dup_mask |= low.isin(dup_labels)
                cancel_mask |= low.isin(cancel_labels)
            if r_stat is not None:
                low = r_stat.fillna("").astype(str).str.lower()
                dup_mask |= low.isin(dup_labels)
                cancel_mask |= low.isin(cancel_labels)
            to_show = resolved_in_window[~(dup_mask | cancel_mask)]

        display_ticket_list(
            to_show,
            server=server_url,
            tz=tz,
            date_col=resolved_date_col,
            date_label="resolved",
            empty_message="No tickets resolved in this window.",
            sort_by="resolved" if resolved_date_col else None,
            ascending=False,
        )
        if not resolved_in_window.empty:
            st.caption(
                f"Showing {len(to_show)} of {len(resolved_in_window)} resolved tickets. "
                "By default, the table includes tickets marked Cancelled/Canceled and Duplicate/Duplicated. "
                "Use the toggle above to exclude them."
            )

    st.markdown("---")
    # Status Distribution (replaces the former 'Tickets Marked as Duplicated' section)
    st.subheader("Status Distribution")
    st.markdown(" ")
    if "status" in df.columns and not df["status"].isnull().all():
        status_series = df["status"].fillna("").astype(str)
        normalized = status_series.apply(normalize_workflow_status)
        counts = normalized.value_counts().rename_axis("status").reset_index(name="count")
        order = [s for s in STATUS_DISPLAY_ORDER if s in counts["status"].values]
        if (counts["status"] == "Unknown").any():
            order = order + ["Unknown"]
        counts = counts.set_index("status").reindex(order).fillna({"count": 0}).reset_index()
        # Height based on number of status categories
        status_rows = len(order) if order else len(counts)
        shared_panel_height = max(180, int(32 * status_rows + 56))
        status_chart = create_bar_chart(counts, "status", "count", None)
        if status_chart:
            st.altair_chart(
                status_chart.properties(height=shared_panel_height),
                width="stretch",
            )
        else:
            st.dataframe(counts, hide_index=True, width="stretch")
    else:
        st.info("No status data available.")

    # Tickets by Status full row with global filter
    st.markdown("---")
    st.subheader("Tickets by Status")
    st.markdown(" ")
    if "status" not in df.columns or df["status"].isnull().all():
        st.info("Status-driven drilldown unavailable.")
    else:
        # Pre-initialize filter session state to reduce initial rerun/focus jump
        if "status_row_assignees" not in st.session_state:
            st.session_state.status_row_assignees = []
        if "status_row_reporters" not in st.session_state:
            st.session_state.status_row_reporters = []
        if "status_row_priorities" not in st.session_state:
            st.session_state.status_row_priorities = []
        if "status_row_obs" not in st.session_state:
            st.session_state.status_row_obs = []
        if "status_row_obs_subsystem" not in st.session_state:
            st.session_state.status_row_obs_subsystem = []
        if "status_row_obs_component" not in st.session_state:
            st.session_state.status_row_obs_component = []
        if "status_row_search" not in st.session_state:
            st.session_state.status_row_search = ""
        if "status_row_ticket_keys" not in st.session_state:
            st.session_state.status_row_ticket_keys = []
        # Coerce legacy values (from prior text input) into list for multiselects
        if not isinstance(st.session_state.status_row_assignees, list):
            st.session_state.status_row_assignees = []
        if not isinstance(st.session_state.status_row_reporters, list):
            st.session_state.status_row_reporters = []
        if not isinstance(st.session_state.status_row_priorities, list):
            st.session_state.status_row_priorities = []
        if not isinstance(st.session_state.status_row_obs, list):
            st.session_state.status_row_obs = []
        if not isinstance(st.session_state.status_row_obs_subsystem, list):
            st.session_state.status_row_obs_subsystem = []
        if not isinstance(st.session_state.status_row_obs_component, list):
            st.session_state.status_row_obs_component = []
        if not isinstance(st.session_state.status_row_ticket_keys, list):
            st.session_state.status_row_ticket_keys = []

        # Normalize status and prepare base frame
        status_series = df["status"].fillna("").astype(str)
        normalized = status_series.apply(normalize_workflow_status)
        df_with_norm = df.copy()
        df_with_norm["__norm_status"] = normalized

        # Global filter expander
        with st.expander("Filter", expanded=False):
            # Row 1: Assignee | Reporter | Priority
            c1, c2, c3 = st.columns(3)
            assignee_opts = (
                sorted(df_with_norm["assignee"].fillna("(Unassigned)").unique())
                if "assignee" in df_with_norm.columns
                else []
            )
            reporter_opts = (
                sorted(df_with_norm["reporter"].dropna().astype(str).unique())
                if "reporter" in df_with_norm.columns
                else []
            )
            priority_opts = (
                sorted(df_with_norm["priority"].dropna().astype(str).unique())
                if "priority" in df_with_norm.columns
                else []
            )
            with c1:
                # Avoid default + session value collision by only using key
                st.session_state.status_row_assignees = [
                    v for v in st.session_state.status_row_assignees if v in assignee_opts
                ]
                sel_assignees = st.multiselect(
                    "Assignee",
                    options=assignee_opts,
                    key="status_row_assignees",
                )
            with c2:
                st.session_state.status_row_reporters = [
                    v for v in st.session_state.status_row_reporters if v in reporter_opts
                ]
                sel_reporters = st.multiselect(
                    "Reporter",
                    options=reporter_opts,
                    key="status_row_reporters",
                )
            with c3:
                st.session_state.status_row_priorities = [
                    v for v in st.session_state.status_row_priorities if v in priority_opts
                ]
                sel_priorities = st.multiselect(
                    "Priority",
                    options=priority_opts,
                    key="status_row_priorities",
                )

            # Row 2: OBS System | OBS Subsystem | OBS Component
            c4, c5, c6 = st.columns(3)
            obs_system_opts = (
                sorted(df_with_norm["obs_system"].dropna().astype(str).unique())
                if "obs_system" in df_with_norm.columns
                else []
            )
            obs_subsystem_opts = (
                sorted(df_with_norm["obs_subsystem"].dropna().astype(str).unique())
                if "obs_subsystem" in df_with_norm.columns
                else []
            )
            obs_component_opts = (
                sorted(df_with_norm["obs_component"].dropna().astype(str).unique())
                if "obs_component" in df_with_norm.columns
                else []
            )
            with c4:
                st.session_state.status_row_obs = [
                    v for v in st.session_state.status_row_obs if v in obs_system_opts
                ]
                sel_obs_systems = st.multiselect(
                    "OBS System",
                    options=obs_system_opts,
                    key="status_row_obs",
                )
            with c5:
                st.session_state.status_row_obs_subsystem = [
                    v for v in st.session_state.status_row_obs_subsystem if v in obs_subsystem_opts
                ]
                sel_obs_subsystems = st.multiselect(
                    "OBS Subsystem",
                    options=obs_subsystem_opts,
                    key="status_row_obs_subsystem",
                )
            with c6:
                st.session_state.status_row_obs_component = [
                    v for v in st.session_state.status_row_obs_component if v in obs_component_opts
                ]
                sel_obs_components = st.multiselect(
                    "OBS Component",
                    options=obs_component_opts,
                    key="status_row_obs_component",
                )

            # Row 3 (full-width): Labels filter
            all_labels: list[str] = []
            if "labels" in df_with_norm.columns:
                # labels is a comma-separated string; split to individual values
                series = df_with_norm["labels"].fillna("").astype(str)
                parts = series.str.split(",")
                values: set[str] = set()
                for lst in parts:
                    if not lst:
                        continue
                    for raw in lst:
                        name = raw.strip()
                        if name:
                            values.add(name)
                all_labels = sorted(values, key=lambda s: s.lower()) if values else []
            if "status_row_labels" not in st.session_state:
                st.session_state.status_row_labels = []
            st.session_state.status_row_labels = [
                v for v in st.session_state.status_row_labels if v in all_labels
            ]
            selected_labels = st.multiselect(
                "Labels",
                options=all_labels,
                key="status_row_labels",
                help="Filter tickets that include any of the selected labels.",
            )

            # Row 4 (full-width): OBS Tickets selector
            all_keys = (
                sorted(df_with_norm["key"].dropna().astype(str).unique())
                if "key" in df_with_norm.columns
                else []
            )
            st.session_state.status_row_ticket_keys = [
                v for v in st.session_state.status_row_ticket_keys if v in all_keys
            ]
            selected_ticket_keys = st.multiselect(
                "OBS Tickets",
                options=all_keys,
                key="status_row_ticket_keys",
                help="Select one or more ticket keys.",
            )

            # Row 5 (full-width): Search Summary
            # Supports keys and comma/space separated lists
            search_text = st.text_input(
                "Search Summary",
                value=st.session_state.status_row_search,
                key="status_row_search",
                help=(
                    "Type text to match in summary, or enter ticket keys (comma/space-separated), "
                    "e.g. OBS-123, OBS-456. Matches keys or summary text."
                ),
            )

        # Apply global filters once
        filtered_base = df_with_norm.copy()
        if sel_assignees and "assignee" in filtered_base.columns:
            filtered_base["assignee"] = filtered_base["assignee"].fillna("(Unassigned)")
            filtered_base = filtered_base[filtered_base["assignee"].isin(sel_assignees)]
        if sel_reporters and "reporter" in filtered_base.columns:
            filtered_base = filtered_base[filtered_base["reporter"].isin(sel_reporters)]
        if sel_priorities and "priority" in filtered_base.columns:
            filtered_base = filtered_base[filtered_base["priority"].astype(str).isin(sel_priorities)]
        if sel_obs_systems and "obs_system" in filtered_base.columns:
            filtered_base = filtered_base[filtered_base["obs_system"].astype(str).isin(sel_obs_systems)]
        if (
            "sel_obs_subsystems" in locals()
            and sel_obs_subsystems
            and "obs_subsystem" in filtered_base.columns
        ):
            filtered_base = filtered_base[filtered_base["obs_subsystem"].astype(str).isin(sel_obs_subsystems)]
        if (
            "sel_obs_components" in locals()
            and sel_obs_components
            and "obs_component" in filtered_base.columns
        ):
            filtered_base = filtered_base[filtered_base["obs_component"].astype(str).isin(sel_obs_components)]
        if search_text:
            q = search_text.strip()
            if q:
                # Parse tokens by comma/space, remove empties
                tokens = [
                    t for t in [x.strip() for x in q.replace("\n", " ").replace(",", " ").split(" ")] if t
                ]
                key_set = {t.upper() for t in tokens if "-" in t}
                word_tokens = [t.lower() for t in tokens if t not in key_set]
                key_mask = (
                    filtered_base["key"].astype(str).str.upper().isin(key_set)
                    if key_set and "key" in filtered_base.columns
                    else False
                )
                sum_mask = (
                    filtered_base["summary"]
                    .astype(str)
                    .str.lower()
                    .apply(lambda s: any(w in s for w in word_tokens))
                    if word_tokens and "summary" in filtered_base.columns
                    else False
                )
                if isinstance(key_mask, pd.Series) and isinstance(sum_mask, pd.Series):
                    filtered_base = filtered_base[key_mask | sum_mask]
                elif isinstance(key_mask, pd.Series):
                    filtered_base = filtered_base[key_mask]
                elif isinstance(sum_mask, pd.Series):
                    filtered_base = filtered_base[sum_mask]
        if selected_labels and "labels" in filtered_base.columns:
            label_series = filtered_base["labels"].fillna("").astype(str)
            filtered_base = filtered_base[
                label_series.apply(
                    lambda s: any(
                        lbl.lower() in [part.strip().lower() for part in s.split(",") if part.strip()]
                        for lbl in selected_labels
                    )
                )
            ]
        if selected_ticket_keys and "key" in filtered_base.columns:
            filtered_base = filtered_base[filtered_base["key"].astype(str).isin(selected_ticket_keys)]

        present = list(filtered_base["__norm_status"].unique()) if not filtered_base.empty else []
        tab_order = [s for s in STATUS_DISPLAY_ORDER if s in present]
        if "Unknown" in present and "Unknown" not in tab_order:
            tab_order.append("Unknown")
        # Build final tabs list with an 'All' tab first when there is any data
        if filtered_base.empty:
            st.info("No tickets match the selected filters.")
        else:
            final_tabs = ["All"] + tab_order
            status_tabs = st.tabs(final_tabs)
            # Choose a date column for display/sort: prefer updated, else created
            date_col = None
            for candidate in ("updated_dt", "updated", "created_dt", "created"):
                if candidate in df.columns:
                    date_col = candidate
                    break
            date_label = (
                "updated" if date_col and "updated" in date_col else ("created" if date_col else None)
            )

            for tab, status_name in zip(status_tabs, final_tabs, strict=False):
                with tab:
                    if status_name == "All":
                        subset = filtered_base
                    else:
                        subset = filtered_base[filtered_base["__norm_status"] == status_name]

                    display_ticket_list(
                        subset,
                        server=server_url,
                        tz=tz,
                        date_col=date_col,
                        date_label=date_label,
                        empty_message=(
                            "No tickets match the selected filters."
                            if status_name == "All"
                            else f"No tickets with status '{status_name}'."
                        ),
                        sort_by=date_label,
                        ascending=False,
                        # Include Top N-like specific metrics in the table
                        extra_columns=[
                            "filtered_comments_count",
                            "ole_comments_count",
                            "filtered_histories_count",
                        ],
                        reorder_like_topn=True,
                        caption=None,
                    )
    # Commonly Reported Faults (moved out of Summary tab)
    st.subheader("Commonly Reported Faults")
    faults_options = {
        "OBS System": "obs_system",
        "OBS Subsystem": "obs_subsystem",
        "OBS Component": "obs_component",
    }
    # Shared Top N control across sub-tabs
    if "summary_faults_top_n" not in st.session_state:
        st.session_state.summary_faults_top_n = 15
    top_default = int(st.session_state.summary_faults_top_n)
    top_default = min(max(top_default, 5), 100)
    if top_default != st.session_state.summary_faults_top_n:
        st.session_state.summary_faults_top_n = top_default
    top_limit = st.number_input(
        "Show top N groups",
        min_value=5,
        max_value=100,
        step=5,
        key="summary_faults_top_n",
    )
    top_limit = int(top_limit)

    # Three sub-tabs for System | Subsystem | Component
    fault_tabs = st.tabs(list(faults_options.keys()))
    for tab, (group_label, group_col) in zip(fault_tabs, faults_options.items(), strict=False):
        with tab:
            faults_df = _prepare_commonly_reported_faults(df, group_col, tz)
            if faults_df.empty:
                st.info("No OBS groups found under the current filters.")
            else:
                # Prepare data for charts
                cols_needed = [
                    group_col,
                    "reported_occurrences",
                    "open_tickets",
                    "avg_open_days",
                    "last_seen",
                ]
                work = faults_df[cols_needed].rename(columns={group_col: group_label})
                for col in ("reported_occurrences", "open_tickets", "avg_open_days"):
                    work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)

                metric_tabs = st.tabs(["Total/Open Tickets", "Avg Days Open"])

                # Tab 1: Open tickets layered over total reported occurrences
                with metric_tabs[0]:
                    work_open = work.sort_values(
                        ["open_tickets", "last_seen"], ascending=[False, False]
                    ).head(top_limit)
                    order_list = work_open[group_label].tolist()
                    bg = (
                        alt.Chart(work_open)
                        .transform_calculate(value="datum.reported_occurrences")
                        .mark_bar(color="#e6e6e6")
                        .encode(
                            y=alt.Y(
                                f"{group_label}:N",
                                sort=order_list,
                                title=group_label,
                            ),
                            x=alt.X("value:Q", title=""),
                            tooltip=[
                                group_label,
                                alt.Tooltip("reported_occurrences:Q", title="Total reported"),
                            ],
                        )
                        .properties(height=alt.Step(22))
                    )
                    fg = (
                        alt.Chart(work_open)
                        .transform_calculate(value="datum.open_tickets")
                        .transform_calculate(
                            open_ratio=(
                                "datum.reported_occurrences == 0 ? 0 : "
                                "datum.open_tickets / datum.reported_occurrences"
                            )
                        )
                        .mark_bar(color="#4472C4")
                        .encode(
                            y=alt.Y(
                                f"{group_label}:N",
                                sort=order_list,
                                title=group_label,
                            ),
                            x=alt.X("value:Q", title=""),
                            tooltip=[
                                group_label,
                                alt.Tooltip("open_tickets:Q", title="Open tickets"),
                                alt.Tooltip("reported_occurrences:Q", title="Total reported"),
                                alt.Tooltip(
                                    "open_ratio:Q",
                                    title="% open",
                                    format=".0%",
                                ),
                            ],
                        )
                        .properties(height=alt.Step(22))
                    )
                    open_chart = bg + fg
                    st.altair_chart(open_chart, width="stretch")
                    st.markdown(
                        "<div style='font-size:12px;color:#666;'>"
                        "<span style='display:inline-block;width:10px;height:10px;background:#e6e6e6;"
                        "margin-right:6px;border:1px solid #ccc;vertical-align:middle'></span>"
                        "Total &nbsp;&nbsp;"
                        "<span style='display:inline-block;width:10px;height:10px;background:#4472C4;"
                        "margin-right:6px;border:1px solid #2f5597;vertical-align:middle'></span>"
                        "Open"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Open vs total reported per group.")

                # Tab 2: Average days open for open tickets
                with metric_tabs[1]:
                    work_age = work.sort_values(
                        ["avg_open_days", "last_seen"], ascending=[False, False]
                    ).head(top_limit)
                    age_chart = (
                        alt.Chart(work_age)
                        .mark_bar(color="#4472C4")
                        .encode(
                            y=alt.Y(f"{group_label}:N", sort="-x", title=group_label),
                            x=alt.X("avg_open_days:Q", title=""),
                            tooltip=[
                                group_label,
                                alt.Tooltip("avg_open_days:Q", title="Avg days open", format=".1f"),
                                alt.Tooltip("open_tickets:Q", title="Open tickets"),
                                alt.Tooltip("reported_occurrences:Q", title="Total reported"),
                            ],
                        )
                        .properties(height=alt.Step(22))
                    )
                    st.altair_chart(age_chart, width="stretch")
                    st.markdown(
                        "<div style='font-size:12px;color:#666;'>"
                        "<span style='display:inline-block;width:10px;height:10px;"
                        "background:#4472C4;margin-right:6px;border:1px solid #2f5597;"
                        "vertical-align:middle'></span>"
                        "Avg days open"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("Average age of open tickets for each group (days).")

    # Reported Occurrences Distribution by OBS Group (moved out of Summary tab)
    st.subheader("Reported Occurrences Distribution by OBS Group")
    st.caption("Note that OLE Comments are a good proxy to measure reported occurrences.")
    # Ensure default Top N is 15 on first render
    if "ole_obs_top_n" not in st.session_state:
        st.session_state.ole_obs_top_n = 15
    ole_top_n = st.number_input(
        "Show top N groups",
        min_value=5,
        max_value=100,
        step=5,
        key="ole_obs_top_n",
    )
    ole_top_n = int(ole_top_n)
    obs_charts = _build_ole_obs_heatmaps(df, start_dt, end_dt, top_n=ole_top_n)
    if not obs_charts:
        st.info("OBS engagement metrics not available for this dataset.")
    else:
        tab_labels = [label for label, _chart, _desc in obs_charts]
        chart_tabs = st.tabs(tab_labels)
        for tab, (chart_label, chart_obj, chart_desc) in zip(chart_tabs, obs_charts, strict=False):
            with tab:
                if chart_obj is not None:
                    st.altair_chart(chart_obj, width="stretch")
                    st.caption(chart_desc)
                else:
                    st.info(f"No data available to plot {chart_label.lower()}.")

    st.markdown("---")

    # Tabs reordered and Summary renamed as requested
    tab_labels = ["Top Tickets", "Aggregations", "Trends", "Flow & Cycle Times", "Word Clouds"]

    # --- Stabilization: pre-initialize widget state BEFORE creating tabs so that
    # first interaction (creating new session_state keys) doesn't cause a rerun
    # that bounces user back to the first tab.
    if "dashboard_segment_field" not in st.session_state:
        st.session_state.dashboard_segment_field = "priority"
    if "summary_obs_level" not in st.session_state:
        st.session_state.summary_obs_level = "OBS System"
    if "summary_status_open_only" not in st.session_state:
        st.session_state.summary_status_open_only = False
    if "summary_faults_obs_level" not in st.session_state:
        st.session_state.summary_faults_obs_level = "OBS System"
    if "summary_faults_top_n" not in st.session_state:
        st.session_state.summary_faults_top_n = 15
    if "ole_obs_top_n" not in st.session_state:
        st.session_state.ole_obs_top_n = 15
    # Initialize drilldown date early if events already available
    if (
        "created_drilldown_date" not in st.session_state
        and "created_events" in locals()
        and isinstance(created_events, pd.DataFrame)
        and not created_events.empty
        and "created_dt" in created_events.columns
    ):
        try:
            st.session_state.created_drilldown_date = created_events["created_dt"].max().date()
        except Exception:
            st.session_state.created_drilldown_date = st.session_state.end_date

    top_tickets_tab, aggregations_tab, trends_tab, flow_tab, wordclouds_tab = st.tabs(tab_labels)

    with flow_tab:
        st.caption(
            "Flow & Cycle Times: resolution cycle time by priority and time-in-status distributions. "
            "Use the controls to select priorities, cap maximum days, and (optionally) focus on "
            "currently open tickets."
        )

        # OBS Group Distribution removed; totals integrated into Faults section below

        # Faults and OBS heatmaps moved above the Summary tabs

        st.markdown("---")
        st.subheader("Resolution Time by Priority")
        if resolved_in_window.empty:
            st.info("Resolution time by priority unavailable — no tickets resolved in this window.")
        else:
            cycle_df = resolved_in_window.copy()
            if "resolution_dt" not in cycle_df.columns:
                if "resolution_date" in cycle_df.columns:
                    cycle_df["resolution_dt"] = pd.to_datetime(
                        cycle_df["resolution_date"], errors="coerce", utc=True
                    ).dt.tz_convert(tz)
                elif "resolutiondate" in cycle_df.columns:
                    cycle_df["resolution_dt"] = pd.to_datetime(
                        cycle_df["resolutiondate"], errors="coerce", utc=True
                    ).dt.tz_convert(tz)
            if "created_dt" not in cycle_df.columns and "created" in cycle_df.columns:
                cycle_df["created_dt"] = pd.to_datetime(
                    cycle_df["created"], errors="coerce", utc=True
                ).dt.tz_convert(tz)
            cycle_df = cycle_df.dropna(subset=["created_dt", "resolution_dt"])
            if cycle_df.empty or "priority" not in cycle_df.columns:
                st.info("Resolution time by priority unavailable — no priority data for resolved tickets.")
            else:
                cycle_df["cycle_time_days"] = (
                    cycle_df["resolution_dt"] - cycle_df["created_dt"]
                ).dt.total_seconds() / 86400.0
                cycle_df = cycle_df[cycle_df["cycle_time_days"] >= 0].copy()
                cycle_df = cycle_df.dropna(subset=["cycle_time_days", "priority"])
                if cycle_df.empty:
                    st.info("Resolution time by priority unavailable — no usable cycle-time data.")
                else:
                    # Normalize priorities (e.g., "Medium (migrated)" -> "Medium")
                    cycle_df["priority"] = cycle_df["priority"].apply(normalize_priority_name)
                    if "available_priorities" in locals() and available_priorities:
                        priority_options = list(available_priorities)
                    else:
                        priority_options = sorted(cycle_df["priority"].unique())
                    # Ensure defaults only include priorities that exist for this window
                    # Order priorities according to DEFAULT_TREND_PRIORITIES where possible
                    ordered_priorities = [p for p in DEFAULT_TREND_PRIORITIES if p in priority_options]
                    ordered_priorities += [p for p in priority_options if p not in ordered_priorities]
                    priority_options = ordered_priorities

                    # Simple multiselect - default to all priorities
                    selected_priorities = st.multiselect(
                        "Select priorities",
                        options=priority_options,
                        default=priority_options,
                    )
                    filtered_cycle = cycle_df.copy()
                    if selected_priorities:
                        filtered_cycle = filtered_cycle[filtered_cycle["priority"].isin(selected_priorities)]
                    if filtered_cycle.empty:
                        st.info("No resolved tickets match the selected filters.")
                    else:
                        grouped = (
                            filtered_cycle.groupby("priority")["cycle_time_days"]
                            .agg(
                                Tickets="count",
                                Mean="mean",
                                P90=lambda s: s.quantile(0.9),
                            )
                            .rename_axis("Priority")
                        )
                        if priority_options:
                            grouped = grouped.reindex(priority_options)
                        stats = grouped.reset_index()
                        if not stats.empty:
                            stats["Tickets"] = stats["Tickets"].fillna(0).astype("Int64")
                            stats["Mean"] = stats["Mean"].round(1)
                            stats["P90"] = stats["P90"].round(1)
                        overall_metrics = {
                            "Priority": "All Selected",
                            "Tickets": int(len(filtered_cycle)),
                            "Mean": (
                                round(filtered_cycle["cycle_time_days"].mean(), 1)
                                if not filtered_cycle.empty
                                else float("nan")
                            ),
                            "P90": (
                                round(filtered_cycle["cycle_time_days"].quantile(0.9), 1)
                                if not filtered_cycle.empty
                                else float("nan")
                            ),
                        }
                        stats = pd.concat(
                            [stats, pd.DataFrame([overall_metrics])],
                            ignore_index=True,
                        )
                        stats["Tickets"] = stats["Tickets"].astype("Int64")
                        stats["Mean"] = stats["Mean"].round(1)
                        stats["P90"] = stats["P90"].round(1)

                        # Bar chart for cycle time by priority (exclude "All Selected" row)
                        chart_data = stats[stats["Priority"] != "All Selected"].copy()
                        if not chart_data.empty:
                            # Melt data for grouped bar chart
                            chart_melted = chart_data.melt(
                                id_vars=["Priority"],
                                value_vars=["Mean", "P90"],
                                var_name="Metric",
                                value_name="Days",
                            )
                            chart_melted = chart_melted.dropna(subset=["Days"])
                            if not chart_melted.empty:
                                # Define priority order for sorting
                                available_priorities = chart_data["Priority"].values
                                priority_order = [p for p in priority_options if p in available_priorities]
                                cycle_chart = (
                                    alt.Chart(chart_melted)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X(
                                            "Priority:N",
                                            sort=priority_order,
                                            title="Priority",
                                        ),
                                        y=alt.Y("Days:Q", title="Days"),
                                        color=alt.Color(
                                            "Metric:N",
                                            scale=alt.Scale(
                                                domain=["Mean", "P90"],
                                                range=["#4472C4", "#ED7D31"],
                                            ),
                                            legend=alt.Legend(title="Metric"),
                                        ),
                                        xOffset="Metric:N",
                                        tooltip=[
                                            alt.Tooltip("Priority:N", title="Priority"),
                                            alt.Tooltip("Metric:N", title="Metric"),
                                            alt.Tooltip("Days:Q", title="Days", format=".1f"),
                                        ],
                                    )
                                    .properties(height=300, title="Resolution Cycle Time by Priority")
                                )
                                st.altair_chart(cycle_chart, width="stretch")
                                st.caption(
                                    "Mean and 90th percentile cycle times (days) per priority. "
                                    "Tickets resolved within the selected date range."
                                )

        st.markdown("---")
        st.subheader("Average Time in Status")
        status_duration_df = build_status_duration_frame(df, tz, now_ts)
        if status_duration_df.empty:
            st.info("Status transition history not available to calculate dwell times.")
        else:
            status_duration_df = status_duration_df.copy()
            status_duration_df["status"] = status_duration_df["status"].astype(str)
            excluded_statuses = {"cancelled", "duplicate", "done", "transferred"}
            status_duration_df = status_duration_df[
                ~status_duration_df["status"].str.lower().isin(excluded_statuses)
            ]
            all_statuses = sorted(status_duration_df["status"].unique())
            if not all_statuses:
                st.info("No eligible statuses available for dwell-time analysis.")
            else:
                # Simple multiselect - default to all statuses
                selected_statuses = st.multiselect(
                    "Select statuses",
                    options=all_statuses,
                    default=all_statuses,
                )
                open_only = st.checkbox(
                    "Focus on currently open tickets only",
                    key="summary_status_open_only",
                )
                filtered = status_duration_df.copy()
                if selected_statuses:
                    filtered = filtered[filtered["status"].isin(selected_statuses)]
                if open_only:
                    filtered = filtered[filtered["is_open"]]

                if filtered.empty:
                    st.info("No status dwell-time data for the selected filters.")
                else:
                    summary_table = (
                        filtered.groupby("status")["duration_days"]
                        .agg(Tickets="count", Mean="mean")
                        .rename_axis("Status")
                        .reset_index()
                    )
                    summary_table["Tickets"] = summary_table["Tickets"].fillna(0).astype("Int64")
                    summary_table["Mean"] = summary_table["Mean"].round(1)

                    # Horizontal bar chart for mean time in status
                    if not summary_table.empty:
                        chart_data = summary_table.dropna(subset=["Mean"]).copy()
                        if not chart_data.empty:
                            # Sort by mean descending for better visualization
                            chart_data = chart_data.sort_values("Mean", ascending=True)
                            status_order = chart_data["Status"].tolist()
                            status_bar_chart = (
                                alt.Chart(chart_data)
                                .mark_bar(color="#4472C4")
                                .encode(
                                    y=alt.Y(
                                        "Status:N",
                                        sort=status_order,
                                        title="Status",
                                        axis=alt.Axis(
                                            labelLimit=300,
                                            labelOverlap="greedy",
                                            labelPadding=5,
                                        ),
                                    ),
                                    x=alt.X("Mean:Q", title="Mean Days in Status"),
                                    tooltip=[
                                        alt.Tooltip("Status:N", title="Status"),
                                        alt.Tooltip("Tickets:Q", title="Tickets", format=",d"),
                                        alt.Tooltip("Mean:Q", title="Mean Days", format=".1f"),
                                    ],
                                )
                                .properties(
                                    height=max(250, len(chart_data) * 40),
                                    title="Average Time in Status",
                                )
                            )
                            st.altair_chart(status_bar_chart, width="stretch")
                            st.caption(
                                "Mean days spent in each workflow status. "
                                "Terminal statuses (Done, Cancelled, etc.) are excluded."
                            )

        # Assignee Workload (Open Tickets) moved to Assignee | Reporter Insights page

    with top_tickets_tab:
        st.caption(
            "Leaderboards highlighting top tickets across activity, comments, time lost, and "
            "critical/blocker. Adjust Top N from the sidebar."
        )
        st.subheader("Leaderboards")
        segments = context.segments if context else {}
        display_leaderboard(
            segments.get("Weighted Activity", pd.DataFrame()),
            title="Weighted Activity",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col="activity_score_weighted",
            metric_label="Weighted Score",
            extra_cols=[
                "comments_in_range",
                "status_changes",
                "other_changes",
                "priority",
                "status",
                "assignee",
                "days_open",
            ],
            caption=(
                "Weights — comment: "
                f"{weights.comment:.2f}, status: {weights.status:.2f}, other: {weights.other:.2f}"
            ),
        )
        display_leaderboard(
            segments.get("Most Active", pd.DataFrame()),
            title="Most Active Tickets",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col="activity_score",
            metric_label="Activity Score",
            extra_cols=[
                "filtered_comments_count",
                "filtered_histories_count",
                "priority",
                "status",
                "assignee",
                "days_since_update",
            ],
            caption="Combined count of comments and history entries in the selected range.",
        )
        display_leaderboard(
            segments.get("Most Commented", pd.DataFrame()),
            title="Most Commented Tickets",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col="filtered_comments_count",
            metric_label="Comments",
            extra_cols=["filtered_histories_count", "priority", "status", "assignee"],
        )
        display_leaderboard(
            segments.get("OLE Comments", pd.DataFrame()),
            title="Tickets with OLE Comments",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col="ole_comments_count",
            metric_label="OLE Comments",
            extra_cols=[
                "comments_in_range",
                "priority",
                "status",
                "assignee",
                "ole_last_comment",
            ],
            caption="Comments submitted via Rubin Jira API Access (OLE).",
        )
        display_leaderboard(
            segments.get("Time Lost", pd.DataFrame()),
            title="Highest Time Lost",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col="time_lost_value",
            metric_label="Time Lost",
            extra_cols=["priority", "status", "assignee", "updated"],
        )
        display_leaderboard(
            segments.get("Blocker/Critical", pd.DataFrame()),
            title="Critical & Blocker Tickets",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col="days_open",
            metric_label="Days Open",
            extra_cols=["priority", "status", "assignee", "days_since_update"],
            caption="Open blocker/critical tickets ordered by age.",
        )
        display_leaderboard(
            segments.get("Testing/Tracking", pd.DataFrame()),
            title="Testing & Tracking Status",
            server=server_url,
            top_n=st.session_state.top_n,
            metric_col=None,
            extra_cols=["status", "priority", "assignee", "days_since_update"],
        )

    with aggregations_tab:
        st.caption(
            "Aggregations by segment (e.g., priority, status, OBS) with optional averages. "
            "Use the tabs to pick a segment and sub-tabs to choose the metric as a bar chart."
        )
        st.subheader("Segment Analysis")
        segment_options = [
            "priority",
            "status",
            "assignee",
            "reporter",
            "obs_system",
            "obs_subsystem",
            "obs_component",
            "labels",
        ]
        seg_tabs = st.tabs([s.replace("_", " ").title() for s in segment_options])

        def _build_segment_agg(seg_field: str) -> tuple[pd.DataFrame, str, list[tuple[str, str, str]]]:
            # Returns (df, label, metrics) where metrics is list of (column, title, help)
            if seg_field not in df.columns:
                return pd.DataFrame(), seg_field.replace("_", " ").title(), []

            working = df.copy()
            # For labels, treat each comma-separated label as an individual segment value
            if seg_field == "labels":
                has_labels = working["labels"].notna() & (working["labels"].astype(str).str.strip() != "")
                working = working[has_labels]
                if working.empty:
                    return pd.DataFrame(), "Labels", []
                # Split comma-separated label strings and explode into separate rows
                split_labels = (
                    working["labels"]
                    .astype(str)
                    .str.split(",")
                    .apply(lambda lst: [item.strip() for item in lst if item.strip()])
                )
                working = working.assign(labels=split_labels).explode("labels")

            aggregations: dict[str, tuple[str, str]] = {"issue_count": ("key", "count")}
            metric_meta: list[tuple[str, str, str]] = [("issue_count", "Tickets", "Total issues in segment.")]
            if "days_open" in working.columns:
                aggregations["avg_days_open"] = ("days_open", "mean")
                metric_meta.append(
                    (
                        "avg_days_open",
                        "Avg days open",
                        "Average lifetime (days) from creation to now for tickets in the segment.",
                    )
                )
            if "time_to_resolution_days" in working.columns:
                aggregations["avg_time_to_resolution_days"] = ("time_to_resolution_days", "mean")
                metric_meta.append(
                    (
                        "avg_time_to_resolution_days",
                        "Avg resolution (days)",
                        "Average days from creation to resolution for resolved tickets in the segment.",
                    )
                )
            if "activity_score_weighted" in working.columns:
                aggregations["avg_activity_score"] = ("activity_score_weighted", "mean")
                metric_meta.append(
                    (
                        "avg_activity_score",
                        "Avg activity score",
                        "Average weighted activity score (comments, status, other).",
                    )
                )
            # Time Lost (hours) sum if available
            time_lost_col = None
            if "time_lost_value" in working.columns:
                time_lost_col = "time_lost_value"
            elif "time_lost" in working.columns:
                time_lost_col = "time_lost"
            if time_lost_col:
                aggregations["time_lost_sum"] = (time_lost_col, "sum")
                metric_meta.append(
                    (
                        "time_lost_sum",
                        "Time Lost (hours)",
                        "Total time lost reported for the segment.",
                    )
                )
            # Activity (range) = comments_in_range + status_changes + other_changes
            # Use sums over the selected window
            have_activity_parts = all(
                c in working.columns for c in ("comments_in_range", "status_changes", "other_changes")
            )
            if have_activity_parts:
                aggregations["comments_in_range_sum"] = ("comments_in_range", "sum")
                aggregations["status_changes_sum"] = ("status_changes", "sum")
                aggregations["other_changes_sum"] = ("other_changes", "sum")

            if not aggregations:
                return pd.DataFrame(), seg_field.replace("_", " ").title(), []

            grouped = working.groupby(seg_field).agg(**aggregations).reset_index()
            # Compose activity_sum from parts and tidy up
            if (
                "comments_in_range_sum" in grouped.columns
                or "status_changes_sum" in grouped.columns
                or "other_changes_sum" in grouped.columns
            ):
                part_cols = [
                    c
                    for c in (
                        "comments_in_range_sum",
                        "status_changes_sum",
                        "other_changes_sum",
                    )
                    if c in grouped.columns
                ]
                if part_cols:
                    grouped["activity_sum"] = grouped[part_cols].sum(axis=1)
                    grouped = grouped.drop(columns=part_cols)
                    metric_meta.append(
                        (
                            "activity_sum",
                            "Activity (range)",
                            "Combined comments, status, and other changes in the selected window.",
                        )
                    )
            seg_label = seg_field.replace("_", " ").title()
            grouped = grouped.rename(columns={seg_field: seg_label})
            return grouped, seg_label, metric_meta

        for tab, seg_field in zip(seg_tabs, segment_options, strict=False):
            with tab:
                agg_df, seg_label, metrics = _build_segment_agg(seg_field)
                if agg_df.empty or not metrics:
                    st.info("No aggregation data available for this segment.")
                    continue
                # Build metric sub-tabs based on available columns
                metric_tabs = st.tabs([title for (_col, title, _help) in metrics])
                for mtab, (col, title, _help) in zip(metric_tabs, metrics, strict=False):
                    with mtab:
                        if col not in agg_df.columns:
                            st.info("Metric not available for this segment.")
                            continue
                        # Sort by selected metric and limit Top N
                        work = agg_df.sort_values(by=col, ascending=False).head(st.session_state.top_n)
                        order_list = work[seg_label].astype(str).tolist()
                        # Horizontal bar chart for readability
                        chart = (
                            alt.Chart(work)
                            .mark_bar(color="#4472C4")
                            .encode(
                                y=alt.Y(f"{seg_label}:N", sort=order_list, title=seg_label),
                                x=alt.X(f"{col}:Q", title=title),
                                tooltip=[
                                    seg_label,
                                    (
                                        alt.Tooltip(f"{col}:Q", title=title, format=".1f")
                                        if work[col].dtype.kind in "fc"
                                        else alt.Tooltip(f"{col}:Q", title=title)
                                    ),
                                ],
                            )
                            .properties(height=alt.Step(22))
                        )
                        st.altair_chart(chart, width="stretch")
                        # Caption below chart explaining the metric
                        metric_help_map = {m_col: m_help for (m_col, _m_title, m_help) in metrics}
                        desc = metric_help_map.get(col) or ""
                        if desc:
                            st.caption(desc)

        st.markdown("---")
        st.subheader("Assignee Spotlight")
        assignee_series = df["assignee"].fillna("(Unassigned)")
        assignee_options = sorted([a for a in assignee_series.unique() if a])
        if not assignee_options:
            st.info("No assignees available for this dataset.")
        else:
            status_series = df["status"].astype(str).apply(normalize_workflow_status)
            open_mask = ~status_series.isin(TERMINAL_STATUSES)
            open_counts = assignee_series[open_mask].value_counts()
            top_assignee = open_counts.idxmax() if not open_counts.empty else assignee_options[0]

            if (
                "assignee_spotlight" not in st.session_state
                or st.session_state.assignee_spotlight not in assignee_options
            ):
                st.session_state.assignee_spotlight = top_assignee
            selected_assignee = st.selectbox(
                "Choose an assignee to view open tickets",
                options=assignee_options,
                key="assignee_spotlight",
            )

            assignee_mask = assignee_series == selected_assignee
            spotlight_df = df[open_mask & assignee_mask].copy()
            if spotlight_df.empty:
                st.info("No open tickets for this assignee in the selected window.")
            else:
                if "days_since_update" in spotlight_df.columns:
                    spotlight_df["days_since_update"] = pd.to_numeric(
                        spotlight_df["days_since_update"], errors="coerce"
                    ).round(1)
                st.caption(f"{len(spotlight_df)} open ticket(s) assigned to {selected_assignee}.")
                display_ticket_list(
                    spotlight_df,
                    server=server_url,
                    tz=tz,
                    date_col="updated_dt" if "updated_dt" in spotlight_df.columns else "updated",
                    date_label="updated",
                    empty_message="No open tickets for this assignee in the selected window.",
                    sort_by="updated",
                    ascending=False,
                )

    with trends_tab:
        st.caption("Time-series charts for issue creation and priority-based activity trends.")

        st.subheader("Created Issues Over Time")
        if created_chart:
            st.altair_chart(created_chart, width="stretch")
            st.caption("Shows all issues created in the selected date window, regardless of priority.")
        else:
            st.info("No created issues in the selected period.")

        st.markdown("---")
        st.subheader("Priority Updates Over Time")

        # Determine smart defaults for priority filter:
        # 1. Blocker + Critical (or one if only one exists)
        # 2. Fallback to Urgent if neither exists
        # 3. Otherwise pick first available priority from data
        def _get_default_trend_priorities(available: list[str]) -> list[str]:
            defaults = []
            for p in ["Blocker", "Critical"]:
                if p in available:
                    defaults.append(p)
            if defaults:
                return defaults
            if "Urgent" in available:
                return ["Urgent"]
            # Fallback: return first available or empty
            return available[:1] if available else []

        # Compute smart default: Blocker/Critical, fallback to Urgent, then first available
        default_trend_prios = _get_default_trend_priorities(available_priorities)

        # Simple multiselect with smart default
        selected_trend_priorities = st.multiselect(
            "Filter by priorities",
            options=available_priorities,
            default=default_trend_prios,
            help="Select which priorities to include in the updates chart below.",
        )

        # Generate chart with selected priorities
        if selected_trend_priorities:
            priority_chart = priority_updates_trend(
                df, start_dt, end_dt, priorities=selected_trend_priorities
            )
            if priority_chart:
                st.altair_chart(priority_chart, width="stretch")
                st.caption(
                    f"Shows updates to issues with selected priorities "
                    f"({', '.join(selected_trend_priorities)}) in the date window."
                )
            else:
                st.info("No updates for the selected priorities in this period.")
        else:
            st.info("Select at least one priority to see the updates chart.")

        if created_events is not None and not created_events.empty:
            st.subheader("Drilldown: Tickets Created On...")
            recent_created = (
                created_events["created_dt"].max().date() if "created_dt" in created_events.columns else None
            )
            # One-time initialization of drilldown date
            # to latest created ticket (or end date fallback)
            min_date = st.session_state.start_date
            max_date = st.session_state.end_date

            def _clamp_date(candidate: date | None, *, lower: date, upper: date) -> date:
                if candidate is None:
                    return upper
                if candidate < lower:
                    return lower
                if candidate > upper:
                    return upper
                return candidate

            if "created_drilldown_date" not in st.session_state:
                st.session_state.created_drilldown_date = _clamp_date(
                    recent_created or st.session_state.end_date,
                    lower=min_date,
                    upper=max_date,
                )
            else:
                stored_date = st.session_state.created_drilldown_date
                st.session_state.created_drilldown_date = _clamp_date(
                    stored_date,
                    lower=min_date,
                    upper=max_date,
                )
            # Use existing state value; avoid passing a dynamic 'value='
            # each rerun to prevent tab focus reset
            drill_date = st.date_input(
                "Select date",
                min_value=min_date,
                max_value=max_date,
                key="created_drilldown_date",
            )
            drill_df = created_events.copy()
            drill_df["created_date"] = drill_df["created_dt"].dt.date
            filtered = drill_df[drill_df["created_date"] == drill_date]
            if not filtered.empty:
                st.markdown(
                    (
                        "<span style='display:inline-block;padding:0.25rem 0.75rem;"
                        "background-color:#1f77b4;color:white;border-radius:999px;"
                        "font-size:0.8rem;font-weight:600;'>"
                        f"{len(filtered)} ticket(s)</span>"
                    ),
                    unsafe_allow_html=True,
                )
                drill_display = filtered.copy()
                # Map created_dt into the canonical 'created' slot if needed
                if "created_dt" in drill_display.columns and "created" not in drill_display.columns:
                    drill_display["created"] = drill_display["created_dt"]
                # Round canonical numeric age columns if present
                for col in ("days_open", "days_since_update"):
                    if col in drill_display.columns:
                        drill_display[col] = pd.to_numeric(drill_display[col], errors="coerce").round(1)
                prepared_drill, drill_cols, drill_cfg = prepare_ticket_table(drill_display, server_url)
                st.dataframe(
                    prepared_drill[drill_cols],
                    hide_index=True,
                    column_config=drill_cfg,
                    width="stretch",
                )
            else:
                st.info("No tickets were created on that date.")

    with wordclouds_tab:
        st.caption("Word cloud of issue summaries to visualize common themes in the loaded dataset.")
        st.subheader("Summary Word Cloud")
        text = " ".join(df["summary"].dropna())
        render_wordcloud(text)
