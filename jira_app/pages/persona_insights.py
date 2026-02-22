"""Unified Assignee | Reporter Insights page.

This module owns the Streamlit page registration and shared data acquisition UI.
Persona-specific tab renderers live in `assignees.py` and `reporter.py` to keep
concerns separated and ease future extension (e.g., adding a "Reviewer" persona).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytz
import streamlit as st
from jira import JIRAError

from jira_app.app import register_page
from jira_app.core.config import (
    DEFAULT_DATE_RANGE_DAYS,
    DEFAULT_HISTORY_FALLBACK_DAYS,
    DEFAULT_PROJECT_KEY,
    TIMEZONE,
)
from jira_app.core.service import ActivityWeights
from jira_app.pages.assignees import render_assignee_tab
from jira_app.pages.reporter import render_reporter_tab
from jira_app.visual.progress import ProgressReporter

logger = logging.getLogger(__name__)

TZ = pytz.timezone(TIMEZONE)

PAGE_KEY = "persona_insights"  # namespace for session keys to avoid collisions


@register_page("Assignee | Reporter Insights")
def render():  # noqa: D401 - Streamlit entrypoint
    """Render the unified persona insights page with tabs for Assignee and Reporter."""

    st.title("Assignee | Reporter Insights")
    st.caption(
        "Workload, throughput, activity, and stale ticket visibility across Assignee and "
        "Reporter personas."
    )

    issue_service = st.session_state.get("issue_service")
    if issue_service is None:
        st.warning("Initialize the Jira connection on the Setup page to use this view.")
        return

    # Persist project key for downstream components (mirrors earlier implementation)
    st.session_state["project_key"] = DEFAULT_PROJECT_KEY

    # Scope selector: Date range vs Whole project (full history)
    weights = st.session_state.get("dashboard_weights") or ActivityWeights()
    now_tz = datetime.now(TZ)
    data_key = f"{PAGE_KEY}_data"

    scope_options = ["Date range", "Whole project (full history)"]
    scope_session_key = f"{PAGE_KEY}_scope"
    scope_default = st.session_state.get(scope_session_key, scope_options[0])
    scope_index = scope_options.index(scope_default) if scope_default in scope_options else 0
    scope_choice = st.radio(
        "Dataset scope",
        scope_options,
        index=scope_index,
        horizontal=True,
        key=scope_session_key,
        help=(
            "Date range: issues updated / created / resolved in the selected window. "
            "Whole project: all issues in project history (may be large)."
        ),
    )

    start_key = f"{PAGE_KEY}_start_date"
    end_key = f"{PAGE_KEY}_end_date"
    default_start = st.session_state.get(start_key) or st.session_state.get("start_date")
    if default_start is None:
        default_start = (now_tz - timedelta(days=DEFAULT_DATE_RANGE_DAYS)).date()
    default_end = st.session_state.get(end_key) or st.session_state.get("end_date") or now_tz.date()
    date_disabled = scope_choice != "Date range"
    start_date = st.date_input("Start date", value=default_start, key=start_key, disabled=date_disabled)
    end_date = st.date_input("End date", value=default_end, key=end_key, disabled=date_disabled)
    if scope_choice == "Date range" and start_date > end_date:
        st.error("Start date must be on or before the end date.")
        return

    if st.button("Fetch dataset", type="primary", key=f"{PAGE_KEY}_fetch_range"):
        reporter = ProgressReporter("Loading issues for the selected dates")
        try:
            if scope_choice == "Date range":
                start_dt = TZ.localize(datetime.combine(start_date, datetime.min.time()))
                end_dt = TZ.localize(datetime.combine(end_date, datetime.max.time()))
                reporter.update("Requesting updated issues from Jira")
                df = issue_service.fetch_and_enrich_range(
                    DEFAULT_PROJECT_KEY,
                    start_dt,
                    end_dt,
                    weights=weights,
                    progress=reporter.callback,
                )
                mode = "Date range"
            else:  # Whole project full history
                reporter.update("Requesting full project issue set from Jira")
                df_raw = issue_service.fetch_full_project(
                    DEFAULT_PROJECT_KEY,
                    progress=reporter.callback,
                    max_days=None,
                )
                # Determine enrichment span (earliest created to now)
                created_col = None
                if "created_dt" in df_raw.columns:
                    created_col = pd.to_datetime(df_raw["created_dt"], errors="coerce", utc=True)
                elif "created" in df_raw.columns:
                    created_col = pd.to_datetime(df_raw["created"], errors="coerce", utc=True)
                earliest = created_col.min() if created_col is not None else None
                if earliest is None or pd.isna(earliest):
                    earliest = datetime.utcnow() - timedelta(days=DEFAULT_HISTORY_FALLBACK_DAYS)
                if hasattr(earliest, "tz_convert"):
                    start_dt = earliest.tz_convert(TZ)
                else:
                    start_dt = TZ.localize(earliest)
                end_dt = TZ.localize(datetime.combine(now_tz.date(), datetime.max.time()))
                reporter.update("Calculating activity metrics over full history")
                df = issue_service.enrich(df_raw, start_dt, end_dt, weights=weights)
                mode = "Whole project"
            # Performance guard: warn if very large dataset (may slow UI rendering)
            LARGE_DATA_WARN_THRESHOLD = 8000  # heuristic; adjust as performance data collected
            if len(df) > LARGE_DATA_WARN_THRESHOLD:
                warn_msg = f"Large dataset loaded ({len(df)} issues). " "This may impact UI responsiveness."
                st.warning(warn_msg)
            st.session_state[data_key] = {
                "scope": mode,
                "df": df,
                "project": DEFAULT_PROJECT_KEY,
                "start": start_dt,
                "end": end_dt,
                "full_project": mode == "Whole project",
            }
            if mode == "Date range":
                reporter.complete(f"Loaded {len(df)} issues updated between {start_date} and {end_date}.")
            else:
                reporter.complete(f"Loaded full history with {len(df)} issues.")
        except JIRAError as exc:
            error_msg = exc.text if hasattr(exc, "text") else str(exc)
            logger.error("Jira API error fetching persona data: %s", error_msg)
            reporter.error(f"Failed to fetch issues: {error_msg}")
            raise
        except (ValueError, KeyError) as exc:
            logger.error("Data processing error in persona insights: %s", exc)
            reporter.error(f"Failed to process issues: {exc}")
            raise

    data_state = st.session_state.get(data_key)
    if not data_state:
        st.info("Use the controls above to load issue data for this view.")
        return

    df = data_state.get("df")
    if df is None or df.empty:
        st.info("No issue data available under the current selection.")
        return

    # High-level persona highlight metrics (champions)
    # Show top 3 each; fastest resolver uses mean (excludes 0d cycles)
    try:
        # Banner heading + scope clarifier
        start_dt = data_state.get("start")
        end_dt = data_state.get("end")
        st.markdown("---")
        st.subheader("Champion Gallery")
        if data_state.get("scope") == "Date range" and start_dt and end_dt:
            st.caption(f"Top 3 per metric for {start_dt.date()} to {end_dt.date()}.")
        else:
            st.caption("Top 3 per metric across full project history.")
            # Visual badge to signal whole-project scope
            st.markdown(
                "<div style='margin-top:2px'>"
                "<span style='display:inline-block;padding:2px 8px;border-radius:12px;"
                "background:#e0e0e0;font-size:0.70rem;font-weight:600;letter-spacing:0.5px;"
                "color:#333;'>FULL HISTORY</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        kpi_cols = st.columns(6)

        def _format_name(name: str | None, placeholder: str) -> str:
            if name is None or str(name).strip() == "" or pd.isna(name):
                return placeholder
            return str(name)

        def _top_entries(series: pd.Series, n: int, sort_desc: bool = True, round_f: float | None = None):
            if series is None or series.empty:
                return []
            ordered = series.sort_values(ascending=False) if sort_desc else series.sort_values(ascending=True)
            entries = []
            for idx, val in list(ordered.items())[:n]:
                if round_f is not None:
                    val = round(float(val), round_f)
                entries.append((_format_name(idx, "(Unassigned)"), val))
            return entries

        # Prepare assignee-filtered frame (exclude unassigned placeholders)
        df_assignee = pd.DataFrame()
        if "assignee" in df.columns:
            exclude_set = {"", "unassigned", "(unassigned)", "none"}
            assignee_mask = df["assignee"].apply(
                lambda x: not (pd.isna(x) or (isinstance(x, str) and x.strip().lower() in exclude_set))
            )
            df_assignee = df.loc[assignee_mask]

        # Build masks relative to scope (window-restricted or full history)
        created_series_full = None
        if "created_dt" in df.columns:
            created_series_full = pd.to_datetime(df["created_dt"], errors="coerce", utc=True)
        elif "created" in df.columns:
            created_series_full = pd.to_datetime(df["created"], errors="coerce", utc=True)
        resolved_series_full = None
        if "resolution_dt" in df.columns:
            resolved_series_full = pd.to_datetime(df["resolution_dt"], errors="coerce", utc=True)
        is_range_scope = data_state.get("scope") == "Date range"
        start_utc = start_dt.astimezone(pytz.UTC) if start_dt else None
        end_utc = end_dt.astimezone(pytz.UTC) if end_dt else None
        if is_range_scope and created_series_full is not None and start_utc and end_utc:
            created_mask = (
                created_series_full.notna()
                & (created_series_full >= start_utc)
                & (created_series_full <= end_utc)
            )
        else:
            created_mask = (
                created_series_full.notna()
                if created_series_full is not None
                else pd.Series([False] * len(df))
            )
        if is_range_scope and resolved_series_full is not None and start_utc and end_utc:
            resolved_mask = (
                resolved_series_full.notna()
                & (resolved_series_full >= start_utc)
                & (resolved_series_full <= end_utc)
            )
        else:
            resolved_mask = (
                resolved_series_full.notna()
                if resolved_series_full is not None
                else pd.Series([False] * len(df))
            )
        # Snapshot open at end: status not done
        if "status" in df.columns:
            status_series_full = df["status"].astype(str).str.lower()
        else:
            status_series_full = pd.Series([""] * len(df))
        done_set_global = {s.lower() for s in ["done", "resolved", "closed", "cancelled", "completed"]}
        open_at_end_mask = ~status_series_full.isin(done_set_global)

        # Activity score (assignee) across all updated-in-range tickets
        activity_top = []
        if {"assignee", "activity_score_weighted"}.issubset(df_assignee.columns):
            grouped = df_assignee.groupby("assignee")["activity_score_weighted"].sum()
            activity_top = _top_entries(grouped, 3, sort_desc=True, round_f=1)

        # Reporter counts: tickets created in the window
        reporter_top = []
        if {"reporter", "key"}.issubset(df.columns) and created_mask.any():
            grouped = df.loc[created_mask].groupby("reporter")["key"].nunique()
            reporter_top = _top_entries(grouped, 3, sort_desc=True)

        # Commenters
        commenter_top = []
        if {"assignee", "comments_in_range"}.issubset(df_assignee.columns):
            grouped = df_assignee.groupby("assignee")["comments_in_range"].sum()
            commenter_top = _top_entries(grouped, 3, sort_desc=True)

        # Open load snapshot (open at end date)
        open_load_top = []
        if {"assignee", "status", "key"}.issubset(df_assignee.columns):
            open_mask = open_at_end_mask.loc[df_assignee.index]
            grouped = df_assignee.loc[open_mask].groupby("assignee")["key"].nunique()
            open_load_top = _top_entries(grouped, 3, sort_desc=True)

        # Resolved ticket champions: assignees closing the most tickets in scope
        resolver_count_top = []
        if {"assignee", "key"}.issubset(df_assignee.columns) and resolved_mask.any():
            resolved_mask_assignee = resolved_mask.loc[df_assignee.index]
            resolved_df_counts = df_assignee.loc[resolved_mask_assignee]
            if not resolved_df_counts.empty:
                grouped_counts = resolved_df_counts.groupby("assignee")["key"].nunique()
                resolver_count_top = _top_entries(grouped_counts, 3, sort_desc=True)

        # Fastest mean resolver: only tickets resolved in range
        mean_resolver_top = []
        if {"resolution_dt", "assignee"}.issubset(df_assignee.columns) and resolved_mask.any():
            resolved_mask_assignee = resolved_mask.loc[df_assignee.index]
            resolved_df = df_assignee.loc[resolved_mask_assignee]
            if not resolved_df.empty:
                res_series = pd.to_datetime(resolved_df["resolution_dt"], errors="coerce", utc=True)
                if "created_dt" in resolved_df.columns:
                    created_raw = resolved_df["created_dt"]
                elif "created" in resolved_df.columns:
                    created_raw = resolved_df["created"]
                else:
                    created_raw = pd.Series([pd.NaT] * len(resolved_df))
                created_series = pd.to_datetime(created_raw, errors="coerce", utc=True)
                cycle_days = (res_series - created_series).dt.total_seconds() / 86400.0
                resolver_df = pd.DataFrame({"assignee": resolved_df.get("assignee"), "cycle": cycle_days})
                resolver_df = resolver_df[resolver_df["cycle"].notna() & (resolver_df["cycle"] > 0.05)]
                if not resolver_df.empty:
                    means = resolver_df.groupby("assignee")["cycle"].mean()
                    means = means.sort_values()  # ascending: fastest first
                    mean_resolver_top = _top_entries(means, 3, sort_desc=False, round_f=1)

        def _render_top(col, title: str, entries, value_fmt: str, help_text: str):
            if not entries:
                col.metric(title, "–")
                return
            # Show #1 prominently with metric
            name1, val1 = entries[0]
            col.metric(title, value_fmt.format(val1), name1, help=help_text)
            # Show remaining entries (2 & 3) as compact markdown list
            if len(entries) > 1:
                lines = []
                for rank, (nm, vv) in enumerate(entries[1:], start=2):
                    lines.append(f"{rank}. {nm} — {value_fmt.format(vv)}")
                col.caption("\n".join(lines))

        _render_top(
            kpi_cols[0],
            "Assignee Activity (Top 3)",
            activity_top,
            "{:.1f}",
            "Sum of weighted activity events (comments, status changes, other) on issues updated in range.",
        )
        _render_top(
            kpi_cols[1],
            "Reporter Creates (Top 3)",
            reporter_top,
            "{}",
            "Distinct issues created in the selected date range (by reporter).",
        )
        _render_top(
            kpi_cols[2],
            "Comments (Top 3 Assignees)",
            commenter_top,
            "{}",
            "Total comments authored in range on issues updated in range (assignee attribution).",
        )
        _render_top(
            kpi_cols[3],
            "Open Load (Top 3 Assignees)",
            open_load_top,
            "{}",
            "Snapshot at end of range: issues not in a done/closed status (current ownership).",
        )
        _render_top(
            kpi_cols[4],
            "Resolved Tickets (Top 3)",
            resolver_count_top,
            "{}",
            "Distinct tickets resolved in the selected scope (assignee attribution).",
        )

        _render_top(
            kpi_cols[5],
            "Fastest Mean Resolve (Top 3)",
            mean_resolver_top,
            "{:.1f}d",
            "Mean creation→resolution duration for tickets resolved in range; excludes cycles ≤0.05d.",
        )
        # Column set complete; future KPIs can extend via an additional row if needed
    except Exception as kpi_exc:  # pragma: no cover - defensive UI guard
        st.warning(f"KPI banner unavailable: {kpi_exc}")

    st.markdown("---")
    assignee_tab, reporter_tab = st.tabs(["Assignee metrics", "Reporter metrics"])
    server = st.session_state.get("jira_server", "")

    with assignee_tab:
        render_assignee_tab(df, server)
    with reporter_tab:
        render_reporter_tab(df, server)
