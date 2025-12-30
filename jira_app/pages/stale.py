"""Stale tickets page.

Fetches open project issues (or reuses cached) and computes stale tickets
based on days since last update.
"""

from __future__ import annotations

import pandas as pd
import pytz
import streamlit as st

from jira_app.app import register_page
from jira_app.core.service import IssueService
from jira_app.core.stale import compute_stale
from jira_app.visual.column_metadata import apply_column_metadata
from jira_app.visual.progress import ProgressReporter
from jira_app.visual.tables import prepare_ticket_table


@register_page("Stale Tickets")
def stale_page():
    st.title("Stale Tickets")
    st.caption("Identify OBS tickets that haven't been updated recently.")
    service: IssueService | None = st.session_state.get("issue_service")
    if service is None:
        st.warning("Initialize connection on Setup page first.")
        return
    project = "OBS"
    st.session_state["project_key"] = project
    stale_days = st.number_input(
        "Mark stale if no update in N days",
        min_value=1,
        value=30,
        step=1,
    )
    refresh = st.button("Fetch Stale", type="primary")
    tz = pytz.timezone("America/Santiago")

    if refresh:
        reporter = ProgressReporter(f"Fetching open issues for {project}")
        try:
            base = service.fetch_project_open(project, progress=reporter.callback)
            if base.empty:
                reporter.complete("No open issues found.")
                st.info("No open issues found.")
                return
            reporter.update("Evaluating tickets for staleness")
            # Derive days_since_update / days_open if not present
            df = base.copy()
            if "updated" in df.columns:
                df["updated_dt"] = pd.to_datetime(df["updated"], errors="coerce", utc=True).dt.tz_convert(tz)
                df["days_since_update"] = (pd.Timestamp.now(tz) - df["updated_dt"]).dt.days
            if "created" in df.columns:
                df["created_dt"] = pd.to_datetime(df["created"], errors="coerce", utc=True).dt.tz_convert(tz)
                df["days_open"] = (pd.Timestamp.now(tz) - df["created_dt"]).dt.days
            stale_df = compute_stale(df, stale_days)
            st.session_state["stale_df"] = stale_df
            reporter.complete(f"Computed stale metrics for {len(stale_df)} ticket(s).")
        except Exception as exc:  # pragma: no cover
            reporter.error(f"Failed to fetch stale tickets: {exc}")
            raise

    stale_df = st.session_state.get("stale_df", pd.DataFrame())
    if stale_df.empty:
        st.info("No stale tickets computed yet.")
        return
    display_df = stale_df.copy()
    for col in ("days_open", "days_since_update"):
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(1)

    server = st.session_state.get("jira_server", "")
    prepared, display_cols, cfg = prepare_ticket_table(display_df, server)
    st.markdown("---")
    st.caption("Open tickets exceeding the selected inactivity threshold.")
    column_config = apply_column_metadata(display_cols, cfg)
    st.dataframe(prepared[display_cols], hide_index=True, column_config=column_config)
    csv = prepared[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Stale CSV",
        data=csv,
        file_name=f"jira_stale_{project}.csv",
        mime="text/csv",
    )
