"""Stale tickets page.

Fetches open project issues (or reuses cached) and computes stale tickets
based on days since last update.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from jira import JIRAError

from jira_app.analytics.metrics.aging import add_aging_metrics
from jira_app.app import register_page
from jira_app.core.config import DEFAULT_PROJECT_KEY, DEFAULT_STALE_DAYS
from jira_app.core.service import IssueService
from jira_app.core.stale import compute_stale
from jira_app.visual.column_metadata import apply_column_metadata
from jira_app.visual.progress import ProgressReporter
from jira_app.visual.tables import prepare_ticket_table

logger = logging.getLogger(__name__)


@register_page("Stale Tickets")
def stale_page():
    st.title("Stale Tickets")
    st.caption(f"Identify {DEFAULT_PROJECT_KEY} tickets that haven't been updated recently.")
    service: IssueService | None = st.session_state.get("issue_service")
    if service is None:
        st.warning("Initialize connection on Setup page first.")
        return
    project = DEFAULT_PROJECT_KEY
    st.session_state["project_key"] = project
    stale_days = st.number_input(
        "Mark stale if no update in N days",
        min_value=1,
        value=DEFAULT_STALE_DAYS,
        step=1,
    )
    refresh = st.button("Fetch Stale", type="primary")

    if refresh:
        reporter = ProgressReporter(f"Fetching open issues for {project}")
        try:
            base = service.fetch_project_open(project, progress=reporter.callback)
            if base.empty:
                reporter.complete("No open issues found.")
                st.info("No open issues found.")
                return
            reporter.update("Computing aging metrics")
            # Use shared aging metrics (consistent float precision with enrichment pipeline)
            df = add_aging_metrics(base)
            reporter.update("Evaluating tickets for staleness")
            stale_df = compute_stale(df, stale_days)
            st.session_state["stale_df"] = stale_df
            reporter.complete(f"Computed stale metrics for {len(stale_df)} ticket(s).")
        except JIRAError as exc:
            error_msg = exc.text if hasattr(exc, "text") else str(exc)
            logger.error("Jira API error fetching stale tickets: %s", error_msg)
            reporter.error(f"Failed to fetch stale tickets: {error_msg}")
            raise
        except (ValueError, KeyError) as exc:
            logger.error("Data processing error in stale tickets: %s", exc)
            reporter.error(f"Failed to process stale tickets: {exc}")
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
