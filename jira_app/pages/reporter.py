"""Reporter-specific metrics rendering used by the combined persona insights page.

Like `assignees.py` this module no longer registers its own Streamlit page. The
unified page is implemented in `persona_insights.py` which imports the
`render_reporter_tab` function defined here.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from jira_app.analytics.persona.metrics import (
    build_activity_summary,
    build_open_workload,
    build_resolution_metrics,
    prepare_open_ticket_detail,
)
from jira_app.visual.column_metadata import apply_column_metadata
from jira_app.visual.tables import prepare_ticket_table

REPORTER_PLACEHOLDER = "(Unknown reporter)"


def _build_reporter_open_workload(df: pd.DataFrame) -> pd.DataFrame:  # internal helper
    return build_open_workload(
        df,
        entity_column="reporter",
        entity_label="Reporter",
        placeholder=REPORTER_PLACEHOLDER,
        include_priority=False,
    )


def _build_reporter_resolved_metrics(df: pd.DataFrame) -> pd.DataFrame:  # internal helper
    return build_resolution_metrics(
        df,
        entity_column="reporter",
        entity_label="Reporter",
        placeholder=REPORTER_PLACEHOLDER,
    )


def _build_reporter_activity_summary(df: pd.DataFrame) -> pd.DataFrame:  # internal helper
    return build_activity_summary(
        df,
        entity_column="reporter",
        entity_label="Reporter",
        placeholder=REPORTER_PLACEHOLDER,
    )


def _render_metric_tabs(
    data: pd.DataFrame,
    *,
    entity_col: str,
    metrics: list[dict[str, object]],
    top_n_key: str,
    top_label: str,
    default_top: int = 15,
) -> None:
    """Render tabbed bar charts for each metric definition provided."""

    available_metrics = [metric for metric in metrics if metric.get("column") in data.columns]
    if not available_metrics:
        st.info("No metrics available for visualization.")
        return

    max_entries = len(data)
    if max_entries == 0:
        st.info("No data available for visualization.")
        return

    default_val = min(default_top, max_entries) if max_entries else 1
    top_n = st.number_input(
        top_label,
        min_value=1,
        max_value=max_entries,
        value=max(default_val, 1),
        step=1,
        key=top_n_key,
    )
    top_n = int(top_n)

    tab_labels = [str(metric.get("label", metric["column"])) for metric in available_metrics]
    metric_tabs = st.tabs(tab_labels)

    for tab, metric in zip(metric_tabs, available_metrics, strict=False):
        column = str(metric["column"])
        label = str(metric.get("label", column))
        help_text = str(metric.get("help", ""))
        sort_desc = bool(metric.get("sort_desc", True))
        value_kind = str(metric.get("kind", "int"))
        color = str(metric.get("color", "#4472C4"))
        axis_format = metric.get("axis_format")
        tooltip_format = metric.get("tooltip_format")

        with tab:
            chart_df = data[[entity_col, column]].copy()
            chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")
            chart_df = chart_df.dropna(subset=[column])
            if chart_df.empty:
                st.info("Metric not available for the current selection.")
                continue

            chart_df = chart_df.sort_values(column, ascending=not sort_desc).head(top_n)
            order_list = chart_df[entity_col].astype(str).tolist()

            axis_kwargs = {}
            if axis_format is not None:
                axis_kwargs["axis"] = alt.Axis(format=str(axis_format))
            else:
                if value_kind == "int":
                    axis_kwargs["axis"] = alt.Axis(format=",d")
                elif value_kind == "percent":
                    axis_kwargs["axis"] = alt.Axis(format=".1f")
                else:
                    axis_kwargs["axis"] = alt.Axis(format=".1f")

            if tooltip_format is None:
                if value_kind == "int":
                    tooltip_format = ",d"
                elif value_kind == "percent":
                    tooltip_format = ".1f"
                else:
                    tooltip_format = ".1f"

            chart = (
                alt.Chart(chart_df)
                .mark_bar(color=color)
                .encode(
                    y=alt.Y(f"{entity_col}:N", sort=order_list, title=entity_col),
                    x=alt.X(f"{column}:Q", title=label, **axis_kwargs),
                    tooltip=[
                        alt.Tooltip(f"{entity_col}:N", title=entity_col),
                        alt.Tooltip(f"{column}:Q", title=label, format=str(tooltip_format)),
                    ],
                )
                .properties(height=alt.Step(int(metric.get("row_height", 22))))
            )

            st.altair_chart(chart, use_container_width=True)
            if help_text:
                st.caption(help_text)


def _prepare_reporter_ticket_detail(df: pd.DataFrame) -> pd.DataFrame:  # internal helper
    detail = prepare_open_ticket_detail(
        df,
        entity_column="reporter",
        placeholder=REPORTER_PLACEHOLDER,
        include_status_filter=True,
    )
    if "reporter" in detail.columns:
        detail["reporter"] = detail["reporter"].fillna(REPORTER_PLACEHOLDER)
    return detail


def render_reporter_tab(df: pd.DataFrame, server: str) -> None:
    """Render the reporter metrics tab contents.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded issue dataset.
    server : str
        Jira base URL used for link construction in ticket tables.
    """

    st.subheader("Open workload by reporter")
    reporter_open = _build_reporter_open_workload(df)
    if reporter_open.empty:
        st.info("No open tickets are associated with reporters in the current dataset.")
    else:
        reporter_open = reporter_open.drop(columns=["Median days open"], errors="ignore")
        metric_defs = [
            {
                "column": "Open tickets",
                "label": "Open tickets",
                "help": "Count of open tickets reported by this person.",
                "kind": "int",
                "sort_desc": True,
            },
            {
                "column": "Avg days open",
                "label": "Average days open",
                "help": "Average age of the reporter's open tickets.",
                "kind": "float",
                "sort_desc": True,
            },
            {
                "column": "Oldest open (days)",
                "label": "Oldest open ticket (days)",
                "help": "Age of the oldest open ticket reported by this person.",
                "kind": "float",
                "sort_desc": True,
            },
            {
                "column": "% updated ≤7d",
                "label": "% updated within 7 days",
                "help": "Percent of the reporter's open tickets updated within the past 7 days.",
                "kind": "percent",
                "sort_desc": True,
            },
        ]

        _render_metric_tabs(
            reporter_open,
            entity_col="Reporter",
            metrics=metric_defs,
            top_n_key="reporter_open_top_n",
            top_label="Show top N reporters",
        )
        st.caption(
            "Snapshot of open-ticket load per reporter with update recency to highlight potential staleness."
        )

    st.markdown("---")
    st.subheader("Resolved output")
    reporter_resolved = _build_reporter_resolved_metrics(df)
    if reporter_resolved.empty:
        st.info("No resolved tickets are available for reporter-level metrics in the dataset.")
    else:
        reporter_resolved = reporter_resolved.drop(columns=["Median resolution (days)"], errors="ignore")
        metric_defs = [
            {
                "column": "Resolved tickets",
                "label": "Resolved tickets",
                "help": "Count of tickets resolved that were opened by this reporter.",
                "kind": "int",
                "sort_desc": True,
            },
            {
                "column": "Mean resolution (days)",
                "label": "Mean resolution time (days)",
                "help": "Average creation-to-resolution time for this reporter's tickets; lower is better.",
                "kind": "float",
                "sort_desc": False,
            },
            {
                "column": "90th percentile (days)",
                "label": "90th percentile resolution (days)",
                "help": "90% of the reporter's tickets resolved within this many days; lower is better.",
                "kind": "float",
                "sort_desc": False,
            },
        ]

        _render_metric_tabs(
            reporter_resolved,
            entity_col="Reporter",
            metrics=metric_defs,
            top_n_key="reporter_resolved_top_n",
            top_label="Show top N reporters",
        )
        st.caption(
            "Throughput indicators derived from tickets resolved within the loaded dataset, "
            "grouped by reporting user."
        )

    st.markdown("---")
    st.subheader("Activity summary")
    reporter_activity = _build_reporter_activity_summary(df)
    if reporter_activity.empty:
        st.info("Activity data is not available for reporter-level aggregation.")
    else:
        metric_defs = [
            {
                "column": "Tickets",
                "label": "Tickets",
                "help": "Number of tickets attributed to the reporter in this dataset.",
                "kind": "int",
                "sort_desc": True,
            },
            {
                "column": "Comments (range)",
                "label": "Comments in range",
                "help": "Comments recorded in the selected window on the reporter's tickets.",
                "kind": "int",
                "sort_desc": True,
            },
            {
                "column": "Status changes",
                "label": "Status changes",
                "help": "Workflow transitions captured in the selected range.",
                "kind": "int",
                "sort_desc": True,
            },
            {
                "column": "Other changes",
                "label": "Other changes",
                "help": "Other field updates applied within the range.",
                "kind": "int",
                "sort_desc": True,
            },
            {
                "column": "Total weighted score",
                "label": "Total weighted score",
                "help": "Sum of weighted activity credited to the reporter's tickets.",
                "kind": "float",
                "sort_desc": True,
            },
            {
                "column": "Avg weighted score",
                "label": "Average weighted score",
                "help": "Average weighted activity score per ticket.",
                "kind": "float",
                "sort_desc": True,
            },
        ]

        _render_metric_tabs(
            reporter_activity,
            entity_col="Reporter",
            metrics=metric_defs,
            top_n_key="reporter_activity_top_n",
            top_label="Show top N reporters",
        )
        st.caption(
            "Aggregated engagement (comments, status changes, other updates, weighted scores) "
            "credited to tickets opened by each reporter."
        )

    st.markdown("---")
    st.subheader("Open ticket details")
    # Backward compat: migrate pre-refactor session keys if present
    if (
        "reporter_detail_filter" in st.session_state
        and "persona_reporter_detail_filter" not in st.session_state
    ):
        st.session_state["persona_reporter_detail_filter"] = st.session_state["reporter_detail_filter"]
    if (
        "reporter_detail_limit" in st.session_state
        and "persona_reporter_detail_limit" not in st.session_state
    ):
        st.session_state["persona_reporter_detail_limit"] = st.session_state["reporter_detail_limit"]

    reporter_detail = _prepare_reporter_ticket_detail(df)
    if reporter_detail.empty:
        st.info("No open ticket details are available for reporter-level review.")
    else:
        reporter_options = sorted(reporter_detail["reporter"].unique())
        default_reporters = reporter_options if len(reporter_options) <= 5 else reporter_options[:5]
        selected_reporters = st.multiselect(
            "Filter reporters",
            options=reporter_options,
            default=default_reporters,
            key="persona_reporter_detail_filter",
        )
        if selected_reporters:
            reporter_filtered = reporter_detail[reporter_detail["reporter"].isin(selected_reporters)]
        else:
            reporter_filtered = reporter_detail

        reporter_limit = st.slider(
            "Show top N by days open",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="persona_reporter_detail_limit",
        )
        reporter_filtered = reporter_filtered.sort_values("days_open", ascending=False).head(reporter_limit)
        if reporter_filtered.empty:
            st.info("No tickets match the current reporter filters.")
        else:
            prepared, display_cols, cfg = prepare_ticket_table(reporter_filtered, server)
            column_config = apply_column_metadata(display_cols, cfg)
            st.dataframe(
                prepared[display_cols],
                hide_index=True,
                use_container_width=True,
                column_config=column_config,
            )
            st.caption(
                "Filtered list of the longest-open tickets, sorted by age to spotlight "
                "where follow-up is needed."
            )
