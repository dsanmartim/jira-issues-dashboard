"""Leaderboard and ticket list display utilities.

This module provides functions for displaying top-N leaderboards
and ticket lists with consistent formatting.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from jira_app.visual.column_metadata import apply_column_metadata
from jira_app.visual.tables import prepare_ticket_table


def _normalize_name(name: str) -> str:
    """Normalize a column name for comparison."""
    return name.strip().lower().replace(" ", "_") if isinstance(name, str) else ""


# Canonical column ordering for ticket tables
CANONICAL_TAIL_COLUMNS = [
    "priority",
    "assignee",
    "status",
    "time_lost",
    "days_open",
    "days_since_update",
    "created",
    "updated",
    "labels",
    "obs_system",
    "obs_subsystem",
    "obs_component",
    "reporter",
]


def display_leaderboard(
    df: pd.DataFrame | None,
    *,
    title: str,
    server: str,
    top_n: int,
    metric_col: str | None,
    metric_label: str | None = None,
    extra_cols: list[str] | None = None,
    caption: str | None = None,
) -> None:
    """Display a leaderboard table with top-N tickets.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame containing ticket data.
    title : str
        Section title to display.
    server : str
        Jira server URL for ticket links.
    top_n : int
        Number of top entries to display.
    metric_col : str or None
        Column to sort by (descending).
    metric_label : str, optional
        Display label for the metric column.
    extra_cols : list[str], optional
        Additional columns to include.
    caption : str, optional
        Caption text below the table.
    """
    st.markdown(f"##### {title}")
    if df is None or df.empty:
        st.info("No data available for this view.")
        return

    work = df.copy()
    display_metric: str | None = None
    if metric_col and metric_col in work.columns:
        work = work.sort_values(metric_col, ascending=False)
        display_metric = metric_label or metric_col
        if metric_label and metric_label != metric_col:
            work = work.rename(columns={metric_col: display_metric})
    elif metric_label and metric_label in work.columns:
        display_metric = metric_label

    trimmed = work.head(top_n).copy()
    if trimmed.empty:
        st.info("No data available for this view.")
        return

    additional_cols: list[str] = []
    raw_specific: list[str] = []
    seen_extra_keys: set[str] = set()

    if display_metric and display_metric in trimmed.columns:
        key = _normalize_name(display_metric)
        if key not in seen_extra_keys:
            additional_cols.append(display_metric)
            raw_specific.append(display_metric)
            seen_extra_keys.add(key)

    if extra_cols:
        for col in extra_cols:
            if col and col in trimmed.columns:
                key = _normalize_name(col)
                if key not in seen_extra_keys:
                    additional_cols.append(col)
                    raw_specific.append(col)
                    seen_extra_keys.add(key)

    numeric_one_decimal = {
        "days_open",
        "days_since_update",
        "time_lost",
        "time_lost_value",
        "Time Lost",
        "Weighted Score",
    }
    for col in numeric_one_decimal:
        if col in trimmed.columns:
            trimmed[col] = pd.to_numeric(trimmed[col], errors="coerce").round(1)

    prepared, base_display_cols, cfg = prepare_ticket_table(trimmed, server, extra_columns=additional_cols)

    display_cols: list[str] = []
    available_cols = set(prepared.columns)
    selected_keys: set[str] = set()

    def _append_if_present(col: str):
        normalized = _normalize_name(col)
        if col in available_cols and normalized not in selected_keys:
            display_cols.append(col)
            selected_keys.add(normalized)

    _append_if_present("Ticket")
    _append_if_present("summary")

    canonical_tail_keys = {_normalize_name(col) for col in CANONICAL_TAIL_COLUMNS}

    specific_metrics: list[str] = [
        col for col in raw_specific if _normalize_name(col) not in canonical_tail_keys
    ]

    for col in specific_metrics:
        _append_if_present(col)

    # Map normalized names to actual labels so renamed canonical columns keep order
    normalized_available = {_normalize_name(col): col for col in prepared.columns}
    for col in CANONICAL_TAIL_COLUMNS:
        actual = normalized_available.get(_normalize_name(col))
        if actual:
            _append_if_present(actual)

    for col in base_display_cols:
        _append_if_present(col)

    column_config = apply_column_metadata(display_cols, cfg)

    st.dataframe(
        prepared[display_cols],
        hide_index=True,
        column_config=column_config,
        width="stretch",
    )

    if caption:
        st.caption(caption)


def display_ticket_list(
    df: pd.DataFrame | None,
    *,
    server: str,
    tz,
    date_col: str | None,
    date_label: str | None,
    empty_message: str,
    sort_by: str | None = None,
    ascending: bool = True,
    extra_columns: list[str] | None = None,
    height: int | str | None = None,
    reorder_like_topn: bool = False,
    caption: str | None = None,
) -> None:
    """Display a ticket list table.

    Parameters
    ----------
    df : pd.DataFrame or None
        DataFrame containing ticket data.
    server : str
        Jira server URL for ticket links.
    tz : timezone
        Timezone for date formatting.
    date_col : str or None
        Column containing dates to format.
    date_label : str or None
        Display label for the date column.
    empty_message : str
        Message to show when no data available.
    sort_by : str, optional
        Column to sort by.
    ascending : bool
        Sort order (default True).
    extra_columns : list[str], optional
        Additional columns to include.
    height : int or str, optional
        Table height. If None, auto-calculated.
    reorder_like_topn : bool
        If True, reorder columns like leaderboard tables.
    caption : str, optional
        Caption text below the table.
    """
    if df is None or df.empty:
        st.info(empty_message)
        return

    table = df.copy()
    target_label = date_label
    if date_col and date_col in table.columns and target_label:
        series = table[date_col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, errors="coerce", utc=True)
            series = series.dt.tz_convert(tz)
        else:
            tzinfo = getattr(series.dt, "tz", None)
            series = series.dt.tz_localize(tz) if tzinfo is None else series.dt.tz_convert(tz)
        table[target_label] = series.dt.strftime("%Y-%m-%d %H:%M")
        if date_col != target_label:
            table = table.drop(columns=[date_col], errors="ignore")
    elif date_col and date_col in table.columns:
        target_label = date_col

    if sort_by and sort_by in table.columns:
        table = table.sort_values(by=sort_by, ascending=ascending)

    additional_cols: list[str] = []
    if extra_columns:
        additional_cols.extend(extra_columns)
    if target_label:
        additional_cols.append(target_label)

    prepared, base_display_cols, cfg = prepare_ticket_table(table, server, extra_columns=additional_cols)
    if height is None:
        # Default height sized to top-N rows to keep tables compact with scroll for overflow
        target_rows = int(st.session_state.get("top_n", 15)) if "top_n" in st.session_state else 15
        visible_rows = min(target_rows, len(prepared))
        resolved_height: int | str = max(200, 46 + visible_rows * 34)
    else:
        resolved_height = height

    # Optional: reorder columns to match the Top N ordering style
    if reorder_like_topn:
        display_cols: list[str] = []
        available_cols = set(prepared.columns)
        selected_keys: set[str] = set()

        def _append_if_present(col: str):
            normalized = _normalize_name(col)
            if col in available_cols and normalized not in selected_keys:
                display_cols.append(col)
                selected_keys.add(normalized)

        _append_if_present("Ticket")
        _append_if_present("summary")

        # Treat extra_columns as the specific metrics to surface next
        if extra_columns:
            for col in extra_columns:
                _append_if_present(col)

        for col in CANONICAL_TAIL_COLUMNS:
            _append_if_present(col)

        for col in base_display_cols:
            _append_if_present(col)

        final_cols = [c for c in display_cols if c in prepared.columns]

        # Merge human-readable labels and hover help like Top Tickets tables
        column_config = apply_column_metadata(final_cols, cfg)

        st.dataframe(
            prepared[final_cols],
            hide_index=True,
            column_config=column_config,
            width="stretch",
            height=resolved_height,
        )
        if caption:
            st.caption(caption)
        return

    # Default: existing display behavior
    column_config = apply_column_metadata(base_display_cols, cfg)
    st.dataframe(
        prepared[base_display_cols],
        hide_index=True,
        column_config=column_config,
        width="stretch",
        height=resolved_height,
    )
    if caption:
        st.caption(caption)
