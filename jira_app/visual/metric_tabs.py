"""Shared metric tabs rendering for persona pages."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def render_metric_tabs(
    data: pd.DataFrame,
    *,
    entity_col: str,
    metrics: list[dict[str, object]],
    top_n_key: str,
    top_label: str,
    default_top: int = 15,
) -> None:
    """Render tabbed bar charts for each metric definition provided.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing entity and metric columns.
    entity_col : str
        Name of the column containing entity names (e.g., "Assignee", "Reporter").
    metrics : list[dict]
        List of metric definitions. Each dict should contain:
        - column: str - Column name in the DataFrame
        - label: str - Display label for the metric (optional)
        - help: str - Help text caption (optional)
        - kind: str - Value type: "int", "float", or "percent" (optional, default "int")
        - sort_desc: bool - Sort descending (optional, default True)
        - color: str - Bar color (optional, default "#4472C4")
        - axis_format: str - Custom axis format (optional)
        - tooltip_format: str - Custom tooltip format (optional)
        - row_height: int - Height per row (optional, default 22)
    top_n_key : str
        Session state key for the top N selector.
    top_label : str
        Label for the top N input widget.
    default_top : int
        Default value for top N (default 15).
    """
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

            st.altair_chart(chart, width="stretch")
            if help_text:
                st.caption(help_text)
