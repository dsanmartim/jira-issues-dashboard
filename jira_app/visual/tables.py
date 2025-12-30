"""Reusable table helpers for Streamlit rendering."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from jira_app.core.column_config import get_columns
from jira_app.core.config import DISPLAY_ORDER_DETAIL


def add_ticket_link(df: pd.DataFrame, server: str, key_col: str = "key", label: str = "Ticket"):
    if df.empty or key_col not in df.columns:
        return df, {}
    out = df.copy()
    base = server.rstrip("/")
    out[label] = out[key_col].astype(str).apply(lambda k: f"{base}/browse/{k}" if k and k != "nan" else "")
    cfg = {
        label: st.column_config.LinkColumn(
            label,
            display_text=r"browse/(.*)$",
            help="Open in Jira",
            width="medium",
        )
    }
    return out, cfg


def prepare_ticket_table(
    df: pd.DataFrame,
    server: str,
    *,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    if df.empty:
        return df, [], {}

    table, cfg = add_ticket_link(df, server)
    canonical = get_columns("ticket_list") or []
    display_cols: list[str] = [col for col in canonical if col in table.columns]

    if extra_columns:
        for col in extra_columns:
            if col in table.columns and col not in display_cols:
                display_cols.append(col)

    if "Ticket" in table.columns and "Ticket" not in display_cols:
        display_cols.insert(0, "Ticket")

    if not display_cols:
        display_cols = [col for col in table.columns if col != "key"]

    return table, display_cols, cfg


def render_issue_table(df: pd.DataFrame, server: str, limit: int = 1000):
    linked, cfg = add_ticket_link(df, server)
    cols = [c for c in DISPLAY_ORDER_DETAIL if c in linked.columns]
    st.dataframe(linked[cols].head(limit), hide_index=True, column_config=cfg)
