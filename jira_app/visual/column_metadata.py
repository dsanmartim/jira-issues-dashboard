"""Central column metadata and helpers for table rendering."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import streamlit as st

# Mapping of raw column keys to (label, help text, format key)
# format key: "int" -> integer, "float1" -> 1 decimal float, None -> default text column
COLUMN_METADATA: dict[str, tuple[str, str, str | None]] = {
    # Core ticket fields
    "summary": ("Summary", "Issue summary from Jira.", None),
    "priority": ("Priority", "Jira priority label assigned to the ticket.", None),
    "status": ("Status", "Current Jira workflow status.", None),
    "assignee": ("Assignee", "Current owner responsible for the issue.", None),
    "reporter": ("Reporter", "Original reporter who created the ticket.", None),
    "created": ("Created", "Timestamp when the ticket was created in Jira.", None),
    "updated": ("Updated", "Timestamp of the most recent update in Jira.", None),
    "labels": ("Labels", "Jira labels associated with the ticket.", None),
    "resolution": ("Resolution", "Jira resolution status of the ticket.", None),
    "obs_system": ("OBS System", "OBS top-level system associated with the ticket.", None),
    "obs_subsystem": ("OBS Subsystem", "OBS subsystem associated with the ticket.", None),
    "obs_component": ("OBS Component", "OBS component associated with the ticket.", None),
    "days_open": ("Days Open", "Days since the ticket was created.", "float1"),
    "days_since_update": (
        "Days Since Update",
        "Days elapsed since the most recent activity captured in the range.",
        "float1",
    ),
    "time_lost": ("Time Lost (hours)", "Total time lost reported on the ticket.", "float1"),
    "time_lost_value": ("Time Lost (hours)", "Total time lost reported on the ticket.", "float1"),
    "Time Lost": ("Time Lost (hours)", "Total time lost reported on the ticket.", "float1"),
    "Days Open": ("Days Open", "Days since the ticket was created.", "float1"),
    # Activity metrics
    "comments_in_range": (
        "Comments",
        "Number of comments added within the selected date range.",
        "int",
    ),
    "filtered_comments_count": (
        "Comments",
        "Number of comments added within the selected date range.",
        "int",
    ),
    "Comments": (
        "Comments",
        "Number of comments added within the selected date range.",
        "int",
    ),
    "status_changes": (
        "Status Changes",
        "Workflow status transitions recorded in the selected range.",
        "int",
    ),
    "other_changes": (
        "Other Changes",
        "Other field updates captured in the selected range.",
        "int",
    ),
    "filtered_histories_count": (
        "History Entries",
        "Workflow history entries captured in the selected range.",
        "int",
    ),
    "History Entries": (
        "History Entries",
        "Workflow history entries captured in the selected range.",
        "int",
    ),
    "ole_comments_count": (
        "OLE Comments",
        "Comments added by Rubin Jira API Access within the selected range.",
        "int",
    ),
    "OLE Comments": (
        "OLE Comments",
        "Comments added by Rubin Jira API Access within the selected range.",
        "int",
    ),
    "ole_last_comment": (
        "Last OLE Comment",
        "Most recent Rubin Jira API Access comment timestamp inside the range.",
        None,
    ),
    "Weighted Score": (
        "Weighted Score",
        "Weighted activity score using the sidebar weights.",
        "float1",
    ),
    "Activity Score": (
        "Activity Score",
        "Raw activity score (comments plus history entries).",
        "int",
    ),
}


def apply_column_metadata(
    columns: Iterable[str],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a column_config dictionary with human labels and hover help."""

    config: dict[str, Any] = dict(existing or {})
    for col in columns:
        if col in config:
            continue
        meta = COLUMN_METADATA.get(col)
        if not meta:
            continue
        label, help_text, fmt = meta
        if fmt == "int":
            config[col] = st.column_config.NumberColumn(label, help=help_text, format="%d")
        elif fmt == "float1":
            config[col] = st.column_config.NumberColumn(label, help=help_text, format="%.1f")
        else:
            config[col] = st.column_config.Column(label, help=help_text)
    return config
