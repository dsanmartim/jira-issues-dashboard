"""Status normalization and categorization utilities.

This module provides centralized status handling functions that can be reused
across all pages and analytics modules. It uses the workflow configuration
from config.py (STATUS_ALIASES, STATUS_DISPLAY_ORDER, TERMINAL_STATUSES).
"""

from __future__ import annotations

from .config import STATUS_ALIASES, STATUS_DISPLAY_ORDER, TERMINAL_STATUSES

# Derive done/terminal statuses (lowercase for matching)
# Includes aliases that map to terminal statuses
# Public constant for efficient DataFrame filtering with .isin()
DONE_STATUSES_LOWER: frozenset[str] = frozenset(
    {s.lower() for s in TERMINAL_STATUSES} | {"resolved", "closed", "completed", "duplicated"}
)


def normalize_workflow_status(value: str | None) -> str:
    """Map raw Jira status to canonical workflow status names.

    Uses STATUS_ALIASES from config for mapping. Returns "Unknown" for any
    unmapped or empty value, which helps surface new/unexpected statuses.

    Parameters
    ----------
    value : str | None
        Raw status string from Jira.

    Returns
    -------
    str
        Canonical status name (e.g., "In Progress", "Done") or "Unknown".

    Examples
    --------
    >>> normalize_workflow_status("in progress")
    'In Progress'
    >>> normalize_workflow_status("resolved")
    'Done'
    >>> normalize_workflow_status("some_new_status")
    'Unknown'
    """
    if not value:
        return "Unknown"
    text = str(value).strip().lower()
    # Check config aliases first
    if text in STATUS_ALIASES:
        return STATUS_ALIASES[text]
    # Check if it matches a display status (case-insensitive)
    for status in STATUS_DISPLAY_ORDER:
        if text == status.lower():
            return status
    return "Unknown"


def clean_status_name(value: str | None) -> str:
    """Sanitize status string, converting null-like values to "Unknown".

    Parameters
    ----------
    value : str | None
        Raw status string.

    Returns
    -------
    str
        Cleaned status string or "Unknown" for empty/null values.
    """
    if not value:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"
    if text.lower() in {"oops", "nan", "none", "null"}:
        return "Unknown"
    return text


def map_status_category(value: str | None) -> str:
    """Map status to high-level category for portfolio views.

    Groups workflow statuses into four categories:
    - "To Do": Reported, To Do
    - "In Progress": In Progress, Testing, Tracking
    - "Done": Done, Cancelled, Duplicate, Transferred
    - "Other": Unknown or unmapped statuses

    Parameters
    ----------
    value : str | None
        Raw or normalized status string.

    Returns
    -------
    str
        One of: "To Do", "In Progress", "Done", "Other"
    """
    normalized = normalize_workflow_status(value)
    if normalized in {"Reported", "To Do"}:
        return "To Do"
    if normalized in {"In Progress", "Testing", "Tracking", "Blocked"}:
        return "In Progress"
    if normalized in TERMINAL_STATUSES:
        return "Done"
    return "Other"


def is_terminal_status(value: str | None) -> bool:
    """Check if status indicates a closed/terminal state.

    Parameters
    ----------
    value : str | None
        Raw or normalized status string.

    Returns
    -------
    bool
        True if the status is terminal (Done, Cancelled, Duplicate, Transferred).
    """
    normalized = normalize_workflow_status(value)
    return normalized in TERMINAL_STATUSES


def is_done_status_lowercase(status_lower: str) -> bool:
    """Check if a lowercase status string represents a done/terminal state.

    This is a fast check for pre-lowercased strings, useful in tight loops.

    Parameters
    ----------
    status_lower : str
        Lowercase status string.

    Returns
    -------
    bool
        True if the status represents a terminal/done state.
    """
    return status_lower in DONE_STATUSES_LOWER
