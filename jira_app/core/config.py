"""Central configuration, constants, feature flags, and shared column definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# =============================================================================
# Jira Connection Settings
# =============================================================================
JIRA_DEFAULT_SERVER = "https://rubinobs.atlassian.net"
TIMEZONE = "America/Santiago"
DEFAULT_PROJECT_KEY = "OBS"

# =============================================================================
# API Comment Authors (OLE/LOVE integration)
# These are special system accounts whose comments are tracked separately.
# =============================================================================
API_COMMENT_AUTHORS: frozenset[str] = frozenset(
    {
        "Rubin Jira API Access",  # OLE API display name
        "love-api",  # LOVE system integration
    }
)

# =============================================================================
# Workflow Status Configuration
# =============================================================================
# Canonical display order for status columns/charts
STATUS_DISPLAY_ORDER: Sequence[str] = (
    "Reported",
    "To Do",
    "In Progress",
    "Testing",
    "Tracking",
    "Blocked",
    "Transferred",
    "Duplicate",
    "Cancelled",
    "Done",
)

# Statuses that indicate a ticket is closed/terminal
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "Done",
        "Cancelled",
        "Duplicate",
        "Transferred",
    }
)

# Map various status strings to canonical display names
# Keys should be lowercase for case-insensitive matching
STATUS_ALIASES: dict[str, str] = {
    # Initial/Open statuses
    "reported": "Reported",
    "to do": "To Do",
    "todo": "To Do",
    "open": "To Do",
    "new": "To Do",
    # In Progress variants
    "in progress": "In Progress",
    "inprogress": "In Progress",
    "in-progress": "In Progress",
    "working": "In Progress",
    # Testing/Tracking
    "testing": "Testing",
    "tracking": "Tracking",
    # Blocked
    "blocked": "Blocked",
    # Cancelled variants
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    # Done variants
    "done": "Done",
    "resolved": "Done",
    "closed": "Done",
    "complete": "Done",
    "completed": "Done",
    # Duplicate/Transferred
    "duplicate": "Duplicate",
    "duplicated": "Duplicate",
    "transferred": "Transferred",
}

# =============================================================================
# UI Default Values
# =============================================================================
DEFAULT_DATE_RANGE_DAYS: int = 30  # Default lookback period for date pickers
DEFAULT_STALE_DAYS: int = 30  # Default threshold for stale ticket detection
DEFAULT_HISTORY_FALLBACK_DAYS: int = 365  # Fallback when no created date available
DEFAULT_TOP_N: int = 20  # Default number of items in top-N lists
DEFAULT_WARN_AGE_DAYS: int = 14  # Default warning threshold for ticket age

# =============================================================================
# Priority Configuration
# =============================================================================
DEFAULT_TREND_PRIORITIES = ["Blocker", "Critical", "High", "Medium", "Low"]

PRIORITY_MAPPING = {
    "Blocker": 5,
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Undefined": 0,
}

# Priority aliases for normalization (lowercase keys)
PRIORITY_ALIASES: dict[str, str] = {
    "blocker": "Blocker",
    "critical": "Critical",
    "urgent": "Urgent",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "undefined": "Undefined",
    "none": "Undefined",
    # Numeric priority IDs (legacy Jira values)
    "5": "Blocker",
    "4": "Critical",
    "3": "High",
    "2": "Medium",
    "1": "Low",
    "0": "Undefined",
}


def normalize_priority_name(priority: str | None) -> str:
    """Normalize a priority name to its canonical form.

    Handles variations like:
    - "(migrated)" suffixes: "Medium (migrated)" -> "Medium"
    - Case variations: "HIGH" -> "High"
    - Whitespace: "  Medium  " -> "Medium"

    Parameters
    ----------
    priority : str or None
        Raw priority string from Jira.

    Returns
    -------
    str
        Canonical priority name, or original string if no mapping found.
    """
    if priority is None:
        return "Undefined"

    # Clean up whitespace
    cleaned = str(priority).strip()
    if not cleaned:
        return "Undefined"

    # Remove "(migrated)" suffix (case-insensitive)
    import re

    cleaned = re.sub(r"\s*\(migrated\)\s*$", "", cleaned, flags=re.IGNORECASE).strip()

    # Try to match against known aliases (case-insensitive)
    lookup_key = cleaned.lower()
    if lookup_key in PRIORITY_ALIASES:
        return PRIORITY_ALIASES[lookup_key]

    # Return cleaned version with title case if no alias match
    return cleaned


# =============================================================================
# Jira Custom Field IDs
# =============================================================================
FIELD_IDS = {
    "obs_hierarchy": "customfield_10476",
    "time_lost": "customfield_10106",
}

# Feature flags / tuning knobs
# If True, every fetched issue will have its comments fully hydrated via an
# individual issue fetch (may increase latency & API usage). If False, we only
# hydrate issues whose embedded comment list appears truncated.
FULL_COMMENT_HYDRATION = True

# Parallel hydration tuning
# Use threads because jira client calls are I/O bound (HTTP) and the
# library is synchronous. Keep worker count moderate to avoid hitting
# Jira rate limits.
COMMENT_HYDRATION_MAX_WORKERS = 8
COMMENT_HYDRATION_MIN_PARALLEL = 4  # below this, stay sequential to reduce overhead

# Canonical field list for Jira fetches (excluding changelog/renderedFields expands)
JIRA_FETCH_BASE_FIELDS = [
    "summary",
    "created",
    "updated",
    "assignee",
    "reporter",
    "priority",
    "status",
    "resolution",
    "resolutiondate",
    "comment",
    "watches",  # Watcher information for Personal View
    FIELD_IDS["time_lost"],
    "issuetype",
    "labels",
    FIELD_IDS["obs_hierarchy"],
]

# Project keys for Personal View page
DEFAULT_PROJECT_KEYS: Sequence[str] = ("OBS", "OSW", "RSO", "DM", "SITCOM")

ISSUE_CORE_COLUMNS: Sequence[str] = (
    "key",
    "summary",
    "priority",
    "priority_value",
    "status",
    "time_lost",
    "labels",
    "obs_system",
    "obs_subsystem",
    "obs_component",
    "days_open",
    "days_since_update",
    "created",
    "updated",
    "assignee",
    "reporter",
    "resolution",
)

DISPLAY_ORDER_DETAIL: Sequence[str] = (
    "Ticket",
    "summary",
    "priority",
    "priority_value",
    "status",
    "time_lost",
    "days_open",
    "labels",
    "obs_system",
    "obs_subsystem",
    "obs_component",
    "days_since_update",
    "created",
    "updated",
    "assignee",
    "reporter",
    "resolution",
)


DISPLAY_ORDER_TICKET_LIST: Sequence[str] = (
    "Ticket",
    "summary",
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
)


@dataclass(slots=True)
class AppSettings:
    max_table_rows: int = 1000
    download_encoding: str = "utf-8"


SETTINGS = AppSettings()
