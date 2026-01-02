"""Central configuration, constants, feature flags, and shared column definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

JIRA_DEFAULT_SERVER = "https://rubinobs.atlassian.net"
TIMEZONE = "America/Santiago"

DEFAULT_TREND_PRIORITIES = ["Blocker", "Critical", "High", "Medium", "Low"]

PRIORITY_MAPPING = {
    "Blocker": 5,
    "Critical": 4,
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "Undefined": 0,
}

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
    FIELD_IDS["time_lost"],
    "issuetype",
    "labels",
    FIELD_IDS["obs_hierarchy"],
]

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
