"""Domain data models for Jira issues, comments, and change histories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class CommentModel:
    author: str | None
    created: datetime | None
    body: str | None


@dataclass(slots=True)
class HistoryItemModel:
    author: str | None
    created: datetime | None
    field: str | None
    items: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class IssueModel:
    key: str
    summary: str | None
    created: datetime | None
    updated: datetime | None
    assignee: str | None
    reporter: str | None
    priority: str | None
    status: str | None
    resolution: str | None
    resolution_date: datetime | None
    issuetype: str | None
    time_lost: float | None
    obs_system: str | None
    obs_subsystem: str | None
    obs_component: str | None
    labels: list[str] = field(default_factory=list)
    comments: list[CommentModel] = field(default_factory=list)
    histories: list[HistoryItemModel] = field(default_factory=list)

    # Derived metrics (populated later)
    priority_value: int | None = None
    days_open: float | None = None
    days_since_update: float | None = None
