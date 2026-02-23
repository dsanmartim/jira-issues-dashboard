"""Mapping raw Jira issue JSON or jira.Issue objects into IssueModel instances."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import pandas as pd

from .config import FIELD_IDS, PRIORITY_MAPPING
from .models import CommentModel, HistoryItemModel, IssueModel

OBS_NAME_ALIASES: dict[str, str] = {
    "s:dome": "Dome",
    "dome": "Dome",
}


def _normalize_obs_name(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        # Non-scalar (e.g., dict) shouldn't route here; fall through
        pass
    text = str(value).strip()
    if not text:
        return None
    text_clean = " ".join(text.split())
    colon_normalized = text_clean.replace(" :", ":").replace(": ", ":")
    key = colon_normalized.casefold()
    if key in OBS_NAME_ALIASES:
        return OBS_NAME_ALIASES[key]
    return text_clean


def map_priority(p: str | None) -> int:
    if not p:
        return -99
    for k, v in PRIORITY_MAPPING.items():
        if p.startswith(k):
            return v
    return -99


def _extract_obs_list(value: Any) -> list[str]:
    names: list[str] = []

    def walk(node: Any):
        if isinstance(node, dict):
            nm = node.get("name")
            if isinstance(nm, str):
                names.append(nm)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    if value is not None:
        walk(value)
    # Deduplicate preserve order
    seen = set()
    out: list[str] = []
    for n in names:
        normalized = _normalize_obs_name(n)
        if normalized is None:
            continue
        if normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _split_obs_hierarchy(lst: list[str]):
    system = lst[0] if len(lst) > 0 else None
    subsystem = lst[1] if len(lst) > 1 else None
    component = lst[2] if len(lst) > 2 else None
    return system, subsystem, component


def map_issue(raw: dict[str, Any]) -> IssueModel:
    fields = raw.get("fields", {})
    obs_value = fields.get(FIELD_IDS["obs_hierarchy"]) if FIELD_IDS.get("obs_hierarchy") else None
    obs_list = _extract_obs_list(obs_value)
    system, subsystem, component = _split_obs_hierarchy(obs_list)

    def parse_dt(val):
        if not val:
            return None
        ts = pd.to_datetime(val, utc=True, errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        return ts.to_pydatetime()

    comments_raw = fields.get("comment", {}).get("comments", []) or []
    comments = [
        CommentModel(
            author=(c.get("author") or {}).get("displayName"),
            created=parse_dt(c.get("created")),
            body=c.get("body"),
        )
        for c in comments_raw
    ]
    histories_raw = (raw.get("changelog") or {}).get("histories", []) or []
    histories = []
    for h in histories_raw:
        items = h.get("items") or []
        first_field = items[0].get("field") if items else None
        histories.append(
            HistoryItemModel(
                author=(h.get("author") or {}).get("displayName"),
                created=parse_dt(h.get("created")),
                field=first_field,
                items=items,
            )
        )

    # Extract watchers if available (may need separate API call for full list)
    watches_raw = fields.get("watches") or {}
    watchers: list[str] = []
    # If the API returns watcher details, extract display names
    if isinstance(watches_raw, dict) and "watchers" in watches_raw:
        for w in watches_raw.get("watchers", []) or []:
            if isinstance(w, dict):
                display_name = w.get("displayName")
                if display_name:
                    watchers.append(display_name)

    issue = IssueModel(
        key=raw.get("key"),
        summary=fields.get("summary"),
        created=parse_dt(fields.get("created")),
        updated=parse_dt(fields.get("updated")),
        assignee=(fields.get("assignee") or {}).get("displayName") if fields.get("assignee") else None,
        reporter=(fields.get("reporter") or {}).get("displayName") if fields.get("reporter") else None,
        priority=(fields.get("priority") or {}).get("name") if fields.get("priority") else None,
        status=(fields.get("status") or {}).get("name") if fields.get("status") else None,
        resolution=(fields.get("resolution") or {}).get("name") if fields.get("resolution") else None,
        resolution_date=parse_dt(fields.get("resolutiondate")),
        issuetype=(fields.get("issuetype") or {}).get("name") if fields.get("issuetype") else None,
        time_lost=fields.get(FIELD_IDS["time_lost"]) if FIELD_IDS.get("time_lost") else None,
        obs_system=system,
        obs_subsystem=subsystem,
        obs_component=component,
        labels=list(fields.get("labels", []) or []),
        watchers=watchers,
        comments=comments,
        histories=histories,
    )
    issue.priority_value = map_priority(issue.priority)
    return issue


def issues_to_dataframe(issues: Iterable[IssueModel]) -> pd.DataFrame:
    rows = []
    for i in issues:
        rows.append(
            {
                "key": i.key,
                "summary": i.summary,
                "created": i.created,
                "updated": i.updated,
                "assignee": i.assignee or "Unassigned",
                "reporter": i.reporter or "Unknown",
                "priority": i.priority or "None",
                "priority_value": i.priority_value,
                "status": i.status,
                "resolution": i.resolution or "Unresolved",
                "resolution_date": i.resolution_date,
                "issuetype": i.issuetype,
                "time_lost": i.time_lost,
                "obs_system": i.obs_system,
                "obs_subsystem": i.obs_subsystem,
                "obs_component": i.obs_component,
                "labels": i.labels,
                "watchers": i.watchers,
                "comments": [asdict(c) for c in i.comments],
                "histories": [asdict(h) for h in i.histories],
            }
        )
    df = pd.DataFrame(rows)
    # Normalize labels list to a stable, comma-separated string for display
    if "labels" in df.columns:

        def _format_labels(val):
            if not val:
                return ""
            try:
                # Deduplicate, sort case-insensitively; keep original casing
                unique = {v for v in val if v}
                ordered = sorted(unique, key=lambda s: s.lower())
                return ", ".join(ordered)
            except Exception:
                return ""

        df["labels"] = df["labels"].apply(_format_labels)
    for col in ("obs_system", "obs_subsystem", "obs_component"):
        if col in df.columns:
            df[col] = df[col].apply(_normalize_obs_name)
    return df
