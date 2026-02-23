"""Jira API client wrapper (REST v3 + enhanced search pagination)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from jira import JIRA, JIRAError


class JiraAPI:
    def __init__(self, server: str, email: str, token: str):
        self.server = server.rstrip("/")
        self.client = JIRA(
            basic_auth=(email, token), options={"server": self.server, "rest_api_version": "3"}
        )
        # Simple in-memory cache: {(hash): (timestamp, data)}
        self._cache: dict[str, tuple[float, list]] = {}
        self._cache_ttl = 300.0  # seconds

    def clear_cache(self) -> None:
        """Reset the in-memory search cache."""
        cache = getattr(self, "_cache", None)
        if isinstance(cache, dict):
            cache.clear()

    def _cache_key(self, jql: str, fields, expand, page_size: int) -> str:
        payload = {
            "jql": jql,
            "fields": fields,
            "expand": expand,
            "page_size": page_size,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def search_enhanced(
        self,
        jql: str,
        fields: list[str] | None = None,
        expand: list[str] | None = None,
        page_size: int = 1000,
    ) -> list[dict[str, Any]]:
        session = getattr(self.client, "_session", None)
        if session is None:
            raise RuntimeError("JIRA session unavailable")
        url = f"{self.server}/rest/api/3/search/jql"
        # Cache check
        key = self._cache_key(jql, fields, expand, page_size)
        now = time.time()
        cached = self._cache.get(key)
        if cached and (now - cached[0]) < self._cache_ttl:
            return cached[1]
        params = {"jql": jql, "maxResults": page_size}
        if fields:
            params["fields"] = ",".join(fields)
        if expand:
            params["expand"] = ",".join(expand)
        out: list[dict[str, Any]] = []
        token = None
        while True:
            qp = dict(params)
            if token:
                qp["nextPageToken"] = token
            resp = session.get(url, params=qp)
            if resp.status_code >= 400:
                raise RuntimeError(f"Enhanced search failed {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            out.extend(data.get("issues", []))
            token = data.get("nextPageToken")
            if not token or data.get("isLast") is True:
                break
        # Store in cache
        self._cache[key] = (now, out)
        return out

    def fetch_issue_raw(self, issue_key: str) -> dict[str, Any]:
        try:
            issue = self.client.issue(issue_key, expand="changelog,renderedFields")
        except JIRAError as exc:  # pragma: no cover - network error path
            raise RuntimeError(f"Failed to fetch issue {issue_key}: {exc}") from exc
        if hasattr(issue, "raw"):
            return issue.raw
        if isinstance(issue, dict):
            return issue
        raise RuntimeError(f"Unexpected issue payload type for {issue_key}: {type(issue)!r}")

    def get_current_user(self) -> str | None:
        """Get the display name of the currently authenticated user.

        Returns
        -------
        str or None
            The display name of the current user, or None if unavailable.
        """
        try:
            user = self.client.myself()
            if hasattr(user, "displayName"):
                return user.displayName
            if isinstance(user, dict):
                return user.get("displayName")
            return None
        except JIRAError:  # pragma: no cover - network error path
            return None

    def get_issue_watchers(self, issue_key: str) -> list[str]:
        """Get the list of watchers for an issue.

        Parameters
        ----------
        issue_key : str
            The Jira issue key (e.g., "OBS-123").

        Returns
        -------
        list[str]
            List of watcher display names.
        """
        try:
            watchers = self.client.watchers(issue_key)
            if hasattr(watchers, "watchers"):
                return [
                    w.displayName if hasattr(w, "displayName") else str(w) for w in watchers.watchers if w
                ]
            return []
        except JIRAError:  # pragma: no cover - network error path
            return []

    def fetch_watchers_batch(self, issue_keys: list[str]) -> dict[str, list[str]]:
        """Fetch watchers for multiple issues.

        Parameters
        ----------
        issue_keys : list[str]
            List of Jira issue keys.

        Returns
        -------
        dict[str, list[str]]
            Mapping of issue key to list of watcher display names.
        """
        result: dict[str, list[str]] = {}
        for key in issue_keys:
            result[key] = self.get_issue_watchers(key)
        return result
