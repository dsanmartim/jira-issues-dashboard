"""IssueService: orchestrates fetching, mapping, and enrichment pipeline."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytz

from jira_app.analytics.metrics.activity import add_weighted_activity, normalize_timestamp
from jira_app.analytics.metrics.aging import add_aging_metrics
from jira_app.analytics.segments.filters import OLE_API_DISPLAY_NAME

from .config import (
    COMMENT_HYDRATION_MAX_WORKERS,
    COMMENT_HYDRATION_MIN_PARALLEL,
    FULL_COMMENT_HYDRATION,
    JIRA_FETCH_BASE_FIELDS,
    TIMEZONE,
)
from .jira_client import JiraAPI
from .mappers import issues_to_dataframe, map_issue

DEFAULT_FIELDS: Sequence[str] = tuple(JIRA_FETCH_BASE_FIELDS)
ProgressCallback = Callable[[str, int | None, int | None], None]


@dataclass(slots=True)
class ActivityWeights:
    comment: float = 1.0
    status: float = 2.0
    other: float = 0.5


class IssueService:
    def __init__(self, api: JiraAPI):
        self.api = api
        self._tz = pytz.timezone(TIMEZONE)

    def get_projects(self):
        """Fetch all projects from Jira."""
        return self.api.client.projects()

    # ------------------ Fetch Methods ------------------
    def fetch_project_open(
        self,
        project_key: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        jql = f"project = {project_key} AND statusCategory != Done"
        if progress:
            progress(f"Querying open issues for {project_key}", None, None)
        raw = self.api.search_enhanced(
            jql,
            fields=list(DEFAULT_FIELDS),
            expand=["changelog", "renderedFields"],
        )
        # Apply comment hydration if enabled (or detect truncation opportunistically)
        if isinstance(raw, list):
            self._inflate_truncated_comments(
                raw,
                force_all=FULL_COMMENT_HYDRATION,
                progress=progress,
            )
        return self._raw_to_df(raw)

    def fetch_updated_between(
        self,
        project_key: str,
        start: datetime,
        end: datetime,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        """
        Fetch issues whose updates or comments fall within the activity window.

        We extend the end timestamp by one day and use a strict upper bound so the
        caller can work with inclusive date pickers. Tickets that were updated
        after the selected window but still contain comments within the window are
        included by explicitly querying comment activity in the same range. Each
        call clears the Jira client cache to guarantee a fresh dataset for the
        current session fetch.
        """
        if hasattr(self.api, "clear_cache"):
            self.api.clear_cache()
            if progress:
                progress("Preparing fresh Jira data", None, None)

        end_plus = end + timedelta(days=1)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end_plus.strftime("%Y-%m-%d")
        updated_jql = f"project = {project_key} AND updated >= '{start_str}' AND updated < '{end_str}'"
        commented_jql = f"project = {project_key} AND commented >= '{start_str}' AND commented < '{end_str}'"
        created_jql = f"project = {project_key} AND created >= '{start_str}' AND created < '{end_str}'"

        if progress:
            progress("Gathering issues recently updated in your window", None, None)
        raw_updated = self.api.search_enhanced(
            updated_jql,
            fields=list(DEFAULT_FIELDS),
            expand=["changelog", "renderedFields"],
        )
        if progress:
            progress("Finding issues with comments during your window", None, None)
        raw_commented = self.api.search_enhanced(
            commented_jql,
            fields=list(DEFAULT_FIELDS),
            expand=["changelog", "renderedFields"],
        )
        if progress:
            progress("Finding issues created during your window", None, None)
        raw_created = self.api.search_enhanced(
            created_jql,
            fields=list(DEFAULT_FIELDS),
            expand=["changelog", "renderedFields"],
        )

        combined_by_key: dict[str, dict[str, Any]] = {}
        for issue in raw_updated:
            key = issue.get("key")
            if key:
                combined_by_key[key] = issue
        for issue in raw_commented:
            key = issue.get("key")
            if key and key not in combined_by_key:
                combined_by_key[key] = issue
        for issue in raw_created:
            key = issue.get("key")
            if key and key not in combined_by_key:
                combined_by_key[key] = issue
        combined = list(combined_by_key.values())

        # Ensure we have the full comment set (search endpoint truncates comment
        # list to a page size even when total is larger). We only inflate those
        # issues whose reported total exceeds the embedded list length. This
        # guarantees consistent comment counts between bulk dataset and
        # single-issue debug fetches.
        # Inflate comments either opportunistically (truncate detection) or
        # unconditionally if FULL_COMMENT_HYDRATION is enabled.
        self._inflate_truncated_comments(
            combined,
            force_all=FULL_COMMENT_HYDRATION,
            progress=progress,
        )

        return self._raw_to_df(combined)

    def fetch_full_project(
        self,
        project_key: str,
        *,
        progress: ProgressCallback | None = None,
        max_days: int | None = None,
    ) -> pd.DataFrame:
        """Fetch the full project issue set (optionally bounded by recent days).

        Parameters
        ----------
        project_key : str
            Jira project key.
        progress : callback, optional
            Progress reporter.
        max_days : int | None
            If provided, restrict to issues created within the last ``max_days`` days
            to protect performance on very large projects.
        """
        if hasattr(self.api, "clear_cache"):
            self.api.clear_cache()
            if progress:
                progress("Preparing fresh Jira data", None, None)
        date_clause = ""
        if max_days is not None and max_days > 0:
            since = datetime.utcnow() - timedelta(days=max_days)
            date_clause = f" AND created >= '{since.strftime('%Y-%m-%d')}'"
        jql = f"project = {project_key}{date_clause}"
        if progress:
            progress("Querying all project issues", None, None)
        raw = self.api.search_enhanced(
            jql,
            fields=list(DEFAULT_FIELDS),
            expand=["changelog", "renderedFields"],
        )
        # Inflate comments opportunistically
        if isinstance(raw, list):
            self._inflate_truncated_comments(
                raw,
                force_all=FULL_COMMENT_HYDRATION,
                progress=progress,
            )
        return self._raw_to_df(raw)

    # ------------------ Enrichment Pipeline ------------------
    def enrich(
        self,
        df: pd.DataFrame,
        start: datetime,
        end: datetime,
        weights: ActivityWeights | None = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df
        weights = weights or ActivityWeights()
        aged = add_aging_metrics(df)
        # Activity window uses timezone-aware boundaries
        start_tz = start.astimezone(self._tz) if start.tzinfo else self._tz.localize(start)
        end_tz = end.astimezone(self._tz) if end.tzinfo else self._tz.localize(end)
        active = add_weighted_activity(
            aged,
            start=start_tz,
            end=end_tz,
            w_comment=weights.comment,
            w_status=weights.status,
            w_other=weights.other,
        )
        # Ensure commonly used columns are present for all downstream views.
        out = active.copy()

        # 1) Comments metric unified
        if "filtered_comments_count" not in out.columns:
            if "comments_in_range" in out.columns:
                out["filtered_comments_count"] = out["comments_in_range"]
            else:
                # Fallback: count comments within range
                def _count_comments(lst):
                    if not isinstance(lst, list):
                        return 0
                    total = 0
                    for c in lst:
                        if not isinstance(c, dict):
                            continue
                        if normalize_timestamp(c.get("created"), self._tz) is None:
                            continue
                        ts = normalize_timestamp(c.get("created"), self._tz)
                        if ts is None or ts < start_tz or ts > end_tz:
                            continue
                        total += 1
                    return total

                if "comments" in out.columns:
                    out["filtered_comments_count"] = out["comments"].apply(_count_comments)
                else:
                    out["filtered_comments_count"] = 0

        # 2) History entries unified
        if "filtered_histories_count" not in out.columns:
            if "status_changes" in out.columns and "other_changes" in out.columns:
                out["filtered_histories_count"] = out["status_changes"] + out["other_changes"]
            else:

                def _count_histories(lst):
                    if not isinstance(lst, list):
                        return 0
                    total = 0
                    for h in lst:
                        if not isinstance(h, dict):
                            continue
                        ts = normalize_timestamp(h.get("created"), self._tz)
                        if ts is None or ts < start_tz or ts > end_tz:
                            continue
                        total += 1
                    return total

                if "histories" in out.columns:
                    out["filtered_histories_count"] = out["histories"].apply(_count_histories)
                else:
                    out["filtered_histories_count"] = 0

        # 3) OLE Comments unified (Rubin Jira API Access / love-api)
        if "ole_comments_count" not in out.columns or "ole_last_comment" not in out.columns:
            allowed_authors = {OLE_API_DISPLAY_NAME, "love-api"}

            def _extract_ole_stats(lst):
                cnt = 0
                last_seen = pd.NaT
                if not isinstance(lst, list):
                    return cnt, last_seen
                for cm in lst:
                    try:
                        author = cm.get("author") if isinstance(cm, dict) else None
                        if isinstance(author, dict):
                            name = author.get("displayName") or author.get("name") or ""
                        else:
                            name = str(author or cm.get("author") or "")
                        if name not in allowed_authors:
                            continue
                        ts = normalize_timestamp(cm.get("created"), self._tz)
                        if ts is None or ts < start_tz or ts > end_tz:
                            continue
                        cnt += 1
                        if pd.isna(last_seen) or ts > last_seen:
                            last_seen = ts
                    except Exception:
                        continue
                return cnt, last_seen

            if "comments" in out.columns:
                stats = out["comments"].apply(_extract_ole_stats)
                out["ole_comments_count"] = stats.apply(lambda p: p[0])
                out["ole_last_comment"] = stats.apply(lambda p: p[1])
            else:
                out["ole_comments_count"] = 0
                out["ole_last_comment"] = pd.NaT

        return out

    def fetch_and_enrich_range(
        self,
        project_key: str,
        start: datetime,
        end: datetime,
        weights: ActivityWeights | None = None,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        df = self.fetch_updated_between(project_key, start, end, progress=progress)
        if progress:
            progress("Calculating activity metrics", None, None)
        return self.enrich(df, start, end, weights=weights)

    def fetch_issue_detail(
        self,
        issue_key: str,
        *,
        progress: ProgressCallback | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        if progress:
            progress(f"Fetching details for {issue_key}", None, None)
        raw = self.api.fetch_issue_raw(issue_key)
        if not raw:
            return pd.DataFrame(), {}
        df = self._raw_to_df([raw])
        return df, raw

    # ------------------ Internal Helpers ------------------
    def _raw_to_df(self, raw_issues) -> pd.DataFrame:
        """Map raw issues to a DataFrame and sort robustly.

        This guard avoids KeyError when the dataset is empty or when the
        'updated' column is missing (e.g., very short time ranges). We prefer
        sorting by 'updated' when available, otherwise fall back to 'created'.
        """
        if not raw_issues:
            return pd.DataFrame()

        issues = [map_issue(r) for r in raw_issues]
        df = issues_to_dataframe(issues)

        if df.empty:
            return df

        # Normalize known datetime columns if present
        if "updated" in df.columns:
            df["updated"] = pd.to_datetime(df["updated"], errors="coerce")
            return df.sort_values(by="updated", ascending=False, na_position="last")

        if "created" in df.columns:
            df["created"] = pd.to_datetime(df["created"], errors="coerce")
            return df.sort_values(by="created", ascending=False, na_position="last")

        # Ensure downstream code can safely reference 'updated'
        df["updated"] = pd.NaT
        return df

    # ------------------ Internal Comment Inflation ------------------
    def _inflate_truncated_comments(
        self,
        raw_issues: list[dict[str, Any]],
        *,
        max_fetch: int | None = None,
        force_all: bool = False,
        progress: ProgressCallback | None = None,
    ) -> None:
        """Replace truncated comment arrays with full lists (in-place).

        Jira search results often embed only the first N comments (e.g. 20) but
        provide the total via ``fields.comment.total``. The single-issue detail
        fetch returns the full set. To unify counts we detect truncation and
        re-fetch only the affected issues.

        Parameters
        ----------
        raw_issues : list[dict]
            Raw issue JSON objects (mutated in place).
        max_fetch : int | None
            Cap on number of refetches (None = no cap).
        force_all : bool
            If True, refetch every issue to guarantee full comments.
        """
        if not raw_issues:
            return
        logger = logging.getLogger(__name__)

        # Build work list of issues needing refetch
        work: list[dict[str, Any]] = []
        PAGE_SIZE_GUESS = 20
        for issue in raw_issues:
            fields = issue.get("fields") or {}
            comment_block = fields.get("comment") or {}
            comments_list = comment_block.get("comments") or []
            total = comment_block.get("total")
            needs_refetch = False
            if (
                force_all
                or isinstance(total, int)
                and total > len(comments_list)
                or total is None
                and isinstance(comments_list, list)
                and len(comments_list) >= PAGE_SIZE_GUESS
            ):
                needs_refetch = True
            if needs_refetch:
                work.append(issue)
        if max_fetch is not None:
            work = work[:max_fetch]
        if not work:
            return

        # Sequential short-circuit
        if len(work) < COMMENT_HYDRATION_MIN_PARALLEL:
            if progress:
                progress("Loading complete comment history", 0, len(work))
            for idx, issue in enumerate(work, start=1):
                self._hydrate_single_issue(issue, logger)
                if progress:
                    progress("Loading complete comment history", idx, len(work))
            return

        # Parallel fetch using threads (I/O bound HTTP calls)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _task(iss: dict[str, Any]):
            return self._hydrate_single_issue(iss, logger)

        if progress:
            progress("Loading complete comment history", 0, len(work))
        completed = 0
        with ThreadPoolExecutor(max_workers=COMMENT_HYDRATION_MAX_WORKERS) as pool:
            futures = [pool.submit(_task, iss) for iss in work]
            for fut in as_completed(futures):  # noqa: F841 (iterate to surface exceptions)
                try:
                    fut.result()
                except Exception as exc:  # pragma: no cover
                    logger.warning("Hydration task failed: %s", exc)
                finally:
                    completed += 1
                    if progress:
                        progress("Loading complete comment history", completed, len(work))

    def _hydrate_single_issue(self, issue: dict[str, Any], logger: logging.Logger) -> None:
        fields = issue.get("fields") or {}
        comment_block = fields.get("comment") or {}
        comments_list = comment_block.get("comments") or []
        key = issue.get("key")
        if not key:
            return
        try:
            detail = self.api.fetch_issue_raw(key)
            d_fields = detail.get("fields", {})
            d_comment_block = d_fields.get("comment") or {}
            full_comments = d_comment_block.get("comments") or []
            if full_comments and len(full_comments) >= len(comments_list):
                comment_block["comments"] = full_comments
                comment_block["total"] = len(full_comments)
                fields["comment"] = comment_block
                issue["fields"] = fields
                # Replace changelog if longer
                detail_changelog = detail.get("changelog") or {}
                existing_histories = (issue.get("changelog") or {}).get("histories", [])
                new_histories = detail_changelog.get("histories") or []
                if len(new_histories) > len(existing_histories):
                    issue["changelog"] = detail_changelog
                logger.debug("Hydrated %s comments: %s -> %s", key, len(comments_list), len(full_comments))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to hydrate issue %s: %s", key, exc)
