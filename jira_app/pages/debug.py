"""Debug page for inspecting comment metadata and fetching single issues."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
import streamlit as st

from jira_app.app import register_page
from jira_app.core.config import TIMEZONE
from jira_app.core.service import ActivityWeights
from jira_app.visual.progress import ProgressReporter

TZ = pytz.timezone(TIMEZONE)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _normalize_datetime(value: Any) -> datetime | None:
    ts = pd.to_datetime(value, errors="coerce")
    if ts is None or pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is None:
        try:
            ts = ts.tz_localize(pytz.UTC)
        except (TypeError, ValueError):  # pragma: no cover - unexpected format
            return None
    try:
        return ts.tz_convert(TZ)
    except (TypeError, ValueError):  # pragma: no cover - unexpected format
        return None


def _build_comment_frames(
    df: pd.DataFrame,
    *,
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or "comments" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    comment_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        ticket_key = row.get("key")
        summary = row.get("summary")
        comments: Iterable[dict[str, Any]] = row.get("comments") or []
        if not isinstance(comments, Iterable):
            comments = []

        sorted_comments: list[dict[str, Any]] = []
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            sorted_comments.append(comment.copy())
        sorted_comments.sort(
            key=lambda c: (_normalize_datetime(c.get("created")) or datetime.min.replace(tzinfo=TZ))
        )

        total_comments = 0
        in_range_comments = 0
        first_seen: datetime | None = None
        last_seen: datetime | None = None

        for idx, comment in enumerate(sorted_comments, start=1):
            created_local = _normalize_datetime(comment.get("created"))
            created_utc = created_local.astimezone(pytz.UTC) if created_local is not None else None
            in_range = False
            if created_local is not None and start_dt is not None and end_dt is not None:
                in_range = start_dt <= created_local <= end_dt
            total_comments += 1
            if in_range:
                in_range_comments += 1
            if created_local is not None:
                if first_seen is None or created_local < first_seen:
                    first_seen = created_local
                if last_seen is None or created_local > last_seen:
                    last_seen = created_local

            body_text = _extract_comment_body(comment.get("body"))
            comment_rows.append(
                {
                    "Ticket": ticket_key,
                    "Summary": summary,
                    "Comment #": idx,
                    "Author": comment.get("author") or "(Unknown)",
                    "Created (local)": created_local,
                    "Created (UTC)": created_utc,
                    "In range": in_range,
                    "Body preview": body_text[:150],
                }
            )

        summary_rows.append(
            {
                "Ticket": ticket_key,
                "Summary": summary,
                "Stored comments_in_range": row.get("comments_in_range"),
                "Computed total comments": total_comments,
                "Computed in-range comments": in_range_comments,
                "Difference (computed - stored)": (
                    in_range_comments - int(row.get("comments_in_range") or 0)
                ),
                "First comment": first_seen,
                "Last comment": last_seen,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    comment_df = pd.DataFrame(comment_rows)
    return summary_df, comment_df


def _get_window_bounds() -> tuple[datetime | None, datetime | None]:
    start_dt = st.session_state.get("dashboard_start_dt")
    end_dt = st.session_state.get("dashboard_end_dt")
    if start_dt is not None and end_dt is not None:
        return start_dt, end_dt

    start_date_str = st.session_state.get("start_date_str")
    end_date_str = st.session_state.get("end_date_str")
    if not start_date_str or not end_date_str:
        return None, None
    start_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str)
    start_dt_local = TZ.localize(datetime.combine(start_date, datetime.min.time()))
    end_dt_local = TZ.localize(datetime.combine(end_date, datetime.max.time()))
    return start_dt_local, end_dt_local


def _extract_comment_body(body: Any) -> str:
    def _walk(node: Any) -> Iterator[str]:
        if node is None:
            return
        if isinstance(node, str):
            yield node
            return
        if isinstance(node, list):
            for item in node:
                yield from _walk(item)
            return
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "text":
                yield node.get("text", "")
            elif node_type == "mention":
                attrs = node.get("attrs") or {}
                mention_text = attrs.get("text") or attrs.get("id")
                if mention_text:
                    yield f"@{mention_text}"
            elif node_type == "emoji":
                attrs = node.get("attrs") or {}
                yield attrs.get("text") or attrs.get("shortName", "")
            for child in node.get("content", []):
                yield from _walk(child)
            if node_type in {"paragraph", "heading", "listItem"}:
                yield "\n"
            return
        yield str(node)

    text = "".join(_walk(body))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@register_page("Debug")
def render():
    st.title("Debug: Comment Tracing & Issue Fetch")

    issue_service = st.session_state.get("issue_service")
    if issue_service is None:
        st.warning("Initialize the Jira connection on the Setup page to use debug tools.")
        return

    st.markdown("### Loaded dataset comment audit")
    current_df = st.session_state.get("enriched_df")
    if current_df is None or current_df.empty:
        st.info("No dataset loaded yet. Use the Dashboard page to fetch issues first.")
    else:
        start_dt, end_dt = _get_window_bounds()
        if start_dt is None or end_dt is None:
            st.warning("Date range not available. Re-run 'Fetch & Analyze' on the Dashboard to set it.")
            summary_df = pd.DataFrame()
            comment_df = pd.DataFrame()
        else:
            needs_recompute = (
                "comments_in_range" not in current_df.columns
                or current_df["comments_in_range"].isnull().any()
            )
            if needs_recompute:
                base_df = st.session_state.get("issues_df")
                weights = st.session_state.get("dashboard_weights")
                if not isinstance(weights, ActivityWeights):
                    weights = ActivityWeights()
                source_df = base_df if isinstance(base_df, pd.DataFrame) and not base_df.empty else current_df
                reporter = ProgressReporter("Recomputing comment activity metrics for the current dataset")
                try:
                    refreshed = issue_service.enrich(source_df, start_dt, end_dt, weights=weights)
                    reporter.complete("Comment activity metrics refreshed.")
                except Exception as exc:  # pragma: no cover
                    reporter.error(f"Failed to recompute metrics: {exc}")
                    raise
                st.session_state.enriched_df = refreshed
                current_df = refreshed
                st.session_state.dashboard_start_dt = start_dt
                st.session_state.dashboard_end_dt = end_dt

            summary_df, comment_df = _build_comment_frames(current_df, start_dt=start_dt, end_dt=end_dt)

        if summary_df.empty:
            st.info("Loaded dataset does not contain comment metadata to audit.")
        else:
            st.caption(
                "Comparison between stored in-range comment counts and freshly computed values for the "
                "current dataset."
            )
            diff_sorted = summary_df.sort_values("Difference (computed - stored)", ascending=False)
            st.dataframe(diff_sorted, hide_index=True, width="stretch")

            default_keys = diff_sorted.head(5)["Ticket"].dropna().astype(str).tolist()
            available_keys = diff_sorted["Ticket"].dropna().astype(str).unique().tolist()
            selected_keys = st.multiselect(
                "Focus on tickets",
                options=available_keys,
                default=default_keys,
            )
            if selected_keys:
                filtered_comments = comment_df[comment_df["Ticket"].astype(str).isin(selected_keys)]
            else:
                filtered_comments = comment_df

            st.markdown("#### Comment timeline detail")
            st.caption(
                "Chronological comment entries for the selected tickets, highlighting whether each one falls "
                "inside the analysis window."
            )
            st.dataframe(
                filtered_comments.sort_values(["Ticket", "Comment #"]),
                hide_index=True,
                width="stretch",
            )

            csv_data = comment_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download comment detail CSV",
                data=csv_data,
                file_name="comment_detail_debug.csv",
                mime="text/csv",
            )

    st.markdown("---")
    st.markdown("### Fetch and store a single issue")
    st.caption(
        "Retrieve the latest data for an individual issue, store the raw JSON, and inspect its comments."
    )
    issue_key_input = st.text_input("Issue key", value=st.session_state.get("debug_issue_key", ""))

    fetch_button = st.button("Fetch issue details", type="primary")
    if fetch_button and issue_key_input.strip():
        issue_key = issue_key_input.strip().upper()
        st.session_state["debug_issue_key"] = issue_key
        reporter = ProgressReporter(f"Fetching {issue_key} and saving raw payload")
        try:
            issue_df, raw_issue = issue_service.fetch_issue_detail(issue_key, progress=reporter.callback)
        except Exception as exc:  # pragma: no cover - network errors
            reporter.error(f"Failed to fetch {issue_key}: {exc}")
            st.error(f"Failed to fetch {issue_key}: {exc}")
        else:
            if not raw_issue:
                reporter.complete(f"No data returned for {issue_key}.")
                st.warning(f"No data returned for {issue_key}.")
            else:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                json_path = DATA_DIR / f"{issue_key}.json"
                with json_path.open("w", encoding="utf-8") as fh:
                    json.dump(raw_issue, fh, indent=2, ensure_ascii=False)
                start_dt, end_dt = _get_window_bounds()
                if start_dt and end_dt:
                    weights = st.session_state.get("dashboard_weights")
                    try:
                        reporter.update("Enriching single-issue metrics")
                        issue_df = issue_service.enrich(issue_df, start_dt, end_dt, weights=weights)
                    except Exception as enrich_exc:  # pragma: no cover
                        reporter.error(f"Enrichment failed: {enrich_exc}")
                        st.warning(f"Enrichment failed: {enrich_exc}")
                    else:
                        reporter.update("Writing raw payload to disk")
                reporter.complete(f"Stored raw JSON in {json_path.relative_to(Path.cwd())}")
                st.session_state["debug_issue_df"] = issue_df
                st.session_state["debug_issue_path"] = str(json_path)

    stored_df: pd.DataFrame | None = st.session_state.get("debug_issue_df")
    stored_path = st.session_state.get("debug_issue_path")
    if stored_df is not None and not stored_df.empty:
        st.markdown("#### Latest fetched issue comment breakdown")
        st.caption(
            "This uses the freshly fetched payload rather than the dashboard cache, "
            "helping compare raw data to derived metrics."
        )
        # Get the main dashboard's date range to ensure a consistent comparison
        start_dt, end_dt = _get_window_bounds()
        summary_df, comment_df = _build_comment_frames(stored_df, start_dt=start_dt, end_dt=end_dt)
        st.caption("Summary of stored versus recomputed comment counts for the fetched issue.")
        st.dataframe(summary_df, hide_index=True, width="stretch")
        st.caption("Comment-level detail including authorship and window membership.")
        st.dataframe(comment_df, hide_index=True, width="stretch")
        if stored_path:
            st.info(f"Raw JSON stored at: {stored_path}")
