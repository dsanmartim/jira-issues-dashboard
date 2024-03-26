from datetime import date, datetime, timedelta
from io import BytesIO

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import pytz
import streamlit as st
from jira import JIRA
from jira.exceptions import JIRAError


@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def _generate_wordcloud_png(
    text: str, width: int = 560, height: int = 210, bg: str = "white"
) -> bytes:
    """Generate a WordCloud PNG (cached by input text and dimensions)."""
    if not text or not text.strip():
        return b""
    try:
        from wordcloud import WordCloud

        wc = WordCloud(width=width, height=height, background_color=bg).generate(text)
        img = wc.to_image()
        bio = BytesIO()
        img.save(bio, format="PNG")
        return bio.getvalue()
    except Exception:
        return b""


class JiraDashboard:
    def __init__(self):
        # Fetch credentials from Streamlit secrets
        email = st.secrets["JIRA_EMAIL"]
        api_token = st.secrets["JIRA_API_TOKEN"]
        server = st.secrets["JIRA_SERVER"]

        self.jira = JIRA(
            basic_auth=(email, api_token),
            # Use REST API v3; Atlassian Cloud has removed some v2 endpoints
            options={"server": server, "rest_api_version": "3"},
        )
        self._server = server
        self.issues_df = None
        self._df_cache = {}

    def map_priority(self, priority_str):
        priority_mapping = {
            "Blocker": 5,
            "Critical": 4,
            "High": 3,
            "Medium": 2,
            "Low": 1,
            "Undefined": 0,
        }
        for key, value in priority_mapping.items():
            if priority_str.startswith(key):
                return value
        return -99  # Default value if no match is found

    def _search_issues_v3_post(
        self, jql_query, fields=None, expand=None, page_size=1000
    ):
        """Fallback search using /rest/api/3/search/jql (enhanced search).

        Uses GET with query parameters and nextPageToken-based pagination.
        Returns a list of raw issue dicts (same shape as Jira search response items).
        """
        # Reuse the authenticated session from jira client
        session = getattr(self.jira, "_session", None)
        if session is None:
            raise RuntimeError("JIRA client session unavailable for direct HTTP calls.")

        base = (self._server or "").rstrip("/")
        url = f"{base}/rest/api/3/search/jql"

        all_issues = []

        # Build base query params
        params = {
            "jql": jql_query,
            "maxResults": page_size,
        }
        if fields:
            # Atlassian accepts comma-separated values for arrays in query params
            params["fields"] = ",".join(fields)
        if expand:
            params["expand"] = (
                ",".join(expand) if isinstance(expand, (list, tuple)) else str(expand)
            )

        next_token = None
        while True:
            local_params = dict(params)
            if next_token:
                local_params["nextPageToken"] = next_token

            resp = session.get(url, params=local_params)
            if resp.status_code >= 400:
                # Raise helpful error surfaced to the caller
                try:
                    details = resp.json()
                except Exception:
                    details = {"text": resp.text}
                raise RuntimeError(
                    f"JQL search (enhanced) failed ({resp.status_code}): {details}"
                )

            data = resp.json()
            issues = data.get("issues", [])
            all_issues.extend(issues)

            next_token = data.get("nextPageToken")
            is_last = data.get("isLast")

            # Stop if there is no next token or API marks it as last page
            if not next_token or is_last is True:
                break

        return all_issues

    def _fetch_project_open_issues(self, project_key):
        """Fetch all open (non-Done) issues for an entire project (project-wide).

        Uses enhanced search with pagination and expands changelog for activity counts.
        """
        jql = f"project = {project_key} AND statusCategory != Done"
        fields = [
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
            "customfield_10106",
            "issuetype",
            "labels",
            "customfield_10476",
        ]
        expand = ["changelog", "renderedFields"]
        try:
            raw_issues = self._search_issues_v3_post(jql, fields=fields, expand=expand)
            return self._issues_to_dataframe(raw_issues)
        except Exception as e:
            st.error(f"Failed to fetch project-wide issues: {e}")
            return pd.DataFrame()

    def _issues_to_dataframe(self, issues):
        """Convert either jira.Issue objects or raw dicts to the expected DataFrame."""
        rows = []
        for issue in issues:
            # Support both Resource objects and raw dicts
            if hasattr(issue, "key") and hasattr(issue, "fields"):
                # jira.Issue object path
                fields = issue.fields
                changelog = getattr(issue, "changelog", None)
                comments = getattr(getattr(fields, "comment", None), "comments", [])
                histories = getattr(changelog, "histories", []) if changelog else []
                # OBS System, Sub-System, and Component (custom field)
                obs_cf = getattr(fields, "customfield_10476", None)
                obs_components = self._parse_obs_components(obs_cf)
                obs_system, obs_subsystem, obs_component = self._extract_obs_hierarchy(
                    obs_cf
                )
                labels = list(getattr(fields, "labels", []) or [])
                issuetype = getattr(getattr(fields, "issuetype", None), "name", None)
                resolutiondate = getattr(fields, "resolutiondate", None)

                row = {
                    "key": issue.key,
                    "summary": getattr(fields, "summary", None),
                    "created": getattr(fields, "created", None),
                    "updated": getattr(fields, "updated", None),
                    "time_lost": getattr(fields, "customfield_10106", None),
                    "assignee": getattr(
                        getattr(fields, "assignee", None), "displayName", "Unassigned"
                    ),
                    "reporter": getattr(
                        getattr(fields, "reporter", None), "displayName", "Unknown"
                    ),
                    "priority": getattr(
                        getattr(fields, "priority", None), "name", "None"
                    ),
                    "status": getattr(getattr(fields, "status", None), "name", None),
                    "resolution": getattr(
                        getattr(fields, "resolution", None), "name", "Unresolved"
                    ),
                    "resolutiondate": resolutiondate,
                    "issuetype": issuetype,
                    "obs_components": obs_components,
                    "labels": labels,
                    "obs_system": obs_system,
                    "obs_subsystem": obs_subsystem,
                    "obs_component": obs_component,
                    "comments": [
                        {
                            "author": getattr(
                                getattr(c, "author", None), "displayName", None
                            ),
                            "created": getattr(c, "created", None),
                            "body": getattr(c, "body", None),
                        }
                        for c in comments
                    ],
                    "histories": [
                        {
                            "author": getattr(
                                getattr(h, "author", None), "displayName", None
                            ),
                            "created": getattr(h, "created", None),
                            "items": getattr(h, "items", []),
                        }
                        for h in histories
                    ],
                }
                rows.append(row)
            else:
                # Raw dict JSON path (from POST /search/jql)
                fields = issue.get("fields", {})
                changelog = issue.get("changelog", {})
                comments = fields.get("comment", {}).get("comments", [])
                histories = changelog.get("histories", [])
                obs_cf = fields.get("customfield_10476")
                obs_components = self._parse_obs_components(obs_cf)
                obs_system, obs_subsystem, obs_component = self._extract_obs_hierarchy(
                    obs_cf
                )
                labels = list(fields.get("labels", []) or [])
                issuetype = (
                    (fields.get("issuetype") or {}).get("name")
                    if fields.get("issuetype")
                    else None
                )
                resolutiondate = fields.get("resolutiondate")

                row = {
                    "key": issue.get("key"),
                    "summary": fields.get("summary"),
                    "created": fields.get("created"),
                    "updated": fields.get("updated"),
                    "time_lost": fields.get("customfield_10106"),
                    "assignee": (
                        (fields.get("assignee") or {}).get("displayName")
                        if fields.get("assignee")
                        else "Unassigned"
                    ),
                    "reporter": (
                        (fields.get("reporter") or {}).get("displayName")
                        if fields.get("reporter")
                        else "Unknown"
                    ),
                    "priority": (
                        (fields.get("priority") or {}).get("name")
                        if fields.get("priority")
                        else "None"
                    ),
                    "status": (fields.get("status") or {}).get("name")
                    if fields.get("status")
                    else None,
                    "resolution": (
                        (fields.get("resolution") or {}).get("name")
                        if fields.get("resolution")
                        else "Unresolved"
                    ),
                    "resolutiondate": resolutiondate,
                    "issuetype": issuetype,
                    "obs_components": obs_components,
                    "labels": labels,
                    "obs_system": obs_system,
                    "obs_subsystem": obs_subsystem,
                    "obs_component": obs_component,
                    "comments": [
                        {
                            "author": (c.get("author") or {}).get("displayName")
                            if c.get("author")
                            else None,
                            "created": c.get("created"),
                            "body": c.get("body"),
                        }
                        for c in comments
                    ],
                    "histories": [
                        {
                            "author": (h.get("author") or {}).get("displayName")
                            if h.get("author")
                            else None,
                            "created": h.get("created"),
                            "items": h.get("items", []),
                        }
                        for h in histories
                    ],
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        # Apply priority mapping if present
        if not df.empty:
            df["priority_value"] = df["priority"].apply(self.map_priority)
        return df

    def _parse_obs_components(self, value):
        """Extract list of names from customfield_10476 structure.

        The field shape appears as { "selection": [[ {"name": "LSSTCam", ...} ]] }.
        We traverse nested lists/dicts and collect any 'name' strings.
        """
        names = []

        def walk(node):
            if isinstance(node, dict):
                if "name" in node and isinstance(node["name"], str):
                    names.append(node["name"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        if value is not None:
            walk(value)
        # Deduplicate while preserving order
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    def _extract_obs_hierarchy(self, value):
        """Return first three hierarchical names (system, subsystem, component)."""
        names = []

        def dfs(node):
            if isinstance(node, dict):
                nm = node.get("name")
                if isinstance(nm, str):
                    names.append(nm)
                for v in node.values():
                    if isinstance(v, (list, dict)):
                        dfs(v)
            elif isinstance(node, list):
                for item in node:
                    if len(names) >= 3:
                        break
                    dfs(item)

        if value is not None:
            dfs(value)
        # Deduplicate preserve order
        seen = set()
        ordered = []
        for n in names:
            if n and n not in seen:
                ordered.append(n)
                seen.add(n)
        system = ordered[0] if len(ordered) > 0 else None
        subsystem = ordered[1] if len(ordered) > 1 else None
        component = ordered[2] if len(ordered) > 2 else None
        return system, subsystem, component

    def _add_aging_metrics(self, df):
        """Add aging and responsiveness metrics; non-mutating."""
        out = df.copy()
        santiago = pytz.timezone("America/Santiago")
        now = datetime.now(tz=santiago)
        # Parse created/updated; ensure tz-aware then convert
        out["created_dt"] = pd.to_datetime(out["created"], utc=True).dt.tz_convert(
            santiago
        )
        out["updated_dt"] = pd.to_datetime(out["updated"], utc=True).dt.tz_convert(
            santiago
        )
        # Compute ages
        out["days_open"] = (now - out["created_dt"]).dt.total_seconds() / 86400.0
        out["hours_since_update"] = (
            now - out["updated_dt"]
        ).dt.total_seconds() / 3600.0
        # Convenience: days since last update
        out["days_since_update"] = out["hours_since_update"] / 24.0
        # Time to resolution (if resolutiondate present)
        if "resolutiondate" in out.columns:
            out["resolution_dt"] = pd.to_datetime(
                out["resolutiondate"], utc=True, errors="coerce"
            ).dt.tz_convert(santiago)
            out["time_to_resolution_hours"] = (
                out["resolution_dt"] - out["created_dt"]
            ).dt.total_seconds() / 3600.0
        else:
            out["time_to_resolution_hours"] = pd.NA
        # First response time: earliest comment created - issue created

        def first_response_hours(row):
            comments = row.get("comments", [])
            if not comments:
                return pd.NA
            try:
                ts = min(pd.to_datetime([c.get("created") for c in comments], utc=True))
                ts = ts.tz_convert(santiago)
                return (ts - row["created_dt"]).total_seconds() / 3600.0
            except Exception:
                return pd.NA

        out["time_to_first_response_hours"] = out.apply(first_response_hours, axis=1)
        return out

    def _compute_weighted_activity(
        self,
        df,
        start_date,
        end_date,
        w_comment=1.0,
        w_status=2.0,
        w_other=0.5,
    ):
        """Compute weighted activity counts within range; non-mutating."""
        out = df.copy()
        # Comments
        out["comments_in_range"] = out["comments"].apply(
            lambda comments: sum(
                self._is_within_range(c.get("created"), start_date, end_date)
                for c in comments
            )
        )
        # Histories broken into status vs other

        def hist_counts(histories):
            status_cnt = 0
            other_cnt = 0
            for h in histories:
                if not self._is_within_range(h.get("created"), start_date, end_date):
                    continue
                items = h.get("items", [])
                if any(
                    (getattr(it, "field", None) or it.get("field")) == "status"
                    for it in items
                ):
                    status_cnt += 1
                else:
                    other_cnt += 1
            return pd.Series({"status_changes": status_cnt, "other_changes": other_cnt})

        hist_df = out["histories"].apply(hist_counts)
        out = pd.concat([out, hist_df], axis=1)
        out["activity_score_weighted"] = (
            out["comments_in_range"] * float(w_comment)
            + out["status_changes"] * float(w_status)
            + out["other_changes"] * float(w_other)
        )
        return out

    # Removed off-hours activity feature as per requirements

    def _aggregate_by_assignee(self, df, limit=20):
        out = df.copy()
        out["time_lost_value"] = pd.to_numeric(
            out.get("time_lost"), errors="coerce"
        ).fillna(0)
        out["total_activity_in_range"] = (
            out.get("comments_in_range", 0)
            + out.get("status_changes", 0)
            + out.get("other_changes", 0)
        )
        agg = (
            out.groupby("assignee", dropna=False)
            .agg(
                issues=("key", "count"),
                time_lost_sum=("time_lost_value", "sum"),
                activity_sum=("total_activity_in_range", "sum"),
            )
            .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
            .head(limit)
        )
        return agg.reset_index()

    def _aggregate_by_obs_component(self, df, limit=20):
        out = df.copy()
        out["obs_components"] = out["obs_components"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        exploded = out.explode("obs_components")
        exploded["time_lost_value"] = pd.to_numeric(
            exploded.get("time_lost"), errors="coerce"
        ).fillna(0)
        exploded["total_activity_in_range"] = (
            exploded.get("comments_in_range", 0)
            + exploded.get("status_changes", 0)
            + exploded.get("other_changes", 0)
        )
        agg = (
            exploded.groupby("obs_components", dropna=False)
            .agg(
                issues=("key", "count"),
                time_lost_sum=("time_lost_value", "sum"),
                activity_sum=("total_activity_in_range", "sum"),
            )
            .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
            .head(limit)
        )
        return agg.reset_index()

    def _aggregate_by_obs_system(self, df, limit=20):
        out = df.copy()
        out["time_lost_value"] = pd.to_numeric(
            out.get("time_lost"), errors="coerce"
        ).fillna(0)
        out["total_activity_in_range"] = (
            out.get("comments_in_range", 0)
            + out.get("status_changes", 0)
            + out.get("other_changes", 0)
        )
        agg = (
            out.groupby("obs_system", dropna=False)
            .agg(
                issues=("key", "count"),
                time_lost_sum=("time_lost_value", "sum"),
                activity_sum=("total_activity_in_range", "sum"),
            )
            .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
            .head(limit)
        )
        return agg.reset_index()

    def _aggregate_by_obs_subsystem(self, df, limit=20):
        out = df.copy()
        out["time_lost_value"] = pd.to_numeric(
            out.get("time_lost"), errors="coerce"
        ).fillna(0)
        out["total_activity_in_range"] = (
            out.get("comments_in_range", 0)
            + out.get("status_changes", 0)
            + out.get("other_changes", 0)
        )
        agg = (
            out.groupby("obs_subsystem", dropna=False)
            .agg(
                issues=("key", "count"),
                time_lost_sum=("time_lost_value", "sum"),
                activity_sum=("total_activity_in_range", "sum"),
            )
            .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
            .head(limit)
        )
        return agg.reset_index()

    def _aggregate_by_obs_component_detail(self, df, limit=20):
        out = df.copy()
        out["time_lost_value"] = pd.to_numeric(
            out.get("time_lost"), errors="coerce"
        ).fillna(0)
        out["total_activity_in_range"] = (
            out.get("comments_in_range", 0)
            + out.get("status_changes", 0)
            + out.get("other_changes", 0)
        )
        agg = (
            out.groupby("obs_component", dropna=False)
            .agg(
                issues=("key", "count"),
                time_lost_sum=("time_lost_value", "sum"),
                activity_sum=("total_activity_in_range", "sum"),
            )
            .sort_values(by=["time_lost_sum", "activity_sum"], ascending=False)
            .head(limit)
        )
        return agg.reset_index()

    def _trend_critical_blocker(self, df, start_date, end_date):
        tz = pytz.timezone("America/Santiago")
        tmp = df.copy()
        tmp = tmp[
            tmp["priority"].apply(
                lambda x: str(x).startswith("Blocker") or str(x).startswith("Critical")
            )
        ]
        tmp = tmp[
            tmp["updated"].apply(
                lambda s: self._is_within_range(s, start_date, end_date)
                if pd.notna(s)
                else False
            )
        ]
        tmp["updated_dt"] = pd.to_datetime(tmp["updated"], utc=True).dt.tz_convert(tz)
        tmp["date"] = tmp["updated_dt"].dt.date
        trend = tmp.groupby("date").size().reset_index(name="count")
        return trend

    def _compute_created_daily(
        self, df: pd.DataFrame, start_date, end_date, priorities=None
    ):
        """Return (full_counts, events_df) for tickets created in range.

        full_counts: DataFrame with columns [date, count, tickets]
        events_df: DataFrame of individual created events with created_dt and date
        """
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame()

        tz = pytz.timezone("America/Santiago")
        tmp = df.copy()
        if priorities:
            tmp = tmp[tmp["priority"].astype(str).isin(priorities)]

        try:
            tmp["created_dt"] = pd.to_datetime(tmp["created"], utc=True).dt.tz_convert(
                tz
            )
        except Exception:
            tmp["created_dt"] = pd.NaT
        tmp = tmp[tmp["created_dt"].notna()]

        # Filter to range and compute date
        tmp = tmp[(tmp["created_dt"] >= start_date) & (tmp["created_dt"] <= end_date)]
        if tmp.empty:
            # Build zero-filled frame for the range
            all_dates = pd.date_range(start_date.date(), end_date.date(), freq="D")
            full_counts = pd.DataFrame(
                {"date": all_dates.date, "count": 0, "tickets": ""}
            )
            return full_counts, pd.DataFrame()

        tmp["date"] = tmp["created_dt"].dt.date

        def concat_tickets(group: pd.DataFrame) -> str:
            seen = set()
            items = []
            for k, s in zip(group["key"], group["summary"]):
                if k not in seen:
                    items.append(f"{k}: {s}")
                    seen.add(k)
            return "; ".join(items)

        agg = (
            tmp.groupby("date")
            .apply(
                lambda g: pd.Series(
                    {"count": int(len(g)), "tickets": concat_tickets(g)}
                )
            )
            .reset_index()
        )

        all_dates = pd.date_range(start_date.date(), end_date.date(), freq="D")
        full = pd.DataFrame({"date": all_dates.date}).merge(agg, on="date", how="left")
        full["count"] = full["count"].fillna(0).astype(int)
        full["tickets"] = full["tickets"].fillna("")
        return full, tmp

    def _build_created_trend_chart(self, full_counts: pd.DataFrame, selected_date=None):
        """Altair chart for created tickets with weekend shading and hover."""
        if full_counts is None or full_counts.empty:
            return None
        # Infer all_dates back from the frame
        all_dates = pd.to_datetime(full_counts["date"])
        color = "#1f77b4"
        line = (
            alt.Chart(full_counts)
            .mark_line(color=color)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("count:Q", title="Tickets created"),
            )
        )
        points = (
            alt.Chart(full_counts)
            .mark_circle(color=color, opacity=0.6, size=60)
            .encode(
                x="date:T",
                y="count:Q",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("tickets:N", title="Tickets"),
                ],
            )
        )
        # Weekend shading
        rng_df = pd.DataFrame({"date": all_dates})
        rng_df["weekday"] = rng_df["date"].dt.weekday
        weekend = rng_df[rng_df["weekday"].isin([5, 6])].copy()
        weekend["date_end"] = weekend["date"] + pd.Timedelta(days=1)
        rect = (
            alt.Chart(weekend)
            .mark_rect(color="#f2f2f2")
            .encode(x="date:T", x2="date_end:T")
        )
        # Selected date marker
        if selected_date is not None:
            sel_df = pd.DataFrame({"date": [pd.to_datetime(selected_date)]})
            rule = (
                alt.Chart(sel_df)
                .mark_rule(color="#444", strokeDash=[4, 4])
                .encode(x="date:T")
            )
            return (rect + line + points + rule).properties(height=300)
        return (rect + line + points).properties(height=300)

    def _add_ticket_link(
        self, df: pd.DataFrame, key_col: str = "key", link_label: str = "Ticket"
    ):
        """Return (df_with_link, column_config) adding a clickable Jira link column.

        - Adds a column with Jira browse URLs built from the issue key.
        - Returns a column_config dict to render it as a clickable link in Streamlit.
        """
        try:
            if df is None or df.empty or key_col not in df.columns:
                return df, {}
            base = str(getattr(self, "_server", "")).rstrip("/")
            out = df.copy()
            out[link_label] = (
                out[key_col]
                .astype(str)
                .apply(lambda k: f"{base}/browse/{k}" if k and k != "nan" else "")
            )
            cfg = {
                link_label: st.column_config.LinkColumn(
                    link_label,
                    display_text=r"browse/(.*)$",
                    help="Open in Jira",
                    width="medium",
                )
            }
            return out, cfg
        except Exception:
            return df, {}

    def fetch_issues_with_details(self, project_key, start_date, end_date):
        try:
            # Format the start and end dates in 'yyyy-MM-dd' format
            end_date_extended = end_date + timedelta(days=1)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_extended_str = end_date_extended.strftime("%Y-%m-%d")

            # Create the JQL query to fetch the issues
            jql_query = (
                f"project = {project_key} AND updated >= '{start_date_str}' "
                f"AND updated <= '{end_date_extended_str}'"
            )
            try:
                # First try using the jira client (v3). Some Cloud sites
                # still accept this path.
                issues = self.jira.search_issues(
                    jql_query,
                    fields=[
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
                        "customfield_10106",
                        "issuetype",
                        "labels",
                        "customfield_10476",
                    ],
                    expand="renderedFields,changelog",
                    maxResults=False,
                )
                self.issues_df = self._issues_to_dataframe(issues)
                return self.issues_df.sort_values(by="updated", ascending=False)
            except JIRAError as je:
                # If the endpoint is removed (410), fall back to the new POST API
                if getattr(je, "status_code", None) == 410 or "has been removed" in str(
                    je
                ):
                    fields = [
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
                        "customfield_10106",
                        "issuetype",
                        "labels",
                        "customfield_10476",
                    ]
                    expand = ["changelog", "renderedFields"]
                    raw_issues = self._search_issues_v3_post(
                        jql_query, fields=fields, expand=expand
                    )
                    self.issues_df = self._issues_to_dataframe(raw_issues)
                    return self.issues_df.sort_values(by="updated", ascending=False)
                # Not a 410 -> surface the original error
                raise

        except Exception as e:
            st.error(f"Failed to fetch data from JIRA: {e}")
            # Return an empty DataFrame in case of failure
            return pd.DataFrame()

    def _is_within_range(self, date_str, start_date, end_date):
        date_format = "%Y-%m-%dT%H:%M:%S.%f%z"
        entry_date = datetime.strptime(date_str, date_format)

        # Convert entry_date to 'America/Santiago' timezone
        santiago_timezone = pytz.timezone("America/Santiago")
        entry_date_santiago = entry_date.astimezone(santiago_timezone)

        # Ensure start_date and end_date are adjusted to 'America/Santiago'
        start_date_santiago = start_date.astimezone(santiago_timezone)
        end_date_santiago = end_date.astimezone(santiago_timezone)

        return start_date_santiago <= entry_date_santiago <= end_date_santiago

    def get_most_active_tickets(self, start_date, end_date):
        # Filter comments and histories within the date range before counting
        self.issues_df["filtered_comments_count"] = self.issues_df["comments"].apply(
            lambda comments: sum(
                self._is_within_range(comment["created"], start_date, end_date)
                for comment in comments
            )
        )
        self.issues_df["filtered_histories_count"] = self.issues_df["histories"].apply(
            lambda histories: sum(
                self._is_within_range(history["created"], start_date, end_date)
                for history in histories
            )
        )
        self.issues_df["activity_score"] = (
            self.issues_df["filtered_comments_count"]
            + self.issues_df["filtered_histories_count"]
        )

        return self.issues_df.sort_values(by="activity_score", ascending=False)

    def get_most_commented_tickets(self, start_date, end_date):
        # Apply the same logic for comments
        self.issues_df["filtered_comments_count"] = self.issues_df["comments"].apply(
            lambda comments: sum(
                self._is_within_range(comment["created"], start_date, end_date)
                for comment in comments
            )
        )

        return self.issues_df.sort_values(by="filtered_comments_count", ascending=False)

    def get_most_active_tickets_from_specific_user(
        self, user_name, start_date, end_date
    ):
        # Count user-specific comments within the date range
        self.issues_df["user_filtered_comment_count"] = self.issues_df[
            "comments"
        ].apply(
            lambda comments: sum(
                self._is_within_range(comment["created"], start_date, end_date)
                and comment["author"] == user_name
                for comment in comments
            )
        )

        # Count user-specific history entries within the date range
        self.issues_df["user_filtered_history_count"] = self.issues_df[
            "histories"
        ].apply(
            lambda histories: sum(
                self._is_within_range(history["created"], start_date, end_date)
                and history["author"] == user_name
                for history in histories
            )
        )

        # Sum the counts to get total activity count for the user
        self.issues_df["user_total_activity_score"] = (
            self.issues_df["user_filtered_comment_count"]
            + self.issues_df["user_filtered_history_count"]
        )

        return self.issues_df.sort_values(
            by="user_total_activity_score", ascending=False
        )

    def get_blocker_critical_tickets(self, start_date, end_date):
        # Filter the DataFrame for tickets with the required priorities
        filtered_df = self.issues_df[
            self.issues_df["priority"].apply(
                lambda x: x.startswith("Blocker") or x.startswith("Critical")
            )
            & (self.issues_df["status"] != "Done")
        ].copy()

        # Calculate total activity (comments + history) within the date range
        filtered_df["total_activity"] = filtered_df["comments"].apply(
            lambda comments: sum(
                self._is_within_range(comment["created"], start_date, end_date)
                for comment in comments
            )
        ) + filtered_df["histories"].apply(
            lambda histories: sum(
                self._is_within_range(history["created"], start_date, end_date)
                for history in histories
            )
        )

        # Return the DataFrame sorted by total activity
        return filtered_df.sort_values(by="total_activity", ascending=False)

    def get_testing_and_tracking_tickets(self, start_date, end_date):
        # Filter the DataFrame for tickets with the required priorities
        filtered_df = self.issues_df[
            self.issues_df["status"].apply(
                lambda x: x.startswith("Testing") or x.startswith("Tracking")
            )
        ].copy()

        # Calculate total activity (comments + history) within the date range
        def calculate_total_activity(row):
            try:
                total_comments = sum(
                    self._is_within_range(comment["created"], start_date, end_date)
                    for comment in row["comments"]
                )
                total_histories = sum(
                    self._is_within_range(history["created"], start_date, end_date)
                    for history in row["histories"]
                )
                return total_comments + total_histories
            except Exception as e:
                print(f"Error in row: {row['key']}, {e}")
                return 0

        filtered_df["total_activity"] = filtered_df.apply(
            calculate_total_activity, axis=1
        )

        # Return the DataFrame sorted by total activity
        return filtered_df.sort_values(by="total_activity", ascending=False)

    def get_most_time_lost_tickets(self, start_date, end_date):
        """Return tickets within range sorted by numeric time loss (descending)."""
        if self.issues_df is None or self.issues_df.empty:
            return pd.DataFrame()

        df = self.issues_df.copy()
        # Filter by updated timestamp within the selected range
        df = df[
            df["updated"].apply(
                lambda s: self._is_within_range(s, start_date, end_date)
                if pd.notna(s)
                else False
            )
        ]
        # Coerce to numeric; non-numeric will become NaN -> 0
        df["time_lost_value"] = pd.to_numeric(
            df.get("time_lost"), errors="coerce"
        ).fillna(0)
        return df.sort_values(by="time_lost_value", ascending=False)

    def visualize_tickets(
        self, df, column, title, explanation, limit=10, show_pie=False
    ):
        if (
            column in df.columns
            and not df.empty
            and not df[column].isnull().any()
            and df[column].sum() > 0
        ):
            st.write(f"## {title}")
            st.write(explanation)

            # Layout: either table-only or table + pie chart
            if show_pie:
                col1, col2 = st.columns([3, 1])
            else:
                col1 = st.container()
                col2 = None

            # Convert 'created' and 'updated' to Santiago time
            santiago_timezone = pytz.timezone("America/Santiago")
            df["created"] = pd.to_datetime(df["created"]).dt.tz_convert(
                santiago_timezone
            )

            df["updated"] = pd.to_datetime(df["updated"]).dt.tz_convert(
                santiago_timezone
            )

            with col1:
                columns = [
                    column,
                    "key",
                    "summary",
                    "priority",
                    "priority_value",
                    "status",
                    "time_lost",
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
                ]

                # Ensure columns exist
                for c in columns:
                    if c not in df.columns:
                        df[c] = pd.NA

                df_display = df[columns].head(limit)
                df_display = df_display.rename(columns={column: "Total Count"})
                # Add clickable Ticket link
                df_linked, col_cfg = self._add_ticket_link(df_display)
                order = [
                    "Ticket",
                    "summary",
                    "priority",
                    "priority_value",
                    "status",
                    "time_lost",
                    "days_open",
                    "obs_system",
                    "obs_subsystem",
                    "obs_component",
                    "days_since_update",
                    "created",
                    "updated",
                    "assignee",
                    "reporter",
                    "resolution",
                    "Total Count",
                ]
                show_cols = [c for c in order if c in df_linked.columns]
                st.dataframe(
                    df_linked[show_cols], hide_index=True, column_config=col_cfg
                )

            if show_pie and col2 is not None:
                with col2:
                    # Create a pie chart
                    fig, ax = plt.subplots()
                    df_head = df.head(limit)  # Limiting to top N for the pie chart
                    ax.pie(
                        df_head[column],
                        labels=df_head["key"],
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    ax.axis("equal")
                    ax.set_title(f"Top {min(limit, len(df_head))} tickets")
                    st.pyplot(fig)

        else:
            st.write(f"## {title}")
            st.write("No data available to display. ")

    def display_all_tickets(self):
        """Render an enriched full table for the currently queried issues."""
        if self.issues_df is None or self.issues_df.empty:
            st.write("No data to display. Please fetch the data first.")
            return

        df_display = self.issues_df.copy()

        # Convert 'created' and 'updated' to Santiago time
        santiago_timezone = pytz.timezone("America/Santiago")
        try:
            df_display["created"] = pd.to_datetime(df_display["created"]).dt.tz_convert(
                santiago_timezone
            )
            df_display["updated"] = pd.to_datetime(df_display["updated"]).dt.tz_convert(
                santiago_timezone
            )
        except Exception:
            pass

        # Pre-compute counts and ensure aging fields exist
        df_display["comment_count"] = df_display.get("comments", []).apply(len)
        df_display["history_count"] = df_display.get("histories", []).apply(len)
        for c in ["days_open", "days_since_update"]:
            if c not in df_display.columns:
                df_display[c] = pd.NA

        st.write("## All Queried Tickets (enriched)")
        st.caption(
            "Full list of issues fetched for the selected period, enriched with aging and counts."
        )
        df_linked, col_cfg = self._add_ticket_link(df_display)
        order = [
            "Ticket",
            "summary",
            "priority",
            "priority_value",
            "status",
            "time_lost",
            "days_open",
            "obs_system",
            "obs_subsystem",
            "obs_component",
            "days_since_update",
            "created",
            "updated",
            "assignee",
            "reporter",
            "resolution",
            "comment_count",
            "history_count",
        ]
        show_cols = [c for c in order if c in df_linked.columns]
        st.dataframe(df_linked[show_cols], hide_index=True, column_config=col_cfg)
        csv_all = df_linked[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Queried CSV",
            data=csv_all,
            file_name="jira_all_queried_enriched.csv",
            mime="text/csv",
        )

    def display(self):
        st.sidebar.title("JIRA Issue Analysis Dashboard")
        project_key = st.sidebar.text_input("Project Key", value="OBS")

        default_end_date = date.today()
        default_start_date = default_end_date - timedelta(days=7)

        start_date = st.sidebar.date_input("Start Date", value=default_start_date)

        end_date = st.sidebar.date_input("End Date", value=default_end_date)

        user_for_comment_analysis = st.sidebar.text_input(
            "User for Comment Analysis", value="Rubin Jira API Access"
        )

        # Number of top tickets to display in tables and charts
        top_n = st.sidebar.number_input(
            "Top N tickets to display",
            min_value=1,
            max_value=1000,
            value=10,
            step=5,
        )

        # Optional pie charts
        show_pies = st.sidebar.checkbox(
            "Show pie charts (optional)",
            value=False,
            help="When enabled, sections will include a pie chart.",
        )

        # Priority filter for trend (affects trend only)
        default_priorities = ["Blocker", "Critical", "High", "Medium", "Low"]
        existing_priorities = []
        if getattr(self, "issues_df", None) is not None and isinstance(
            self.issues_df, pd.DataFrame
        ):
            try:
                existing_priorities = (
                    self.issues_df.get("priority")
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                )
            except Exception:
                existing_priorities = []
        options = existing_priorities or default_priorities
        selected_priorities = st.sidebar.multiselect(
            "Trend priorities",
            options=options,
            default=options,
            help=(
                "Which priorities to include in the combined trend. "
                "Updated (CB) always uses Blocker/Critical subset when present."
            ),
        )

        # Stale tickets threshold moved to dedicated Stale Tickets view

        # Advanced analysis controls
        with st.sidebar.expander("Advanced metrics", expanded=False):
            weight_comment = st.number_input(
                "Weight: comment",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Points added per comment within the selected period.",
            )
            weight_status = st.number_input(
                "Weight: status change",
                min_value=0.0,
                max_value=10.0,
                value=2.0,
                step=0.1,
                help=(
                    "Points added per status transition recorded in the changelog "
                    "within the selected period."
                ),
            )
            weight_other = st.number_input(
                "Weight: other change",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.1,
                help="Points added per non-status changelog entry within the selected period.",
            )
            crit_age_days = st.number_input(
                "Flag Critical/Blocker older than (days)",
                min_value=0,
                max_value=365,
                value=7,
                help="Threshold for days_open used to flag aging Critical/Blocker tickets.",
            )

        if st.sidebar.button("Fetch and Analyze"):
            with st.spinner("Fetching and analyzing data..."):
                # Convert input dates to datetime objects with
                # 'America/Santiago' timezone
                santiago_timezone = pytz.timezone("America/Santiago")

                start_datetime = santiago_timezone.localize(
                    datetime.combine(start_date, datetime.min.time())
                )

                end_datetime = santiago_timezone.localize(
                    datetime.combine(end_date, datetime.max.time())
                )

                # Fetch the data from JIRA
                fetched_data = self.fetch_issues_with_details(
                    project_key, start_datetime, end_datetime
                )

                if fetched_data.empty:
                    st.error(
                        "No data fetched. Please check the JIRA connection "
                        "and query."
                    )
                else:
                    # Enrich with metrics and activity weights
                    enriched_df = self._add_aging_metrics(self.issues_df)
                    enriched_df = self._compute_weighted_activity(
                        enriched_df,
                        start_datetime,
                        end_datetime,
                        w_comment=weight_comment,
                        w_status=weight_status,
                        w_other=weight_other,
                    )
                    # Store enriched
                    self.issues_df = enriched_df

                    # Combined Critical/Blocker activity + aging
                    blocker_critical_df = self.get_blocker_critical_tickets(
                        start_datetime, end_datetime
                    )
                    st.write("## Critical & Blocker: Activity and Aging")
                    st.caption(
                        "Blocker/Critical tickets (excluding Done). Activity within the selected "
                        "period, plus age fields."
                    )
                    if not blocker_critical_df.empty:
                        # Warn on aging items over threshold
                        aging_mask = blocker_critical_df.get("days_open", 0) > float(
                            crit_age_days
                        )
                        aging_count = int(aging_mask.sum())
                        if aging_count > 0:
                            st.warning(
                                f"{aging_count} Critical/Blocker tickets older than {crit_age_days} days."
                            )

                        # Prepare display table with days_since_update
                        disp = blocker_critical_df.copy()
                        disp["days_open"] = (
                            pd.to_numeric(disp.get("days_open", 0), errors="coerce")
                            .fillna(0)
                            .astype(int)
                        )
                        disp["days_since_update"] = (
                            pd.to_numeric(
                                disp.get("days_since_update", 0), errors="coerce"
                            )
                            .fillna(0)
                            .astype(int)
                        )
                        # Convert timestamps to local timezone for display
                        tz = pytz.timezone("America/Santiago")
                        try:
                            disp["created"] = pd.to_datetime(
                                disp["created"]
                            ).dt.tz_convert(tz)
                            disp["updated"] = pd.to_datetime(
                                disp["updated"]
                            ).dt.tz_convert(tz)
                        except Exception:
                            pass
                        if show_pies:
                            col1, col2 = st.columns([3, 1])
                        else:
                            col1 = st.container()
                            col2 = None
                        with col1:
                            base_cols = [
                                "total_activity",
                                "key",
                                "summary",
                                "priority",
                                "priority_value",
                                "status",
                                "time_lost",
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
                            ]
                            for c in base_cols:
                                if c not in disp.columns:
                                    disp[c] = pd.NA
                            disp_linked, col_cfg = self._add_ticket_link(disp)
                            order = [
                                "Ticket",
                                "summary",
                                "priority",
                                "priority_value",
                                "status",
                                "time_lost",
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
                            ]
                            show_cols = [c for c in order if c in disp_linked.columns]
                            st.dataframe(
                                disp_linked[show_cols].head(int(top_n)),
                                hide_index=True,
                                column_config=col_cfg,
                            )
                        if show_pies and col2 is not None:
                            with col2:
                                fig, ax = plt.subplots()
                                head = disp.head(int(top_n))
                                ax.pie(
                                    head["total_activity"],
                                    labels=head["key"],
                                    autopct="%1.1f%%",
                                    startangle=90,
                                )
                                ax.axis("equal")
                                ax.set_title(
                                    f"Top {min(int(top_n), len(head))} tickets"
                                )
                                st.pyplot(fig)
                    else:
                        st.info(
                            "No Critical/Blocker tickets found in the selected period."
                        )

                    # Display Time Loss focused view
                    time_lost_df = self.get_most_time_lost_tickets(
                        start_datetime, end_datetime
                    )
                    if not time_lost_df.empty:
                        self.visualize_tickets(
                            time_lost_df,
                            "time_lost_value",
                            "Tickets with Highest Time Lost",
                            "This view ranks tickets by the total reported time lost.",
                            limit=int(top_n),
                            show_pie=show_pies,
                        )
                    else:
                        st.write("## Tickets with Highest Time Lost")
                        st.write("No time lost data available in this period.")

                    # Weighted activity view
                    weighted_df = self.issues_df.sort_values(
                        by="activity_score_weighted", ascending=False
                    )
                    self.visualize_tickets(
                        weighted_df,
                        "activity_score_weighted",
                        "Weighted Activity",
                        "Weighted sum of comments, status changes, and other changes in the selected period.",
                        limit=int(top_n),
                        show_pie=show_pies,
                    )

                    # Display Testing and Tracking tickets
                    testing_tracking_df = self.get_testing_and_tracking_tickets(
                        start_datetime, end_datetime
                    )
                    if not testing_tracking_df.empty:
                        self.visualize_tickets(
                            testing_tracking_df,
                            "total_activity",
                            "Testing and Tracking Tickets Activity",
                            "This table shows the total activity (comments + history) for Testing and "
                            "Tracking tickets.",
                            limit=int(top_n),
                            show_pie=show_pies,
                        )
                    else:
                        st.write("## Testing and Tracking Tickets Activity")
                        st.write(
                            "No Testing or Tracking tickets found in the specified period."
                        )

                    # Display the most attended tickets
                    attended_df = self.get_most_active_tickets(
                        start_datetime, end_datetime
                    )
                    self.visualize_tickets(
                        attended_df,
                        "activity_score",
                        "Most Active Tickets",
                        "This metric combines the total number of comments "
                        " and history entries to identify the most engaged "
                        "tickets.",
                        limit=int(top_n),
                        show_pie=show_pies,
                    )

                    # Store the fetched data in the session state
                    st.session_state["issues_df"] = self.issues_df

                    # Display the most commented tickets
                    commented_df = self.get_most_commented_tickets(
                        start_datetime, end_datetime
                    )
                    self.visualize_tickets(
                        commented_df,
                        "filtered_comments_count",
                        "Most Commented Tickets",
                        "This metric ranks tickets based on the total "
                        " number of comments they received.",
                        limit=int(top_n),
                        show_pie=show_pies,
                    )

                    # Display tickets with most comments from a specific user
                    user_commented_df = self.get_most_active_tickets_from_specific_user(
                        user_for_comment_analysis, start_datetime, end_datetime
                    )
                    self.visualize_tickets(
                        user_commented_df,
                        "user_total_activity_score",
                        "Most Active Tickets from Specific User",
                        "Tickets with the highest activity from the specified user (comments + history).",
                        limit=int(top_n),
                        show_pie=show_pies,
                    )

                    # Aggregations by OBS levels only (date-scoped)
                    st.write("## Aggregation by OBS System/Sub-System/Component")
                    st.caption("Groups by OBS System, Sub-System, and Component.")
                    col_sys, col_sub, col_comp = st.columns(3)
                    with col_sys:
                        st.caption("OBS System")
                        agg_sys = self._aggregate_by_obs_system(
                            self.issues_df, limit=50
                        )
                        st.dataframe(agg_sys.head(int(top_n)), hide_index=True)
                    with col_sub:
                        st.caption("OBS Sub-System")
                        agg_sub = self._aggregate_by_obs_subsystem(
                            self.issues_df, limit=50
                        )
                        st.dataframe(agg_sub.head(int(top_n)), hide_index=True)
                    with col_comp:
                        st.caption("OBS Component")
                        agg_comp = self._aggregate_by_obs_component_detail(
                            self.issues_df, limit=50
                        )
                        st.dataframe(agg_comp.head(int(top_n)), hide_index=True)

                    # Cache views for later (word cloud, no re-query on select)
                    st.session_state["views"] = {
                        "Blocker/Critical": blocker_critical_df,
                        "Time Lost": time_lost_df,
                        "Weighted Activity": weighted_df,
                        "Testing/Tracking": testing_tracking_df,
                        "Most Active": attended_df,
                        "Most Commented": commented_df,
                        "Agg OBS System": agg_sys,
                        "Agg OBS Sub-System": agg_sub,
                        "Agg OBS Component": agg_comp,
                    }

                    # Daily Created Trend (with details on selected day)
                    st.write("## Daily New Tickets (Created)")
                    st.caption("New tickets created per day in the selected period.")
                    full_counts, created_events = self._compute_created_daily(
                        self.issues_df,
                        start_datetime,
                        end_datetime,
                        priorities=selected_priorities,
                    )
                    # Date selector to view details
                    sel_date = st.date_input(
                        "Select a day to view new tickets",
                        value=end_date,
                        min_value=start_date,
                        max_value=end_date,
                        key="created_trend_date",
                    )
                    created_chart = self._build_created_trend_chart(
                        full_counts, selected_date=sel_date
                    )
                    if created_chart is not None:
                        st.altair_chart(
                            created_chart.interactive(), use_container_width=True
                        )
                    # Details table for the selected date
                    if not created_events.empty and sel_date is not None:
                        day_rows = created_events[created_events["date"] == sel_date]
                        if not day_rows.empty:
                            st.subheader(f"New tickets on {sel_date}")
                            cols = [
                                "key",
                                "summary",
                                "priority",
                                "assignee",
                                "reporter",
                                "status",
                                "labels",
                                "obs_system",
                                "obs_subsystem",
                                "obs_component",
                                "created_dt",
                            ]
                            for c in cols:
                                if c not in day_rows.columns:
                                    day_rows[c] = pd.NA
                            linked_rows, col_cfg = self._add_ticket_link(day_rows)
                            order = [
                                "Ticket",
                                "summary",
                                "priority",
                                "assignee",
                                "reporter",
                                "status",
                                "labels",
                                "obs_system",
                                "obs_subsystem",
                                "obs_component",
                                "created_dt",
                            ]
                            show_cols = [c for c in order if c in linked_rows.columns]
                            st.dataframe(
                                linked_rows[show_cols].sort_values("created_dt"),
                                hide_index=True,
                                column_config=col_cfg,
                            )

                    # Full enriched table for queried issues
                    self.display_all_tickets()

            st.success("Analysis completed!")

        # If we already have views cached (prior fetch), re-render main sections on rerun
        elif "views" in st.session_state:
            # Restore issues for trend rendering across navigation
            if (
                getattr(self, "issues_df", None) is None
                or (isinstance(self.issues_df, pd.DataFrame) and self.issues_df.empty)
            ) and "issues_df" in st.session_state:
                self.issues_df = st.session_state["issues_df"]
            # Recompute start/end datetimes from sidebar dates
            santiago_timezone = pytz.timezone("America/Santiago")
            start_datetime = santiago_timezone.localize(
                datetime.combine(start_date, datetime.min.time())
            )
            end_datetime = santiago_timezone.localize(
                datetime.combine(end_date, datetime.max.time())
            )
            views = st.session_state.get("views", {})
            # Critical & Blocker (re-render with same columns as Time Lost)
            st.write("## Critical & Blocker: Activity and Aging")
            st.caption(
                "Blocker/Critical tickets (excluding Done). Activity within the selected "
                "period, plus age fields."
            )
            blocker_critical_df = views.get("Blocker/Critical", pd.DataFrame()).copy()
            if not blocker_critical_df.empty:
                disp = blocker_critical_df.copy()
                disp["days_open"] = (
                    pd.to_numeric(disp.get("days_open", 0), errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                disp["days_since_update"] = (
                    pd.to_numeric(disp.get("days_since_update", 0), errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                tz = pytz.timezone("America/Santiago")
                try:
                    disp["created"] = pd.to_datetime(disp["created"]).dt.tz_convert(tz)
                    disp["updated"] = pd.to_datetime(disp["updated"]).dt.tz_convert(tz)
                except Exception:
                    pass
                cols = [
                    "total_activity",
                    "key",
                    "summary",
                    "priority",
                    "priority_value",
                    "status",
                    "time_lost",
                    "days_open",
                    "days_since_update",
                    "created",
                    "updated",
                    "assignee",
                    "reporter",
                    "resolution",
                ]
                for c in cols:
                    if c not in disp.columns:
                        disp[c] = pd.NA
                disp_linked, col_cfg = self._add_ticket_link(disp)
                order = [
                    "Ticket",
                    "summary",
                    "priority",
                    "priority_value",
                    "status",
                    "time_lost",
                    "days_open",
                    "days_since_update",
                    "created",
                    "updated",
                    "assignee",
                    "reporter",
                    "resolution",
                ]
                show_cols = [c for c in order if c in disp_linked.columns]
                st.dataframe(
                    disp_linked[show_cols].head(int(top_n)),
                    hide_index=True,
                    column_config=col_cfg,
                )
            else:
                st.info("No Critical/Blocker tickets found in the selected period.")

            # Other sections from cached views
            time_lost_df = views.get("Time Lost", pd.DataFrame())
            if not time_lost_df.empty:
                self.visualize_tickets(
                    time_lost_df,
                    "time_lost_value",
                    "Tickets with Highest Time Lost",
                    "This view ranks tickets by the total reported time lost.",
                    limit=int(top_n),
                    show_pie=show_pies,
                )
            weighted_df = views.get("Weighted Activity", pd.DataFrame())
            if not weighted_df.empty:
                self.visualize_tickets(
                    weighted_df,
                    "activity_score_weighted",
                    "Weighted Activity",
                    "Weighted sum of comments, status changes, and other changes in the selected period.",
                    limit=int(top_n),
                    show_pie=show_pies,
                )
            testing_tracking_df = views.get("Testing/Tracking", pd.DataFrame())
            if not testing_tracking_df.empty:
                self.visualize_tickets(
                    testing_tracking_df,
                    "total_activity",
                    "Testing and Tracking Tickets Activity",
                    "This table shows the total activity (comments + history) "
                    "for Testing and Tracking tickets.",
                    limit=int(top_n),
                    show_pie=show_pies,
                )
            attended_df = views.get("Most Active", pd.DataFrame())
            if not attended_df.empty:
                self.visualize_tickets(
                    attended_df,
                    "activity_score",
                    "Most Active Tickets",
                    "This metric combines the total number of comments and history entries "
                    "to identify the most engaged tickets.",
                    limit=int(top_n),
                    show_pie=show_pies,
                )
            commented_df = views.get("Most Commented", pd.DataFrame())
            if not commented_df.empty:
                self.visualize_tickets(
                    commented_df,
                    "filtered_comments_count",
                    "Most Commented Tickets",
                    "This metric ranks tickets based on the total  number of comments they received.",
                    limit=int(top_n),
                    show_pie=show_pies,
                )

            # Daily Created Trend also in cached view
            if (
                getattr(self, "issues_df", None) is not None
                and not self.issues_df.empty
            ):
                st.write("## Daily New Tickets (Created)")
                st.caption("New tickets created per day in the selected period.")
                full_counts, created_events = self._compute_created_daily(
                    self.issues_df,
                    start_datetime,
                    end_datetime,
                    priorities=selected_priorities,
                )
                sel_date = st.date_input(
                    "Select a day to view new tickets",
                    value=end_date,
                    min_value=start_date,
                    max_value=end_date,
                    key="created_trend_date_cached",
                )
                created_chart = self._build_created_trend_chart(
                    full_counts, selected_date=sel_date
                )
                if created_chart is not None:
                    st.altair_chart(
                        created_chart.interactive(), use_container_width=True
                    )
                if not created_events.empty and sel_date is not None:
                    day_rows = created_events[created_events["date"] == sel_date]
                    if not day_rows.empty:
                        st.subheader(f"New tickets on {sel_date}")
                        cols = [
                            "key",
                            "summary",
                            "priority",
                            "assignee",
                            "reporter",
                            "status",
                            "labels",
                            "obs_system",
                            "obs_subsystem",
                            "obs_component",
                            "created_dt",
                        ]
                        for c in cols:
                            if c not in day_rows.columns:
                                day_rows[c] = pd.NA
                        linked_rows, col_cfg = self._add_ticket_link(day_rows)
                        order = [
                            "Ticket",
                            "summary",
                            "priority",
                            "assignee",
                            "reporter",
                            "status",
                            "labels",
                            "obs_system",
                            "obs_subsystem",
                            "obs_component",
                            "created_dt",
                        ]
                        show_cols = [c for c in order if c in linked_rows.columns]
                        st.dataframe(
                            linked_rows[show_cols].sort_values("created_dt"),
                            hide_index=True,
                            column_config=col_cfg,
                        )

            # Also show the enriched full table if we still have data
            if (
                getattr(self, "issues_df", None) is not None
                and not self.issues_df.empty
            ):
                self.display_all_tickets()

        # Word Cloud section (outside fetch) – uses cached views without re-query
        if "views" in st.session_state:
            st.write("## Quick Word Clouds")
            cols = st.columns(3)
            quick_keys = [
                "Blocker/Critical",
                "Time Lost",
                "Most Active",
            ]
            labels = [
                "Critical/Blocker",
                "Time Lost",
                "Most Active",
            ]
            for i, key_name in enumerate(quick_keys):
                with cols[i]:
                    df_src = st.session_state["views"].get(key_name, pd.DataFrame())
                    txt = " ".join(
                        (
                            df_src.head(int(top_n))["summary"].dropna().astype(str)
                        ).tolist()
                    )
                    if txt.strip():
                        png_small = _generate_wordcloud_png(
                            txt, width=360, height=160, bg="white"
                        )
                        if png_small:
                            st.image(
                                png_small, caption=labels[i], use_container_width=True
                            )
                        else:
                            st.caption(f"{labels[i]}: wordcloud unavailable")
                    else:
                        st.caption(f"{labels[i]}: no titles")

    def display_stale_tickets(self):
        """Standalone page to list stale tickets across a project."""
        st.sidebar.title("Stale Tickets")
        project_key = st.sidebar.text_input(
            "Project Key", value="OBS", key="stale_project_key"
        )
        stale_days_threshold = st.sidebar.number_input(
            "Mark stale if no update in N days",
            min_value=1,
            max_value=365,
            value=30,
            step=5,
            help="Threshold in days since last update to consider an open ticket stale.",
            key="stale_days_threshold",
        )
        fetch = st.sidebar.button("Fetch Stale Tickets", key="fetch_stale_btn")

        if fetch:
            with st.spinner(
                "Fetching project-wide open issues and computing staleness..."
            ):
                stale_all = self._fetch_project_open_issues(project_key)
                if stale_all.empty:
                    st.info("No project-wide open issues found or fetch failed.")
                    return

                # Enrich to compute days_since_update and counts
                stale_all = self._add_aging_metrics(stale_all)
                stale_all["comment_count"] = stale_all["comments"].apply(len)
                stale_all["history_count"] = stale_all["histories"].apply(len)
                stale_all["total_activity_all"] = (
                    stale_all["comment_count"] + stale_all["history_count"]
                )
                # Filter by threshold
                stale_all["days_since_update"] = pd.to_numeric(
                    stale_all["days_since_update"], errors="coerce"
                ).fillna(0)
                stale_all = stale_all[
                    stale_all["days_since_update"] >= float(stale_days_threshold)
                ]
                # Sort by days since update desc, low activity first
                stale_all = stale_all.sort_values(
                    by=["days_since_update", "total_activity_all"],
                    ascending=[False, True],
                )

                st.write("## Stale Tickets (project-wide)")
                st.caption(
                    "Open tickets across the whole project, filtered by days since last update, "
                    "and sorted by days since update (desc) and total activity (asc)."
                )
                cols = [
                    "key",
                    "summary",
                    "priority",
                    "priority_value",
                    "status",
                    "created",
                    "updated",
                    "days_open",
                    "days_since_update",
                    "obs_system",
                    "obs_subsystem",
                    "obs_component",
                    "assignee",
                    "reporter",
                    "comment_count",
                    "history_count",
                    "total_activity_all",
                    "issuetype",
                    "labels",
                ]
                for c in cols:
                    if c not in stale_all.columns:
                        stale_all[c] = pd.NA
                # Convert datetimes to local tz
                tz = pytz.timezone("America/Santiago")
                try:
                    stale_all["created"] = pd.to_datetime(
                        stale_all["created"]
                    ).dt.tz_convert(tz)
                    stale_all["updated"] = pd.to_datetime(
                        stale_all["updated"]
                    ).dt.tz_convert(tz)
                except Exception:
                    pass
                stale_linked, col_cfg = self._add_ticket_link(stale_all)
                show_cols = [
                    c
                    for c in [
                        "Ticket",
                        "summary",
                        "priority",
                        "priority_value",
                        "status",
                        "created",
                        "updated",
                        "days_open",
                        "days_since_update",
                        "obs_system",
                        "obs_subsystem",
                        "obs_component",
                        "assignee",
                        "reporter",
                        "comment_count",
                        "history_count",
                        "total_activity_all",
                        "issuetype",
                        "labels",
                    ]
                    if c in stale_linked.columns
                ]
                st.dataframe(
                    stale_linked[show_cols].head(1000),
                    hide_index=True,
                    column_config=col_cfg,
                )
                csv_stale = stale_all[cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Stale CSV",
                    data=csv_stale,
                    file_name=f"jira_stale_{project_key}.csv",
                    mime="text/csv",
                )

    def display_project_aggregations(self):
        """Standalone page to show project-wide aggregations and data hygiene lists.

        Uses the same project-wide open-issues fetch as the Stale Tickets page.
        """
        st.sidebar.title("Assignees & OBS Hierarchy")
        project_key = st.sidebar.text_input(
            "Project Key", value="OBS", key="agg_project_key"
        )
        mode = st.sidebar.radio(
            "Scope", ["All open issues", "By date range"], index=0, key="agg_scope"
        )
        if mode == "By date range":
            default_end = date.today()
            default_start = default_end - timedelta(days=30)
            start_date = st.sidebar.date_input(
                "Start Date", value=default_start, key="agg_start"
            )
            end_date = st.sidebar.date_input(
                "End Date", value=default_end, key="agg_end"
            )
        fetch = st.sidebar.button("Fetch Tickets", key="fetch_aggs_btn")

        if fetch:
            with st.spinner("Fetching issues and computing aggregations..."):
                if mode == "All open issues":
                    proj_open = self._fetch_project_open_issues(project_key)
                else:
                    # Date-scoped: query project for updated range (non-Done)
                    end_date_extended = end_date + timedelta(days=1)
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date_extended.strftime("%Y-%m-%d")
                    jql_query = (
                        f"project = {project_key} AND statusCategory != Done "
                        f"AND updated >= '{start_str}' AND updated < '{end_str}'"
                    )
                    try:
                        raw = self._search_issues_v3_post(
                            jql_query,
                            fields=[
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
                                "customfield_10106",
                                "issuetype",
                                "labels",
                                "customfield_10476",
                            ],
                            expand=["changelog", "renderedFields"],
                        )
                        proj_open = self._issues_to_dataframe(raw)
                    except Exception as e:
                        st.error(f"Failed to fetch issues: {e}")
                        proj_open = pd.DataFrame()
                if proj_open.empty:
                    st.info("No project-wide open issues found or fetch failed.")
                    return

                # Enrich for aging and activity counts
                proj_open = self._add_aging_metrics(proj_open)
                proj_open["comment_count"] = proj_open.get("comments", []).apply(len)
                proj_open["history_count"] = proj_open.get("histories", []).apply(len)
                proj_open["total_activity_all"] = (
                    proj_open["comment_count"] + proj_open["history_count"]
                )

                # Aggregations (project-wide, open issues)
                scope_lbl = (
                    "Project-wide (Open Issues)"
                    if mode == "All open issues"
                    else "Open Issues in Range"
                )
                st.write(f"## Aggregations — {scope_lbl}")
                agg_assignee = self._aggregate_by_assignee(proj_open, limit=200)
                st.subheader("By Assignee")
                st.dataframe(agg_assignee, hide_index=True)

                # Assignee detail explorer (moved up for visibility)
                assignee_options = (
                    agg_assignee["assignee"]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                    .tolist()
                    if "assignee" in agg_assignee.columns and not agg_assignee.empty
                    else []
                )
                if assignee_options:
                    # Pick top assignee by issue count (fallback: first alphabetical)
                    top_assignee = None
                    try:
                        if (
                            not agg_assignee.empty
                            and "issues" in agg_assignee.columns
                            and "assignee" in agg_assignee.columns
                        ):
                            top_row = agg_assignee.sort_values(
                                "issues", ascending=False
                            ).iloc[0]
                            top_assignee = (
                                str(top_row["assignee"])
                                if pd.notna(top_row["assignee"])
                                else None
                            )
                    except Exception:
                        top_assignee = None

                    # Build list with optional (None) sentinel retained
                    select_list = ["(None)"] + assignee_options
                    default_index = 0
                    if top_assignee and top_assignee in assignee_options:
                        default_index = 1 + assignee_options.index(top_assignee)

                    selected_assignee = st.selectbox(
                        "Select an assignee to view their open tickets",
                        select_list,
                        index=default_index,
                        key="assignee_detail_select",
                    )
                    if selected_assignee and selected_assignee != "(None)":
                        detail_df = proj_open[
                            proj_open["assignee"].astype(str) == selected_assignee
                        ].copy()
                        if not detail_df.empty:
                            for col in ["days_open", "days_since_update"]:
                                if col in detail_df.columns:
                                    try:
                                        detail_df[col] = (
                                            pd.to_numeric(
                                                detail_df[col], errors="coerce"
                                            )
                                            .fillna(0)
                                            .astype(int)
                                        )
                                    except Exception:
                                        pass
                            tz = pytz.timezone("America/Santiago")
                            for dt_col in ["created", "updated"]:
                                if dt_col in detail_df.columns:
                                    try:
                                        detail_df[dt_col] = pd.to_datetime(
                                            detail_df[dt_col], utc=True
                                        ).dt.tz_convert(tz)
                                    except Exception:
                                        pass
                            display_cols = [
                                "key",
                                "summary",
                                "priority",
                                "priority_value",
                                "status",
                                "time_lost",
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
                            ]
                            for c in display_cols:
                                if c not in detail_df.columns:
                                    detail_df[c] = pd.NA
                            if "updated" in detail_df.columns:
                                try:
                                    detail_df = detail_df.sort_values(
                                        "updated", ascending=False
                                    )
                                except Exception:
                                    pass
                            linked, link_cfg = self._add_ticket_link(detail_df)
                            order = [
                                "Ticket",
                                "summary",
                                "priority",
                                "priority_value",
                                "status",
                                "time_lost",
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
                            ]
                            show_cols = [c for c in order if c in linked.columns]
                            st.subheader(f"Tickets Assigned to {selected_assignee}")
                            st.dataframe(
                                linked[show_cols].head(1000),
                                hide_index=True,
                                column_config=link_cfg,
                            )
                            csv_detail = (
                                linked[show_cols].to_csv(index=False).encode("utf-8")
                            )
                            st.download_button(
                                "Download Assignee Tickets CSV",
                                data=csv_detail,
                                file_name=f"jira_assignee_{selected_assignee.replace(' ', '_')}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.info("No tickets found for the selected assignee.")
                else:
                    st.info("No assignees available for detail view.")

                col_sys, col_sub, col_comp = st.columns(3)
                with col_sys:
                    st.subheader("OBS System")
                    agg_sys = self._aggregate_by_obs_system(proj_open, limit=200)
                    st.dataframe(agg_sys, hide_index=True)
                with col_sub:
                    st.subheader("OBS Sub-System")
                    agg_sub = self._aggregate_by_obs_subsystem(proj_open, limit=200)
                    st.dataframe(agg_sub, hide_index=True)
                with col_comp:
                    st.subheader("OBS Component")
                    agg_comp = self._aggregate_by_obs_component_detail(
                        proj_open, limit=200
                    )
                    st.dataframe(agg_comp, hide_index=True)

                # Missing OBS fields (only not Done/open issues as requested)
                st.write("## Data Hygiene: Missing OBS Fields (Open Issues)")
                # Treat empty strings or NaN as missing

                def _is_missing(series):
                    return series.isna() | (series.astype(str).str.strip() == "")

                missing_sys = proj_open[
                    _is_missing(proj_open.get("obs_system", pd.Series([])))
                ]
                missing_sub = proj_open[
                    _is_missing(proj_open.get("obs_subsystem", pd.Series([])))
                ]

                # Common columns with links
                base_cols = [
                    "key",
                    "summary",
                    "priority",
                    "status",
                    "assignee",
                    "reporter",
                    "labels",
                    "created",
                    "updated",
                    "days_open",
                    "days_since_update",
                ]
                # Missing OBS System table
                st.subheader("Tickets Missing OBS System")
                for c in base_cols:
                    if c not in missing_sys.columns:
                        missing_sys[c] = pd.NA
                sys_linked, sys_cfg = self._add_ticket_link(missing_sys)
                sys_order = [
                    "Ticket",
                    "summary",
                    "priority",
                    "status",
                    "assignee",
                    "reporter",
                    "labels",
                    "obs_system",
                    "obs_subsystem",
                    "obs_component",
                    "created",
                    "updated",
                    "days_open",
                    "days_since_update",
                ]
                sys_show = [c for c in sys_order if c in sys_linked.columns]
                st.dataframe(
                    sys_linked[sys_show].head(1000),
                    hide_index=True,
                    column_config=sys_cfg,
                )
                csv_sys = sys_linked[sys_show].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Missing OBS System CSV",
                    data=csv_sys,
                    file_name=f"jira_missing_obs_system_{project_key}.csv",
                    mime="text/csv",
                )

                # Missing OBS Sub-System table
                st.subheader("Tickets Missing OBS Sub-System")
                for c in base_cols:
                    if c not in missing_sub.columns:
                        missing_sub[c] = pd.NA
                sub_linked, sub_cfg = self._add_ticket_link(missing_sub)
                sub_order = [
                    "Ticket",
                    "summary",
                    "priority",
                    "status",
                    "assignee",
                    "reporter",
                    "labels",
                    "obs_system",
                    "obs_subsystem",
                    "obs_component",
                    "created",
                    "updated",
                    "days_open",
                    "days_since_update",
                ]
                sub_show = [c for c in sub_order if c in sub_linked.columns]
                st.dataframe(
                    sub_linked[sub_show].head(1000),
                    hide_index=True,
                    column_config=sub_cfg,
                )
                csv_sub = sub_linked[sub_show].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Missing OBS Sub-System CSV",
                    data=csv_sub,
                    file_name=f"jira_missing_obs_subsystem_{project_key}.csv",
                    mime="text/csv",
                )

                # Cache project aggregations
                st.session_state["project_aggs"] = {
                    "assignee": agg_assignee,
                    "system": agg_sys,
                    "subsystem": agg_sub,
                    "component": agg_comp,
                    "missing_system": sys_linked,
                    "missing_subsystem": sub_linked,
                    "proj_open": proj_open,
                }
        # Reactive re-render (user changes selectbox after initial fetch)
        elif "project_aggs" in st.session_state:
            cache = st.session_state["project_aggs"]
            proj_open = cache.get("proj_open", pd.DataFrame())
            if proj_open.empty:
                st.info("No cached project issues to display. Fetch tickets first.")
                return
            agg_assignee = cache.get("assignee", pd.DataFrame())
            agg_sys = cache.get("system", pd.DataFrame())
            agg_sub = cache.get("subsystem", pd.DataFrame())
            agg_comp = cache.get("component", pd.DataFrame())
            sys_linked = cache.get("missing_system", pd.DataFrame())
            sub_linked = cache.get("missing_subsystem", pd.DataFrame())

            scope_lbl = (
                "Project-wide (Open Issues)"
                if mode == "All open issues"
                else "Open Issues in Range"
            )
            st.write(f"## Aggregations — {scope_lbl}")
            st.subheader("By Assignee")
            st.dataframe(agg_assignee, hide_index=True)

            assignee_options = (
                agg_assignee["assignee"]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
                if "assignee" in agg_assignee.columns and not agg_assignee.empty
                else []
            )
            if assignee_options:
                # Reuse prior selection from session state
                selected_assignee = st.selectbox(
                    "Select an assignee to view their open tickets",
                    ["(None)"] + assignee_options,
                    key="assignee_detail_select",
                )
                if selected_assignee and selected_assignee != "(None)":
                    detail_df = proj_open[
                        proj_open["assignee"].astype(str) == selected_assignee
                    ].copy()
                    if not detail_df.empty:
                        for col in ["days_open", "days_since_update"]:
                            if col in detail_df.columns:
                                try:
                                    detail_df[col] = (
                                        pd.to_numeric(detail_df[col], errors="coerce")
                                        .fillna(0)
                                        .astype(int)
                                    )
                                except Exception:
                                    pass
                        tz = pytz.timezone("America/Santiago")
                        for dt_col in ["created", "updated"]:
                            if dt_col in detail_df.columns:
                                try:
                                    detail_df[dt_col] = pd.to_datetime(
                                        detail_df[dt_col], utc=True
                                    ).dt.tz_convert(tz)
                                except Exception:
                                    pass
                        display_cols = [
                            "key",
                            "summary",
                            "priority",
                            "priority_value",
                            "status",
                            "time_lost",
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
                        ]
                        for c in display_cols:
                            if c not in detail_df.columns:
                                detail_df[c] = pd.NA
                        if "updated" in detail_df.columns:
                            try:
                                detail_df = detail_df.sort_values(
                                    "updated", ascending=False
                                )
                            except Exception:
                                pass
                        linked, link_cfg = self._add_ticket_link(detail_df)
                        order = [
                            "Ticket",
                            "summary",
                            "priority",
                            "priority_value",
                            "status",
                            "time_lost",
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
                        ]
                        show_cols = [c for c in order if c in linked.columns]
                        st.subheader(f"Tickets Assigned to {selected_assignee}")
                        st.dataframe(
                            linked[show_cols].head(1000),
                            hide_index=True,
                            column_config=link_cfg,
                        )
                        csv_detail = (
                            linked[show_cols].to_csv(index=False).encode("utf-8")
                        )
                        st.download_button(
                            "Download Assignee Tickets CSV",
                            data=csv_detail,
                            file_name=f"jira_assignee_{selected_assignee.replace(' ', '_')}.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("No tickets found for the selected assignee.")
            else:
                st.info("No assignees available for detail view.")

            col_sys, col_sub, col_comp = st.columns(3)
            with col_sys:
                st.subheader("OBS System")
                st.dataframe(agg_sys, hide_index=True)
            with col_sub:
                st.subheader("OBS Sub-System")
                st.dataframe(agg_sub, hide_index=True)
            with col_comp:
                st.subheader("OBS Component")
                st.dataframe(agg_comp, hide_index=True)

            st.write("## Data Hygiene: Missing OBS Fields (Open Issues)")
            st.subheader("Missing OBS System")
            if isinstance(sys_linked, pd.DataFrame) and not sys_linked.empty:
                st.dataframe(sys_linked.head(1000), hide_index=True)
            else:
                st.caption("None missing.")
            st.subheader("Missing OBS Sub-System")
            if isinstance(sub_linked, pd.DataFrame) and not sub_linked.empty:
                st.dataframe(sub_linked.head(1000), hide_index=True)
            else:
                st.caption("None missing.")
