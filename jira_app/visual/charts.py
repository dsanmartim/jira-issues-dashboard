"""Chart builders (Altair) for trends."""

from __future__ import annotations

import altair as alt
import pandas as pd
import pytz

from jira_app.core.config import TIMEZONE


def _format_ticket_list(group: pd.DataFrame) -> str:
    seen: set[str] = set()
    items: list[str] = []
    for _, row in group.iterrows():
        key = str(row.get("key") or "").strip()
        if not key or key in seen:
            continue
        summary_val = row.get("summary")
        summary = str(summary_val).strip() if summary_val is not None else ""
        items.append(f"{key}: {summary}" if summary else key)
        seen.add(key)
    return "\n".join(items)


def created_trend(df: pd.DataFrame, start, end, priorities=None):
    if df.empty:
        return None, pd.DataFrame()
    tz = pytz.timezone(TIMEZONE)
    tmp = df.copy()
    if priorities:
        tmp = tmp[tmp["priority"].astype(str).isin(priorities)]
    # Prefer an existing created_dt column if present; otherwise derive from created
    if "created_dt" in tmp.columns:
        created_series = pd.to_datetime(tmp["created_dt"], utc=True, errors="coerce")
    else:
        created_series = pd.to_datetime(tmp.get("created"), utc=True, errors="coerce")
    tmp["created_dt"] = created_series.dt.tz_convert(tz)
    tmp = tmp[(tmp["created_dt"] >= start) & (tmp["created_dt"] <= end)]
    tmp["date"] = tmp["created_dt"].dt.date
    if tmp.empty:
        return None, tmp

    agg = (
        tmp.groupby("date")
        .apply(
            lambda g: pd.Series({"count": int(len(g)), "tickets": _format_ticket_list(g)}),
            include_groups=False,
        )
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"])

    all_dates = pd.date_range(start.date(), end.date(), freq="D")
    chart_df = pd.DataFrame({"date": all_dates})
    chart_df = chart_df.merge(agg, on="date", how="left")
    chart_df["count"] = chart_df["count"].fillna(0).astype(int)
    chart_df["tickets"] = chart_df["tickets"].fillna("").astype(str)
    chart_df["date"] = pd.to_datetime(chart_df["date"])

    base_line = (
        alt.Chart(chart_df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Tickets Created"),
        )
    )
    points = (
        alt.Chart(chart_df)
        .mark_circle(color="#1f77b4", opacity=0.75, size=70)
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

    shading = alt.Chart(pd.DataFrame()).mark_rect()  # default empty rect
    unique_dates = chart_df[["date"]].drop_duplicates()
    unique_dates = unique_dates.assign(weekday=unique_dates["date"].dt.weekday)
    weekend = unique_dates[unique_dates["weekday"].isin([5, 6])].copy()
    if not weekend.empty:
        weekend = weekend.assign(date_end=weekend["date"] + pd.Timedelta(days=1))
        shading = alt.Chart(weekend).mark_rect(color="#f2f2f2").encode(x="date:T", x2="date_end:T")

    chart = (shading + base_line + points).properties(height=300)
    return chart, tmp


def blocker_critical_trend(df: pd.DataFrame, start, end):
    if df.empty:
        return None
    tz = pytz.timezone(TIMEZONE)
    tmp = df.copy()
    tmp = tmp[tmp["priority"].astype(str).str.startswith(("Blocker", "Critical"))]
    tmp["updated_dt"] = pd.to_datetime(tmp["updated"], utc=True, errors="coerce").dt.tz_convert(tz)
    tmp = tmp[(tmp["updated_dt"] >= start) & (tmp["updated_dt"] <= end)]
    if tmp.empty:
        return None
    tmp["date"] = tmp["updated_dt"].dt.date
    agg = (
        tmp.groupby("date")
        .apply(
            lambda g: pd.Series({"count": int(len(g)), "tickets": _format_ticket_list(g)}),
            include_groups=False,
        )
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"])
    agg["tickets"] = agg["tickets"].fillna("").astype(str)
    chart = (
        alt.Chart(agg)
        .mark_bar(color="#d62728")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title="Blocker/Critical Updates"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("count:Q", title="Updates"),
                alt.Tooltip("tickets:N", title="Tickets"),
            ],
        )
        .properties(height=220)
    )
    return chart
