"""Chart builders (Altair) for trends."""

from __future__ import annotations

import altair as alt
import pandas as pd
import pytz

from jira_app.core.config import TIMEZONE


def create_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
) -> alt.Chart | None:
    """Create a simple bar chart.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot.
    x_col : str
        Column name for x-axis (categorical).
    y_col : str
        Column name for y-axis (quantitative).
    title : str, optional
        Chart title.

    Returns
    -------
    alt.Chart or None
        Altair chart object, or None if data is empty/invalid.
    """
    if data is None or data.empty or x_col not in data or y_col not in data:
        return None
    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{x_col}:N", sort="-y", title=x_col.replace("_", " ").title()),
            y=alt.Y(f"{y_col}:Q", title=y_col.replace("_", " ").title()),
            tooltip=[x_col, y_col],
        )
        .properties(height=280)
    )
    if title:
        chart = chart.properties(title=title)
    return chart


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


def priority_updates_trend(df: pd.DataFrame, start, end, priorities: list[str] | None = None):
    """Create a bar chart showing updates for selected priorities over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing issue data.
    start, end : datetime-like
        Date range for filtering.
    priorities : list[str] | None
        List of priority names to include. If None, defaults to Blocker/Critical.

    Returns
    -------
    alt.Chart or None
        Altair chart object, or None if no data matches.
    """
    if df.empty:
        return None
    tz = pytz.timezone(TIMEZONE)
    tmp = df.copy()

    # Filter by selected priorities
    if priorities:
        tmp = tmp[tmp["priority"].astype(str).isin(priorities)]
    else:
        # Legacy fallback: Blocker/Critical
        tmp = tmp[tmp["priority"].astype(str).str.startswith(("Blocker", "Critical"))]

    tmp["updated_dt"] = pd.to_datetime(tmp["updated"], utc=True, errors="coerce").dt.tz_convert(tz)
    tmp = tmp[(tmp["updated_dt"] >= start) & (tmp["updated_dt"] <= end)]
    if tmp.empty:
        return None
    tmp["date"] = tmp["updated_dt"].dt.date
    tmp["priority_str"] = tmp["priority"].astype(str)

    # Group by date and priority for color-coded stacked bars
    agg = (
        tmp.groupby(["date", "priority_str"])
        .apply(
            lambda g: pd.Series({"count": int(len(g)), "tickets": _format_ticket_list(g)}),
            include_groups=False,
        )
        .reset_index()
    )
    agg["date"] = pd.to_datetime(agg["date"])
    agg["tickets"] = agg["tickets"].fillna("").astype(str)

    # Define priority color mapping - distinct color for each priority
    priority_colors = {
        "Blocker": "#d62728",  # Red
        "Critical": "#c0392b",  # Dark Red
        "Urgent": "#ff7f0e",  # Orange
        "High": "#f39c12",  # Amber
        "Medium": "#f1c40f",  # Yellow
        "Low": "#27ae60",  # Green
        "Undefined": "#95a5a6",  # Gray
    }
    # Fallback colors for unknown priorities
    fallback_colors = ["#3498db", "#9b59b6", "#1abc9c", "#e74c3c", "#34495e"]

    # Build color scale domain and range
    unique_priorities = agg["priority_str"].unique().tolist()
    color_domain = unique_priorities
    fallback_idx = 0
    color_range = []
    for p in unique_priorities:
        if p in priority_colors:
            color_range.append(priority_colors[p])
        else:
            color_range.append(fallback_colors[fallback_idx % len(fallback_colors)])
            fallback_idx += 1

    # Dynamic title based on selected priorities
    if priorities:
        title_suffix = "/".join(priorities) if len(priorities) <= 3 else f"{len(priorities)} priorities"
    else:
        title_suffix = "Blocker/Critical"

    chart = (
        alt.Chart(agg)
        .mark_bar()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("count:Q", title=f"{title_suffix} Updates", stack=True),
            color=alt.Color(
                "priority_str:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=alt.Legend(title="Priority"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("priority_str:N", title="Priority"),
                alt.Tooltip("count:Q", title="Updates"),
                alt.Tooltip("tickets:N", title="Tickets"),
            ],
        )
        .properties(height=220)
    )
    return chart


# Keep old name for backwards compatibility
blocker_critical_trend = priority_updates_trend
