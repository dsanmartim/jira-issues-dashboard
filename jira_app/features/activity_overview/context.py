"""Pure helpers to build dashboard context for testing (no Streamlit)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from jira_app.analytics.segments import filters as seg
from jira_app.visual.charts import created_trend


@dataclass(slots=True)
class DashboardContext:
    segments: dict[str, pd.DataFrame]
    created_events: pd.DataFrame
    aging_warning_count: int
    drilldown_rows: pd.DataFrame


def build_context(
    df: pd.DataFrame,
    start: datetime,
    end: datetime,
    top_n: int = 25,
    priorities: Iterable[str] | None = None,
    aging_warn_days: int = 14,
    drilldown_day=None,
) -> DashboardContext:
    if df.empty:
        return DashboardContext({}, pd.DataFrame(), 0, pd.DataFrame())

    most_active_df = seg.most_active(df, start, end)
    most_commented_df = seg.most_commented(df, start, end)
    ole_commented_df = seg.ole_commented(df, start, end)
    blocker_critical_df = seg.blocker_critical(df)
    testing_tracking_df = seg.testing_tracking(df)
    time_lost_df = seg.most_time_lost(df, start, end)
    weighted_df = seg.weighted_activity(df)

    chart_created, created_events = created_trend(
        df,
        start,
        end,
    )
    # drilldown
    drill_rows = pd.DataFrame()
    if drilldown_day is not None and not created_events.empty and "created_dt" in created_events.columns:
        created_events["date"] = created_events["created_dt"].dt.date
        drill_rows = created_events[created_events["date"] == drilldown_day]

    aging_warning = 0
    if not blocker_critical_df.empty and "days_open" in blocker_critical_df.columns:
        aging_warning = int((blocker_critical_df["days_open"] > aging_warn_days).sum())

    segs = {
        "Weighted Activity": weighted_df.head(top_n),
        "Most Active": most_active_df.head(top_n),
        "Most Commented": most_commented_df.head(top_n),
        "OLE Comments": ole_commented_df.head(top_n),
        "Testing/Tracking": testing_tracking_df.head(top_n),
        "Time Lost": time_lost_df.head(top_n),
        "Blocker/Critical": blocker_critical_df.head(top_n),
    }
    return DashboardContext(segs, created_events, aging_warning, drill_rows)
