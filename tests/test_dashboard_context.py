from datetime import datetime, timedelta

import pandas as pd
import pytz

from jira_app.features.activity_overview.context import build_context


def _make_df():
    tz = pytz.UTC
    now = datetime.now(tz)
    rows = []
    for i in range(10):
        rows.append(
            {
                "key": f"OBS-{i}",
                "priority": "Critical" if i % 3 == 0 else "High",
                "status": "In Progress",
                "created": (now - timedelta(days=5 - (i % 5))).isoformat(),
                "updated": (now - timedelta(hours=i)).isoformat(),
                "comments": [{"created": (now - timedelta(hours=i + j)).isoformat()} for j in range(2)],
                "histories": [
                    {
                        "created": (now - timedelta(hours=i + j + 1)).isoformat(),
                        "items": [{"field": "status"}],
                    }
                    for j in range(1)
                ],
                "days_open": i + 1,
                "summary": f"Dash issue {i}",
                "time_lost": i * 5,
                "activity_score_weighted": i * 2,
            }
        )
    return pd.DataFrame(rows)


def test_dashboard_context_basic():
    df = _make_df()
    tz = pytz.UTC
    start = datetime.now(tz) - timedelta(days=7)
    end = datetime.now(tz)
    ctx = build_context(
        df,
        start,
        end,
        top_n=5,
        priorities=["Critical"],
        aging_warn_days=3,
        drilldown_day=end.date(),
    )
    # segments present
    assert set(ctx.segments.keys()) == {
        "Weighted Activity",
        "Most Active",
        "Most Commented",
        "OLE Comments",
        "Testing/Tracking",
        "Time Lost",
        "Blocker/Critical",
    }
    # aging warning positive (some days_open > 3)
    assert ctx.aging_warning_count >= 1
    # drilldown rows subset of created events
    if not ctx.created_events.empty:
        assert set(ctx.drilldown_rows.columns).issubset(set(ctx.created_events.columns))
