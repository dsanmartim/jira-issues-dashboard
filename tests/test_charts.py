from datetime import datetime, timedelta

import pandas as pd
import pytz

from jira_app.visual.charts import blocker_critical_trend, created_trend


def _sample_df():
    tz = pytz.timezone("America/Santiago")
    now = tz.localize(datetime.now())
    rows = []
    for i in range(5):
        rows.append(
            {
                "key": f"OBS-{i+1}",
                "summary": f"Sample ticket {i+1}",
                "priority": "Critical" if i % 2 == 0 else "High",
                "status": "In Progress",
                "created": now - timedelta(days=5 - i),
                "updated": now - timedelta(days=5 - i) + timedelta(hours=2),
            }
        )
    return pd.DataFrame(rows)


def test_created_trend_shapes():
    df = _sample_df()
    start = df["created"].min()
    end = df["created"].max()
    chart, events = created_trend(df, start, end, priorities=["Critical", "High"])
    assert events is not None
    assert "created_dt" in events.columns
    assert chart is not None


def test_blocker_critical_trend():
    df = _sample_df()
    start = df["created"].min()
    end = df["created"].max()
    chart = blocker_critical_trend(df, start, end)
    assert chart is not None
