from datetime import datetime, timedelta

import pandas as pd
import pytz

from jira_app.analytics.segments import filters as seg


def _sample_df():
    tz = pytz.UTC
    now = datetime.now(tz)
    data = [
        {
            "key": f"OBS-{i}",
            "priority": p,
            "status": s,
            "updated": (now - timedelta(hours=i)).isoformat(),
            "created": (now - timedelta(days=1, hours=i)).isoformat(),
            "comments": [{"created": (now - timedelta(hours=i + j)).isoformat()} for j in range(2)],
            "histories": [{"created": (now - timedelta(hours=i + j + 1)).isoformat()} for j in range(1)],
            "summary": f"Issue {i}",
            "time_lost": i * 10,
            "activity_score_weighted": i * 3,
        }
        for i, (p, s) in enumerate(
            [
                ("Blocker", "In Progress"),
                ("Critical", "Tracking"),
                ("High", "Testing"),
                ("Medium", "Done"),
                ("Low", "To Do"),
            ]
        )
    ]
    return pd.DataFrame(data)


def test_segments_basic():
    df = _sample_df()
    tz = pytz.UTC
    start = datetime.now(tz) - timedelta(days=2)
    end = datetime.now(tz)

    assert not seg.most_active(df, start, end).empty
    assert not seg.most_commented(df, start, end).empty
    assert not seg.blocker_critical(df).empty
    assert not seg.testing_tracking(df).empty
    assert not seg.most_time_lost(df, start, end).empty
    assert not seg.weighted_activity(df).empty
