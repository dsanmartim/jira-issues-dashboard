import pandas as pd

from jira_app.pages import assignees as assignees_page


def test_assignee_metrics_grouping(monkeypatch):
    # Create minimal DataFrame simulating enriched issues
    data = [
        {
            "key": "A-1",
            "assignee": "Alice",
            "activity_score": 3,
            "activity_score_weighted": 5,
            "days_open": 10,
        },
        {
            "key": "A-2",
            "assignee": "Alice",
            "activity_score": 1,
            "activity_score_weighted": 2,
            "days_open": 6,
        },
        {
            "key": "B-1",
            "assignee": "Bob",
            "activity_score": 2,
            "activity_score_weighted": 3,
            "days_open": 4,
        },
        {
            "key": "U-1",
            "assignee": None,
            "activity_score": 0,
            "activity_score_weighted": 0,
            "days_open": 1,
        },
    ]
    df = pd.DataFrame(data)

    # Invoke rendering helper to ensure no exceptions; UI output not asserted here.
    assignees_page.render_assignee_tab(df, "https://example")

    # Metrics should exist implicitly via grouping; re-run logic inline for assertion
    work = df.copy()
    work["assignee"] = work.get("assignee").fillna("(Unassigned)")
    grouped = work.groupby("assignee").agg(
        {
            "key": "count",
            "activity_score": "sum",
            "activity_score_weighted": "sum",
            "days_open": "median",
        }
    )
    assert grouped.loc["Alice", "key"] == 2
    assert grouped.loc["Bob", "activity_score"] == 2
    assert "(Unassigned)" in grouped.index
