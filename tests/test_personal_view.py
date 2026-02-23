"""Tests for the Personal View feature module."""

import pandas as pd

from jira_app.features.personal_view import filters as pv
from jira_app.features.personal_view.context import build_personal_context


def _sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "key": ["TEST-1", "TEST-2", "TEST-3", "TEST-4", "TEST-5"],
            "summary": ["Issue 1", "Issue 2", "Issue 3", "Issue 4", "Issue 5"],
            "assignee": ["Alice", "Bob", "Alice", "Charlie", "Alice"],
            "reporter": ["Bob", "Alice", "Charlie", "Alice", "Bob"],
            "status": ["In Progress", "Blocked", "Done", "In Progress", "Testing"],
            "priority": ["High", "Blocker", "Medium", "Critical", "Low"],
            "priority_value": [3, 5, 2, 4, 1],
            "days_since_update": [5, 35, 10, 2, 45],
            "time_lost": [1.5, 0, 2.0, 0.5, 0],
            "watchers": [["Bob"], ["Alice", "Charlie"], [], ["Alice"], ["Bob"]],
            "comments": [
                [{"author": "Bob", "body": "Comment 1"}],
                [{"author": "Alice", "body": "Hi Charlie, check this"}],
                [],
                [{"author": "Alice", "body": "Working on it"}],
                [{"author": "Charlie", "body": "Done"}],
            ],
        }
    )


def test_assigned_to():
    df = _sample_df()
    result = pv.assigned_to(df, "Alice")
    assert len(result) == 3
    assert set(result["key"]) == {"TEST-1", "TEST-3", "TEST-5"}


def test_reported_by():
    df = _sample_df()
    result = pv.reported_by(df, "Bob")
    assert len(result) == 2
    assert set(result["key"]) == {"TEST-1", "TEST-5"}


def test_watching():
    df = _sample_df()
    result = pv.watching(df, "Alice")
    assert len(result) == 2
    assert set(result["key"]) == {"TEST-2", "TEST-4"}


def test_mentioned_in():
    df = _sample_df()
    result = pv.mentioned_in(df, "Charlie")
    assert len(result) == 1
    assert result.iloc[0]["key"] == "TEST-2"


def test_commented_on():
    df = _sample_df()
    result = pv.commented_on(df, "Alice")
    assert len(result) == 2
    assert set(result["key"]) == {"TEST-2", "TEST-4"}


def test_stale_issues():
    df = _sample_df()
    result = pv.stale_issues(df, stale_days=30)
    # Only open issues that are stale (TEST-2 is Blocked=open, TEST-5 is Testing=open)
    assert len(result) == 2
    assert set(result["key"]) == {"TEST-2", "TEST-5"}


def test_blocked_issues():
    df = _sample_df()
    result = pv.blocked_issues(df)
    assert len(result) == 1
    assert result.iloc[0]["key"] == "TEST-2"


def test_needs_attention():
    df = _sample_df()
    result = pv.needs_attention(df, "Alice", stale_days=30)
    # Alice's open assigned issues that are blocked, stale, or high priority (Blocker/Critical)
    # TEST-1: High priority (not Blocker/Critical) - not included
    # TEST-5: Stale (45 days) - included
    # TEST-3 is Done so not included
    assert len(result) == 1
    assert set(result["key"]) == {"TEST-5"}


def test_build_personal_context():
    df = _sample_df()
    ctx = build_personal_context(df, "Alice", stale_days=30, top_n=10)

    assert ctx.user == "Alice"
    assert len(ctx.assigned) == 3
    assert len(ctx.reported) == 2  # Alice reported TEST-2 and TEST-4
    assert len(ctx.watching) == 2
    assert ctx.total_assigned_open == 2  # TEST-1 and TEST-5 (TEST-3 is Done)
    assert ctx.total_time_lost == 3.5  # 1.5 + 2.0 + 0 = 3.5
