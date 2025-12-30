from datetime import UTC, datetime

import pandas as pd

from jira_app.analytics.metrics.activity import add_weighted_activity
from jira_app.analytics.metrics.aging import add_aging_metrics
from jira_app.core.mappers import _extract_obs_list, _split_obs_hierarchy, issues_to_dataframe
from jira_app.core.models import IssueModel


def test_aging_metrics_basic():
    df = pd.DataFrame(
        {
            "created": ["2024-09-09T10:00:00.000+0000"],
            "updated": ["2024-09-09T12:00:00.000+0000"],
        }
    )
    out = add_aging_metrics(df)
    assert "days_open" in out.columns
    assert out.loc[0, "days_open"] >= 0


def test_activity_weighting_counts():
    start = datetime(2024, 9, 1, tzinfo=UTC)
    end = datetime(2024, 9, 30, tzinfo=UTC)
    df = pd.DataFrame(
        {
            "comments": [[{"created": "2024-09-10T10:00:00.000+0000"}]],
            "histories": [[{"created": "2024-09-11T10:00:00.000+0000", "items": [{"field": "status"}]}]],
        }
    )
    out = add_weighted_activity(df, start=start, end=end)
    assert out.loc[0, "comments_in_range"] == 1
    assert out.loc[0, "status_changes"] == 1
    assert out.loc[0, "activity_score_weighted"] > 0


def test_obs_parsing():
    value = {"name": "Sys", "child": {"name": "Sub", "child": {"name": "Comp"}}}
    lst = _extract_obs_list(value)
    assert lst[:3] == ["Sys", "Sub", "Comp"]
    system, subsystem, component = _split_obs_hierarchy(lst)
    assert system == "Sys" and subsystem == "Sub" and component == "Comp"


def test_obs_alias_normalization():
    value = {"name": "OBS", "child": {"name": "S: Dome", "child": {"name": "S:Dome"}}}
    lst = _extract_obs_list(value)
    # Both hierarchical entries should collapse to the canonical "Dome"
    assert lst == ["OBS", "Dome"]


def test_issues_to_dataframe_normalizes_obs_aliases():
    issue = IssueModel(
        key="OBS-1",
        summary=None,
        created=None,
        updated=None,
        assignee=None,
        reporter=None,
        priority=None,
        status=None,
        resolution=None,
        resolution_date=None,
        issuetype=None,
        time_lost=None,
        obs_system="S: Dome",
        obs_subsystem="S:Dome",
        obs_component="S : Dome",
    )
    df = issues_to_dataframe([issue])
    assert df.loc[0, "obs_system"] == "Dome"
    assert df.loc[0, "obs_subsystem"] == "Dome"
    assert df.loc[0, "obs_component"] == "Dome"
