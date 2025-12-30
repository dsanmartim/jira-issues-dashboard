from datetime import UTC, datetime

from jira_app.core.jira_client import JiraAPI
from jira_app.core.service import ActivityWeights, IssueService


class DummyAPI(JiraAPI):
    def __init__(self):
        self.server = "https://example.atlassian.net"

    def search_enhanced(self, jql, fields=None, expand=None, page_size=1000):
        # Return minimal fake issue
        return [
            {
                "key": "OBS-1",
                "fields": {
                    "summary": "Test",
                    "created": "2024-09-01T10:00:00.000+0000",
                    "updated": "2024-09-02T10:00:00.000+0000",
                    "assignee": {"displayName": "Alice"},
                    "reporter": {"displayName": "Bob"},
                    "priority": {"name": "High"},
                    "status": {"name": "In Progress"},
                    "resolution": None,
                    "resolutiondate": None,
                    "issuetype": {"name": "Task"},
                    "customfield_10106": 5,
                    "labels": ["x"],
                    "customfield_10476": {"name": "Sys", "child": {"name": "Sub", "child": {"name": "Comp"}}},
                    "comment": {"comments": []},
                },
                "changelog": {"histories": []},
            }
        ]


def test_fetch_and_enrich_range():
    api = DummyAPI()
    svc = IssueService(api)
    start = datetime(2024, 9, 1, tzinfo=UTC)
    end = datetime(2024, 9, 3, tzinfo=UTC)
    df = svc.fetch_and_enrich_range("OBS", start, end, weights=ActivityWeights())
    assert not df.empty
    assert "days_open" in df.columns
    assert "activity_score_weighted" in df.columns
