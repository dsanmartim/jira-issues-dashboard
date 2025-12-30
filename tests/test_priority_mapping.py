from jira_app.core.mappers import map_priority


def test_priority_mapping():
    assert map_priority("Blocker") == 5
    assert map_priority("Critical - something") == 4
    assert map_priority("High") == 3
    assert map_priority(None) == -99
