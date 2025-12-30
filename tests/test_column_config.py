from jira_app.core.column_config import get_columns, load_column_sets


def test_column_sets_load():
    sets = load_column_sets()
    assert "detail" in sets and "core" in sets
    assert isinstance(get_columns("detail"), list)
