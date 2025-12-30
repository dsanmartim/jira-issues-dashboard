from datetime import datetime

import pandas as pd
import pytz

from jira_app.analytics.aggregations.obs import (
    aggregate_by_obs_component,
    aggregate_by_obs_subsystem,
    aggregate_by_obs_system,
)
from jira_app.visual.tables import add_ticket_link


def _sample_df():
    tz = pytz.timezone("America/Santiago")
    base = tz.localize(datetime(2024, 12, 1, 10, 0, 0))
    data = []
    for i in range(6):
        data.append(
            {
                "key": f"OBS-{i+10}",
                "summary": f"Obs ticket {i}",
                "priority": "High",
                "status": "In Progress",
                "created": base,
                "updated": base,
                "obs_system": "SYSA" if i < 3 else "SYSB",
                "obs_subsystem": "SUB1" if i % 2 == 0 else "SUB2",
                "obs_component": f"C{i%3}",
            }
        )
    return pd.DataFrame(data)


def test_aggregate_system():
    df = _sample_df()
    out = aggregate_by_obs_system(df, limit=10)
    assert not out.empty
    assert "count" in out.columns


def test_aggregate_subsystem():
    df = _sample_df()
    out = aggregate_by_obs_subsystem(df, limit=10)
    assert not out.empty
    assert "count" in out.columns


def test_aggregate_component():
    df = _sample_df()
    out = aggregate_by_obs_component(df, limit=10)
    assert not out.empty
    assert "count" in out.columns


def test_ticket_link_injection():
    df = _sample_df().head(1)
    server = "https://example.atlassian.net"
    linked, cfg = add_ticket_link(df, server)
    assert "Ticket" in linked.columns
    assert linked.loc[linked.index[0], "Ticket"].startswith(server + "/browse/OBS-")
    assert "Ticket" in cfg
