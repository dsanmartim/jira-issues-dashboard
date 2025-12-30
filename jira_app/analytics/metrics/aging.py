"""Aging metrics computation (pure functions)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytz

from jira_app.core.config import TIMEZONE


def add_aging_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz=tz)
    out["created_dt"] = pd.to_datetime(out["created"], utc=True, errors="coerce").dt.tz_convert(tz)
    out["updated_dt"] = pd.to_datetime(out["updated"], utc=True, errors="coerce").dt.tz_convert(tz)
    out["days_open"] = (now - out["created_dt"]).dt.total_seconds() / 86400.0
    out["days_since_update"] = (now - out["updated_dt"]).dt.total_seconds() / 86400.0
    resolution_source = None
    if "resolution_date" in out.columns:
        resolution_source = "resolution_date"
    elif "resolutiondate" in out.columns:
        resolution_source = "resolutiondate"
    if resolution_source:
        out["resolution_dt"] = pd.to_datetime(
            out[resolution_source], utc=True, errors="coerce"
        ).dt.tz_convert(tz)
        out["time_to_resolution_days"] = (
            out["resolution_dt"] - out["created_dt"]
        ).dt.total_seconds() / 86400.0
    return out
