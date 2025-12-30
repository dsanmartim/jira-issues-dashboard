"""Stale ticket helper functions."""

from __future__ import annotations

import pandas as pd


def compute_stale(df: pd.DataFrame, days_threshold: int) -> pd.DataFrame:
    if df.empty or "days_since_update" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["days_since_update"] = pd.to_numeric(out["days_since_update"], errors="coerce").fillna(0)
    return out[out["days_since_update"] >= float(days_threshold)].sort_values(
        by="days_since_update", ascending=False
    )
