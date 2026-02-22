"""Shared derived column computations for analytics."""

from __future__ import annotations

import pandas as pd


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add commonly used derived metrics to a DataFrame.

    Adds the following columns:
        - time_lost_value: Numeric version of time_lost (0 if missing/invalid)
        - total_activity_in_range: Sum of comments_in_range + status_changes + other_changes

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with optional columns: time_lost, comments_in_range,
        status_changes, other_changes.

    Returns
    -------
    pd.DataFrame
        Copy of input with derived columns added.
    """
    out = df.copy()

    # Convert time_lost to numeric, handling missing column gracefully
    if "time_lost" in out.columns:
        out["time_lost_value"] = pd.to_numeric(out["time_lost"], errors="coerce").fillna(0)
    else:
        out["time_lost_value"] = 0

    # Sum activity metrics (each defaults to 0 if missing)
    out["total_activity_in_range"] = (
        out.get("comments_in_range", 0) + out.get("status_changes", 0) + out.get("other_changes", 0)
    )

    return out
