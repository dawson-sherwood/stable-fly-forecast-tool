# lag_features.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd


def precip_total_2_to_7_weeks(
    df_daily: pd.DataFrame,
    today: Optional[date] = None,
) -> tuple[float, int]:
    """
    Sum precipitation over [today-49, today-14] inclusive.

    Parameters
    ----------
    df_daily:
        DataFrame containing at least `date` and `total_precip_mm`.
    today:
        Reference date. Defaults to system date.

    Returns
    -------
    (total_mm, missing_day_count)
        total_mm: summed precipitation across available non-null days in window.
        missing_day_count: count of days in the window that are absent or null.
    """

    if today is None:
        today = date.today()

    start = today - timedelta(days=49)
    end = today - timedelta(days=14)

    window_dates = pd.date_range(start=start, end=end, freq="D").date
    expected_days = set(window_dates)

    if df_daily is None or df_daily.empty:
        return 0.0, len(expected_days)

    work = df_daily[["date", "total_precip_mm"]].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date

    work = work[(work["date"] >= start) & (work["date"] <= end)]

    if work.empty:
        return 0.0, len(expected_days)

    # If duplicate dates exist, keep the last observation.
    work = work.drop_duplicates(subset=["date"], keep="last")

    present_non_null = set(work.loc[work["total_precip_mm"].notna(), "date"])
    missing_day_count = len(expected_days - present_non_null)

    total_mm = float(work["total_precip_mm"].fillna(0.0).sum())

    return total_mm, missing_day_count
