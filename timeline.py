# timeline.py

import pandas as pd

def build_master_timeline(
    df_ytd_dd,
    df_forecast_dd,
    daily_dd_stats,
    horizon_days=45
):
    """
    Returns unified daily dataframe spanning past observed, 
    NWS forecast, and P50 climatology fill.
    """
    parts = []

    # 1. Observed (ACIS YTD + Lag)
    if not df_ytd_dd.empty:
        obs = df_ytd_dd.copy()
        obs["source"] = "observed"
        parts.append(obs)

    # 2. Forecast (NWS 7-Day)
    if not df_forecast_dd.empty:
        fc = df_forecast_dd.copy()
        fc["source"] = "forecast"
        parts.append(fc)

    # 3. Future Climatology Fill (ACIS 8-Year P50)
    # Start filling from the day after our last known data point
    if parts:
        last_date = pd.to_datetime(pd.concat(parts)["date"]).max()
    else:
        last_date = pd.Timestamp.today()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=horizon_days,
    )

    rows = []
    for d in future_dates:
        doy = d.dayofyear
        
        # SAFELY HANDLE LEAP YEARS: Normalize day 366 to 365
        if doy == 366:
            doy = 365 

        row = daily_dd_stats[daily_dd_stats["doy"] == doy]
        
        if row.empty:
            continue

        rows.append({
            "date": d.date(),
            "dd15": row["dd15_p50"].values[0],
            "avg_temp_c": row["avg_temp_p50"].values[0],
            "total_precip_mm": 0.0,
            "source": "climatology",
        })

    if rows:
        clim = pd.DataFrame(rows)
        parts.append(clim)

    # Merge, sort, and deduplicate
    if parts:
        timeline = pd.concat(parts, ignore_index=True)
        # In case NWS and ACIS overlap today, keep the NWS forecast (first)
        timeline = timeline.drop_duplicates(subset=["date"], keep="last")
        timeline = timeline.sort_values("date").reset_index(drop=True)
    else:
        timeline = pd.DataFrame()

    return timeline