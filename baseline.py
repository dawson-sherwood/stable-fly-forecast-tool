# baseline.py
import pandas as pd
from processing import compute_daily_dd


# ----------------------------
# Formatting
# ----------------------------

def acis_to_daily_schema(df):

    if df is None or df.empty:
        return pd.DataFrame(
            columns=["date", "min_temp_c", "max_temp_c", "total_precip_mm"]
        )

    out = df.rename(
        columns={
            "mint_c": "min_temp_c",
            "maxt_c": "max_temp_c",
            "pcpn_mm": "total_precip_mm",
        }
    )

    return out[[
        "date",
        "min_temp_c",
        "max_temp_c",
        "total_precip_mm",
    ]].copy()


def baseline_to_long(baseline_by_year):

    parts = []

    for yr, df in baseline_by_year.items():

        if df is None or df.empty:
            continue

        tmp = df.copy()
        tmp["year"] = int(yr)

        parts.append(tmp)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


# ----------------------------
# DD baselines
# ----------------------------

def baseline_with_dd(
    baseline_long,
    base_c,
    upper_c
):

    if baseline_long.empty:
        return baseline_long

    df = baseline_long.copy()

    df["doy"] = pd.to_datetime(df["date"]).dt.dayofyear

    daily = acis_to_daily_schema(df)

    daily = compute_daily_dd(
        daily,
        base_c,
        upper_c
    )

    df["dd15"] = daily["dd15"].values
    
    return df

def build_daily_dd_percentiles(df):
    grouped = df.groupby("doy")
    stats = grouped.agg(
        dd15_p50=("dd15", "median"),
        avg_temp_p50=("avg_temp_c", "median"),
    ).reset_index()

    # Normalize to 365-day climatology (drop leap-day)
    stats = stats[stats["doy"] != 366].reset_index(drop=True)
    return stats