# processing.py

import pandas as pd
from config import DEFAULT_BASE_C, DEFAULT_UPPER_C

# ----------------------------
# Unit coercion
# ----------------------------

def coerce_temp_to_celsius(df, uom):
    if df.empty:
        return df

    out = df.copy().dropna(subset=["value", "validTime"])
    
    # Cast to float first
    out["value"] = out["value"].astype(float)

    # Vectorized F to C conversion
    if uom == "degF":
        out["value"] = (out["value"] - 32.0) * (5.0 / 9.0)

    return out


def coerce_precip_to_mm(df, uom):
    if df.empty:
        return df

    out = df.copy().dropna(subset=["value", "validTime"])
    
    # Cast to float first
    out["value"] = out["value"].astype(float)

    # Vectorized inches to mm conversion
    if uom == "in":
        out["value"] = out["value"] * 25.4

    return out


# ----------------------------
# Hourly → Daily
# ----------------------------

def process_hourly_to_daily(df_temp, df_precip):
    """
    Aggregates hourly NWS data into daily min/max/sum.
    Optimized to use vectorized string splitting instead of row-by-row iteration.
    """
    
    # Temp
    if not df_temp.empty:
        # NWS validTime looks like "2024-07-25T14:00:00+00:00/PT1H". 
        # We split on '/' and grab the first part (the timestamp) instantly across the whole column.
        df_temp["time"] = pd.to_datetime(df_temp["validTime"].str.split("/").str[0])
        df_temp["date"] = df_temp["time"].dt.date
        df_temp["temp_c"] = df_temp["value"].astype(float)
        
        daily_t = df_temp.groupby("date").agg(
            min_temp_c=("temp_c", "min"),
            max_temp_c=("temp_c", "max"),
        )
    else:
        daily_t = pd.DataFrame(columns=["min_temp_c", "max_temp_c"])
        daily_t.index.name = "date"

    # Precip
    if not df_precip.empty:
        df_precip["time"] = pd.to_datetime(df_precip["validTime"].str.split("/").str[0])
        df_precip["date"] = df_precip["time"].dt.date
        df_precip["precip_mm"] = df_precip["value"].astype(float)
        
        daily_p = df_precip.groupby("date").agg(
            total_precip_mm=("precip_mm", "sum")
        )
    else:
        daily_p = pd.DataFrame(columns=["total_precip_mm"])
        daily_p.index.name = "date"

    return daily_t.join(daily_p, how="outer").reset_index()


# ----------------------------
# Degree days
# ----------------------------

def compute_daily_dd(df, base=DEFAULT_BASE_C, cap=DEFAULT_UPPER_C):
    """
    Vectorized Degree Day calculation. Exclusively uses DD15 as established 
    by Taylor et al. (2017) to track active adult stable flies.
    """
    out = df.copy()

    # Calculate average temp
    out["avg_temp_c"] = (out["min_temp_c"] + out["max_temp_c"]) / 2.0

    # Vectorized capping
    capped = out["avg_temp_c"].clip(upper=cap) if cap else out["avg_temp_c"]
    
    # Vectorized Degree Day calculation (max(0, capped - base))
    out["dd15"] = (capped - base).clip(lower=0.0)

    # Vectorized Display helpers
    out["min_temp_f"] = out["min_temp_c"] * 1.8 + 32.0
    out["max_temp_f"] = out["max_temp_c"] * 1.8 + 32.0
    out["avg_temp_f"] = out["avg_temp_c"] * 1.8 + 32.0
    out["total_precip_in"] = out["total_precip_mm"] / 25.4

    return out