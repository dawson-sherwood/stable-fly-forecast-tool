# data_sources.py

import requests
import pandas as pd
import streamlit as st
from datetime import date, timedelta

from config import USER_AGENT, ACIS_GRIDDATA_URL
from utils import normalize_uom

# ----------------------------
# ZIP → Lat/Lon
# ----------------------------

@st.cache_data(show_spinner=False)
def get_lat_lon_from_zip(zip_code: str):
    url = f"https://api.zippopotam.us/us/{zip_code}"
    
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None, None
            
        data = resp.json()
        place = data["places"][0]
        
        lat = float(place["latitude"])
        lon = float(place["longitude"])
    except requests.exceptions.RequestException:
        # Catches timeouts, connection errors, etc.
        return None, None
    except (KeyError, ValueError, IndexError):
        # Catches malformed JSON or missing data
        return None, None

    return lat, lon


# ----------------------------
# NWS
# ----------------------------

@st.cache_data(show_spinner=False, ttl=3600)
def get_nws_grid_info(lat: float, lon: float):
    url = f"https://api.weather.gov/points/{lat},{lon}"
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
            
        props = resp.json().get("properties", {})
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None

    return {
        "gridId": props.get("gridId"),
        "gridX": props.get("gridX"),
        "gridY": props.get("gridY"),
        "forecastGridData": props.get("forecastGridData"),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_nws_grid_data(grid_url: str):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json",
    }

    try:
        resp = requests.get(grid_url, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
            
        props = resp.json().get("properties", {})
    except requests.exceptions.RequestException:
        return None
    except ValueError:
        return None

    temp = props.get("temperature", {})
    precip = props.get("quantitativePrecipitation", {})

    temp_uom = normalize_uom(temp.get("uom", ""))
    precip_uom = normalize_uom(precip.get("uom", ""))

    df_temp = pd.DataFrame(temp.get("values", []))
    df_precip = pd.DataFrame(precip.get("values", []))

    return df_temp, df_precip, temp_uom, precip_uom


# ----------------------------
# ACIS
# ----------------------------

def _acis_safe_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()

    if s.upper() == "M":
        return None
    if s.upper() == "T":
        return 0.0
    if s.endswith("A"):
        s = s[:-1]

    try:
        return float(s)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def acis_griddata_daily(lat, lon, sdate, edate, grid_code="1", include_precip=True):
    """
    Dynamically requests precipitation only when needed to save massive payload
    overhead during the 8-year baseline fetch.
    """
    elems = [
        {"name": "maxt", "units": "degreeC"},
        {"name": "mint", "units": "degreeC"},
    ]
    if include_precip:
        elems.append({"name": "pcpn", "units": "mm"})

    payload = {
        "loc": f"{lon:.4f},{lat:.4f}",
        "grid": grid_code,
        "sdate": sdate,
        "edate": edate,
        "elems": elems,
    }

    try:
        resp = requests.post(
            ACIS_GRIDDATA_URL,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return None, payload, {"status": resp.status_code}
            
        data = resp.json()
    except requests.exceptions.RequestException:
        return None, payload, {"status": "Network Error"}
    except ValueError:
        return None, payload, {"status": "JSON Parse Error"}

    rows = data.get("data", [])
    parsed = []

    for r in rows:
        d = r[0]
        maxt = _acis_safe_float(r[1])
        mint = _acis_safe_float(r[2])
        
        if include_precip:
            pcpn = _acis_safe_float(r[3])
            parsed.append((d, maxt, mint, pcpn))
        else:
            # Fill with 0.0 to maintain the expected downstream dataframe schema
            parsed.append((d, maxt, mint, 0.0)) 

    df = pd.DataFrame(
        parsed,
        columns=["date", "maxt_c", "mint_c", "pcpn_mm"]
    )

    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df, payload, {"ok": True}


def ytd_date_range():
    """
    Ensure the start date covers the 60-day lag window even early in the year,
    allowing us to consolidate the YTD and Lag API calls.
    """
    today = date.today()
    yesterday = today - timedelta(days=1)
    
    lag_start = today - timedelta(days=60)
    jan1 = date(today.year, 1, 1)
    
    s = min(jan1, lag_start)
    return s.isoformat(), yesterday.isoformat()


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def acis_baseline_8yr(lat, lon, n_years=8, grid_code="1"):
    
    today = date.today()
    start_year = today.year - n_years
    end_year = today.year - 1

    s = date(start_year, 1, 1).isoformat()
    e = date(end_year, 12, 31).isoformat()

    # Pass include_precip=False to drop the unused precipitation overhead
    df, payload, meta = acis_griddata_daily(
        lat, lon, s, e, grid_code, include_precip=False
    )

    if df is None or df.empty:
        return {}, {}

    df["year"] = pd.to_datetime(df["date"]).dt.year

    baseline_by_year = {
        yr: subdf.drop(columns=["year"]).reset_index(drop=True)
        for yr, subdf in df.groupby("year")
    }

    meta_by_year = {yr: meta for yr in baseline_by_year.keys()}

    return baseline_by_year, meta_by_year