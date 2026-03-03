# utils.py

import re

# ----------------------------
# Validation
# ----------------------------

def validate_zip(zip_code: str) -> bool:
    from config import ZIP_RE
    return bool(re.match(ZIP_RE, (zip_code or "").strip()))

# ----------------------------
# Unit conversions
# ----------------------------

def f_to_c(f: float) -> float:
    return (f - 32.0) * (5.0 / 9.0)

def c_to_f(c: float) -> float:
    return c * (9.0 / 5.0) + 32.0

def mm_to_in(mm: float) -> float:
    return mm / 25.4


# ----------------------------
# Time parsing (NWS)
# ----------------------------

def parse_iso_duration_to_hours(duration: str) -> float:
    if not duration or not duration.startswith("P"):
        return 0.0

    days = hours = minutes = 0.0

    if "T" in duration:
        date_part, time_part = duration.split("T", 1)
    else:
        date_part, time_part = duration, ""

    m = re.search(r"(\d+(\.\d+)?)D", date_part)
    if m:
        days = float(m.group(1))

    m = re.search(r"(\d+(\.\d+)?)H", time_part)
    if m:
        hours = float(m.group(1))

    m = re.search(r"(\d+(\.\d+)?)M", time_part)
    if m:
        minutes = float(m.group(1))

    return days * 24.0 + hours + minutes / 60.0


def split_valid_time(valid_time: str):
    if not valid_time or "/" not in valid_time:
        return valid_time, 0.0

    start, dur = valid_time.split("/", 1)
    return start, parse_iso_duration_to_hours(dur)


def normalize_uom(uom: str) -> str:
    if not uom:
        return ""

    u = uom.lower().strip()

    if "degc" in u:
        return "degC"
    if "degf" in u:
        return "degF"
    if "kg m-2" in u or "kg m**-2" in u or "kg m^-2" in u:
        return "mm"
    if "mm" in u:
        return "mm"
    if "inch" in u:
        return "in"

    return uom