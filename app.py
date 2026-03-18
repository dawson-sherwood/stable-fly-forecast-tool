import concurrent.futures
from datetime import date, timedelta
from pathlib import Path
from PIL import Image

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from baseline import acis_to_daily_schema, baseline_to_long, build_daily_dd_percentiles
from config import DEFAULT_BASE_C, DEFAULT_UPPER_C
from data_sources import (
    acis_baseline_8yr,
    acis_griddata_daily,
    fetch_nws_grid_data,
    get_lat_lon_from_zip,
    get_nws_grid_info,
    ytd_date_range,
)
from lag_features import precip_total_2_to_7_weeks
from processing import (
    coerce_precip_to_mm,
    coerce_temp_to_celsius,
    compute_daily_dd,
    process_hourly_to_daily,
)
from risk_score import compute_risk_score
from timeline import build_master_timeline
from utils import validate_zip

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "ShieldStrongLogoGreen_2026.png"

# ----------------------------
# Brand / Theme helpers
# ----------------------------
BRAND_TEAL = "#10a295"
BRAND_NAVY = "#002854"
BRAND_YELLOW = "#feb81c"
BRAND_ORANGE = "#f58220"
BRAND_RED = "#d9534f"
WHITE = "#ffffff"
GRAY = "#5b6673"

BAND_STEPS = [
    (0, 10, "Minimal", BRAND_TEAL),
    (11, 40, "Building", BRAND_YELLOW),
    (41, 79, "Elevated", BRAND_ORANGE),
    (80, 100, "Peak", BRAND_RED),
]

MILK_LOSS_PAPER_URL = "https://www.mdpi.com/2306-7381/12/11/1035"
MILK_LOSS_FOOTNOTE = "https://www.mdpi.com/2306-7381/12/11/1035"

st.set_page_config(page_title="Fly Pressure Forecast Tool", layout="centered")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;800;900&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMarkdownContainer"],
[data-testid="stTextInputRootElement"], [data-testid="stButton"], .stRadio, .stCheckbox, .stSelectbox,
div, p, label, input, button, h1, h2, h3, h4, h5, h6 {{
    font-family: 'Roboto', sans-serif !important;
}}

.material-icons,
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp,
[data-testid="stStatusWidget"] i,
[data-testid="stStatusWidget"] span.material-symbols-outlined,
[data-testid="stStatusWidget"] span.material-symbols-rounded,
[data-testid="stStatusWidget"] span.material-symbols-sharp {{
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
}}

.main .block-container {{
    padding-top: 0.75rem;
    padding-bottom: 1rem;
    max-width: 760px;
}}

.shs-header {{
    text-align: center;
    padding: 0;
    margin-bottom: 0.35rem;
}}

.shs-brand-title {{
    font-family: 'Roboto', sans-serif !important;
    font-weight: 800;
    color: {BRAND_TEAL};
    font-size: 2rem;
    line-height: 1.1;
    margin: 0;
}}

.shs-h2 {{
    font-family: 'Roboto', sans-serif !important;
    margin: 0 0 0.35rem 0;
    color: {BRAND_NAVY};
    font-weight: 800;
    font-size: 1.9rem;
}}

.shs-desc {{
    font-family: 'Roboto', sans-serif !important;
    font-size: 1rem;
    color: {GRAY};
    margin: 0.15rem 0 0.9rem 0;
    line-height: 1.5;
    white-space: pre-line;
    max-width: 86%;
    font-style: italic;
}}

.question-text {{
    font-family: 'Roboto', sans-serif !important;
    font-size: 1.14rem;
    font-weight: 700;
    color: {BRAND_NAVY};
    margin-bottom: 0.55rem;
    line-height: 1.45;
}}

.question-extra {{
    font-family: 'Roboto', sans-serif !important;
    font-size: 0.98rem;
    color: {GRAY};
    margin-top: -0.1rem;
    margin-bottom: 1rem;
    line-height: 1.45;
    max-width: 86%;
}}

.score-card {{
    border: 2px solid rgba(16, 162, 149, 0.28);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    background: {WHITE};
}}

.score-card-label {{
    font-size: 1rem;
    font-weight: 700;
    color: {BRAND_NAVY};
    margin-bottom: 0.45rem;
}}

.score-card-value {{
    font-size: 2.75rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.4rem;
}}

.score-card-band {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {BRAND_NAVY};
}}

.summary-top-card {{
    border: 2px solid rgba(16, 162, 149, 0.28);
    border-radius: 14px;
    background: {WHITE};
    padding: 0.8rem 0.9rem 0.55rem 0.9rem;
    margin-bottom: 1rem;
}}

.summary-score-card {{
    border-radius: 14px;
    background: {WHITE};
    min-height: 318px;
    padding: 1.25rem 1.2rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.assessment-top-card {{
    border: 2px solid {BRAND_TEAL};
    border-radius: 14px;
    background: {WHITE};
    padding: 1rem 1rem 0.6rem 1rem;
    margin-bottom: 1.25rem;
}}

.gauge-legend {{
    display:flex;
    justify-content:center;
    gap:1rem;
    flex-wrap:wrap;
    font-size:0.92rem;
    margin-top:-0.1rem;
    margin-bottom:0.55rem;
}}

.gauge-legend span.dot {{
    display:inline-block;
    width:10px;
    height:10px;
    border-radius:50%;
    margin-right:6px;
}}


.assessment-box {{
    border: 2px solid {BRAND_TEAL};
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.25rem;
    background: {WHITE};
}}

.small-note {{
    color: {GRAY};
    font-size: 0.86rem;
    line-height: 1.45;
}}

.stProgress > div > div > div > div {{
    background-color: {BRAND_TEAL};
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------
# Session State Management
# ----------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "zip_code" not in st.session_state:
    st.session_state.zip_code = ""
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None
if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None


def reset_all():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ----------------------------
# Questionnaire Data Dictionary
# ----------------------------
MANURE_DESC = (
    "Refers to practices and strategies to minimize the presence of manure and moisture for a prolonged period "
    "of time in highly populated areas. Examples are flushing, scraping, and/or raking of animal confinement areas."
)

QUESTIONS = [
    {
        "id": "q1_sanitation",
        "title": "General Sanitation",
        "desc": (
            "Refers to overall cleanliness and management of the dairy facility. Examples include high traffic areas, "
            "alleyways or other animal handling areas, feed storage, water troughs, overgrown vegetation, and other "
            "areas where moisture is present."
        ),
        "text": "1. How thoroughly and how often are sanitation tasks performed on your dairy farm?",
        "type": "radio",
        "options": {
            "Thoroughly, at least once per day": 0.25,
            "Moderately, several times per week": 0.50,
            "Lightly, once per week": 0.75,
            "Not performed regularly": 1.0,
        },
    },
    {
        "id": "q2_manure_tasks",
        "title": "Manure Management",
        "desc": MANURE_DESC,
        "text": "2. How thoroughly and how often are manure management tasks performed on your dairy operation?",
        "type": "radio",
        "options": {
            "Thoroughly, at least once per day": 0.25,
            "Moderately, several times per week": 0.50,
            "Lightly, once per week": 0.75,
            "Not performed regularly": 1.0,
        },
    },
    {
        "id": "q3_calves",
        "title": "Manure Management",
        "desc": MANURE_DESC + "\n\n<br>Calf housing setups impact moisture retention and overall fly breeding grounds.",
        "text": "3. Are there calves housed on farm? (Select all that apply)",
        "type": "multiselect",
        "options": {
            "No": 0.0,
            "Yes, in hutches": 1.0,
            "Yes, calves in group housing": 0.75,
            "Yes, older calves over 600lbs in group pens": 0.50,
        },
    },
    {
        "id": "q4_manure_store",
        "title": "Manure Management",
        "desc": MANURE_DESC + "\n\n<br>How manure is stored heavily dictates fly emergence rates.",
        "text": "4. Where is manure being managed or stored?",
        "type": "radio",
        "options": {
            "Removed from the farm entirely": 0.0,
            "Flushed/scraped into lagoon/digester or separator on-site": 0.50,
            "Composted on-farm": 1.0,
        },
    },
    {
        "id": "q5_fogging",
        "title": "Prevention & Control",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "5. How thoroughly and how often are you <u>fogging</u> in and around the dairy?",
        "extra": "Examples include use of portable thermal foggers, handheld units or integrated high-pressure systems.",
        "type": "radio",
        "options": {
            "Thoroughly, at least once per day": 0.25,
            "Moderately, several times per week": 0.50,
            "Lightly, once per week": 0.75,
            "Not performed regularly": 1.0,
        },
    },
    {
        "id": "q6_spraying",
        "title": "Prevention & Control",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "6. How thoroughly and how often are you <u>spraying or pouring animals</u>?",
        "type": "radio",
        "options": {
            "Thoroughly, at least once per day": 0.25,
            "Moderately, several times per week": 0.50,
            "Lightly, once per week": 0.75,
            "Not performed regularly": 1.0,
        },
    },
    {
        "id": "q7_feedthrough",
        "title": "Prevention & Control",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "7. Are you using a <u>feed-through</u> product (IGR or Larvicide) to prevent fly larvae from maturing in manure?",
        "type": "radio",
        "options": {"Yes": 0.25, "No": 1.0},
    },
    {
        "id": "q8_wasps",
        "title": "Prevention & Control",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "8. Are you using fly predators (parasitic wasps) around the dairy?",
        "type": "radio",
        "options": {"Yes": 0.25, "No": 1.0},
    },
    {
        "id": "q9_bait",
        "title": "Prevention & Control",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "9. Are you using fly bait (scatter, paint-on or spray)?",
        "type": "radio",
        "options": {"Yes": 0.25, "No": 1.0},
    },
]

for q in QUESTIONS:
    if q["id"] not in st.session_state:
        st.session_state[q["id"]] = [] if q["type"] == "multiselect" else list(q["options"].keys())[0]


# ----------------------------
# Helpers
# ----------------------------
def fetch_and_process_nws(lat, lon, base_c, upper_c):
    grid = get_nws_grid_info(lat, lon)
    fetched = fetch_nws_grid_data(grid["forecastGridData"]) if grid else None
    if fetched:
        df_temp, df_precip, tuom, puom = fetched
        return compute_daily_dd(
            process_hourly_to_daily(
                coerce_temp_to_celsius(df_temp, tuom),
                coerce_precip_to_mm(df_precip, puom),
            ),
            base_c,
            upper_c,
        )
    return pd.DataFrame()


def fetch_and_process_ytd(lat, lon, base_c, upper_c):
    start_date, end_date = ytd_date_range()
    df_ytd, _, _ = acis_griddata_daily(lat, lon, start_date, end_date, include_precip=True)
    df_ytd_daily = acis_to_daily_schema(df_ytd)
    df_ytd_dd = compute_daily_dd(df_ytd_daily, base_c, upper_c)
    p_mm_2_7w_total, _ = precip_total_2_to_7_weeks(df_ytd_daily, today=date.today())
    return df_ytd_dd, p_mm_2_7w_total


def fetch_and_process_baseline(lat, lon, base_c, upper_c):
    baseline_by_year, _ = acis_baseline_8yr(lat, lon)
    daily = compute_daily_dd(acis_to_daily_schema(baseline_to_long(baseline_by_year)), base_c, upper_c)
    daily["doy"] = pd.to_datetime(daily["date"]).dt.dayofyear
    return build_daily_dd_percentiles(daily)


def get_band(score):
    score = float(score)
    if score <= 10:
        return "Minimal"
    if score <= 40:
        return "Building"
    if score <= 79:
        return "Elevated"
    return "Peak"


def get_band_color(band):
    return {
        "Minimal": BRAND_TEAL,
        "Building": BRAND_YELLOW,
        "Elevated": BRAND_ORANGE,
        "Peak": BRAND_RED,
    }.get(band, BRAND_NAVY)


def build_gauge(score, height=360):
    score = float(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=score,
            title={"text": "<b>Fly Risk Score</b>", "font": {"size": 24, "color": BRAND_NAVY}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickmode": "array",
                    "tickvals": [0, 10, 40, 80, 100],
                    "ticktext": ["0", "10", "40", "80", "100"],
                    "tickfont": {"size": 11, "color": GRAY},
                    "tickwidth": 1,
                    "tickcolor": GRAY,
                },
                "bar": {"color": BRAND_NAVY, "thickness": 0.15},
                "steps": [
                    {"range": [0, 10], "color": BRAND_TEAL},
                    {"range": [10, 40], "color": BRAND_YELLOW},
                    {"range": [40, 80], "color": BRAND_ORANGE},
                    {"range": [80, 100], "color": BRAND_RED},
                ],
                "threshold": {
                    "line": {"color": "#555555", "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.105,
        xref="paper",
        yref="paper",
        text=f"<b>{int(round(score))}</b>",
        showarrow=False,
        font={"size": 56, "color": BRAND_NAVY},
        bordercolor=BRAND_NAVY,
        borderwidth=3,
        borderpad=10,
        bgcolor="white",
    )

    fig.update_layout(
        margin=dict(l=24, r=24, t=70, b=18),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_gauge_legend():
    st.markdown(
        f"""
        <div class="gauge-legend">
            <span><span class="dot" style="background:{BRAND_TEAL};"></span>Minimal</span>
            <span><span class="dot" style="background:{BRAND_YELLOW};"></span>Building</span>
            <span><span class="dot" style="background:{BRAND_ORANGE};"></span>Elevated</span>
            <span><span class="dot" style="background:{BRAND_RED};"></span>Peak</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_question_header(question):
    title_col, logo_col = st.columns([4.0, 1.15])
    with title_col:
        st.markdown(f'<h2 class="shs-h2">{question["title"]}</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="shs-desc">{question["desc"]}</div>', unsafe_allow_html=True)
    with logo_col:
        try:
            st.image(Image.open(LOGO_PATH), use_container_width=True)
        except Exception:
            pass


def render_question_text(question):
    st.markdown(f'<div class="question-text">{question["text"]}</div>', unsafe_allow_html=True)
    if question.get("extra"):
        st.markdown(f'<div class="question-extra">{question["extra"]}</div>', unsafe_allow_html=True)


# ----------------------------
# Step tracker & progress
# ----------------------------
TOTAL_STEPS = 13
if st.session_state.step < 13:
    st.progress(st.session_state.step / TOTAL_STEPS)


# ----------------------------
# STEP 1: ZIP CODE
# ----------------------------
if st.session_state.step == 1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(Image.open(LOGO_PATH), use_container_width=True)
        except Exception:
            st.write("*(Logo placeholder)*")

    st.markdown(
        """
        <div class="shs-header">
            <div class="shs-brand-title">Fly Pressure Forecast Tool</div>
        </div>
        <h2 class="shs-h2" style="text-align:center; margin-top:0.1rem;">Let's Get Started!</h2>
        """,
        unsafe_allow_html=True,
    )

    zip_in = st.text_input(
        "Enter Farm ZIP Code. We'll use this to pull your local weather data.",
        value=st.session_state.zip_code,
        placeholder="e.g., 30542",
        max_chars=10,
    ).strip()

    if st.button("Continue", type="primary", use_container_width=True):
        # Normalize a 9-digit zip to 5 digits for the API
        clean_zip = ''.join(filter(str.isdigit, zip_in))[:5]
        
        if not validate_zip(clean_zip):
            st.error("Please enter a valid 5-digit ZIP code.")
        else:
            with st.spinner("Locating..."):
                lat, lon = get_lat_lon_from_zip(clean_zip)
                if lat is None or lon is None:
                    st.error("ZIP lookup failed. Please try again.")
                else:
                    st.session_state.zip_code = clean_zip
                    st.session_state.lat = float(lat)
                    st.session_state.lon = float(lon)
                    st.session_state.step = 2
                    st.rerun()

# ----------------------------
# STEP 2: TRANSITION STATE
# ----------------------------
elif st.session_state.step == 2:
    st.markdown('<h2 class="shs-h2" style="text-align:center;">Location Confirmed</h2>', unsafe_allow_html=True)
    st.write(f"We've located the historical and forecasted weather data for **{st.session_state.zip_code}**.")
    st.write(
        "Next, we need a little information about your farm management practices that can have an impact on fly populations. "
        "We'll ask you 9 quick questions so we can tailor your final risk score."
    )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("Start Questions", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ----------------------------
# STEPS 3-11: QUESTIONS
# ----------------------------
elif 3 <= st.session_state.step <= 11:
    q_idx = st.session_state.step - 3
    q = QUESTIONS[q_idx]

    render_question_header(q)
    render_question_text(q)

    if q["type"] == "radio":
        # Native key-binding eliminates the stuttering issue completely
        st.radio(
            "Select one:",
            options=list(q["options"].keys()),
            key=q["id"],
            label_visibility="collapsed",
        )

    elif q["type"] == "multiselect":
        # Streamlit handles the checkbox state natively via keys
        for opt in q["options"].keys():
            chk_key = f"ui_{q['id']}_{opt}"
            # Initialize default state for this specific option if not set
            if chk_key not in st.session_state:
                st.session_state[chk_key] = (opt in st.session_state[q["id"]])
            
            st.checkbox(opt, key=chk_key)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back", use_container_width=True):
            st.session_state.step -= 1
            st.rerun()
    with col2:
        btn_text = "Next" if st.session_state.step < 11 else "Calculate Risk Score"
        if st.button(btn_text, type="primary", use_container_width=True):
            st.session_state.step += 1
            st.rerun()

# ----------------------------
# STEP 12: COMPUTE & RESULTS GAUGE
# ----------------------------
elif st.session_state.step == 12:
    if not st.session_state.result:
        answers_0_1 = {}
        for q in QUESTIONS:
            if q["type"] == "radio":
                ans = st.session_state[q["id"]]
                answers_0_1[q["id"]] = q["options"][ans]
            elif q["type"] == "multiselect":
                # Gather multiselect responses dynamically
                ans = [opt for opt in q["options"].keys() if st.session_state.get(f"ui_{q['id']}_{opt}", False)]
                vals = [q["options"][a] for a in ans]
                answers_0_1[q["id"]] = max(vals, default=0.0)

        st.session_state.running = True
        lat, lon = st.session_state.lat, st.session_state.lon
        base_c, upper_c = DEFAULT_BASE_C, DEFAULT_UPPER_C

        try:
            with st.status("Computing your risk score. This takes a second.", expanded=True) as status:
                status.update(label="Calculating your risk score. This takes a second.")

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_nws = executor.submit(fetch_and_process_nws, lat, lon, base_c, upper_c)
                    future_ytd = executor.submit(fetch_and_process_ytd, lat, lon, base_c, upper_c)
                    future_baseline = executor.submit(fetch_and_process_baseline, lat, lon, base_c, upper_c)

                    daily_fc = future_nws.result()
                    df_ytd_dd, p_mm_2_7w_total = future_ytd.result()
                    daily_dd_stats = future_baseline.result()

                status.update(label="Calculating your custom 30-day forecast...")
                timeline = build_master_timeline(df_ytd_dd, daily_fc, daily_dd_stats, horizon_days=30)

                end_30 = date.today() + timedelta(days=29)
                window_30 = timeline[(timeline["date"] >= date.today()) & (timeline["date"] <= end_30)]

                t30_c = float(window_30["avg_temp_c"].mean()) if not window_30.empty else 15.0
                dd15_sum_30 = float(window_30["dd15"].sum()) if not window_30.empty else 120.0

                result = compute_risk_score(t30_c, dd15_sum_30, float(p_mm_2_7w_total), answers_0_1)
                result["band"] = get_band(result["risk_score"])

                st.session_state.result = result
                status.update(label="Analysis complete.", state="complete", expanded=False)

        except Exception as exc:
            status.update(label="An error occurred during calculation. Try again in a second.", state="error", expanded=True)
            st.error(f"Error details: {exc}")
            st.stop()
        finally:
            st.session_state.running = False

    if st.session_state.result:
        res = st.session_state.result
        score = float(res["risk_score"])
        fig = build_gauge(score, height=360)

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        render_gauge_legend()
        st.markdown(
            f"<div style='text-align:center; font-size:30px; font-weight:800; color:{BRAND_NAVY}; margin-top:-16px; margin-bottom:10px;'>"
            f"Risk Level: {res['band']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; color:#555;'><em>Click Next to view your full assessment and recommended actions.</em></p>",
            unsafe_allow_html=True,
        )

        st.write("")
        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("Back", use_container_width=True, type="secondary"):
                st.session_state.result = None
                st.session_state.step = 11
                st.rerun()
        with col_next:
            if st.button("Next", use_container_width=True, type="primary"):
                st.session_state.step = 13
                st.rerun()

# ----------------------------
# STEP 13: SUMMARY PAGE
# ----------------------------
elif st.session_state.step == 13:
    res = st.session_state.result
    score = int(round(float(res["risk_score"])))
    band = res["band"]
    band_color = get_band_color(band)

    band_explanations = {
        "Minimal": "Your score is Minimal, indicating that current conditions and your management practices are helping keep fly pressure low. This is still the right time to stay proactive so you are not caught reacting later.",
        "Building": "Your score is Building, meaning conditions are beginning to support fly emergence and pressure may rise over the next 30 days. This is the ideal window to tighten prevention before the population accelerates.",
        "Elevated": "Your score is Elevated, meaning weather and on-farm conditions are favorable for meaningful fly pressure. Prevention should already be active, and targeted intervention is recommended now.",
        "Peak": "Your score is Peak, meaning strong fly pressure is likely already underway or imminent. Immediate action is recommended to limit further economic impact on the herd.",
    }

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown("<h2 class='shs-h2' style='margin-top:10px;'>Your Farm's Assessment</h2>", unsafe_allow_html=True)
    with header_col2:
        try:
            st.image(Image.open(LOGO_PATH), use_container_width=True)
        except Exception:
            pass

    top_card = st.container(border=True)
    with top_card:
        st.plotly_chart(
            build_gauge(score, height=285),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        render_gauge_legend()

        st.markdown(
            f"""
            <hr style="border:none; border-top:1px solid rgba(0,40,84,0.10); margin:0.35rem 0 0.75rem 0;">
            <div style="
                display:flex;
                align-items:flex-start;
                justify-content:space-between;
                gap:1.2rem;
                padding: 0.1rem 0.15rem 0.25rem 0.15rem;
                flex-wrap:wrap;
            ">
                <div style="min-width:155px;">
                    <div style="font-size:0.95rem; font-weight:700; color:{BRAND_NAVY}; margin-bottom:0.3rem;">
                        Estimated Risk Score
                    </div>
                    <div style="font-size:2.8rem; font-weight:900; line-height:1; color:{band_color};">
                        {score}
                    </div>
                    <div style="font-size:1.25rem; font-weight:800; color:{BRAND_NAVY}; margin-top:0.15rem;">
                        {band}
                    </div>
                </div>
                <div style="
                    flex:1;
                    min-width:240px;
                    font-size:1.02rem;
                    color:{GRAY};
                    line-height:1.65;
                    padding-top:0.15rem;
                ">
                    Scores in this range suggest <strong>{band.lower()}</strong> fly pressure relative to your current weather pattern and management inputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="assessment-box">
            <div style="font-size:1.55rem; font-weight:800; color:{BRAND_NAVY}; margin-bottom:0.9rem;">What does this mean?</div>
            <div style="font-size:1rem; color:{BRAND_NAVY}; line-height:1.65; margin-bottom:0.9rem;">
                We look at the next 7 days of weather combined with 8 years of historical climate data for your ZIP code to
                predict how local temperatures and precipitation will drive fly emergence over the next 30 days.
            </div>
            <div style="font-size:1rem; color:{BRAND_NAVY}; line-height:1.65; margin-bottom:0.9rem;">
                <strong>{band_explanations.get(band, '')}</strong>
            </div>
            <div style="font-size:1rem; color:{BRAND_NAVY}; line-height:1.65; margin-bottom:0.9rem;">
                <strong>Did You Know...</strong> stable flies can lower milk production by 15-30%<sup>1</sup>. Prevention and early intervention
                measures are key to a sound fly control strategy. Don't wait until you see cows bunching, by then the economic loss has already begun.
            </div>
            <div class="small-note" style="margin-bottom:1.1rem;"><sup>1</sup><a href='{MILK_LOSS_PAPER_URL}' target='_blank'>{MILK_LOSS_FOOTNOTE}</a></div>
            <div style="font-size:1.55rem; font-weight:800; color:{BRAND_NAVY}; margin-bottom:0.9rem;">Take Action</div>
            <div style="font-size:1rem; color:{BRAND_NAVY}; line-height:1.65;">
                Protect your herd and your bottom line with the <strong>ShieldStrong® Automated Fly Control System.</strong> Our set-and-forget
                fly control system eliminates the need for manual application, reduces employee exposure to chemicals and consistently
                applies the recommended dose of insecticide per cow, every time. The ShieldStrong system, combined with our unique
                rotation of insecticides, are an unstoppable duo for your fly control program. Simply purchase the recommended products
                and we'll do the rest, including product delivery and refills. Contact us now for a FREE consultation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Over", use_container_width=True, type="secondary"):
            reset_all()
    with c2:
        st.link_button(
            "Contact Us",
            "https://specialtyherdsolutions.com/contact/",
            type="primary",
            use_container_width=True,
        )