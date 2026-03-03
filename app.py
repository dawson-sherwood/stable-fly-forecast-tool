# app.py

import streamlit as st
import pandas as pd
import concurrent.futures
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "ShieldStrongLogoGreen_2026.png"

from utils import validate_zip, mm_to_in
from data_sources import (
    get_lat_lon_from_zip,
    get_nws_grid_info,
    fetch_nws_grid_data,
    acis_griddata_daily,
    acis_baseline_8yr,
    ytd_date_range,
)
from processing import (
    coerce_temp_to_celsius,
    coerce_precip_to_mm,
    process_hourly_to_daily,
    compute_daily_dd,
)
from baseline import (
    acis_to_daily_schema,
    baseline_to_long,
    build_daily_dd_percentiles,
)
from timeline import build_master_timeline
from lag_features import precip_total_2_to_7_weeks
from config import DEFAULT_BASE_C, DEFAULT_UPPER_C
from risk_score import compute_risk_score

# ----------------------------
# Brand / Theme helpers
# ----------------------------

BRAND_TEAL = "#10a295"
BRAND_NAVY = "#002854"
BRAND_YELLOW = "#feb81c"
BRAND_ORANGE = "#f58220"
BRAND_RED = "#d9534f"
WHITE = "#ffffff"

st.set_page_config(
    page_title="Fly Pressure Forecasting Tool",
    layout="centered",
)

CSS = f"""
<style>
.main .block-container {{
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 700px;
}}
.shs-header {{
    text-align: center;
    padding: 10px;
    background: {WHITE};
    border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.06);
    margin-bottom: 15px;
}}
.shs-brand .title {{
    font-weight: 800;
    color: {BRAND_TEAL};
    font-size: 22px;
    margin-top: 10px;
}}
.shs-h2 {{
    margin: 0 0 5px 0;
    color: {BRAND_NAVY};
    font-weight: 800;
    font-size: 24px;
}}
.shs-desc {{
    font-size: 14px;
    color: #555555;
    margin-bottom: 15px;
    font-style: italic;
}}
.question-text {{
    font-size: 18px;
    font-weight: 600;
    color: {BRAND_NAVY};
    margin-bottom: 15px;
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
QUESTIONS = [
    {
        "id": "q1_sanitation",
        "title": "General Sanitation",
        "desc": "Refers to overall cleanliness and management of the dairy facility. Examples include high traffic areas, alleyways or other animal handling areas, feed storage, water troughs, overgrown vegetation, and other areas where moisture is present.",
        "text": "1. How thoroughly and how often are sanitation tasks performed on your dairy farm?",
        "type": "radio",
        "options": {"Thoroughly, at least once per day": 0.0, "Moderately, several times per week": 0.33, "Lightly, once per week": 0.66, "Not performed regularly": 1.0}
    },
    {
        "id": "q2_manure_tasks",
        "title": "Manure Management",
        "desc": "Refers to practices and strategies to minimize the presence of manure and moisture for a prolonged period of time in highly populated areas. Examples are flushing, scraping, and/or raking of animal confinement areas.",
        "text": "2. How thoroughly and how often are manure management tasks performed on your dairy farm?",
        "type": "radio",
        "options": {"Thoroughly, at least once per day": 0.0, "Moderately, several times per week": 0.33, "Lightly, once per week": 0.66, "Not performed regularly": 1.0}
    },
    {
        "id": "q3_calves",
        "title": "Calf Housing",
        "desc": "Calf housing setups impact moisture retention and overall fly breeding grounds.",
        "text": "3. Are there calves housed on farm? (Select all that apply)",
        "type": "multiselect",
        "options": {"No": 0.0, "Yes, in hutches": 1.0, "Yes, calves in group housing": 0.75, "Yes, older calves over 600lbs in group pens": 0.5}
    },
    {
        "id": "q4_manure_store",
        "title": "Manure Storage",
        "desc": "How manure is stored heavily dictates stable fly emergence rates.",
        "text": "4. Where is manure being managed or stored?",
        "type": "radio",
        "options": {"Removed from the farm entirely": 0.0, "Flushed/scraped into lagoon/digester or separator on-site": 0.5, "Composted on-farm": 1.0}
    },
    {
        "id": "q5_fogging",
        "title": "Prevention: Fogging",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "5. How thoroughly and how often are you fogging in and around the dairy?",
        "type": "radio",
        "options": {"Thoroughly, at least once per day": 0.0, "Moderately, several times per week": 0.33, "Lightly, once per week": 0.66, "Not performed regularly": 1.0}
    },
    {
        "id": "q6_spraying",
        "title": "Prevention: Spraying",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "6. How thoroughly and how often are you spraying or pouring animals?",
        "type": "radio",
        "options": {"Thoroughly, at least once per day": 0.0, "Moderately, several times per week": 0.33, "Lightly, once per week": 0.66, "Not performed regularly": 1.0}
    },
    {
        "id": "q7_feedthrough",
        "title": "Prevention: Feed-through",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "7. Are you using a feed-through product during the fly season?",
        "type": "radio",
        "options": {"Yes": 0.0, "No": 1.0}
    },
    {
        "id": "q8_wasps",
        "title": "Prevention: Fly Predators",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "8. Are you using fly predators (parasitic wasps) around the dairy?",
        "type": "radio",
        "options": {"Yes": 0.0, "No": 1.0}
    },
    {
        "id": "q9_bait",
        "title": "Prevention: Fly Bait",
        "desc": "Refers to practices and strategies currently in place intended to prevent and/or control the presence of flies.",
        "text": "9. Are you using fly bait (scatter, paint-on or spray)?",
        "type": "radio",
        "options": {"Yes": 0.0, "No": 1.0}
    }
]

for q in QUESTIONS:
    if q["id"] not in st.session_state:
        st.session_state[q["id"]] = [] if q["type"] == "multiselect" else list(q["options"].keys())[0]

# ----------------------------
# CONCURRENT API WRAPPERS
# ----------------------------
def fetch_and_process_nws(lat, lon, base_c, upper_c):
    grid = get_nws_grid_info(lat, lon)
    fetched = fetch_nws_grid_data(grid["forecastGridData"]) if grid else None
    if fetched:
        df_temp, df_precip, tuom, puom = fetched
        return compute_daily_dd(process_hourly_to_daily(coerce_temp_to_celsius(df_temp, tuom), coerce_precip_to_mm(df_precip, puom)), base_c, upper_c)
    return pd.DataFrame()

def fetch_and_process_ytd(lat, lon, base_c, upper_c):
    s, e = ytd_date_range()
    df_ytd, _, _ = acis_griddata_daily(lat, lon, s, e, include_precip=True)
    df_ytd_daily = acis_to_daily_schema(df_ytd)
    df_ytd_dd = compute_daily_dd(df_ytd_daily, base_c, upper_c)
    p_mm_2_7w_total, _ = precip_total_2_to_7_weeks(df_ytd_daily, today=date.today())
    return df_ytd_dd, p_mm_2_7w_total

def fetch_and_process_baseline(lat, lon, base_c, upper_c):
    baseline_by_year, _ = acis_baseline_8yr(lat, lon)
    daily = compute_daily_dd(acis_to_daily_schema(baseline_to_long(baseline_by_year)), base_c, upper_c)
    daily["doy"] = pd.to_datetime(daily["date"]).dt.dayofyear
    return build_daily_dd_percentiles(daily)

# ----------------------------
# STEP TRACKER & UI PROGRESS
# ----------------------------
TOTAL_STEPS = 13 # Zip(1) + Transition(2) + Qs(3-11) + Gauge(12) + Summary(13)

# Only show the progress bar if we are not on the final summary screen
if st.session_state.step < 13:
    st.progress(st.session_state.step / TOTAL_STEPS)

# ----------------------------
# STEP 1: ZIP CODE
# ----------------------------
if st.session_state.step == 1:
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(str(LOGO_PATH), use_container_width=True)
        except Exception:
            st.write("*(Logo placeholder)*")
            
    st.markdown(
        f"""
        <div class="shs-header" style="border:none; background:transparent;">
            <div class="shs-brand"><div class="title" style="margin-top:0;">Fly Pressure Forecasting Tool</div></div>
        </div>
        <h2 class="shs-h2" style="text-align:center;">Let's Get Started!</h2>
        """,
        unsafe_allow_html=True,
    )
    
    zip_in = st.text_input("Enter Farm ZIP Code—we'll use this to pull your local weather data", value=st.session_state.zip_code, placeholder="e.g., 30542", max_chars=10).strip()
    
    if st.button("Continue", type="primary", use_container_width=True):
        if not validate_zip(zip_in):
            st.error("Please enter a valid 5-digit ZIP code.")
        else:
            with st.spinner("Locating..."):
                lat, lon = get_lat_lon_from_zip(zip_in)
                if lat is None or lon is None:
                    st.error("ZIP lookup failed. Please try again.")
                else:
                    st.session_state.zip_code = zip_in
                    st.session_state.lat = float(lat)
                    st.session_state.lon = float(lon)
                    st.session_state.step = 2 # Move to transition state
                    st.rerun()

# ----------------------------
# STEP 2: TRANSITION STATE
# ----------------------------
elif st.session_state.step == 2:
    st.markdown('<h2 class="shs-h2" style="text-align:center;">Location Confirmed</h2>', unsafe_allow_html=True)
    st.write(f"We've located the historical and forecasted weather data for **{st.session_state.zip_code}**.")
    st.write("Next, we need a little information about your facility. We'll ask you 9 quick questions about your current management practices so we can tailor your final risk score.")
    
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
    
    # Top-right logo placement using columns
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown(f'<h2 class="shs-h2" style="margin-top:10px;">{q["title"]}</h2>', unsafe_allow_html=True)
    with header_col2:
        try:
            st.image(str(LOGO_PATH), use_container_width=True)
        except Exception:
            pass
            
    st.markdown(f'<div class="shs-desc">{q["desc"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="question-text">{q["text"]}</div>', unsafe_allow_html=True)
    
    if q["type"] == "radio":
        current_val = st.session_state[q["id"]]
        current_idx = list(q["options"].keys()).index(current_val)
        selected = st.radio("Select one:", options=list(q["options"].keys()), index=current_idx, key=f"ui_{q['id']}", label_visibility="collapsed")
        st.session_state[q["id"]] = selected

    elif q["type"] == "multiselect":
        selected_options = []
        for opt in q["options"].keys():
            is_checked = opt in st.session_state[q["id"]]
            if st.checkbox(opt, value=is_checked, key=f"ui_{q['id']}_{opt}"):
                selected_options.append(opt)
        st.session_state[q["id"]] = selected_options
        
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
            ans = st.session_state[q["id"]]
            if q["type"] == "radio":
                answers_0_1[q["id"]] = q["options"][ans]
            elif q["type"] == "multiselect":
                vals = [q["options"][a] for a in ans]
                answers_0_1[q["id"]] = max(vals, default=0.0)

        st.session_state.running = True
        lat, lon = st.session_state.lat, st.session_state.lon
        base_c, upper_c = DEFAULT_BASE_C, DEFAULT_UPPER_C

        try:
            with st.status("Computing your risk score, this takes a second—hang tight!", expanded=True) as status:
                
                status.update(label="Calculating your risk score, this takes a second-hang tight!")
                
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
                
                T30_c = float(window_30["avg_temp_c"].mean()) if not window_30.empty else 15.0
                dd15_sum_30 = float(window_30["dd15"].sum()) if not window_30.empty else 120.0

                result = compute_risk_score(T30_c, dd15_sum_30, float(p_mm_2_7w_total), answers_0_1)
                
                # Dynamic band mapping based on the updated 4-tier system
                score_val = result["risk_score"]
                if score_val <= 30:
                    result["band"] = "Minimal"
                elif score_val <= 60:
                    result["band"] = "Building"
                elif score_val <= 79:
                    result["band"] = "Elevated"
                else:
                    result["band"] = "Peak"
                    
                st.session_state.result = result
                status.update(label="Analysis complete!", state="complete", expanded=False)
                
        except Exception as e:
            status.update(label="An error occurred during calculation. Try again in a second!", state="error", expanded=True)
            st.error(f"Error details: {e}")
            st.stop() 
        finally:
            st.session_state.running = False

    if st.session_state.result:
        res = st.session_state.result
        score = float(res["risk_score"])
        
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=score,
            title={"text": "<b>Fly Risk Score</b>", "font": {"size": 30, "color": BRAND_NAVY}},            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": BRAND_NAVY},
                "bar": {"color": BRAND_NAVY, "thickness": 0.15},
                "steps": [
                    {"range": [0, 30], "color": BRAND_TEAL},
                    {"range": [30, 60], "color": BRAND_YELLOW},
                    {"range": [60, 80], "color": BRAND_ORANGE},
                    {"range": [80, 100], "color": BRAND_RED},
                ],
                "threshold": {
                    "line": {"color": "#555555", "width": 4}, 
                    "thickness": 0.75, 
                    "value": score
                }
            }
        ))
        
        fig.add_annotation(
            x=0.5, y=0.08, 
            text=f"<b>{int(score)}</b>",
            showarrow=False,
            font=dict(size=55, color=BRAND_NAVY),
            bordercolor=BRAND_NAVY,
            borderwidth=3,
            borderpad=10,
            bgcolor="white",
        )
        
        fig.update_layout(margin=dict(l=20, r=20, t=80, b=20), height=350) 
        
        st.plotly_chart(fig, width="stretch")
        
        st.markdown(f"<div style='text-align:center; font-size:32px; font-weight:800; color:{BRAND_NAVY}; margin-top:-30px; margin-bottom:10px;'>Risk Level: {res['band']}</div>", unsafe_allow_html=True)
        
        st.markdown("<p style='text-align:center; color:#555;'><em>Click Next to view your full assessment and recommended actions.</em></p>", unsafe_allow_html=True)
            
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
# STEP 13: NEW SUMMARY PAGE
# ----------------------------
elif st.session_state.step == 13:
    res = st.session_state.result
    
    # Top-right logo placement
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown('<h2 class="shs-h2" style="margin-top:10px;">Your Farm\'s Assessment</h2>', unsafe_allow_html=True)
    with header_col2:
        try:
            st.image(str(LOGO_PATH), use_container_width=True)
        except Exception:
            pass
    
    # Dynamic styling based on Risk Band
    band_color = {
        "Minimal": BRAND_TEAL,
        "Building": BRAND_YELLOW,
        "Elevated": BRAND_ORANGE,
        "Peak": BRAND_RED
    }.get(res['band'], BRAND_NAVY)
    
    st.markdown(
        f"""
        <div style="text-align:center; padding: 20px; border: 2px solid {band_color}; border-radius: 10px; margin-bottom: 20px;">
            <div style="font-size: 48px; font-weight: 900; color: {band_color};">{res['risk_score']}</div>
            <div style="font-size: 24px; font-weight: 700; color: {BRAND_NAVY};">Category: {res['band']}</div>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Dynamic text logic mapping
    band_explanations = {
        "Minimal": "Your score is Minimal, indicating that weather conditions and your current management practices are keeping fly pressure low. However, staying proactive is the best way to prevent unexpected flare-ups.",
        "Building": "Your score is Building, meaning the environment is starting to prime for fly development. Now is the time to ensure your preventative measures are fully active before the population explodes.",
        "Elevated": "Your score is Elevated. Conditions are highly favorable for stable fly emergence, and you may start seeing an economic impact on your herd. Immediate, targeted action is recommended.",
        "Peak": "Your score is Peak. The combination of weather and available breeding sites suggests severe fly pressure. Aggressive intervention is suggested right now to protect your herd's health and milk production."
    }
    
    st.markdown(
        f"""
        ### What does this mean?
        We look at the next 7 days of weather combined with 8 years of historical climate data for your ZIP code to predict how local temperatures and precipitation will drive stable fly emergence over the next 30 days. 
        
        **{band_explanations.get(res['band'], "")}**
        
        Because stable flies can lower milk production by 15-30%, early intervention is the most economically sound strategy. Don't wait until you see your cows bunching—by then, the economic loss has already occurred.
        
        ### Take Action
        Protect your herd and your bottom line. Let our custom-designed, set-and-forget fly control system get the job done for you at zero upfront cost.
        """
    )
    
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Over", use_container_width=True, type="secondary"):
            reset_all()
    with c2:
        st.link_button("Contact Us", "https://specialtyherdsolutions.com/contact/", type="primary", use_container_width=True)