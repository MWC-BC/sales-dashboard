# -------------------------------------------------------------------
# dashboard.py  (stable version: password gate + fixed date filters)
# -------------------------------------------------------------------

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------
def check_password() -> bool:
    """Simple password gate using Streamlit secrets."""
    def password_entered():
        if st.session_state.get("password") == st.secrets["access"]["code"]:
            st.session_state["password_correct"] = True
            # don't keep the password in session state
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # First run: ask for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    # Wrong password entered previously
    if not st.session_state["password_correct"]:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("Incorrect access code.")
        return False

    # Password correct
    return True


# STOP app until password is correct
if not check_password():
    st.stop()

# -------------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Call Coaching Dashboard",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    # -----------------------
    # Intent column
    # -----------------------
    if "intent" not in df.columns:
        # fall back to LLM / rule intent
        llm = df["llm_cpd_intent"] if "llm_cpd_intent" in df.columns else np.nan
        rule = df["rule_intent"] if "rule_intent" in df.columns else np.nan
        df["intent"] = llm.fillna(rule)
    df["intent"] = df["intent"].fillna("Unknown").astype(str)

    # -----------------------
    # Sales rep
    # -----------------------
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    else:
        df["sales_rep"] = df["sales_rep"].fillna("Unassigned").astype(str)

    # -----------------------
    # Provider (carrier)
    # -----------------------
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # -----------------------
    # Call datetime (make tz-naive so comparisons don't explode)
    # -----------------------
    if "call_datetime" in df.columns:
        dt = pd.to_datetime(df["call_datetime"], utc=True, errors="coerce")
        # remove timezone info -> naive datetime64[ns]
        df["call_datetime"] = dt.dt.tz_localize(None)
        df = df[df["call_datetime"].notna()]
    else:
        df["call_datetime"] = pd.NaT

    # -----------------------
    # Call type flags (voicemail, too short, no-answer)
    # -----------------------
    if "is_voicemail" not in df.columns:
        df["is_voicemail"] = False
    if "duration_seconds" not in df.columns:
        # fall back to duration if present, else zero
        if "duration" in df.columns:
            df["duration_seconds"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
        else:
            df["duration_seconds"] = 0
    if "status" not in df.columns:
        df["status"] = ""

    df["call_type"] = "Standard"
    df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"
    df.loc[df["status"].astype(str).str.lower() == "no-answer", "call_type"] = "No Answer"

    # -----------------------
    # Coaching scores
    # -----------------------
    # Some pipelines use "coaching_total_sc" instead of "coaching_total_score"
    if "coaching_total_score" not in df.columns and "coaching_total_sc" in df.columns:
        df["coaching_total_score"] = df["coaching_total_sc"]

    score_cols: List[str] = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure numeric
    df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")

    # Text fields for coaching section
    if "coaching_summary" not in df.columns:
        df["coaching_summary"] = ""
    if "coaching_improvement_points" not in df.columns:
        df["coaching_improvement_points"] = ""

    if "transcript" not in df.columns:
        df["transcript"] = ""

    return df


df = load_data()

if df.empty:
    st.error("No data available in the CSV. Please check the pipeline/export.")
    st.stop()

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------
st.sidebar.header("Filters")

# ----- Date range -----
min_dt = df["call_datetime"].min()
max_dt = df["call_datetime"].max()

# Use dates (not datetimes) for the widget
min_date = min_dt.date()
max_date = max_dt.date()

date_range = st.sidebar.date_input(
    "Date Range",
    (min_date, max_date),
)

# normalise what Streamlit returns (can be a date or list of dates)
if isinstance(date_range, (list, tuple)):
    if len(date_range) == 2:
        start_date, end_date = date_range
    elif len(date_range) == 1:
        start_date = end_date = date_range[0]
    else:
        start_date = end_date = min_date
else:
    start_date = end_date = date_range

start_ts = pd.to_datetime(start_date)
# include full end day
end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

# ----- Sales rep filter -----
rep_values = sorted(df["sales_rep"].dropna().unique().tolist())
rep_options = ["All Reps"] + rep_values
selected_rep = st.sidebar.selectbox("Sales Rep", rep_options)

# ----- Carrier filter -----
carrier_values = sorted(df["provider"].dropna().unique().tolist())
carrier_options = ["All"] + carrier_values
selected_carrier = st.sidebar.selectbox("Carrier", carrier_options)

# ----- Call type toggles -----
include_voicemail = st.sidebar.checkbox("Include Voicemail", True)
include_no_answer = st.sidebar.checkbox("Include No Answer", True)
include_too_short = st.sidebar.checkbox("Include Too Short (<1s)", True)

# -------------------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------------------
filtered = df.copy()

# Date range
filtered = filtered[
    (filtered["call_datetime"] >= start_ts)
    & (filtered["call_datetime"] <= end_ts)
]

# Carrier
if selected_carrier != "All":
    filtered = filtered[filtered["provider"] == selected_carrier]

# Sales rep
if selected_rep != "All Reps":
    filtered = filtered[filtered["sales_rep"] == selected_rep]

# Call types
exclude_types = []
if not include_voicemail:
    exclude_types.append("Voicemail")
if not include_no_answer:
    exclude_types.append("No Answer")
if not include_too_short:
    exclude_types.append("Too Short")

if exclude_types:
    filtered = filtered[~filtered["call_type"].isin(exclude_types)]

# -------------------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------------------
st.title("Sales Call Coaching Dashboard")
st.write("Use the sidebar to filter by Sales Rep, date range, carrier, and call type.")

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])

# -------------------------------------------------------------------
# BUYING INTENT TAB
# -------------------------------------------------------------------
with tab_intent:
    st.subheader("Intent Overview")

    total_calls = int(len(filtered))
    distinct_intents = int(filtered["intent"].nunique())

    c1, c2 = st.columns(2)
    c1.metric("Total Calls", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    if filtered.empty:
        st.warning("No calls match the selected filters.")
    else:
        intent_counts = (
            filtered["intent"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "intent", "intent": "count"})
        )

        base = alt.Chart(intent_counts)

        bars = base.mark_bar().encode(
            x=alt.X("intent:N", title="Intent"),
            y=alt.Y("count:Q", title="Calls"),
            color=alt.Color("intent:N", legend=None),
            tooltip=["intent", "count"],
        )

        labels = base.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=13,
        ).encode(
            x="intent:N",
            y="count:Q",
            text="count:Q",
        )

        st.altair_chart(bars + labels, use_container_width=True)

# -------------------------------------------------------------------
# COACHING TAB
# -------------------------------------------------------------------
with tab_coaching:
    st.subheader("Coaching Scores")

    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]

    coach_df = filtered[score_cols].dropna(how="all")

    if coach_df.empty:
        st.warning("No coaching data for the selected filters.")
    else:
        avg_scores = coach_df.mean().reset_index()
        avg_scores.columns = ["pillar", "score"]
        avg_scores["score_rounded"] = avg_scores["score"].round(1)

        base = alt.Chart(avg_scores)

        bars = base.mark_bar().encode(
            x=alt.X("pillar:N", title="Coaching Pillar"),
            y=alt.Y("score:Q", title="Average Score (out of 5)"),
            color=alt.Color("pillar:N", legend=None),
            tooltip=["pillar", "score_rounded"],
        )

        labels = base.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=13,
        ).encode(
            x="pillar:N",
            y="score:Q",
            text="score_rounded:Q",
        )

        st.altair_chart(bars + labels, use_container_width=True)

    # -------- Improvement bullets --------
    st.subheader("Improvement Points")
    improvements = (
        filtered["coaching_improvement_points"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    improvements = improvements[improvements != ""]

    if improvements.empty:
        st.info("No coaching improvement points available for these filters.")
    else:
        for item in improvements.unique().tolist():
            st.write(f"- {item}")

    # -------- Coaching call table --------
    st.subheader("Coached Calls")

    table_cols = [
        "call_datetime",
        "sales_rep",
        "intent",
        "coaching_summary",
        "coaching_improvement_points",
        "transcript",
    ]
    table_cols = [c for c in table_cols if c in filtered.columns]

    if table_cols:
        table_df = (
            filtered[table_cols]
            .sort_values("call_datetime", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(table_df, use_container_width=True, height=450)
    else:
        st.info("No detailed coaching columns available in this dataset.")
