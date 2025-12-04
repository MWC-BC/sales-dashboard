# -------------------------------------------------------------------
# dashboard.py  (stable Streamlit Cloud version with password gate)
# -------------------------------------------------------------------

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------
def check_password():
    """Simple password gate using Streamlit secrets."""
    def password_entered():
        if st.session_state["password"] == st.secrets["access"]["code"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    if not st.session_state["password_correct"]:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("Incorrect access code.")
        return False

    return True


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
def load_data():
    df = pd.read_csv(DATA_CSV)

    # --- Create missing unified intent column ------------------------
    if "intent" not in df.columns:
        df["intent"] = df["rule_intent"].fillna("").astype(str).str.strip()
        df.loc[df["intent"] == "", "intent"] = (
            df["llm_cpd_intent"].fillna("").astype(str).str.strip()
        )
        df["intent"].replace("", pd.NA, inplace=True)

    # Make provider pretty (Twilio / Telzio)
    if "provider" in df.columns:
        df["provider"] = df["provider"].str.title()

    # Create simple call_type category
    df["call_type"] = "Standard"
    df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short (<1s)"
    df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # Parse datetime
    df["call_datetime"] = pd.to_datetime(df["call_datetime"], errors="coerce")

    # Fix coaching scores if missing
    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]
    for c in score_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df


df = load_data()

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------

st.sidebar.header("Filters")

# Date range
min_date = df["call_datetime"].min()
max_date = df["call_datetime"].max()

date_range = st.sidebar.date_input(
    "Date Range", [min_date, max_date]
)

# Sales Rep filter
rep_list = ["All Reps"] + sorted(df["sales_rep"].dropna().unique().tolist())
selected_rep = st.sidebar.selectbox("Sales Rep", rep_list)

# Carrier filter
carrier_list = ["All", "Twilio", "Telzio"]
selected_carrier = st.sidebar.selectbox("Carrier", carrier_list)

# Call type toggles
include_voicemail = st.sidebar.checkbox("Include Voicemail", True)
include_no_answer = st.sidebar.checkbox("Include No Answer", True)
include_too_short = st.sidebar.checkbox("Include Too Short (<1s)", True)

# -------------------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------------------
filtered = df.copy()

# Date range filter
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered = filtered[
    (filtered["call_datetime"] >= start)
    & (filtered["call_datetime"] <= end)
]

# Carrier filter
if selected_carrier != "All":
    filtered = filtered[filtered["provider"] == selected_carrier]

# Rep filter
if selected_rep != "All Reps":
    filtered = filtered[filtered["sales_rep"] == selected_rep]

# Call type filters
exclude = []
if not include_voicemail:
    exclude.append("Voicemail")
if not include_no_answer:
    exclude.append("No Answer")
if not include_too_short:
    exclude.append("Too Short (<1s)")

if exclude:
    filtered = filtered[~filtered["call_type"].isin(exclude)]

# -------------------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------------------
st.title("Sales Call Coaching Dashboard")
st.write("Use the sidebar to filter by Sales Rep, date range, carrier, and call type.")

tabs = st.tabs(["Buying Intent", "Coaching"])

# -------------------------------------------------------------------
# BUYING INTENT TAB
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("Intent Overview")

    total_calls = len(filtered)
    total_intents = filtered["intent"].nunique()

    col1, col2 = st.columns(2)
    col1.metric("Total Calls in Filter", total_calls)
    col2.metric("Distinct CPD Intents", total_intents)

    # Bar: Count per intent
    if total_calls > 0:
        intent_counts = (
            filtered["intent"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "intent", "intent": "count"})
        )

        chart = (
            alt.Chart(intent_counts)
            .mark_bar()
            .encode(
                x="intent:N",
                y="count:Q",
                tooltip=["intent", "count"],
                color=alt.Color("intent:N", legend=None),
            )
        )

        labels = (
            chart.mark_text(
                align="center",
                baseline="bottom",
                dy=-5,
                fontSize=14
            ).encode(text="count:Q")
        )

        st.altair_chart(chart + labels, use_container_width=True)

# -------------------------------------------------------------------
# COACHING TAB
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Coaching Scores")

    # Score data
    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]
    score_df = filtered[score_cols].dropna()

    if len(score_df) == 0:
        st.warning("No coaching data available for these filters.")
    else:
        avg_scores = score_df.mean().reset_index()
        avg_scores.columns = ["pillar", "score"]

        chart = (
            alt.Chart(avg_scores)
            .mark_bar()
            .encode(
                x=alt.X("pillar:N", title="Coaching Pillar"),
                y=alt.Y("score:Q", title="Avg Score"),
                tooltip=["pillar", "score"],
                color=alt.Color("pillar:N", legend=None),
            )
        )

        labels = chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=14
        ).encode(text="score:Q")

        st.altair_chart(chart + labels, use_container_width=True)

    st.subheader("Improvement Points")
    if "coaching_improvement_points" in filtered.columns:
        improvements = filtered["coaching_improvement_points"].dropna().tolist()
        for item in improvements:
            st.write(f"- {item}")
