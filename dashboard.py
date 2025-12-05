# -------------------------------------------------------------------
# dashboard.py  (stable working version with password gate + fixes)
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

    # First visit → ask for password
    if "password_correct" not in st.session_state:
        st.text_input("Enter access code:", type="password",
                      on_change=password_entered, key="password")
        return False

    # Wrong password entered
    if not st.session_state["password_correct"]:
        st.text_input("Enter access code:", type="password",
                      on_change=password_entered, key="password")
        st.error("Incorrect access code.")
        return False

    return True


# STOP here until password is correct
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

    # ---- FIX REQUIRED COLUMNS ----

    # intent
    if "intent" not in df.columns:
        df["intent"] = df["llm_cpd_intent"].fillna(df["rule_intent"])
    df["intent"] = df["intent"].fillna("Unknown")

    # sales_rep
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    df["sales_rep"] = df["sales_rep"].fillna("Unassigned")

    # provider
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # datetime
    df["call_datetime"] = pd.to_datetime(df["call_datetime"], errors="coerce")
    df = df[df["call_datetime"].notna()]

    # call_type
    df["call_type"] = "Standard"
    df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"
    df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # coaching score columns
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
    "Date Range",
    value=[min_date, max_date]
)

# Sales rep list
rep_list = ["All Reps"] + sorted(df["sales_rep"].dropna().unique().tolist())
selected_rep = st.sidebar.selectbox("Sales Rep", rep_list)

# Carrier list
carrier_list = ["All"] + sorted(df["provider"].dropna().unique().tolist())
selected_carrier = st.sidebar.selectbox("Carrier", carrier_list)

# Call type toggles
include_voicemail = st.sidebar.checkbox("Include Voicemail", True)
include_no_answer = st.sidebar.checkbox("Include No Answer", True)
include_too_short = st.sidebar.checkbox("Include Too Short (<1s)", True)

# -------------------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------------------
filtered = df.copy()

# Date filter
start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered = filtered[
    (filtered["call_datetime"] >= start)
    & (filtered["call_datetime"] <= end)
]

# Rep filter
if selected_rep != "All Reps":
    filtered = filtered[filtered["sales_rep"] == selected_rep]

# Carrier
if selected_carrier != "All":
    filtered = filtered[filtered["provider"] == selected_carrier]

# Call type filters
exclude = []
if not include_voicemail:
    exclude.append("Voicemail")
if not include_no_answer:
    exclude.append("No Answer")
if not include_too_short:
    exclude.append("Too Short")

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
    distinct_intents = filtered["intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    # FIXED INTENT COUNTS
    intent_counts = (
        filtered["intent"]
        .fillna("Unknown")
        .value_counts()
        .rename_axis("intent")
        .reset_index(name="call_count")
    )

    chart = (
        alt.Chart(intent_counts)
        .mark_bar()
        .encode(
            x="intent:N",
            y="call_count:Q",
            tooltip=["intent", "call_count"],
            color=alt.Color("intent:N", legend=None),
        )
    )

    labels = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=13
    ).encode(text="call_count:Q")

    st.altair_chart(chart + labels, use_container_width=True)


# -------------------------------------------------------------------
# COACHING TAB
# -------------------------------------------------------------------
with tabs[1]:
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
        st.warning("No coaching data for selected filters.")
    else:
        avg_scores = coach_df.mean().reset_index()
        avg_scores.columns = ["pillar", "score"]

        bar = (
            alt.Chart(avg_scores)
            .mark_bar()
            .encode(
                x=alt.X("pillar:N", title="Coaching Pillar"),
                y=alt.Y("score:Q", title="Average Score"),
                tooltip=["pillar", "score"],
                color=alt.Color("pillar:N", legend=None),
            )
        )

        bar_labels = bar.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=14,
        ).encode(text="score:Q")

        st.altair_chart(bar + bar_labels, use_container_width=True)

    st.subheader("Improvement Points")
    if "coaching_improvement_points" in filtered.columns:
        points = (
            filtered["coaching_improvement_points"]
            .dropna()
            .astype(str)
            .tolist()
        )
        for p in points:
            st.write(f"- {p}")
