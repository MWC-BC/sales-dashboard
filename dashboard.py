# -------------------------------------------------------------------
# dashboard.py  (working version: password gate + fixed filters/charts)
# -------------------------------------------------------------------

from pathlib import Path
from typing import List, Tuple

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
        if st.session_state["password"] == st.secrets["access"]["code"]:
            st.session_state["password_correct"] = True
            # remove the password from session state
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


# Stop app until password is correct
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
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ---------- datetime ----------
    # Normalize to timezone-naive and drop rows with invalid datetime
    if "call_datetime" in df.columns:
        dt = pd.to_datetime(df["call_datetime"], utc=True, errors="coerce")
        # strip timezone info to avoid tz-naive / tz-aware comparison errors
        df["call_datetime"] = dt.dt.tz_convert(None)
        df = df[df["call_datetime"].notna()]
    else:
        df["call_datetime"] = pd.NaT

    # ---------- intent ----------
    if "intent" not in df.columns:
        # prefer LLM intent, fall back to rule intent
        df["intent"] = df.get("llm_cpd_intent", pd.Series(index=df.index))
        df["intent"] = df["intent"].fillna(df.get("rule_intent", ""))
    df["intent"] = df["intent"].fillna("Unknown").replace("", "Unknown")

    # ---------- sales rep ----------
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    df["sales_rep"] = df["sales_rep"].fillna("Unassigned").astype(str)

    # ---------- provider / carrier ----------
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # ---------- call type ----------
    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    if "duration_seconds" in df.columns:
        df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"
    if "status" in df.columns:
        df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # ---------- coaching scores ----------
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


df = load_data(DATA_CSV)

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------
st.sidebar.header("Filters")

# ----- date range -----
if df["call_datetime"].notna().any():
    min_date = df["call_datetime"].dt.date.min()
    max_date = df["call_datetime"].dt.date.max()
else:
    # fallback if something goes wrong
    min_date = pd.to_datetime("2000-01-01").date()
    max_date = pd.to_datetime("2000-01-01").date()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
)

# ----- sales rep filter -----
unique_reps = sorted(
    {r for r in df["sales_rep"].dropna().unique() if r != "Unassigned"}
)
rep_options: List[str] = ["All Reps"] + unique_reps
if "Unassigned" in df["sales_rep"].values:
    rep_options.append("Unassigned")

selected_rep = st.sidebar.selectbox("Sales Rep", rep_options)

# ----- carrier filter -----
carrier_list = ["All"] + sorted(df["provider"].dropna().unique().tolist())
selected_carrier = st.sidebar.selectbox("Carrier", carrier_list)

# ----- call type toggles -----
include_voicemail = st.sidebar.checkbox("Include Voicemail", True)
include_no_answer = st.sidebar.checkbox("Include No Answer", True)
include_too_short = st.sidebar.checkbox("Include Too Short (<1s)", True)

# -------------------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------------------
filtered = df.copy()

# Date range – compare by DATE only (no tz problems)
start_date, end_date = date_range
filtered = filtered[
    (filtered["call_datetime"].dt.date >= start_date)
    & (filtered["call_datetime"].dt.date <= end_date)
]

# Carrier filter
if selected_carrier != "All":
    filtered = filtered[filtered["provider"] == selected_carrier]

# Rep filter
if selected_rep != "All Reps":
    filtered = filtered[filtered["sales_rep"] == selected_rep]

# Call type filters
exclude_types: List[str] = []
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
st.write(
    "Use the sidebar to filter by Sales Rep, date range, carrier, and call type."
)

tabs = st.tabs(["Buying Intent", "Coaching"])

# -------------------------------------------------------------------
# BUYING INTENT TAB
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("Intent Overview")

    total_calls = len(filtered)
    distinct_intents = filtered["intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls", int(total_calls))
    c2.metric("Distinct CPD Intents", int(distinct_intents))

    # Intent distribution – avoid duplicate column names
    intent_counts = (
        filtered["intent"]
        .fillna("Unknown")
        .replace("", "Unknown")
        .value_counts()
        .reset_index()
    )
    intent_counts.columns = ["intent", "call_count"]

    if not intent_counts.empty:
        base = alt.Chart(intent_counts)

        bars = (
            base.mark_bar()
            .encode(
                x=alt.X("intent:N", title="Intent"),
                y=alt.Y("call_count:Q", title="Calls"),
                tooltip=["intent", "call_count"],
                color=alt.Color("intent:N", legend=None),
            )
        )

        labels = (
            base.mark_text(
                align="center",
                baseline="bottom",
                dy=-5,
                fontSize=14,
            )
            .encode(
                x="intent:N",
                y="call_count:Q",
                text="call_count:Q",
            )
        )

        st.altair_chart(bars + labels, use_container_width=True)
    else:
        st.info("No calls match the selected filters.")

# -------------------------------------------------------------------
# COACHING TAB
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Coaching Scores")

    scoring_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]

    coach_df = filtered[scoring_cols].dropna(how="all")

    if coach_df.empty:
        st.warning("No coaching data for selected filters.")
    else:
        avg_scores = coach_df.mean().reset_index()
        avg_scores.columns = ["pillar", "score"]

        base = alt.Chart(avg_scores)

        bars = (
            base.mark_bar()
            .encode(
                x=alt.X("pillar:N", title="Coaching Pillar"),
                y=alt.Y("score:Q", title="Average Score"),
                tooltip=["pillar", "score"],
                color=alt.Color("pillar:N", legend=None),
            )
        )

        labels = (
            base.mark_text(
                align="center",
                baseline="bottom",
                dy=-5,
                fontSize=14,
            )
            .encode(
                x="pillar:N",
                y="score:Q",
                text="score:Q",
            )
        )

        st.altair_chart(bars + labels, use_container_width=True)

    st.subheader("Improvement Points")
    if "coaching_improvement_points" in filtered.columns:
        imp_list = (
            filtered["coaching_improvement_points"]
            .dropna()
            .astype(str)
            .tolist()
        )
        if not imp_list:
            st.info("No coaching improvement points for selected filters.")
        else:
            for item in imp_list:
                st.write(f"- {item}")
    else:
        st.info("No coaching improvement data in this dataset.")
