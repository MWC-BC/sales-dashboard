# -------------------------------------------------------------------
# dashboard.py  – stable version for Streamlit Cloud
#  - Password gate (secrets: [access] code = "...")
#  - Date / Rep / Carrier / Call-type filters
#  - Buying Intent chart
#  - Coaching scores + improvement points
# -------------------------------------------------------------------

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# STREAMLIT CONFIG  (must be first Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Call Coaching Dashboard",
    layout="wide",
)
alt.data_transformers.disable_max_rows()

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------
def check_password() -> bool:
    """Simple password gate using Streamlit secrets."""
    def password_entered():
        correct = st.session_state.get("password") == st.secrets["access"]["code"]
        st.session_state["password_correct"] = correct
        # Clear the password from session state for safety
        st.session_state["password"] = ""

    # First run: ask for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    # If password was incorrect, ask again
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


# Stop app here until password is correct
if not check_password():
    st.stop()

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"

# -------------------------------------------------------------------
# LOAD & CLEAN DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    # ---------- DATETIME ----------
    if "call_datetime" in df.columns:
        # Parse as UTC then drop tz to ensure *naive* timestamps
        dt = pd.to_datetime(df["call_datetime"], errors="coerce", utc=True)
        df["call_datetime"] = dt.dt.tz_localize(None)
        df = df[df["call_datetime"].notna()]
    else:
        # If somehow missing, create a dummy column so the app still runs
        df["call_datetime"] = pd.Timestamp.now()

    # ---------- INTENT ----------
    if "intent" not in df.columns:
        # Fallback: combine llm + rule intent
        base_intent = df.get("llm_cpd_intent", pd.Series(index=df.index))
        rule_intent = df.get("rule_intent", pd.Series(index=df.index))
        df["intent"] = base_intent.fillna(rule_intent)
    df["intent"] = df["intent"].fillna("Unknown").astype(str)

    # ---------- SALES REP ----------
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    df["sales_rep"] = df["sales_rep"].fillna("Unassigned").astype(str)

    # ---------- PROVIDER / CARRIER ----------
    if "provider" not in df.columns:
        df["provider"] = "Unknown"
    df["provider"] = df["provider"].fillna("Unknown").astype(str).str.title()

    # ---------- CALL TYPE ----------
    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    if "duration_seconds" in df.columns:
        df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"
    if "status" in df.columns:
        df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # ---------- COACHING SCORES ----------
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

    # ---------- TEXT FIELDS ----------
    for col in [
        "coaching_summary",
        "coaching_improvement_points",
        "transcript",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


df = load_data()

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------
st.sidebar.header("Filters")

# Date range
min_dt = df["call_datetime"].min()
max_dt = df["call_datetime"].max()
default_start = min_dt.date() if not pd.isna(min_dt) else pd.Timestamp.now().date()
default_end = max_dt.date() if not pd.isna(max_dt) else pd.Timestamp.now().date()

date_range = st.sidebar.date_input(
    "Date Range",
    [default_start, default_end],
)

# Sales Rep filter
rep_options = sorted(df["sales_rep"].dropna().unique().tolist())
rep_list = ["All Reps"] + rep_options
selected_rep = st.sidebar.selectbox("Sales Rep", rep_list)

# Carrier filter (use actual providers present)
provider_options = sorted(df["provider"].dropna().unique().tolist())
carrier_list = ["All"] + provider_options
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
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[
        (filtered["call_datetime"] >= start)
        & (filtered["call_datetime"] < end)
    ]

# Carrier filter
if selected_carrier != "All":
    filtered = filtered[filtered["provider"] == selected_carrier]

# Rep filter
if selected_rep != "All Reps":
    filtered = filtered[filtered["sales_rep"] == selected_rep]

# Call type filters
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

    # Intent distribution
    if total_calls == 0:
        st.warning("No calls match the selected filters.")
    else:
        intent_counts = (
            filtered["intent"]
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        intent_counts.columns = ["intent", "count"]

        base = (
            alt.Chart(intent_counts)
            .mark_bar()
            .encode(
                x=alt.X("intent:N", title="Intent", sort="-y"),
                y=alt.Y("count:Q", title="Calls"),
                tooltip=["intent", "count"],
                color=alt.Color("intent:N", legend=None),
            )
            .properties(height=350)
        )

        st.altair_chart(base, use_container_width=True)

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

        label_map = {
            "coaching_opening_score": "Opening",
            "coaching_discovery_score": "Discovery",
            "coaching_value_score": "Value",
            "coaching_closing_score": "Closing",
            "coaching_total_score": "Total",
        }
        avg_scores["pillar_label"] = avg_scores["pillar"].map(label_map)

        bar = (
            alt.Chart(avg_scores)
            .mark_bar()
            .encode(
                x=alt.X("pillar_label:N", title="Coaching Pillar"),
                y=alt.Y("score:Q", title="Average Score"),
                tooltip=["pillar_label", "score"],
                color=alt.Color("pillar_label:N", legend=None),
            )
            .properties(height=350)
        )

        st.altair_chart(bar, use_container_width=True)

    # ----- Improvement Points -----
    st.subheader("Improvement Points")
    if "coaching_improvement_points" in filtered.columns:
        points = (
            filtered["coaching_improvement_points"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not points:
            st.write("No improvement points available for the selected filters.")
        else:
            for item in points:
                st.write(f"- {item}")

    # ----- Optional: Call-level table -----
    st.subheader("Call-Level Coaching Details")
    detail_cols = [
        "call_datetime",
        "sales_rep",
        "intent",
        "coaching_summary",
        "coaching_improvement_points",
    ]
    detail_cols = [c for c in detail_cols if c in filtered.columns]

    if detail_cols:
        st.dataframe(
            filtered[detail_cols].sort_values("call_datetime", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No detailed coaching columns available.")
