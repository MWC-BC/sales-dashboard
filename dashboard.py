# -------------------------------------------------------------------
# dashboard.py  – stable version
#  - Password gate
#  - Simple Altair charts (no overlays)
#  - Date / Rep / Carrier / Call-type filters
# -------------------------------------------------------------------

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.write("BUILD CHECK: V9")

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------
def check_password() -> bool:
    """Simple password gate using Streamlit secrets."""

    def password_entered():
        code = st.secrets.get("access", {}).get("code")
        if code is None:
            # If secrets aren't set correctly, fail closed.
            st.session_state["password_correct"] = False
            return

        if st.session_state["password"] == code:
            st.session_state["password_correct"] = True
            # We don't need to keep the password in session_state
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # First visit: ask for password
    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    # Wrong password: ask again
    if not st.session_state["password_correct"]:
        st.text_input(
            "Enter access code:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("Incorrect access code.")
        return False

    # Correct password
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
# LOAD + PREP DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # --------- Intent column (unified) ----------
    if "intent" not in df.columns:
        if "llm_cpd_intent" in df.columns or "rule_intent" in df.columns:
            df["intent"] = df["llm_cpd_intent"].fillna(df.get("rule_intent"))
        else:
            df["intent"] = "Unknown"

    df["intent"] = df["intent"].fillna("Unknown").astype(str)

    # --------- Sales rep column ----------
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    df["sales_rep"] = df["sales_rep"].fillna("Unassigned").astype(str)

    # --------- Provider / carrier ----------
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # --------- Datetime ----------
    df["call_datetime"] = pd.to_datetime(
        df.get("call_datetime"),
        errors="coerce"
    )

    # Make sure datetime is tz-naive to avoid comparison errors
    if pd.api.types.is_datetime64tz_dtype(df["call_datetime"]):
        df["call_datetime"] = df["call_datetime"].dt.tz_convert(None)

    df = df[df["call_datetime"].notna()].copy()

    # --------- Call type flags ----------
    df["call_type"] = "Standard"

    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"

    if "duration_seconds" in df.columns:
        df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"

    if "status" in df.columns:
        df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # --------- Coaching score columns ----------
    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]
    for col in score_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Ensure improvement text exists
    if "coaching_improvement_points" not in df.columns:
        df["coaching_improvement_points"] = np.nan

    return df


df = load_data(DATA_CSV)

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------
st.sidebar.header("Filters")

# Date range
min_date = df["call_datetime"].dt.date.min()
max_date = df["call_datetime"].dt.date.max()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
)

# Normalise to timestamps (inclusive end-of-day)
start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

# Sales Rep filter
rep_options = ["All Reps"] + sorted(df["sales_rep"].dropna().unique().tolist())
selected_rep = st.sidebar.selectbox("Sales Rep", rep_options)

# Carrier filter
carrier_options = ["All"] + sorted(df["provider"].dropna().unique().tolist())
selected_carrier = st.sidebar.selectbox("Carrier", carrier_options)

# Call type toggles
include_voicemail = st.sidebar.checkbox("Include Voicemail", True)
include_no_answer = st.sidebar.checkbox("Include No Answer", True)
include_too_short = st.sidebar.checkbox("Include Too Short (<1s)", True)

# -------------------------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------------------------
filtered = df.copy()

# Date filter
filtered = filtered[
    (filtered["call_datetime"] >= start) &
    (filtered["call_datetime"] <= end)
]

# Carrier filter
if selected_carrier != "All":
    filtered = filtered[filtered["provider"] == selected_carrier]

# Rep filter
if selected_rep != "All Reps":
    filtered = filtered[filtered["sales_rep"] == selected_rep]

# Call type exclusions
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

    total_calls = len(filtered)
    distinct_intents = filtered["intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls", int(total_calls))
    c2.metric("Distinct CPD Intents", int(distinct_intents))

    # Intent distribution chart
    intent_counts = (
        filtered["intent"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Intent", "intent": "Count"})
    )

    if intent_counts.empty:
        st.info("No calls match the current filters.")
    else:
        base = (
            alt.Chart(intent_counts)
            .mark_bar()
            .encode(
                x=alt.X("Intent:N", title="Intent", sort="-y"),
                y=alt.Y("Count:Q", title="Calls"),
                tooltip=["Intent", "Count"],
                color=alt.Color("Intent:N", legend=None),
            )
            .properties(height=350)
        )

        # IMPORTANT: no overlay / mark_text combo – keep it simple + robust
        st.altair_chart(base, use_container_width=True)

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
        avg_scores.columns = ["Pillar", "Score"]

        bar = (
            alt.Chart(avg_scores)
            .mark_bar()
            .encode(
                x=alt.X("Pillar:N", title="Coaching Pillar"),
                y=alt.Y("Score:Q", title="Average Score"),
                tooltip=["Pillar", "Score"],
                color=alt.Color("Pillar:N", legend=None),
            )
            .properties(height=350)
        )

        st.altair_chart(bar, use_container_width=True)

    # -------------- Improvement points list --------------
    st.subheader("Improvement Points")

    improvements = (
        filtered["coaching_improvement_points"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not improvements:
        st.info("No coaching improvement points available for the selected filters.")
    else:
        for item in improvements:
            st.write(f"- {item}")
