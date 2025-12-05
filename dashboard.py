# -------------------------------------------------------------------
# dashboard.py  – Sales Call Coaching Dashboard
# -------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------
def check_password() -> bool:
    """Simple password gate using Streamlit secrets."""
    def password_entered():
        if st.session_state["password"] == st.secrets["access"]["code"]:
            st.session_state["password_correct"] = True
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

    # Password was entered but wrong
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
REP_MAP_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# -------------------------------------------------------------------
# HELPER FUNCTIONS – SALES REP ASSIGNMENT
# -------------------------------------------------------------------
def canonical_phone(num: Optional[Any]) -> Optional[str]:
    """
    Canonicalize a phone-like value to the last 10 digits.

    - Ignores Twilio client identities like 'client:instapulse_...'
    - Returns None if there are fewer than 10 digits.
    """
    if num is None:
        return None
    if isinstance(num, float) and np.isnan(num):
        return None

    s = str(num).strip()
    if s.startswith("client:"):
        return None

    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 10:
        return None

    # Use last 10 digits so formats +1XXXXXXXXXX / XXXXXXXXXX both work
    return digits[-10:]


def canonical_from_mapping(num: Optional[Any]) -> Optional[str]:
    """Canonicalization specifically for the mapping CSV."""
    return canonical_phone(num)


def load_sales_rep_map(path: Path) -> Dict[str, str]:
    """
    Load Config/sales_rep_phone_map.csv into a dict:
    {canonical_phone: sales_rep_name}
    """
    if not path.exists():
        return {}

    df_map = pd.read_csv(path)
    if not {"phone_number", "sales_rep"}.issubset(df_map.columns):
        return {}

    mapping: Dict[str, str] = {}
    for _, row in df_map.iterrows():
        canon = canonical_from_mapping(row["phone_number"])
        rep = str(row["sales_rep"]).strip()
        if canon and rep:
            mapping[canon] = rep

    return mapping


def get_rep_candidate_number(row: pd.Series) -> Optional[str]:
    """
    Decide which endpoint to use as the rep's phone number for this row,
    then canonicalize it.

    Mirrors the high-level behaviour of assign_sales_rep_final.py:
    - Twilio dialer / outbound → FROM side
    - Inbound → TO side
    - Fallback → FROM then TO
    """
    direction_business = str(row.get("direction_business") or "").lower()
    direction = str(row.get("direction") or "").lower()

    raw_from = row.get("from")
    raw_to = row.get("to")

    # Outbound from the Twilio dialer
    if "twilio dialer" in direction_business:
        return canonical_phone(raw_from)

    # Generic outbound / inbound logic
    if "outbound" in direction:
        return canonical_phone(raw_from)

    if "inbound" in direction:
        return canonical_phone(raw_to)

    # Fallback: try FROM first, then TO
    cand = canonical_phone(raw_from)
    if cand:
        return cand
    return canonical_phone(raw_to)


def assign_sales_reps(df: pd.DataFrame, rep_map: Dict[str, str]) -> pd.DataFrame:
    """
    Add a 'sales_rep' column to df using the mapping and inference rules.
    """
    # Compute canonical rep phone per row
    df = df.copy()
    df["rep_phone_canon"] = df.apply(get_rep_candidate_number, axis=1)

    # Map to rep names
    def lookup_rep(canon: Optional[str]) -> str:
        if canon is None or not isinstance(canon, str):
            return "Unassigned"
        return rep_map.get(canon, "Unassigned")

    df["sales_rep"] = df["rep_phone_canon"].apply(lookup_rep)
    df.drop(columns=["rep_phone_canon"], inplace=True)

    return df

# -------------------------------------------------------------------
# LOAD & PREP DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    # Intent column (rule + LLM)
    if "intent" not in df.columns:
        df["intent"] = df["llm_cpd_intent"].fillna(df["rule_intent"])
        df["intent"] = df["intent"].fillna("Unknown")

    # Provider normalisation
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # Make call_datetime timezone-naive for safe filtering
    dt = pd.to_datetime(df["call_datetime"], utc=True, errors="coerce")
    df["call_datetime"] = dt.dt.tz_convert(None)
    df = df[df["call_datetime"].notna()]

    # Derive call_type
    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    if "duration_seconds" in df.columns:
        df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"
    if "status" in df.columns:
        df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # Ensure coaching score columns exist
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

    # Add sales_rep via mapping
    rep_map = load_sales_rep_map(REP_MAP_CSV)
    df = assign_sales_reps(df, rep_map)

    return df


df = load_data()

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------
st.sidebar.header("Filters")

# Date range
min_date = df["call_datetime"].min().date()
max_date = df["call_datetime"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
)

# Sales Rep filter (Unassigned at the end)
all_reps = sorted(
    {r for r in df["sales_rep"].dropna().unique().tolist() if r != "Unassigned"}
)
if "Unassigned" in df["sales_rep"].values:
    all_reps.append("Unassigned")

rep_options = ["All Reps"] + all_reps
selected_rep = st.sidebar.selectbox("Sales Rep", rep_options)

# Carrier filter
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

# Date range filter (inclusive)
start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
filtered = filtered[
    (filtered["call_datetime"] >= start) & (filtered["call_datetime"] <= end)
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
    c1.metric("Total Calls in Filter", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    # Intent distribution
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
                x=alt.X("intent:N", title="Intent"),
                y=alt.Y("count:Q", title="Calls"),
                tooltip=["intent", "count"],
                color=alt.Color("intent:N", legend=None),
            )
        )

        labels = chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=12,
        ).encode(text="count:Q")

        st.altair_chart(chart + labels, use_container_width=True)
    else:
        st.info("No calls found for the current filters.")

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

        pillar_labels = {
            "coaching_opening_score": "Opening",
            "coaching_discovery_score": "Discovery",
            "coaching_value_score": "Value",
            "coaching_closing_score": "Closing",
            "coaching_total_score": "Total",
        }
        avg_scores["pillar"] = avg_scores["pillar"].map(
            pillar_labels
        ).fillna(avg_scores["pillar"])

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
            fontSize=12,
        ).encode(text="score:Q")

        st.altair_chart(bar + bar_labels, use_container_width=True)

    # ----------------- Improvement bullets -----------------
    st.subheader("Improvement Points")
    if "coaching_improvement_points" in filtered.columns:
        imp_series = (
            filtered["coaching_improvement_points"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not imp_series:
            st.info("No improvement points available for the current filters.")
        else:
            for item in imp_series:
                st.write(f"- {item}")
    else:
        st.info("No improvement points column found in the data.")

    # ----------------- Coaching details table -----------------
    st.subheader("Coaching Details")

    detail_cols_preferred = [
        "call_datetime",
        "sales_rep",
        "intent",
        "coaching_summary",
        "coaching_improvement_points",
        "transcript",
    ]
    available_cols = [c for c in detail_cols_preferred if c in filtered.columns]

    if not available_cols:
        st.info("No detailed coaching fields available to display.")
    else:
        detail_df = filtered[available_cols].copy()
        if "call_datetime" in detail_df.columns:
            detail_df = detail_df.sort_values("call_datetime", ascending=False)
            detail_df["call_datetime"] = detail_df["call_datetime"].dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
        )
