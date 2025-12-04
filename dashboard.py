# -------------------------------------------------------------------
# dashboard.py  (working version with password gate + tz fix)
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
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)

    # ----- intent column (unified) -----
    if "intent" not in df.columns:
        df["intent"] = df.get("llm_cpd_intent")
        if "rule_intent" in df.columns:
            df["intent"] = df["intent"].fillna(df["rule_intent"])
        df["intent"] = df["intent"].fillna("Unknown")

    # ----- sales_rep -----
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"

    # ----- provider (carrier) -----
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # ----- call_datetime (parse + strip timezone) -----
    df["call_datetime"] = pd.to_datetime(df["call_datetime"], errors="coerce")

    # If this is timezone-aware, drop the tz so comparisons work
    try:
        dtype = df["call_datetime"].dtype
        tz = getattr(dtype, "tz", None)
        if tz is not None:
            df["call_datetime"] = df["call_datetime"].dt.tz_localize(None)
    except (AttributeError, TypeError):
        # Already tz-naive; nothing to do
        pass

    df = df[df["call_datetime"].notna()]

    # ----- call_type -----
    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    if "duration_seconds" in df.columns:
        df.loc[df["duration_seconds"] < 1, "call_type"] = "Too Short"
    if "status" in df.columns:
        df.loc[df["status"] == "no-answer", "call_type"] = "No Answer"

    # ----- coaching score columns -----
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

# Date range (use just the date objects)
min_date = df["call_datetime"].min().date()
max_date = df["call_datetime"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
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

# Date range filter (convert to pandas Timestamps – both tz-naive)
start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1])

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
exclude_types: list[str] = []
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
# TAB 1: BUYING INTENT
# -------------------------------------------------------------------
with tabs[0]:
    st.subheader("Intent Overview")

    total_calls = len(filtered)
    distinct_intents = filtered["intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    # Intent distribution chart
    intent_counts = (
        filtered["intent"]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
        .rename(columns={"index": "intent", "intent": "count"})
    )

    if len(intent_counts) > 0:
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

        labels = chart.mark_text(
            align="center",
            baseline="bottom",
            dy=-5,
            fontSize=14,
        ).encode(text="count:Q")

        st.altair_chart(chart + labels, use_container_width=True)

    # Simple call browser based on intent
    st.subheader("Call Browser (Intent)")
    filtered_sorted = filtered.sort_values("call_datetime", ascending=False)

    if not filtered_sorted.empty:
        options = (
            filtered_sorted["call_datetime"].astype(str)
            + " | "
            + filtered_sorted["sales_rep"]
            + " | "
            + filtered_sorted["intent"]
        )
        selected = st.selectbox("Select a call:", options)
        row = filtered_sorted.iloc[options.tolist().index(selected)]

        left, right = st.columns(2)

        with left:
            st.markdown("### Intent Details")
            st.write(f"**Intent:** {row['intent']}")
            st.write(f"**LLM Reason:** {row.get('llm_cpd_reason', '')}")
            st.write(f"**Rule Reason:** {row.get('rule_intent_reason', '')}")

        with right:
            st.markdown("### Transcript")
            st.write(row.get("transcript", ""))

# -------------------------------------------------------------------
# TAB 2: COACHING
# -------------------------------------------------------------------
with tabs[1]:
    st.subheader("Coaching Overview")

    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]

    scored = filtered.dropna(subset=["coaching_total_score"], how="all")

    if scored.empty:
        st.warning("No coaching data in current filters.")
    else:
        # Per-rep summary table
        st.subheader("Per-Rep Coaching Summary")
        rep_summary = scored.groupby("sales_rep")[score_cols].mean()
        rep_summary["Calls Coached"] = scored.groupby("sales_rep").size()
        st.dataframe(rep_summary.style.format("{:.2f}"), use_container_width=True)

        # Side-by-side pillar chart by rep (no total)
        st.subheader("Coaching Scores by Pillar (Side-by-Side by Rep)")
        melted = scored.melt(
            id_vars=["sales_rep"],
            value_vars=score_cols[:-1],  # exclude total
            var_name="Pillar",
            value_name="Score",
        )

        if not melted.empty:
            pillar_chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    x=alt.X("Pillar:N", title="Pillar"),
                    y=alt.Y("Score:Q", title="Average Score"),
                    color=alt.Color("sales_rep:N", title="Sales Rep"),
                    column=alt.Column("sales_rep:N", title=None),
                    tooltip=["sales_rep", "Pillar", "Score"],
                )
            )
            st.altair_chart(pillar_chart, use_container_width=True)

    st.subheader("Improvement Points")
    if "coaching_improvement_points" in filtered.columns:
        imp_list = (
            filtered["coaching_improvement_points"]
            .dropna()
            .astype(str)
            .tolist()
        )
        for item in imp_list:
            st.write(f"- {item}")
