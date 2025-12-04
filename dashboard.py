import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt

# =================================================================
# PASSWORD GATE
# =================================================================

def check_password():
    """Simple password gate using Streamlit Cloud secrets."""
    def password_entered():
        if st.session_state["password"] == st.secrets["access"]["code"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter access code:", type="password",
                       on_change=password_entered, key="password")
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Enter access code:", type="password",
                       on_change=password_entered, key="password")
        st.error("Incorrect password.")
        return False

    return True


if not check_password():
    st.stop()


# =================================================================
# PATHS
# =================================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"
MAP_PATH = BASE_DIR / "Config" / "sales_rep_phone_map.csv"


# =================================================================
# PHONE NORMALIZATION
# =================================================================

def canonical(num):
    """Convert any phone-like value to canonical 11-digit '1XXXXXXXXXX'."""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return None
    s = str(num)
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) == 10:
        return "1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return digits
    return None


# =================================================================
# LOAD MAPPING
# =================================================================

def load_rep_mapping():
    """Load mapping file and return dict phone -> rep."""
    df_map = pd.read_csv(MAP_PATH)
    rep_map = {}
    for _, r in df_map.iterrows():
        phone = canonical(r["phone_number"])
        if phone:
            rep_map[phone] = str(r["sales_rep"]).strip()
    return rep_map


# =================================================================
# REP INFERENCE (Option A — ALWAYS override using mapping)
# =================================================================

def infer_rep(df, rep_map):
    reps = []
    for _, r in df.iterrows():
        # Try FROM number
        rep = rep_map.get(canonical(r.get("from_number")))
        if not rep:
            # Try TO number
            rep = rep_map.get(canonical(r.get("to_number")))
        reps.append(rep if rep else "Unassigned")
    return reps


# =================================================================
# LOAD DATA
# =================================================================

def load_data():
    df = pd.read_csv(CSV_PATH)

    # Force UTC timestamps into datetime
    if "call_datetime" in df.columns:
        df["call_datetime"] = pd.to_datetime(df["call_datetime"], errors="coerce")

    # Load mapping and infer reps
    rep_map = load_rep_mapping()
    df["sales_rep"] = infer_rep(df, rep_map)

    return df


# =================================================================
# DASHBOARD UI
# =================================================================

st.title("Sales Call Coaching Dashboard")
st.write("Use the sidebar to filter by Sales Rep, date range, carrier, and call type.")

df = load_data()

# ---------------------------
# SIDEBAR FILTERS
# ---------------------------

st.sidebar.header("Filters")

# Date range
min_date = df["call_datetime"].min()
max_date = df["call_datetime"].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Sales Rep filter
all_reps = ["All Reps"] + sorted(df["sales_rep"].unique())
rep_filter = st.sidebar.selectbox("Sales Rep", all_reps)

# Carrier filter
all_carriers = ["All", "Twilio", "Telzio"]
carrier_filter = st.sidebar.selectbox("Carrier", all_carriers)

# Call type toggles
include_vm = st.sidebar.checkbox("Include Voicemail", True)
include_noanswer = st.sidebar.checkbox("Include No Answer", True)
include_short = st.sidebar.checkbox("Include Too Short (<1s)", True)

# ---------------------------
# APPLY FILTERS
# ---------------------------

df_f = df.copy()

# Date range
df_f = df_f[(df_f["call_datetime"].dt.date >= date_range[0]) &
            (df_f["call_datetime"].dt.date <= date_range[1])]

# Rep filter
if rep_filter != "All Reps":
    df_f = df_f[df_f["sales_rep"] == rep_filter]

# Carrier filter
if carrier_filter != "All":
    df_f = df_f[df_f["carrier"] == carrier_filter]

# Call type filtering
if not include_vm:
    df_f = df_f[df_f["intent"] != "voicemail"]

if not include_noanswer:
    df_f = df_f[df_f["intent"] != "no_answer"]

if not include_short:
    df_f = df_f[df_f["call_duration"] > 1]


# =================================================================
# BUYING INTENT TAB
# =================================================================

tab_intent, tab_coach = st.tabs(["Buying Intent", "Coaching"])

with tab_intent:
    st.header("Intent Overview")

    total_calls = len(df_f)
    total_intents = df_f["intent"].nunique()

    col1, col2 = st.columns(2)
    col1.metric("Total Calls in Filter", total_calls)
    col2.metric("Distinct CPD Intents", total_intents)

    # Bar chart of intents
    intent_counts = df_f["intent"].value_counts().reset_index()
    intent_counts.columns = ["intent", "count"]

    chart = (
        alt.Chart(intent_counts)
        .mark_bar()
        .encode(
            x=alt.X("intent:N", sort="-y"),
            y="count:Q",
            tooltip=["intent", "count"]
        )
    )

    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=14
    ).encode(
        text="count:Q"
    )

    st.altair_chart(chart + text, use_container_width=True)


# =================================================================
# COACHING TAB
# =================================================================

with tab_coach:
    st.header("Coaching Scores by Pillar")

    if "coaching_opening" in df.columns:
        melted = df_f.melt(
            id_vars=["sales_rep"],
            value_vars=["coaching_opening", "coaching_discovery",
                        "coaching_value", "coaching_closing"],
            var_name="pillar",
            value_name="score"
        )

        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("pillar:N", title="Pillar"),
                y=alt.Y("mean(score):Q", title="Avg Score (0–5)"),
                color=alt.Color("sales_rep:N", legend=alt.Legend(title="Rep")),
                column=alt.Column("sales_rep:N", title="Rep")
            )
        )

        st.altair_chart(chart, use_container_width=True)

    st.subheader("Call Browser")

    call_ids = df_f["unique_call_id"].astype(str).tolist()
    selected = st.selectbox("Select a call", call_ids)

    row = df_f[df_f["unique_call_id"].astype(str) == selected].iloc[0]

    st.markdown("### Coaching Summary")
    st.write(row.get("coaching_summary", "No summary."))

    st.markdown("### Transcript")
    st.write(row.get("transcript", "No transcript available."))

