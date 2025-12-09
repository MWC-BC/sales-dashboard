# -------------------------------------------------------------------
# dashboard.py  – Sales Call Coaching Dashboard
# Version: V7 – Dropbox auto-loading + Last Updated + Hide GitHub Link
# -------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------

def check_password() -> bool:
    """Simple password gate using Streamlit secrets (access.code)."""

    def password_entered():
        correct = st.session_state.get("password") == st.secrets["access"]["code"]
        st.session_state["password_correct"] = bool(correct)
        if "password" in st.session_state:
            del st.session_state["password"]

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter access code:",
            type="password",
            key="password",
            on_change=password_entered,
        )
        return False

    if not st.session_state["password_correct"]:
        st.text_input(
            "Enter access code:",
            type="password",
            key="password",
            on_change=password_entered,
        )
        st.error("Incorrect access code.")
        return False

    return True

if not check_password():
    st.stop()

# -------------------------------------------------------------------
# STREAMLIT CONFIG & STYLING
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Call Coaching Dashboard",
    layout="wide",
)

alt.data_transformers.disable_max_rows()

# Global styling
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    h1 { font-size: 40px !important; }
    h2, h3 { font-size: 30px !important; }
    .stMetric label { font-size: 22px !important; }
    .stMetric span { font-size: 28px !important; }
    .stDataFrame table tbody tr td { font-size: 18px !important; }
    .stDataFrame table thead tr th { font-size: 19px !important; }
    .stSelectbox label, .stDateInput label {
        font-size: 20px !important;
    }
    .stTabs [data-baseweb="tab"] { font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)

# Hide GitHub link / View source button
st.markdown("""
<style>
    /* Hide the GitHub icon and "View source" link */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    header a[href*="github"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATA_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"
MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# -------------------------------------------------------------------
# PHONE / IDENTITY HELPERS
# -------------------------------------------------------------------

def canonical_phone(num: Optional[Any]) -> Optional[str]:
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return None
    if isinstance(num, (int, np.integer)):
        digits = str(int(num))
    elif isinstance(num, (float, np.floating)):
        digits = str(int(num))
    else:
        s = str(num)
        if s.startswith("client:"):
            return None
        digits = "".join(ch for ch in s if ch.isdigit())

    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("1"):
        return digits
    if len(digits) == 10:
        return "1" + digits
    if len(digits) > 11:
        digits = digits[-11:]
        if digits.startswith("1"):
            return digits
    if len(digits) < 10:
        return None

    return digits

def canonical_from_mapping(num: Optional[Any]) -> Optional[str]:
    if num is None:
        return None
    digits = "".join(ch for ch in str(num) if ch.isdigit())
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("1"):
        return digits
    if len(digits) == 10:
        return "1" + digits
    if len(digits) > 11:
        digits = digits[-11:]
        if digits.startswith("1"):
            return digits
    return None

def load_mapping(path: Path):
    mapping_numbers, mapping_identities = {}, {}
    if not path.exists():
        return mapping_numbers, mapping_identities

    df_map = pd.read_csv(path)
    has_identity = "identity" in df_map.columns

    for _, r in df_map.iterrows():
        rep = str(r["sales_rep"]).strip()

        canon = canonical_from_mapping(r["phone_number"])
        if canon:
            mapping_numbers[canon] = rep

        if has_identity:
            ident = r.get("identity")
            if isinstance(ident, str) and ident.strip():
                mapping_identities[ident.strip()] = rep

    return mapping_numbers, mapping_identities

def map_endpoint_to_rep(endpoint, mapping_numbers, mapping_identities):
    if endpoint is None or (isinstance(endpoint, float) and np.isnan(endpoint)):
        return None
    if isinstance(endpoint, str) and endpoint.startswith("client:"):
        return mapping_identities.get(endpoint.strip())
    canon = canonical_phone(endpoint)
    if canon:
        return mapping_numbers.get(canon)
    return None

def infer_rep_from_row(row, mapping_numbers, mapping_identities):
    direction = str(row.get("direction", "")).lower()
    direction_business = str(row.get("direction_business", "")).lower()

    raw_from = row.get("from")
    raw_to = row.get("to")

    if "twilio dialer" in direction_business:
        primary = raw_from
    elif "outbound" in direction:
        primary = raw_from
    elif "inbound" in direction:
        primary = raw_to
    else:
        primary = raw_from

    rep = map_endpoint_to_rep(primary, mapping_numbers, mapping_identities)
    if rep:
        return rep
    rep = map_endpoint_to_rep(raw_to, mapping_numbers, mapping_identities)
    return rep or "Unassigned"

# -------------------------------------------------------------------
# DATA PROCESSING
# -------------------------------------------------------------------

def process_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["intent"] = df.get("llm_cpd_intent", df.get("rule_intent")).fillna("unknown")
    df["provider"] = df.get("provider", "Unknown").astype(str).str.title()
    df["call_datetime"] = pd.to_datetime(df.get("call_datetime"), errors="coerce")
    df = df[df["call_datetime"].notna()].copy()

    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    if "status" in df.columns:
        df.loc[df["status"].astype(str).str.lower() == "no-answer", "call_type"] = "No Answer"

    dur = pd.to_numeric(df.get("duration_seconds", df.get("duration")), errors="coerce")
    df.loc[dur < 1, "call_type"] = "Too Short"

    for c in [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    mapping_numbers, mapping_identities = load_mapping(MAPPING_CSV)
    df["sales_rep"] = df.apply(
        lambda r: infer_rep_from_row(r, mapping_numbers, mapping_identities),
        axis=1,
    )

    return df

# -------------------------------------------------------------------
# LOAD DATA (LOCAL → DROPBOX → UPLOAD)
# -------------------------------------------------------------------

def load_data() -> pd.DataFrame:

    # 1: Local CSV
    if LOCAL_DATA_CSV.exists():
        try:
            df_raw = pd.read_csv(LOCAL_DATA_CSV)
            return process_raw_dataframe(df_raw)
        except Exception as e:
            st.error(f"Local CSV found but could not load: {e}")

    # 2: Dropbox CSV
    dropbox_url = None
    try:
        dropbox_url = st.secrets["data"]["csv_url"]
    except Exception:
        dropbox_url = None

    if dropbox_url:
        st.info("Loading latest CSV from Dropbox…")
        try:
            df_raw = pd.read_csv(dropbox_url)
            df_proc = process_raw_dataframe(df_raw)
            if not df_proc.empty:
                return df_proc
        except Exception as e:
            st.error(f"Could not load CSV from Dropbox: {e}")

    # 3: Upload fallback
    st.info("Please upload the enriched CSV to continue.")
    uploaded = st.file_uploader(
        "Upload enriched calls CSV",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            return process_raw_dataframe(df_raw)
        except Exception as e:
            st.error(f"Uploaded CSV could not be read: {e}")

    return pd.DataFrame()

# -------------------------------------------------------------------
# LOAD ALL DATA
# -------------------------------------------------------------------

df_all = load_data()

if df_all.empty:
    st.title("Sales Call Coaching Dashboard")
    st.warning("No calls found. Upload a CSV or check Dropbox.")
    st.stop()

global_min_dt = df_all["call_datetime"].min()
global_max_dt = df_all["call_datetime"].max()
last_updated = global_max_dt

# -------------------------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    min_dt = df_all["call_datetime"].min()
    max_dt = df_all["call_datetime"].max()

    date_range = st.date_input(
        "Call Date Range",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )
    start_date, end_date = date_range

    selected_carrier = st.selectbox(
        "Carrier", ["All"] + sorted(df_all["provider"].unique())
    )

    selected_rep = st.selectbox(
        "Sales Rep", ["All Reps"] + sorted(df_all["sales_rep"].unique())
    )

    include_voicemail = st.checkbox("Include Voicemail", value=True)
    include_no_answer = st.checkbox("Include No Answer", value=True)
    include_too_short = st.checkbox("Include Too Short", value=False)

# -------------------------------------------------------------------
# FILTER DATA
# -------------------------------------------------------------------

df = df_all.copy()

df = df[
    (df["call_datetime"].dt.date >= start_date)
    & (df["call_datetime"].dt.date <= end_date)
]

if selected_carrier != "All":
    df = df[df["provider"] == selected_carrier]

if selected_rep != "All Reps":
    df = df[df["sales_rep"] == selected_rep]

exclude_types = []
if not include_voicemail:
    exclude_types.append("Voicemail")
if not include_no_answer:
    exclude_types.append("No Answer")
if not include_too_short:
    exclude_types.append("Too Short")

df = df[~df["call_type"].isin(exclude_types)]

if df.empty:
    st.title("Sales Call Coaching Dashboard")
    st.warning("No matching calls.")
    st.stop()

# -------------------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------------------

st.title("Sales Call Coaching Dashboard")

# Last updated caption
if pd.notna(last_updated):
    st.caption(
        f"Use the sidebar to filter calls. Last updated: "
        f"{last_updated.strftime('%Y-%m-%d %H:%M')}."
    )
else:
    st.caption("Use the sidebar to filter calls.")

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])

# -------------------------------------------------------------------
# TAB 1 — BUYING INTENT
# -------------------------------------------------------------------

with tab_intent:
    st.subheader("Intent Overview")

    total_calls = len(df)
    distinct_intents = df["intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls in Filter", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    intent_counts = (
        df["intent"]
        .value_counts()
        .rename_axis("Intent")
        .reset_index(name="Count")
    )

    bar = (
        alt.Chart(intent_counts)
        .mark_bar()
        .encode(
            x=alt.X("Intent:N", sort="-y", axis=alt.Axis(labelAngle=0)),
            y="Count:Q",
            color="Intent:N",
        )
    )

    labels = (
        alt.Chart(intent_counts)
        .mark_text(dy=-10, fontSize=18, fontWeight="bold")
        .encode(x="Intent:N", y="Count:Q", text="Count:Q")
    )

    st.altair_chart(bar + labels, use_container_width=True)

    # Table
    cols = [
        "call_datetime", "sales_rep", "provider",
        "intent", "llm_cpd_intent", "rule_intent", "call_type"
    ]
    cols = [c for c in cols if c in df.columns]

    st.markdown("### Calls by Intent")
    st.dataframe(
        df[cols].sort_values("call_datetime", ascending=False),
        height=350,
        use_container_width=True,
    )

    # Detail view
    st.markdown("### Call Detail")
    df_sel = df.copy()
    df_sel["_label"] = (
        df_sel["call_datetime"].astype(str)
        + " | " + df_sel["sales_rep"]
        + " | " + df_sel["intent"]
    )
    chosen = st.selectbox("Select a call", df_sel["_label"])
    row = df_sel[df_sel["_label"] == chosen].iloc[0]

    left, right = st.columns(2)

    with left:
        st.markdown("#### Intent Info")
        st.write(f"**Unified Intent:** {row['intent']}")
        st.write(f"**LLM Intent:** {row.get('llm_cpd_intent', '')}")
        st.write(f"**Rule Intent:** {row.get('rule_intent', '')}")
        st.write("")

        st.markdown("#### Metadata")
        st.write(f"**Rep:** {row['sales_rep']}")
        st.write(f"**Provider:** {row['provider']}")
        st.write(f"**Direction:** {row.get('direction', '')}")
        st.write(f"**Call Type:** {row['call_type']}")

    with right:
        st.markdown("#### Transcript")
        st.write(row.get("transcript", ""))

# -------------------------------------------------------------------
# TAB 2 — COACHING
# -------------------------------------------------------------------

with tab_coaching:
    st.subheader("Average Sales Call Score by Rep")

    df_coached = df[df["coaching_total_score"].notna()]

    if df_coached.empty:
        st.info("No coached calls available.")
        st.stop()

    rep_avg = (
        df_coached.groupby("sales_rep")["coaching_total_score"]
        .mean()
        .reset_index()
        .rename(columns={"coaching_total_score": "AvgScore"})
    )

    bar = (
        alt.Chart(rep_avg)
        .mark_bar()
        .encode(
            x="sales_rep:N",
            y="AvgScore:Q",
            color="sales_rep:N",
        )
    )

    labels = (
        alt.Chart(rep_avg)
        .mark_text(dy=-10, fontSize=18, fontWeight="bold")
        .encode(x="sales_rep:N", y="AvgScore:Q",
               text=alt.Text("AvgScore:Q", format=".1f"))
    )

    st.altair_chart(bar + labels, use_container_width=True)

    # Pillars
    st.markdown("### Pillar Breakdown")

    records = []
    for _, r in df_coached.iterrows():
        for pillar, col in {
            "Opening": "coaching_opening_score",
            "Discovery": "coaching_discovery_score",
            "Value": "coaching_value_score",
            "Closing": "coaching_closing_score",
        }.items():
            if pd.notna(r.get(col)):
                records.append({
                    "sales_rep": r["sales_rep"],
                    "Pillar": pillar,
                    "Score": float(r[col])
                })

    pillar_df = pd.DataFrame(records)
    pillar_df["Pillar"] = pd.Categorical(
        pillar_df["Pillar"],
        categories=["Opening", "Discovery", "Value", "Closing"],
        ordered=True
    )

    summary = (
        pillar_df.groupby(["sales_rep", "Pillar"], observed=False)["Score"]
        .mean()
        .reset_index()
    )

    bar = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x="sales_rep:N",
            xOffset="Pillar:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
            color="Pillar:N",
        )
    )

    labels = (
        alt.Chart(summary)
        .mark_text(dy=-12, fontSize=18, fontWeight="bold")
        .encode(
            x="sales_rep:N",
            xOffset="Pillar:N",
            y="Score:Q",
            text=alt.Text("Score:Q", format=".1f"),
        )
    )

    st.altair_chart(bar + labels, use_container_width=True)

    # Table
    st.markdown("### Coaching View — Calls")

    call_cols = [
        "call_datetime",
        "sales_rep",
        "provider",
        "intent",
        "coaching_total_score",
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "call_type",
    ]

    call_cols = [c for c in call_cols if c in df_coached.columns]

    st.dataframe(
        df_coached[call_cols].sort_values("call_datetime", ascending=False),
        height=350,
        use_container_width=True,
    )

    # Coaching detail
    st.markdown("### Coaching Detail")

    df_det = df_coached.copy()
    df_det["_label"] = (
        df_det["call_datetime"].astype(str)
        + " | " + df_det["sales_rep"]
        + " | " + df_det["intent"]
    )

    chosen2 = st.selectbox("Select a coached call", df_det["_label"])
    row = df_det[df_det["_label"] == chosen2].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Coaching Summary")
        st.write(row.get("coaching_summary", ""))

        st.markdown("#### Improvement Points")
        imp = row.get("coaching_improvement_points", "")
        if isinstance(imp, str) and imp.strip():
            for line in imp.split("\n"):
                line = line.strip()
                if line:
                    st.markdown(f"- {line}")
        else:
            st.write("No improvement points provided.")

        st.markdown("#### Scores")
        st.write(f"**Total:** {row.get('coaching_total_score', '')}")
        st.write(f"**Opening:** {row.get('coaching_opening_score', '')}")
        st.write(f"**Discovery:** {row.get('coaching_discovery_score', '')}")
        st.write(f"**Value:** {row.get('coaching_value_score', '')}")
        st.write(f"**Closing:** {row.get('coaching_closing_score', '')}")

    with col2:
        st.markdown("#### Transcript")
        st.write(row.get("transcript", ""))
