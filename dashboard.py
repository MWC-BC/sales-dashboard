# -------------------------------------------------------------------
# dashboard.py  – Sales Call Coaching Dashboard
# Version: V6 – Cloud-ready with Dropbox auto-loading + last-updated stamp
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

st.markdown(
    """
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
    """,
    unsafe_allow_html=True,
)

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
    """Normalize phone numbers into canonical 1XXXXXXXXXX format."""
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


def load_mapping(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    mapping_numbers, mapping_identities = {}, {}

    if not path.exists():
        return mapping_numbers, mapping_identities

    df_map = pd.read_csv(path)
    has_identity = "identity" in df_map.columns

    for _, r in df_map.iterrows():
        rep = str(r["sales_rep"]).strip()

        # phone mapping
        canon = canonical_from_mapping(r["phone_number"])
        if canon:
            mapping_numbers[canon] = rep

        # softphone identity mapping
        if has_identity:
            ident = r.get("identity")
            if isinstance(ident, str) and ident.strip():
                mapping_identities[ident.strip()] = rep

    return mapping_numbers, mapping_identities


def map_endpoint_to_rep(endpoint: Any, mapping_numbers, mapping_identities):
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

    # fallback: try the other endpoint
    rep = map_endpoint_to_rep(raw_to, mapping_numbers, mapping_identities)
    return rep or "Unassigned"


# -------------------------------------------------------------------
# PROCESS RAW DATA
# -------------------------------------------------------------------


def process_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Intent unification
    df["intent"] = df.get("llm_cpd_intent", df.get("rule_intent"))
    df["intent"] = df["intent"].fillna("unknown")

    # Provider
    df["provider"] = df.get("provider", "Unknown").astype(str).str.title()

    # call datetime
    df["call_datetime"] = pd.to_datetime(df.get("call_datetime"), errors="coerce")
    df = df[df["call_datetime"].notna()].copy()

    # Call type
    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"

    if "status" in df.columns:
        df.loc[df["status"].astype(str).str.lower() == "no-answer", "call_type"] = "No Answer"

    # Too-short
    dur = pd.to_numeric(df.get("duration_seconds", df.get("duration")), errors="coerce")
    df.loc[dur < 1, "call_type"] = "Too Short"

    # Coaching fields
    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]

    for col in score_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # Sales rep mapping
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
    """
    Order:
    1. Local CSV (your laptop)
    2. Dropbox CSV via secrets
    3. Upload fallback
    """

    # 1. LOCAL FILE
    if LOCAL_DATA_CSV.exists():
        try:
            df_raw = pd.read_csv(LOCAL_DATA_CSV)
            return process_raw_dataframe(df_raw)
        except Exception as e:
            st.error(f"Local CSV found but could not load: {e}")

    # 2. DROPBOX (STREAMLIT CLOUD)
    dropbox_url = None
    try:
        if "data" in st.secrets and "csv_url" in st.secrets["data"]:
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
            else:
                st.warning("Dropbox file loaded, but produced an empty dataset.")
        except Exception as e:
            st.error(f"Could not load CSV from Dropbox: {e}")

    # 3. UPLOAD FALLBACK
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
    st.warning("No calls found yet. Upload a CSV or check the Dropbox link.")
    st.stop()

# Global date range & last-updated info (based on call_datetime)
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
    st.warning("No matching calls found.")
    st.stop()

# -------------------------------------------------------------------
# PAGE STRUCTURE
# -------------------------------------------------------------------

st.title("Sales Call Coaching Dashboard")

# Build a nice caption with date range + last updated
if pd.notna(global_min_dt) and pd.notna(global_max_dt):
    if global_min_dt.date() == global_max_dt.date():
        date_span_text = f"Data for {global_min_dt.date():%Y-%m-%d}"
    else:
        date_span_text = (
            f"Data from {global_min_dt.date():%Y-%m-%d} "
            f"to {global_max_dt.date():%Y-%m-%d}"
        )
else:
    date_span_text = "Data date range unavailable"

if pd.notna(last_updated):
    last_updated_text = last_updated.strftime("%Y-%m-%d %H:%M")
    st.caption(
        f"Use the sidebar to filter calls. {date_span_text}. "
        f"Last updated: {last_updated_text} (based on latest call time)."
    )
else:
    st.caption(f"Use the sidebar to filter calls. {date_span_text}.")

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

    st.markdown("### Calls by Intent")
    intent_cols = [
        "call_datetime",
        "sales_rep",
        "provider",
        "intent",
        "llm_cpd_intent",
        "rule_intent",
        "call_type",
    ]
    intent_cols = [c for c in intent_cols if c in df.columns]

    st.dataframe(
        df[intent_cols].sort_values("call_datetime", ascending=False),
        height=350,
        use_container_width=True,
    )

    st.markdown("### Call Detail (Intent)")
    df_sel = df.copy()
    df_sel["_label"] = (
        df_sel["call_datetime"].astype(str)
        + " | " + df_sel["sales_rep"]
        + " | " + df_sel["intent"]
    )
    chosen = st.selectbox("Select a call", df_sel["_label"])
    row = df_sel[df_sel["_label"] == chosen].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Intent Info")
        st.write(f"**Unified Intent:** {row['intent']}")
        st.write(f"**LLM Intent:** {row.get('llm_cpd_intent', '')}")
        st.write(f"**Rule Intent:** {row.get('rule_intent', '')}")

        st.markdown("#### Metadata")
        st.write(f"**Rep:** {row['sales_rep']}")
        st.write(f"**Provider:** {row['provider']}")
        st.write(f"**Direction:** {row.get('direction', '')}")
        st.write(f"**Call Type:** {row['call_type']}")

    with col2:
        st.markdown("#### Transcript")
        st.write(row.get("transcript", ""))


# -------------------------------------------------------------------
# TAB 2 – COACHING
# -------------------------------------------------------------------

with tab_coaching:
    st.subheader("Average Sales Call Score by Rep")

    score_cols = {
        "Opening": "coaching_opening_score",
        "Discovery": "coaching_discovery_score",
        "Value": "coaching_value_score",
        "Closing": "coaching_closing_score",
    }

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

    rep_bar = (
        alt.Chart(rep_avg)
        .mark_bar()
        .encode(
            x="sales_rep:N",
            y="AvgScore:Q",
            color="sales_rep:N",
            tooltip=["sales_rep", "AvgScore"],
        )
    )

    rep_labels = (
        alt.Chart(rep_avg)
        .mark_text(dy=-10, fontSize=18, fontWeight="bold")
        .encode(
            x="sales_rep:N",
            y="AvgScore:Q",
            text=alt.Text("AvgScore:Q", format=".1f"),
        )
    )

    st.altair_chart(rep_bar + rep_labels, use_container_width=True)

    st.markdown("### Sales Call Scores by Pillar")

    records = []
    for _, r in df_coached.iterrows():
        rep = r["sales_rep"]
        for pillar, col in score_cols.items():
            val = r.get(col)
            if pd.notna(val):
                records.append({"sales_rep": rep, "Pillar": pillar, "Score": float(val)})

    pillar_df = pd.DataFrame(records)

    pillar_order = ["Opening", "Discovery", "Value", "Closing"]
    pillar_df["Pillar"] = pd.Categorical(
        pillar_df["Pillar"], categories=pillar_order, ordered=True
    )

    summary = (
        pillar_df.groupby(["sales_rep", "Pillar"], observed=False)["Score"]
        .mean()
        .reset_index()
    )

    bars = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x="sales_rep:N",
            xOffset="Pillar:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
            color="Pillar:N",
            tooltip=["sales_rep", "Pillar", "Score"],
        )
    )

    labels = (
        alt.Chart(summary)
        .mark_text(dy=-12, fontWeight="bold", fontSize=18)
        .encode(
            x="sales_rep:N",
            xOffset="Pillar:N",
            y="Score:Q",
            text=alt.Text("Score:Q", format=".1f"),
        )
    )

    st.altair_chart(bars + labels, use_container_width=True)

    st.markdown("### Coached Calls")

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

    st.markdown("### Coaching Detail")

    df_det = df_coached.copy()
    df_det["_label"] = (
        df_det["call_datetime"].astype(str)
        + " | " + df_det["sales_rep"]
        + " | " + df_det["intent"]
    )

    chosen2 = st.selectbox("Select a coached call", df_det["_label"])
    row_c = df_det[df_det["_label"] == chosen2].iloc[0]

    left, right = st.columns(2)

    with left:
        st.markdown("#### Coaching Summary")
        st.write(row_c.get("coaching_summary", ""))

        st.markdown("#### Improvement Points")
        imp = row_c.get("coaching_improvement_points", "")
        if isinstance(imp, str) and imp.strip():
            for line in imp.split("\n"):
                line = line.strip()
                if line:
                    st.markdown(f"- {line}")
        else:
            st.write("No improvement points provided.")

        st.markdown("#### Scores")
        st.write(f"**Total:** {row_c.get('coaching_total_score', '')}")
        st.write(f"**Opening:** {row_c.get('coaching_opening_score', '')}")
        st.write(f"**Discovery:** {row_c.get('coaching_discovery_score', '')}")
        st.write(f"**Value:** {row_c.get('coaching_value_score', '')}")
        st.write(f"**Closing:** {row_c.get('coaching_closing_score', '')}")

    with right:
        st.markdown("#### Transcript")
        st.write(row_c.get("transcript", ""))
