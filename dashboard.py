# -------------------------------------------------------------------
# dashboard.py  – Sales Call Coaching Dashboard
# Version: V3 – Streamlit Cloud Compatible (relative paths)
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
# PATHS (LOCAL & CLOUD-FRIENDLY)
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Local CSV location when you run on your machine
DATA_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"

# Sales rep mapping CSV (safe to keep in repo)
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
        if len(digits) == 11 and digits.startswith("1"):
            return digits
    if len(digits) < 10:
        return None

    return digits


def canonical_from_mapping(num: Optional[Any]) -> Optional[str]:
    """Normalize phone numbers from mapping CSV."""
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
        if len(digits) == 11 and digits.startswith("1"):
            return digits
    if len(digits) < 10:
        return None

    return digits


def load_mapping(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    mapping_numbers: Dict[str, str] = {}
    mapping_identities: Dict[str, str] = {}

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
    direction_business = str(row.get("direction_business", "") or "").lower()
    direction = str(row.get("direction", "") or "").lower()

    raw_from = row.get("from")
    raw_to = row.get("to")

    if "twilio dialer" in direction_business:
        primary = raw_from
        secondary = raw_to
    elif "outbound" in direction:
        primary = raw_from
        secondary = raw_to
    elif "inbound" in direction:
        primary = raw_to
        secondary = raw_from
    else:
        primary = raw_from
        secondary = raw_to

    rep = map_endpoint_to_rep(primary, mapping_numbers, mapping_identities)
    if rep:
        return rep

    rep = map_endpoint_to_rep(secondary, mapping_numbers, mapping_identities)
    if rep:
        return rep

    return "Unassigned"


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # Try to load local CSV; if missing (e.g., in Streamlit Cloud), return empty df
    try:
        df = pd.read_csv(DATA_CSV)
    except FileNotFoundError:
        return pd.DataFrame()

    # Intent unification
    if "llm_cpd_intent" in df.columns:
        df["intent"] = df["llm_cpd_intent"].fillna(df.get("rule_intent"))
    else:
        df["intent"] = df.get("rule_intent")
    df["intent"] = df["intent"].fillna("unknown")

    # Provider
    df["provider"] = df.get("provider", "Unknown").astype(str).str.title()

    # Datetime
    df["call_datetime"] = pd.to_datetime(df.get("call_datetime"), errors="coerce")
    df = df[df["call_datetime"].notna()].copy()

    # Call type
    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"
    if "status" in df.columns:
        df.loc[
            df["status"].astype(str).str.lower() == "no-answer", "call_type"
        ] = "No Answer"

    # Too short
    if "duration_seconds" in df.columns:
        dur = pd.to_numeric(df["duration_seconds"], errors="coerce")
        df.loc[dur < 1, "call_type"] = "Too Short"
    elif "duration" in df.columns:
        dur = pd.to_numeric(df["duration"], errors="coerce")
        df.loc[dur < 1, "call_type"] = "Too Short"

    # Coaching fields
    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]
    for c in score_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # Sales rep mapping
    mapping_numbers, mapping_identities = load_mapping(MAPPING_CSV)
    df["sales_rep"] = df.apply(
        lambda r: infer_rep_from_row(r, mapping_numbers, mapping_identities),
        axis=1,
    )

    return df


df_all = load_data()

if df_all.empty:
    st.title("Sales Call Coaching Dashboard")
    st.warning(
        "No calls found in the data file. "
        "If you are running in Streamlit Cloud, the local CSV is not available yet."
    )
    st.stop()

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

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range

    selected_carrier = st.selectbox(
        "Carrier",
        ["All"] + sorted(df_all["provider"].unique()),
    )
    selected_rep = st.selectbox(
        "Sales Rep",
        ["All Reps"] + sorted(df_all["sales_rep"].unique()),
    )

    include_voicemail = st.checkbox("Include Voicemail")
    include_no_answer = st.checkbox("Include No Answer")
    include_too_short = st.checkbox("Include Too Short")

# -------------------------------------------------------------------
# APPLY FILTERS
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
    st.warning("No calls match the current filters.")
    st.stop()

# -------------------------------------------------------------------
# PAGE STRUCTURE
# -------------------------------------------------------------------

st.title("Sales Call Coaching Dashboard")
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
        .fillna("unknown")
        .value_counts()
        .rename_axis("Intent")
        .reset_index(name="Count")
    )

    if not intent_counts.empty:
        bar = (
            alt.Chart(intent_counts)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Intent:N",
                    sort="-y",
                    title="Intent",
                    axis=alt.Axis(labelAngle=0, labelLimit=200),
                ),
                y=alt.Y("Count:Q", title="Number of Calls"),
                color=alt.Color("Intent:N", legend=None),
            )
        )

        labels = (
            alt.Chart(intent_counts)
            .mark_text(
                align="center",
                baseline="bottom",
                dy=-10,
                fontSize=18,
                fontWeight="bold",
                color="black",
            )
            .encode(
                x=alt.X(
                    "Intent:N",
                    sort="-y",
                    axis=alt.Axis(labelAngle=0, labelLimit=200),
                ),
                y="Count:Q",
                text="Count:Q",
            )
        )

        st.altair_chart(alt.layer(bar, labels), use_container_width=True)

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

    st.markdown("### Calls by Intent")
    st.dataframe(
        df[intent_cols].sort_values("call_datetime", ascending=False),
        use_container_width=True,
        height=350,
    )

    st.markdown("### Call Detail (Intent)")
    df_sel = df.copy()
    df_sel["_label"] = (
        df_sel["call_datetime"].astype(str)
        + " | "
        + df_sel["sales_rep"].astype(str)
        + " | "
        + df_sel["intent"].astype(str)
    )
    chosen = st.selectbox("Select a call", df_sel["_label"].tolist())
    row = df_sel[df_sel["_label"] == chosen].iloc[0]

    left, right = st.columns(2)

    with left:
        st.markdown("#### Intent Info")
        st.write(f"**Unified Intent:** {row['intent']}")
        st.write(f"**LLM Intent:** {row.get('llm_cpd_intent', '')}")
        st.write(f"**Rule Intent:** {row.get('rule_intent', '')}")
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

    score_cols = {
        "Opening": "coaching_opening_score",
        "Discovery": "coaching_discovery_score",
        "Value": "coaching_value_score",
        "Closing": "coaching_closing_score",
    }

    df_coached = df[df["coaching_total_score"].notna()].copy()

    if df_coached.empty:
        st.info("No coached calls available.")
    else:
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
                y=alt.Y("AvgScore:Q", title="Avg Total Score (0–20)"),
                color=alt.Color("sales_rep:N", legend=None),
                tooltip=["sales_rep", "AvgScore"],
            )
        )

        rep_labels = (
            alt.Chart(rep_avg)
            .mark_text(
                align="center",
                baseline="bottom",
                dy=-10,
                fontSize=18,
                fontWeight="bold",
                color="black",
            )
            .encode(
                x="sales_rep:N",
                y="AvgScore:Q",
                text=alt.Text("AvgScore:Q", format=".1f"),
            )
        )

        st.altair_chart(rep_bar + rep_labels, use_container_width=True)

        st.markdown("### Sales Call Scores by Pillar (Side-by-Side by Rep)")

        records = []
        for _, r in df_coached.iterrows():
            rep = r["sales_rep"]
            for pillar, col in score_cols.items():
                val = r.get(col)
                if pd.notna(val):
                    records.append(
                        {"sales_rep": rep, "Pillar": pillar, "Score": float(val)}
                    )

        pillar_df = pd.DataFrame(records)

        if pillar_df.empty:
            st.info("No pillar-level scores available.")
        else:
            pillar_order = ["Opening", "Discovery", "Value", "Closing"]
            pillar_df["Pillar"] = pd.Categorical(
                pillar_df["Pillar"],
                categories=pillar_order,
                ordered=True,
            )

            pillar_summary = (
                pillar_df.groupby(["sales_rep", "Pillar"], observed=False)["Score"]
                .mean()
                .reset_index()
            )

            base = alt.Chart(pillar_summary)

            bars = base.mark_bar().encode(
                x="sales_rep:N",
                xOffset="Pillar:N",
                y=alt.Y(
                    "Score:Q",
                    title="Average Score (0–5)",
                    scale=alt.Scale(domain=[0, 5]),
                ),
                color=alt.Color("Pillar:N", title="Pillar", sort=pillar_order),
                tooltip=["sales_rep", "Pillar", "Score"],
            )

            labels = base.mark_text(
                dy=-12,
                fontSize=18,
                fontWeight="bold",
                color="black",
                align="center",
                baseline="bottom",
            ).encode(
                x="sales_rep:N",
                xOffset="Pillar:N",
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 5])),
                text=alt.Text("Score:Q", format=".1f"),
            )

            st.altair_chart(bars + labels, use_container_width=True)

        st.markdown("### Calls in Current Filter (Coaching View)")

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
            use_container_width=True,
            height=350,
        )

        st.markdown("### Call Detail (Coaching)")

        df_det = df_coached.copy()
        df_det["_label"] = (
            df_det["call_datetime"].astype(str)
            + " | "
            + df_det["sales_rep"].astype(str)
            + " | "
            + df_det["intent"].astype(str)
        )

        chosen2 = st.selectbox("Select a coached call", df_det["_label"].tolist())
        row_c = df_det[df_det["_label"] == chosen2].iloc[0]

        left_c, right_c = st.columns(2)

        with left_c:
            st.markdown("#### Coaching Summary")
            st.write(row_c.get("coaching_summary", ""))

            st.markdown("#### Top Improvement Points")
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

        with right_c:
            st.markdown("#### Transcript")
            st.write(row_c.get("transcript", ""))
