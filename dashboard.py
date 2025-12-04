# -----------------------------------------------------------
# dashboard.py  (clean Streamlit Cloud–safe version)
print("STREAMLIT BOOTED")
# -----------------------------------------------------------

from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -----------------------------------------------------------
# PASSWORD GATE
# -----------------------------------------------------------
def check_password():
    """Simple password gate using Streamlit secrets."""

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
        st.error("Incorrect access code.")
        return False

    return True


# Stop app before rendering anything
if not check_password():
    st.stop()


# -----------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -----------------------------------------------------------

st.set_page_config(page_title="Sales Call Coaching Dashboard",
                   layout="wide")

alt.data_transformers.disable_max_rows()

# Some basic CSS for bigger fonts
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-size: 22px !important;
        }
        h1 { font-size: 48px !important; }
        h2, h3 { font-size: 36px !important; }
        .stMetric label { font-size: 26px !important; }
        .stMetric span { font-size: 34px !important; }
        .stTabs [data-baseweb="tab"] { font-size: 24px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# FILE PATHS
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
COACHED_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"
MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"


# -----------------------------------------------------------
# PHONE + REP MAPPING HELPERS
# -----------------------------------------------------------

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
        if len(digits) == 11:
            return digits
    return None


def load_mapping(path: Path):
    numbers: Dict[str, str] = {}
    identities: Dict[str, str] = {}
    if not path.exists():
        return numbers, identities
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rep = str(r["sales_rep"]).strip()
        phone = canonical_phone(r["phone_number"])
        if phone:
            numbers[phone] = rep
        ident = r.get("identity")
        if isinstance(ident, str) and ident.strip():
            identities[ident.strip()] = rep
    return numbers, identities


def map_endpoint_to_rep(endpoint, numbers, identities):
    if endpoint is None or (isinstance(endpoint, float) and np.isnan(endpoint)):
        return None
    if isinstance(endpoint, str) and endpoint.startswith("client:"):
        return identities.get(endpoint.strip())
    phone = canonical_phone(endpoint)
    if phone:
        return numbers.get(phone)
    return None


def infer_rep(row, numbers, identities):
    direction = str(row.get("direction", "") or "").lower()
    raw_from = row.get("from")
    raw_to = row.get("to")

    if direction == "outbound":
        primary = raw_from
        secondary = raw_to
    elif direction == "inbound":
        primary = raw_to
        secondary = raw_from
    else:
        primary = raw_from
        secondary = raw_to

    rep = map_endpoint_to_rep(primary, numbers, identities)
    if rep:
        return rep
    rep = map_endpoint_to_rep(secondary, numbers, identities)
    if rep:
        return rep
    return "Unassigned"


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(COACHED_CSV)

    numbers, identities = load_mapping(MAPPING_CSV)
    df["sales_rep"] = df.apply(lambda r: infer_rep(r, numbers, identities), axis=1)

    if "call_datetime" in df.columns:
        df["call_datetime"] = pd.to_datetime(df["call_datetime"], utc=True)

    # make sure coaching columns are numeric
    for c in [
        "coaching_total_score",
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


df_all = load_data()

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    reps = ["All Reps"] + sorted(df_all["sales_rep"].unique())
    rep_filter = st.selectbox("Sales Rep", reps)

    min_d = df_all["call_datetime"].min().date()
    max_d = df_all["call_datetime"].max().date()
    start_date, end_date = st.date_input("Call Date Range", (min_d, max_d))

# Filter by date + rep
df = df_all[
    (df_all["call_datetime"].dt.date >= start_date)
    & (df_all["call_datetime"].dt.date <= end_date)
]

if rep_filter != "All Reps":
    df = df[df["sales_rep"] == rep_filter]

# -----------------------------------------------------------
# TABS
# -----------------------------------------------------------

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])


# -----------------------------------------------------------
# TAB: COACHING
# -----------------------------------------------------------

with tab_coaching:
    st.title("Sales Call Coaching Dashboard")

    pillar_cols = {
        "Opening": "coaching_opening_score",
        "Discovery": "coaching_discovery_score",
        "Value": "coaching_value_score",
        "Closing": "coaching_closing_score",
    }

    # Average coaching score by rep (all pillars)
    st.subheader("Average Coaching Score by Rep")

    numeric_cols = [c for c in pillar_cols.values() if c in df.columns]
    if numeric_cols:
        df_tmp = df.copy()
        df_tmp["pillar_avg"] = df_tmp[numeric_cols].mean(axis=1)
        rep_avg = df_tmp.groupby("sales_rep")["pillar_avg"].mean().reset_index()

        chart_rep = (
            alt.Chart(rep_avg)
            .mark_bar()
            .encode(
                x=alt.X("sales_rep:N", title="Sales Rep"),
                y=alt.Y("pillar_avg:Q", title="Average Coaching Score",
                        scale=alt.Scale(domain=[0, 5])),
                tooltip=["sales_rep", alt.Tooltip("pillar_avg:Q", format=".1f")]
            )
            .properties(height=300)
        )
        st.altair_chart(chart_rep, use_container_width=True)
    else:
        st.info("No coaching score columns found to compute averages.")

    # Coaching scores by pillar (compare reps)
    st.subheader("Coaching Scores by Pillar (compare reps per pillar)")

    records = []
    for rep, grp in df.groupby("sales_rep"):
        for pillar, col in pillar_cols.items():
            if col in grp:
                scores = grp[col].dropna()
                if len(scores) > 0:
                    records.append(
                        {
                            "sales_rep": rep,
                            "Pillar": pillar,
                            "Average Score": scores.mean(),
                        }
                    )
    prdf = pd.DataFrame(records)

    if not prdf.empty:
        chart_pillar = (
            alt.Chart(prdf)
            .mark_bar()
            .encode(
                x=alt.X("Pillar:N", title="Pillar"),
                y=alt.Y("Average Score:Q",
                        scale=alt.Scale(domain=[0, 5])),
                color=alt.Color("sales_rep:N", title="Sales Rep"),
                column=alt.Column("sales_rep:N", title="Sales Rep"),
                tooltip=[
                    "sales_rep",
                    "Pillar",
                    alt.Tooltip("Average Score:Q", format=".1f")
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_pillar, use_container_width=True)
    else:
        st.info("No coaching data available to plot pillar scores.")

    # Calls table + detail
    st.subheader("Calls in Current Filter")

    cols_calls = [
        "call_datetime",
        "sales_rep",
        "direction",
        "llm_cpd_intent",
        "coaching_total_score",
    ]
    cols_calls = [c for c in cols_calls if c in df.columns]

    df_display = df.sort_values("call_datetime", ascending=False)
    st.dataframe(df_display[cols_calls], use_container_width=True, height=350)

    st.subheader("Call Detail")

    if not df_display.empty:
        df_display = df_display.copy()
        df_display["_label"] = (
            df_display["call_datetime"].astype(str)
            + " | "
            + df_display["sales_rep"]
        )

        selected = st.selectbox("Select a call", df_display["_label"].tolist())
        row = df_display[df_display["_label"] == selected].iloc[0]

        left, right = st.columns(2)

        with left:
            st.markdown("### Coaching Summary")
            st.write(row.get("coaching_summary", ""))

        with right:
            st.markdown("### Transcript")
            st.write(row.get("transcript", ""))
