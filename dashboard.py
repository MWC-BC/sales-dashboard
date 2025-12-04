# -----------------------------------------------------------------------------------
# dashboard.py — Full Production Dashboard (Streamlit Cloud–Safe)
# With Password Gate + Altair Charts + Bar Labels
# -----------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Any, Dict, Tuple
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -----------------------------------------------------------------------------------
# PASSWORD GATE
# -----------------------------------------------------------------------------------
def check_password():
    """Simple password gate using Streamlit Cloud secrets."""

    def password_entered():
        if "access" in st.secrets and "code" in st.secrets["access"]:
            if st.session_state["password"] == st.secrets["access"]["code"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Enter access code:",
            type="password",
            key="password",
            on_change=password_entered
        )
        return False

    if not st.session_state["password_correct"]:
        st.text_input(
            "Enter access code:",
            type="password",
            key="password",
            on_change=password_entered
        )
        st.error("Incorrect access code.")
        return False

    return True


if not check_password():
    st.stop()

# -----------------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Call Coaching Dashboard",
    layout="wide",
)

# Remove Altair row limit
alt.data_transformers.disable_max_rows()

# Make fonts larger
st.markdown("""
<style>
html, body, [class*="css"]  { font-size: 22px !important; }
h1 { font-size: 48px !important; }
h2, h3 { font-size: 36px !important; }
.stDataFrame table tbody tr td { font-size: 20px !important; }
.stDataFrame table thead tr th { font-size: 22px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
COACHED_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"
MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# -----------------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------------

def canonical_phone(num: Optional[Any]) -> Optional[str]:
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return None
    s = str(num)
    if s.startswith("client:"):
        return None
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 10:
        return "1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return digits
    return None


def load_mapping(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    numbers = {}
    identities = {}
    if not path.exists():
        return numbers, identities
    
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rep = str(r["sales_rep"]).strip()
        p = canonical_phone(r["phone_number"])
        if p:
            numbers[p] = rep
        ident = r.get("identity")
        if isinstance(ident, str) and ident.strip():
            identities[ident.strip()] = rep
    return numbers, identities


def infer_rep_from_row(row, num_map, id_map):
    primary = row.get("from")
    secondary = row.get("to")

    # Try primary first (outbound from rep)
    rep = map_endpoint(primary, num_map, id_map)
    if rep:
        return rep
    rep = map_endpoint(secondary, num_map, id_map)
    return rep if rep else "Unassigned"


def map_endpoint(value, num_map, id_map):
    if not value:
        return None
    if isinstance(value, str) and value.startswith("client:"):
        return id_map.get(value.strip())
    p = canonical_phone(value)
    return num_map.get(p)


# -----------------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(COACHED_CSV)

    # Parse datetime
    if "call_datetime" in df.columns:
        df["call_datetime"] = pd.to_datetime(df["call_datetime"], errors="coerce", utc=True)
    elif "start_time" in df.columns:
        df["call_datetime"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    else:
        df["call_datetime"] = pd.NaT

    num_map, id_map = load_mapping(MAPPING_CSV)

    df["sales_rep"] = df.apply(lambda r: infer_rep_from_row(r, num_map, id_map), axis=1)

    # Convert coaching columns to numeric
    score_cols = [
        "coaching_total_score",
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
    ]
    for c in score_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


df_all = load_data()

# -----------------------------------------------------------------------------------
# Sidebar Filters
# -----------------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    reps = sorted(df_all["sales_rep"].unique())
    rep_filter = st.selectbox("Sales Rep", ["All Reps"] + reps)

    min_dt = df_all["call_datetime"].min().date()
    max_dt = df_all["call_datetime"].max().date()

    start, end = st.date_input(
        "Call Date Range",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
    )

    include_vm = st.checkbox("Include Voicemails", False)
    include_noans = st.checkbox("Include No-Answer Calls", False)

# Apply filters
df = df_all[
    (df_all["call_datetime"].dt.date >= start) &
    (df_all["call_datetime"].dt.date <= end)
]

if rep_filter != "All Reps":
    df = df[df["sales_rep"] == rep_filter]

if not include_vm and "is_voicemail" in df.columns:
    df = df[~df["is_voicemail"]]

if not include_noans and "no_answer" in df.columns:
    df = df[~df["no_answer"]]

# -----------------------------------------------------------------------------------
# Layout Tabs
# -----------------------------------------------------------------------------------

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])

# -----------------------------------------------------------------------------------
# TAB 1 — INTENT
# -----------------------------------------------------------------------------------

with tab_intent:

    st.title("Sales Call Coaching Dashboard")
    st.caption("Use the sidebar to filter by Sales Rep and date range.")

    st.subheader("Intent Overview")

    total_calls = len(df)
    intents = df["llm_cpd_intent"].fillna("unknown")

    c1, c2 = st.columns(2)
    c1.metric("Total Calls in Filter", total_calls)
    c2.metric("Distinct CPD Intents", intents.nunique())

    # Intent chart
    intent_df = intents.value_counts().reset_index()
    intent_df.columns = ["Intent", "Count"]

    chart = (
        alt.Chart(intent_df)
        .mark_bar()
        .encode(
            x=alt.X("Intent:N", sort="-y", title="Intent"),
            y=alt.Y("Count:Q", title="Number of Calls"),
            tooltip=["Intent", "Count"]
        )
    )

    labels = (
        alt.Chart(intent_df)
        .mark_text(dy=-10, size=20)
        .encode(
            x="Intent:N",
            y="Count:Q",
            text="Count:Q"
        )
    )

    st.altair_chart(chart + labels, use_container_width=True)

    # Calls table
    st.subheader("Calls by Intent")

    cols = ["call_datetime", "sales_rep", "llm_cpd_intent", "llm_cpd_reason", "direction"]
    cols = [c for c in cols if c in df.columns]

    st.dataframe(df[cols].sort_values("call_datetime", ascending=False), height=350)

# -----------------------------------------------------------------------------------
# TAB 2 — COACHING
# -----------------------------------------------------------------------------------

with tab_coaching:

    st.subheader("Average Coaching Score by Rep")

    score_cols = {
        "Opening": "coaching_opening_score",
        "Discovery": "coaching_discovery_score",
        "Value": "coaching_value_score",
        "Closing": "coaching_closing_score",
    }

    df_tmp = df.copy()
    score_values = [c for c in score_cols.values() if c in df_tmp.columns]

    if score_values:
        df_tmp["avg_pillar"] = df_tmp[score_values].mean(axis=1)

        rep_avg = df_tmp.groupby("sales_rep")["avg_pillar"].mean().reset_index()

        chart = (
            alt.Chart(rep_avg)
            .mark_bar(color="#4a90e2")
            .encode(
                x=alt.X("sales_rep:N", title="Sales Rep"),
                y=alt.Y("avg_pillar:Q", title="Average Score", scale=alt.Scale(domain=[0, 5])),
                tooltip=["sales_rep", "avg_pillar"]
            )
        )

        labels = (
            alt.Chart(rep_avg)
            .mark_text(dy=-10, size=20)
            .encode(
                x="sales_rep:N",
                y="avg_pillar:Q",
                text=alt.Text("avg_pillar:Q", format=".1f")
            )
        )

        st.altair_chart(chart + labels, use_container_width=True)

    st.subheader("Coaching Scores by Pillar (Compare Reps)")

    rows = []
    for rep, grp in df.groupby("sales_rep"):
        for label, col in score_cols.items():
            if col in grp:
                vals = grp[col].dropna()
                if len(vals) > 0:
                    rows.append({"sales_rep": rep, "Pillar": label, "Score": vals.mean()})

    prdf = pd.DataFrame(rows)

    if not prdf.empty:
        chart = (
            alt.Chart(prdf)
            .mark_bar()
            .encode(
                x=alt.X("Pillar:N", title="Pillar"),
                y=alt.Y("Score:Q", title="Average Score", scale=alt.Scale(domain=[0, 5])),
                color="sales_rep:N",
                tooltip=["sales_rep", "Pillar", "Score"]
            )
        )

        labels = (
            alt.Chart(prdf)
            .mark_text(size=18, dy=-10)
            .encode(
                x="Pillar:N",
                y="Score:Q",
                detail="sales_rep:N",
                text=alt.Text("Score:Q", format=".1f")
            )
        )

        st.altair_chart(chart + labels, use_container_width=True)

    st.subheader("Calls in Current Filter")

    cols = ["call_datetime", "sales_rep", "direction", "llm_cpd_intent",
            "coaching_total_score", "is_voicemail"]
    cols = [c for c in cols if c in df.columns]

    st.dataframe(df[cols].sort_values("call_datetime", ascending=False), height=350)

    st.subheader("Call Detail")

    if not df.empty:
        df_sel = df.assign(
            _label=lambda d: d["call_datetime"].astype(str)
            + " | " + d["sales_rep"]
            + " | " + d["llm_cpd_intent"].fillna("unknown")
        )

        selected = st.selectbox("Select a call", df_sel["_label"].tolist())
        row = df_sel[df_sel["_label"] == selected].iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### Coaching Summary")
            st.write(row.get("coaching_summary", ""))

            st.markdown("### Top Improvement Points")
            pts = row.get("coaching_improvement_points", "")
            if isinstance(pts, str):
                for p in pts.split("\n"):
                    if p.strip():
                        st.markdown(f"- {p.strip()}")

        with c2:
            st.markdown("### Transcript")
            st.write(row.get("transcript", ""))
