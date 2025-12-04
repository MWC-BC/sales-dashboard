# dashboard.py – Sales Team Dashboard (Streamlit Cloud + password gate)

from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Call Coaching Dashboard",
    layout="wide",
)

# Allow Altair to show more than 5000 rows
alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# PASSWORD GATE
# -----------------------------------------------------------------------------
def check_password() -> bool:
    """Simple password gate using Streamlit secrets."""
    def password_entered():
        try:
            if st.session_state["password"] == st.secrets["access"]["code"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]
            else:
                st.session_state["password_correct"] = False
        except KeyError:
            # secrets not configured correctly
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
        st.error(
            'Incorrect access code or missing secret. '
            'If you are the owner, check [access] -> code in app secrets.'
        )
        return False

    return True


# Stop app here until password is correct
if not check_password():
    st.stop()

# -----------------------------------------------------------------------------
# STYLES
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-size: 22px !important;
        }

        h1 {
            font-size: 48px !important;
        }

        h2, h3 {
            font-size: 36px !important;
        }

        .stMetric label {
            font-size: 26px !important;
        }

        .stMetric span {
            font-size: 34px !important;
        }

        .stDataFrame table tbody tr td {
            font-size: 20px !important;
        }

        .stDataFrame table thead tr th {
            font-size: 22px !important;
        }

        .stSelectbox label, .stDateInput label {
            font-size: 22px !important;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 24px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
COACHED_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"
MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# -----------------------------------------------------------------------------
# PHONE / IDENTITY HELPERS (same logic as your working version)
# -----------------------------------------------------------------------------
def canonical_phone(num: Optional[Any]) -> Optional[str]:
    """Normalize phone numbers into canonical 11-digit format '1XXXXXXXXXX'."""
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
    """Normalize phone numbers coming from the mapping CSV."""
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
    """
    Load sales rep mapping.

    Expected CSV columns:
        phone_number,sales_rep
        [optional] identity  (e.g. client:instapulse_mikemasterworkscoachingco)
    """
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


def map_endpoint_to_rep(
    endpoint: Any,
    mapping_numbers: Dict[str, str],
    mapping_identities: Dict[str, str],
) -> Optional[str]:
    """Map a single endpoint value (phone or client identity) to a rep name."""
    if endpoint is None or (isinstance(endpoint, float) and np.isnan(endpoint)):
        return None

    if isinstance(endpoint, str) and endpoint.startswith("client:"):
        rep = mapping_identities.get(endpoint.strip())
        if rep:
            return rep

    canon = canonical_phone(endpoint)
    if canon:
        rep = mapping_numbers.get(canon)
        if rep:
            return rep

    return None


def infer_rep_from_row(
    row: pd.Series,
    mapping_numbers: Dict[str, str],
    mapping_identities: Dict[str, str],
) -> str:
    """
    Determine which endpoint belongs to the SALES REP and map to rep name.
    """
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


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not COACHED_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(COACHED_CSV)

    # Build call_datetime
    if "call_datetime" in df.columns:
        df["call_datetime"] = pd.to_datetime(
            df["call_datetime"], utc=True, errors="coerce"
        )
    elif "start_time" in df.columns:
        df["call_datetime"] = pd.to_datetime(
            df["start_time"], utc=True, errors="coerce"
        )
    else:
        df["call_datetime"] = pd.NaT

    # Mapping
    mapping_numbers, mapping_identities = load_mapping(MAPPING_CSV)

    df["sales_rep"] = df.apply(
        lambda r: infer_rep_from_row(r, mapping_numbers, mapping_identities),
        axis=1,
    )

    # Ensure coaching fields numeric
    numeric_cols = [
        "coaching_total_score",
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure boolean flags exist
    if "is_voicemail" not in df.columns:
        df["is_voicemail"] = False
    if "no_answer" not in df.columns:
        df["no_answer"] = False

    return df


df_all = load_data()

if df_all.empty:
    st.error(
        "No data loaded. Make sure "
        "`Output/all_calls_recordings_enriched_CPD_coached.csv` "
        "exists in the repository."
    )
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    unique_reps = sorted(r for r in df_all["sales_rep"].unique() if r)
    rep_filter = st.selectbox("Sales Rep", ["All Reps"] + unique_reps)

    if df_all["call_datetime"].notna().any():
        min_date = df_all["call_datetime"].min().date()
        max_date = df_all["call_datetime"].max().date()
    else:
        today = pd.Timestamp.utcnow().date()
        min_date = max_date = today

    date_range = st.date_input(
        "Call Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    show_vm = st.checkbox("Include Voicemails", False)
    show_noans = st.checkbox("Include No-Answer Calls", False)

# Apply filters
df_filtered = df_all.copy()

df_filtered = df_filtered[
    (df_filtered["call_datetime"].dt.date >= start_date)
    & (df_filtered["call_datetime"].dt.date <= end_date)
]

if rep_filter != "All Reps":
    df_filtered = df_filtered[df_filtered["sales_rep"] == rep_filter]

if not show_vm and "is_voicemail" in df_filtered.columns:
    df_filtered = df_filtered[~df_filtered["is_voicemail"]]

if not show_noans and "no_answer" in df_filtered.columns:
    df_filtered = df_filtered[~df_filtered["no_answer"]]

# -----------------------------------------------------------------------------
# HELPER: ALTair bar with labels
# -----------------------------------------------------------------------------
def bar_with_labels(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: Optional[str] = None,
    y_max: Optional[float] = None,
    text_format: str = ".0f",
    x_offset: Optional[str] = None,
    color_title: Optional[str] = None,
):
    base = alt.Chart(data)

    bar = base.mark_bar().encode(
        x=alt.X(x, title=None),
        y=alt.Y(y, title=None, scale=alt.Scale(domain=(0, y_max)) if y_max else alt.Undefined),
        tooltip=[x, y] + ([color] if color else []),
        color=alt.Color(
            color,
            title=color_title,
            scale=alt.Scale(scheme="tableau10"),
        )
        if color
        else alt.value("#4C78A8"),
        xOffset=x_offset,
    )

    text = base.mark_text(
        dy=-5,
        fontSize=16,
    ).encode(
        x=alt.X(x),
        y=alt.Y(y),
        text=alt.Text(y, format=text_format),
        color=alt.value("black"),
        xOffset=x_offset,
    )

    chart = (bar + text).properties(title=title, height=350)
    return chart


# -----------------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------------
st.title("Sales Call Coaching Dashboard")
st.caption("Use the sidebar to filter by Sales Rep and date range.")

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])

# -----------------------------------------------------------------------------
# TAB 1 – INTENT
# -----------------------------------------------------------------------------
with tab_intent:
    st.subheader("Intent Overview")

    total_calls = len(df_filtered)
    distinct_intents = df_filtered["llm_cpd_intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls in Filter", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    if not df_filtered.empty:
        intent_df = (
            df_filtered["llm_cpd_intent"]
            .fillna("unknown")
            .value_counts()
            .reset_index()
        )
        intent_df.columns = ["Intent", "Count"]
        y_max = float(intent_df["Count"].max() * 1.25)

        chart = bar_with_labels(
            intent_df,
            x="Intent:N",
            y="Count:Q",
            title="Intent Summary",
            y_max=y_max,
            text_format=".0f",
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
        ).configure_title(fontSize=24)

        st.altair_chart(chart, use_container_width=True)

    st.subheader("Calls by Intent")
    cols_intent = [
        "call_datetime",
        "sales_rep",
        "llm_cpd_intent",
        "llm_cpd_reason",
        "direction",
        "is_voicemail",
    ]
    cols_intent = [c for c in cols_intent if c in df_filtered.columns]

    st.dataframe(
        df_filtered[cols_intent].sort_values("call_datetime", ascending=False),
        use_container_width=True,
        height=350,
    )

    st.subheader("Call Detail (Intent View)")
    if not df_filtered.empty:
        df_sel = df_filtered.assign(
            _label=lambda d: d["call_datetime"].astype(str)
            + " | "
            + d["sales_rep"]
            + " | "
            + d["llm_cpd_intent"].fillna("unknown")
        )

        selected = st.selectbox(
            "Select a call (Intent view)",
            options=df_sel["_label"].tolist(),
        )

        row = df_sel[df_sel["_label"] == selected].iloc[0]

        left, right = st.columns(2)

        with left:
            st.markdown("### CPD Intent")
            st.write(f"**Intent:** {row.get('llm_cpd_intent', '')}")
            st.write(f"**Reason:** {row.get('llm_cpd_reason', '')}")

        with right:
            st.markdown("### Transcript")
            st.write(row.get("transcript", ""))

# -----------------------------------------------------------------------------
# TAB 2 – COACHING
# -----------------------------------------------------------------------------
with tab_coaching:
    st.subheader("Average Coaching Score by Rep")

    pillar_cols = {
        "Opening": "coaching_opening_score",
        "Discovery": "coaching_discovery_score",
        "Value": "coaching_value_score",
        "Closing": "coaching_closing_score",
    }

    df_tmp = df_filtered.copy()
    numeric_pillars = [c for c in pillar_cols.values() if c in df_tmp.columns]

    if numeric_pillars:
        df_tmp["pillar_avg"] = df_tmp[numeric_pillars].mean(axis=1)
        rep_avg = (
            df_tmp.groupby("sales_rep", as_index=False)["pillar_avg"]
            .mean()
            .rename(columns={"pillar_avg": "Average Coaching Score"})
        )

        y_max = 5.0

        rep_chart = bar_with_labels(
            rep_avg,
            x="sales_rep:N",
            y="Average Coaching Score:Q",
            title="Average Coaching Score by Rep",
            y_max=y_max,
            text_format=".1f",
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
        ).configure_title(fontSize=24)

        st.altair_chart(rep_chart, use_container_width=True)
    else:
        st.info("No coaching score columns found to compute averages.")

    st.subheader("Coaching Scores by Pillar (compare reps per pillar)")

    records = []
    for rep, grp in df_filtered.groupby("sales_rep"):
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
        y_max = 5.0

        base = alt.Chart(prdf)

        bar = base.mark_bar().encode(
            x=alt.X("Pillar:N", title=None),
            y=alt.Y(
                "Average Score:Q",
                title=None,
                scale=alt.Scale(domain=(0, y_max)),
            ),
            color=alt.Color(
                "sales_rep:N",
                title="Sales Rep",
                scale=alt.Scale(scheme="tableau10"),
            ),
            xOffset="sales_rep:N",
            tooltip=["sales_rep", "Pillar", "Average Score"],
        )

        text = base.mark_text(
            dy=-5,
            fontSize=16,
        ).encode(
            x=alt.X("Pillar:N"),
            y=alt.Y("Average Score:Q"),
            text=alt.Text("Average Score:Q", format=".1f"),
            xOffset="sales_rep:N",
            color=alt.value("black"),
        )

        pillar_chart = (bar + text).properties(
            title="Coaching Scores by Pillar (compare reps per pillar)",
            height=400,
        ).configure_axis(
            labelFontSize=16,
            titleFontSize=18,
        ).configure_legend(
            titleFontSize=18,
            labelFontSize=16,
        ).configure_title(fontSize=24)

        st.altair_chart(pillar_chart, use_container_width=True)
    else:
        st.info("No coaching data available to plot pillar scores.")

    st.subheader("Calls in Current Filter")
    cols_calls = [
        "call_datetime",
        "sales_rep",
        "direction",
        "llm_cpd_intent",
        "coaching_total_score",
        "is_voicemail",
    ]
    cols_calls = [c for c in cols_calls if c in df_filtered.columns]

    st.dataframe(
        df_filtered[cols_calls].sort_values("call_datetime", ascending=False),
        use_container_width=True,
        height=350,
    )

    st.subheader("Call Detail")

    if not df_filtered.empty:
        df_sel = df_filtered.assign(
            _label=lambda d: d["call_datetime"].astype(str)
            + " | "
            + d["sales_rep"]
            + " | "
            + d["llm_cpd_intent"].fillna("unknown")
        )

        selected = st.selectbox("Select a call", df_sel["_label"].tolist())
        row = df_sel[df_sel["_label"] == selected].iloc[0]

        left, right = st.columns(2)

        with left:
            st.markdown("### Coaching Summary")
            st.write(row.get("coaching_summary", ""))

            st.markdown("### Top Improvement Points")
            imp = row.get("coaching_improvement_points", "")
            if isinstance(imp, str):
                for p in imp.split("\n"):
                    if p.strip():
                        st.markdown(f"- {p.strip()}")

        with right:
            st.markdown("### Transcript")
            st.write(row.get("transcript", ""))
