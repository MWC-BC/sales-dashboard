# dashboard.py

from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# -------------------------------------------------------------------------
# Streamlit & Altair global config
# -------------------------------------------------------------------------

st.set_page_config(
    page_title="Sales Call Coaching Dashboard",
    layout="wide",
)

# Allow more than 5000 rows in Altair charts
alt.data_transformers.disable_max_rows()

# Bigger fonts (similar to your original)
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

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
COACHED_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"
MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# -------------------------------------------------------------------------
# Phone / identity helpers (same logic as your working version)
# -------------------------------------------------------------------------


def canonical_phone(num: Optional[Any]) -> Optional[str]:
    """Normalize phone numbers into canonical 11-digit format '1XXXXXXXXXX'."""
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return None

    # numeric types first (handle scientific notation / .0)
    if isinstance(num, (int, np.integer)):
        digits = str(int(num))
    elif isinstance(num, (float, np.floating)):
        digits = str(int(num))
    else:
        s = str(num)
        # ignore Twilio client identities here
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

        # phone number mapping
        canon = canonical_from_mapping(r["phone_number"])
        if canon:
            mapping_numbers[canon] = rep

        # identity mapping (for inbound Keap / Twilio client IDs)
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
    """
    Try to map a single endpoint value (phone or client identity) to a rep name.

    - If it looks like 'client:instapulse_...', use the identity mapping.
    - Otherwise, normalize as phone and look up in phone mapping.
    """
    if endpoint is None or (isinstance(endpoint, float) and np.isnan(endpoint)):
        return None

    # Identity-based mapping (client:instapulse_...)
    if isinstance(endpoint, str) and endpoint.startswith("client:"):
        rep = mapping_identities.get(endpoint.strip())
        if rep:
            return rep

    # Phone-based mapping
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

    Priority:
        1) Choose the most likely rep side based on direction.
        2) Try to map that side (identity, then phone).
        3) If no match, try the other side.
        4) If still no match, return 'Unassigned'.
    """
    direction_business = str(row.get("direction_business", "") or "").lower()
    direction = str(row.get("direction", "") or "").lower()

    raw_from = row.get("from")
    raw_to = row.get("to")

    # 1) Decide which endpoint is most likely the rep side
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
        # Unknown direction – pick FROM first, then TO
        primary = raw_from
        secondary = raw_to

    # 2) Try primary endpoint
    rep = map_endpoint_to_rep(primary, mapping_numbers, mapping_identities)
    if rep:
        return rep

    # 3) Fallback: try the other side
    rep = map_endpoint_to_rep(secondary, mapping_numbers, mapping_identities)
    if rep:
        return rep

    # 4) Nothing matched
    return "Unassigned"


# -------------------------------------------------------------------------
# Safe CSV loader
# -------------------------------------------------------------------------


def safe_load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Could not find CSV at: {path}")
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error reading CSV at {path}:\n{e}")
        st.stop()


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = safe_load_csv(COACHED_CSV)

    # Build call_datetime if needed
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

    # If sales_rep already exists (from pipeline), respect it.
    if "sales_rep" not in df.columns:
        mapping_numbers, mapping_identities = load_mapping(MAPPING_CSV)
        df["sales_rep"] = df.apply(
            lambda r: infer_rep_from_row(r, mapping_numbers, mapping_identities),
            axis=1,
        )
    else:
        df["sales_rep"] = df["sales_rep"].fillna("Unassigned")

    # Ensure numeric coaching fields
    for c in [
        "coaching_total_score",
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Make sure llm_cpd_intent exists (fallback to 'unknown' if missing)
    if "llm_cpd_intent" not in df.columns:
        df["llm_cpd_intent"] = "unknown"

    return df


df_all = load_data()

# -------------------------------------------------------------------------
# Rep color scale – consistent across all charts
# -------------------------------------------------------------------------

all_reps = sorted(r for r in df_all["sales_rep"].unique() if r)

# Nice 10-color palette; will loop if we ever have more than 10 reps.
base_colors = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
]
rep_colors = (base_colors * ((len(all_reps) // len(base_colors)) + 1))[: len(all_reps)]

rep_color_scale = alt.Scale(domain=all_reps, range=rep_colors)

# -------------------------------------------------------------------------
# Sidebar filters
# -------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    rep_filter = st.selectbox("Sales Rep", ["All Reps"] + all_reps)

    min_date = df_all["call_datetime"].min().date()
    max_date = df_all["call_datetime"].max().date()

    date_range = st.date_input(
        "Call Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    start_date, end_date = date_range

    show_vm = st.checkbox("Include Voicemails", False)
    show_noans = st.checkbox("Include No-Answer Calls", False)

# -------------------------------------------------------------------------
# Apply filters
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------------

st.title("Sales Call Coaching Dashboard")
st.caption("Use the sidebar to filter by Sales Rep and date range.")

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])

# -------------------------------------------------------------------------
# TAB 1 – BUYING INTENT
# -------------------------------------------------------------------------

with tab_intent:
    st.subheader("Intent Overview")

    total_calls = len(df_filtered)
    distinct_intents = df_filtered["llm_cpd_intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls in Filter", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    # Intent Summary chart
    if not df_filtered.empty:
        intent_df = (
            df_filtered["llm_cpd_intent"]
            .fillna("unknown")
            .value_counts()
            .reset_index()
        )
        intent_df.columns = ["Intent", "Count"]

        intent_chart = (
            alt.Chart(intent_df)
            .mark_bar()
            .encode(
                x=alt.X("Intent:N", sort="-y", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q"),
                color=alt.Color("Intent:N", legend=None),
            )
        )

        intent_labels = intent_chart.mark_text(
            dy=-10,
            fontSize=14,
            fontWeight="bold",
            color="black",
        ).encode(text=alt.Text("Count:Q", format="d"))

        st.altair_chart(intent_chart + intent_labels, use_container_width=True)

    # Calls per Rep chart
    st.subheader("Calls Per Rep")

    if not df_filtered.empty:
        rep_counts = (
            df_filtered.groupby("sales_rep")
            .size()
            .reset_index(name="call_count")
        )

        calls_rep_chart = (
            alt.Chart(rep_counts)
            .mark_bar()
            .encode(
                x=alt.X("sales_rep:N", title="Sales Rep", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("call_count:Q", title="Number of Calls"),
                color=alt.Color("sales_rep:N", scale=rep_color_scale, legend=None),
            )
        )

        calls_rep_labels = calls_rep_chart.mark_text(
            dy=-10,
            fontSize=14,
            fontWeight="bold",
            color="black",
        ).encode(text=alt.Text("call_count:Q", format="d"))

        st.altair_chart(calls_rep_chart + calls_rep_labels, use_container_width=True)
    else:
        st.info("No calls in the current filter.")

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

    if cols_intent:
        st.dataframe(
            df_filtered[cols_intent].sort_values(
                "call_datetime", ascending=False
            ),
            use_container_width=True,
            height=350,
        )

    st.subheader("Call Detail (Intent View)")

    if not df_filtered.empty:
        df_sel_intent = df_filtered.assign(
            _label=lambda d: d["call_datetime"].astype(str)
            + " | "
            + d["sales_rep"]
            + " | "
            + d["llm_cpd_intent"].fillna("unknown")
        )

        selected_intent = st.selectbox(
            "Select a call (Intent view)",
            options=df_sel_intent["_label"].tolist(),
        )

        row_intent = df_sel_intent[df_sel_intent["_label"] == selected_intent].iloc[0]

        left, right = st.columns(2)

        with left:
            st.markdown("### CPD Intent")
            st.write(f"**Intent:** {row_intent['llm_cpd_intent']}")
            st.write(f"**Reason:** {row_intent['llm_cpd_reason']}")

        with right:
            st.markdown("### Transcript")
            st.write(row_intent.get("transcript", ""))

# -------------------------------------------------------------------------
# TAB 2 – COACHING
# -------------------------------------------------------------------------

with tab_coaching:
    st.subheader("Average Coaching Score by Rep")

    pillar_cols = {
        "Opening": "coaching_opening_score",
        "Discovery": "coaching_discovery_score",
        "Value": "coaching_value_score",
        "Closing": "coaching_closing_score",
    }

    df_tmp = df_filtered.copy()
    numeric_cols = [c for c in pillar_cols.values() if c in df_tmp.columns]

    if numeric_cols:
        df_tmp["pillar_avg"] = df_tmp[numeric_cols].mean(axis=1)
        rep_avg = (
            df_tmp.groupby("sales_rep")["pillar_avg"]
            .mean()
            .reset_index()
            .rename(columns={"pillar_avg": "Average Score"})
        )

        if not rep_avg.empty:
            rep_avg_chart = (
                alt.Chart(rep_avg)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "sales_rep:N",
                        title="Sales Rep",
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y(
                        "Average Score:Q",
                        scale=alt.Scale(domain=[0, 5]),
                        title="Average Coaching Score",
                    ),
                    color=alt.Color(
                        "sales_rep:N",
                        scale=rep_color_scale,
                        legend=None,
                    ),
                )
            )

            rep_avg_labels = rep_avg_chart.mark_text(
                dy=-10,
                fontSize=14,
                fontWeight="bold",
                color="black",
            ).encode(text=alt.Text("Average Score:Q", format=".1f"))

            st.altair_chart(rep_avg_chart + rep_avg_labels, use_container_width=True)
        else:
            st.info("No coaching scores available to compute averages.")
    else:
        st.info("No coaching score columns found to compute averages.")

    st.subheader("Coaching Scores by Pillar (compare reps per pillar)")

    # Build per-rep per-pillar averages
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

    pillar_chart_data = pd.DataFrame(records)

    if not pillar_chart_data.empty:
        base = alt.Chart(pillar_chart_data).encode(
            x=alt.X("Pillar:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "Average Score:Q",
                scale=alt.Scale(domain=[0, 5]),
                title="Average Score",
            ),
            color=alt.Color(
                "sales_rep:N",
                scale=rep_color_scale,
                title="Sales Rep",
            ),
            column=alt.Column("sales_rep:N", title="Sales Rep"),
        )

        # Bars
        bars = base.mark_bar()

        # Labels (score above each bar)
        labels = base.mark_text(
            dy=-10,
            fontSize=14,
            fontWeight="bold",
            color="black",
        ).encode(text=alt.Text("Average Score:Q", format=".1f"))

        chart = (bars + labels).properties(height=300).configure_axis(
            labelFontSize=14,
            titleFontSize=16,
        )

        st.altair_chart(chart, use_container_width=True)
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

    if cols_calls:
        st.dataframe(
            df_filtered[cols_calls].sort_values(
                "call_datetime", ascending=False
            ),
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
