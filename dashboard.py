# -------------------------------------------------------------------
# dashboard.py  – Sales Call Coaching Dashboard
# Version: V41 – Coaching table: FLEX widths (use spare room)
#              – Call Time + Intent no-wrap + ellipsis (but wider)
#              – Score headers show full words when room exists
#              – Headers still NO-WRAP + ellipsis (never wrap)
#              – Intent tab kept stable
# -------------------------------------------------------------------

from pathlib import Path
from typing import Any, Optional
import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# -------------------------------------------------------------------
# STREAMLIT CONFIG & GLOBAL STYLING
# -------------------------------------------------------------------

st.set_page_config(page_title="Sales Call Coaching Dashboard", layout="wide")
alt.data_transformers.disable_max_rows()

st.markdown(
    """
<style>
    html, body, [class*="css"]  { font-size: 20px !important; }
    h1 { font-size: 40px !important; }
    h2, h3 { font-size: 30px !important; }
    .stMetric label { font-size: 22px !important; }
    .stMetric span { font-size: 28px !important; }
    .stSelectbox label, .stDateInput label { font-size: 20px !important; }
    .stTabs [data-baseweb="tab"] { font-size: 22px !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    [data-testid="stToolbar"] { display: none !important; }
    a[href*="github.com"] { display: none !important; }
    button[title*="GitHub"],
    button[aria-label*="GitHub"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# AXIS STYLES
# -------------------------------------------------------------------

def axis_style_x_bold() -> alt.Axis:
    return alt.Axis(
        labelAngle=0,
        labelLimit=0,
        labelFontWeight="bold",
        labelFontSize=14,
        titleFontWeight="bold",
        titleFontSize=16,
    )


def axis_style_y_bold() -> alt.Axis:
    return alt.Axis(
        labelLimit=0,
        labelFontWeight="bold",
        labelFontSize=14,
        titleFontWeight="bold",
        titleFontSize=16,
    )

# -------------------------------------------------------------------
# PASSWORD GATE
# -------------------------------------------------------------------

def check_password() -> bool:
    def password_entered():
        correct = st.session_state.get("password") == st.secrets["access"]["code"]
        st.session_state["password_correct"] = bool(correct)
        if "password" in st.session_state:
            del st.session_state["password"]

    if "password_correct" not in st.session_state:
        st.text_input("Enter access code:", type="password", key="password", on_change=password_entered)
        return False

    if not st.session_state["password_correct"]:
        st.text_input("Enter access code:", type="password", key="password", on_change=password_entered)
        st.error("Incorrect access code.")
        return False

    return True


if not check_password():
    st.stop()

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

LOCAL_DATA_CSV = Path(
    r"C:\Users\guy\OneDrive\2026\Python Project\Sales Team Call Analysis\Output\all_calls_recordings_enriched_CPD_coached.csv"
)

MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# -------------------------------------------------------------------
# PHONE / IDENTITY HELPERS
# -------------------------------------------------------------------

def canonical_phone(num: Optional[Any]) -> Optional[str]:
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return None
    if isinstance(num, (int, np.integer)):
        digits = str(int(num))
    elif isinstance(num, float):
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
    return None


def canonical_from_mapping(num: Optional[Any]) -> Optional[str]:
    if num is None:
        return None
    digits = "".join(ch for ch in str(num) if ch.isdigit())
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

    if "Intent" in df.columns:
        df["intent"] = df["Intent"].fillna("unknown")
    elif "final_cpd_intent" in df.columns:
        df["intent"] = df["final_cpd_intent"].fillna("unknown")
    else:
        df["intent"] = df.get("llm_cpd_intent", df.get("rule_intent")).fillna("unknown")

    df["provider"] = df.get("provider", "Unknown").astype(str).str.title()

    df["call_datetime"] = pd.to_datetime(df.get("call_datetime"), errors="coerce")
    df = df[df["call_datetime"].notna()].copy()

    df["call_type"] = "Standard"
    if "is_voicemail" in df.columns:
        df.loc[df["is_voicemail"] == True, "call_type"] = "Voicemail"  # noqa: E712
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
    df["sales_rep"] = df.apply(lambda r: infer_rep_from_row(r, mapping_numbers, mapping_identities), axis=1)

    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    if "intent" not in df.columns:
        df["intent"] = "unknown"

    return df


def load_data() -> pd.DataFrame:
    if LOCAL_DATA_CSV.exists():
        try:
            df_raw = pd.read_csv(LOCAL_DATA_CSV)
            return process_raw_dataframe(df_raw)
        except Exception as e:
            st.error(f"Local pipeline CSV found but could not load: {e}")

    st.info("Local enriched CSV not found. Upload CSV.")
    uploaded = st.file_uploader("Upload enriched CSV", type=["csv"])
    if uploaded:
        try:
            return process_raw_dataframe(pd.read_csv(uploaded))
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

    return pd.DataFrame()

# -------------------------------------------------------------------
# LOAD ALL DATA
# -------------------------------------------------------------------

df_all = load_data()

if df_all.empty:
    st.title("Sales Call Coaching Dashboard")
    st.warning("No calls found.")
    st.stop()

last_updated = df_all["call_datetime"].max()

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

    if isinstance(date_range, tuple):
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date = end_date = date_range[0]
        else:
            start_date, end_date = min_dt.date(), max_dt.date()
    else:
        start_date = end_date = date_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    selected_carrier = st.selectbox("Carrier", ["All"] + sorted(df_all["provider"].unique()))
    selected_rep = st.selectbox("Sales Rep", ["All Reps"] + sorted(df_all["sales_rep"].unique()))

    include_voicemail = st.checkbox("Include Voicemail", True)
    include_no_answer = st.checkbox("Include No Answer", True)
    include_too_short = st.checkbox("Include Too Short", False)

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

if pd.notna(last_updated):
    st.caption(
        f"Using local pipeline CSV: {LOCAL_DATA_CSV} · {len(df_all)} total calls · "
        f"Last updated: {last_updated.strftime('%Y-%m-%d %H:%M')}."
    )

tab_intent, tab_coaching = st.tabs(["Buying Intent", "Coaching"])

# -------------------------------------------------------------------
# AGGRID CSS (HEADERS: NO-WRAP + ELLIPSIS)
# -------------------------------------------------------------------

AGGRID_CSS_BIG = {
    ".ag-header-cell-label": {
        "font-size": "26px !important;",
        "font-weight": "700 !important;",
        "white-space": "nowrap !important;",
        "overflow": "hidden !important;",
        "text-overflow": "ellipsis !important;",
    },
    ".ag-header-cell-text": {
        "white-space": "nowrap !important;",
        "overflow": "hidden !important;",
        "text-overflow": "ellipsis !important;",
    },
    ".ag-cell": {
        "font-size": "28px !important;",
        "line-height": "32px !important;",
        "padding": "8px 10px !important;",
    },
    ".ag-selection-checkbox": {"transform": "scale(1.8);", "margin-top": "6px;"},
}

# -------------------------------------------------------------------
# TAB 1 — BUYING INTENT (stable)
# -------------------------------------------------------------------

with tab_intent:
    st.subheader("Intent Overview")

    total_calls = len(df)
    distinct_intents = df["intent"].nunique()

    c1, c2 = st.columns(2)
    c1.metric("Total Calls in Filter", total_calls)
    c2.metric("Distinct CPD Intents", distinct_intents)

    intent_counts = df["intent"].value_counts().rename_axis("Intent").reset_index(name="Count")

    bar = (
        alt.Chart(intent_counts)
        .mark_bar()
        .encode(
            x=alt.X("Intent:N", sort="-y", axis=alt.Axis(labelAngle=0, labelLimit=0)),
            y=alt.Y("Count:Q", axis=axis_style_y_bold()),
            color=alt.Color("Intent:N", title="Intent"),
        )
    )

    labels = (
        alt.Chart(intent_counts)
        .mark_text(dy=-10, fontSize=18, fontWeight="bold")
        .encode(x="Intent:N", y="Count:Q", text="Count:Q")
    )

    st.altair_chart(bar + labels, width="stretch")

    st.markdown("### Calls by Intent (tick a box to see details below)")

    cols = ["call_datetime", "sales_rep", "provider", "intent", "call_type"]
    cols = [c for c in cols if c in df.columns]
    df_intent_calls = df[cols].sort_values("call_datetime", ascending=False)

    gb_int = GridOptionsBuilder.from_dataframe(df_intent_calls)
    gb_int.configure_selection("single", use_checkbox=True)
    gb_int.configure_pagination(enabled=False)
    gb_int.configure_default_column(resizable=True, wrapText=True, autoHeight=True, sortable=True, filter=True, minWidth=160)
    gb_int.configure_grid_options(rowHeight=64, headerHeight=72, suppressSizeToFit=True, alwaysShowHorizontalScroll=True, domLayout="normal")

    grid_resp_int = AgGrid(
        df_intent_calls,
        gridOptions=gb_int.build(),
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=420,
        fit_columns_on_grid_load=False,
        custom_css=AGGRID_CSS_BIG,
        theme="streamlit",
    )

    selected_rows_int = grid_resp_int.get("selected_rows", None)

    sel_int = None
    if isinstance(selected_rows_int, list) and len(selected_rows_int) > 0:
        sel_int = selected_rows_int[0]
    elif isinstance(selected_rows_int, pd.DataFrame) and not selected_rows_int.empty:
        sel_int = selected_rows_int.iloc[0].to_dict()

    if sel_int is not None:
        key_dt = pd.to_datetime(sel_int.get("call_datetime"))
        key_rep = sel_int.get("sales_rep", "")
        key_intent = sel_int.get("intent", "")

        mask_int = (df["call_datetime"] == key_dt) & (df["sales_rep"] == key_rep) & (df["intent"] == key_intent)
        if not mask_int.any():
            mask_int = df["call_datetime"] == key_dt

        row = df[mask_int].iloc[0]

        st.markdown("---")
        st.markdown('<h2 style="font-size: 36px; font-weight: 700;">Call Detail</h2>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            st.markdown('<h3 style="font-size: 32px; font-weight: 700;">Intent Info</h3>', unsafe_allow_html=True)

            unified_intent = html.escape(str(row.get("intent", "")))
            llm_intent = html.escape(str(row.get("llm_cpd_intent", "")))
            rule_intent = html.escape(str(row.get("rule_intent", "")))

            st.markdown(f'<div style="font-size: 30px;"><strong>Unified Intent:</strong> {unified_intent}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px;"><strong>LLM Intent:</strong> {llm_intent}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px;"><strong>Rule Intent:</strong> {rule_intent}</div>', unsafe_allow_html=True)

            st.markdown('<h3 style="font-size: 32px; font-weight: 700; margin-top: 1.25rem;">Metadata</h3>', unsafe_allow_html=True)

            rep = html.escape(str(row.get("sales_rep", "")))
            provider = html.escape(str(row.get("provider", "")))
            direction = html.escape(str(row.get("direction", "")))
            call_type = html.escape(str(row.get("call_type", "")))

            st.markdown(f'<div style="font-size: 30px;"><strong>Rep:</strong> {rep}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px;"><strong>Provider:</strong> {provider}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px;"><strong>Direction:</strong> {direction}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px;"><strong>Call Type:</strong> {call_type}</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<h3 style="font-size: 32px; font-weight: 700;">Transcript</h3>', unsafe_allow_html=True)
            transcript_html = html.escape(str(row.get("transcript", "") or ""))
            st.markdown(f'<div style="font-size: 30px; white-space: pre-wrap;">{transcript_html}</div>', unsafe_allow_html=True)
    else:
        st.info("Tick a checkbox in the table above to view the call detail and transcript.")

# -------------------------------------------------------------------
# TAB 2 — COACHING (FIXED: flex widths + wider score headers)
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

    chart_rep = (
        alt.Chart(rep_avg)
        .mark_bar()
        .encode(
            x=alt.X("sales_rep:N", title="Sales Rep", axis=axis_style_x_bold()),
            y=alt.Y("AvgScore:Q", title="Avg Score", axis=axis_style_y_bold()),
            color=alt.Color("sales_rep:N", title="Sales Rep"),
        )
    )
    rep_labels = (
        alt.Chart(rep_avg)
        .mark_text(dy=-10, fontSize=18, fontWeight="bold")
        .encode(x="sales_rep:N", y="AvgScore:Q", text=alt.Text("AvgScore:Q", format=".1f"))
    )
    st.altair_chart(chart_rep + rep_labels, width="stretch")

    st.markdown("### Coaching View — Calls (tick a box to see details below)")

    df_calls = df_coached[
        [
            "call_datetime",
            "sales_rep",
            "intent",
            "coaching_total_score",
            "coaching_opening_score",
            "coaching_discovery_score",
            "coaching_value_score",
            "coaching_closing_score",
        ]
    ].copy()

    df_calls["call_time"] = pd.to_datetime(df_calls["call_datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    df_calls["intent_display"] = df_calls.get("intent", "").astype(str)

    display_cols = [
        "call_time",
        "sales_rep",
        "intent_display",
        "coaching_total_score",
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
    ]
    df_calls = df_calls[display_cols].sort_values("call_time", ascending=False)

    gb_calls = GridOptionsBuilder.from_dataframe(df_calls)
    gb_calls.configure_selection("single", use_checkbox=True)
    gb_calls.configure_pagination(enabled=False)
    gb_calls.configure_default_column(resizable=True, sortable=True, filter=True)

    nowrap_ellipsis_style = {"white-space": "nowrap", "overflow": "hidden", "text-overflow": "ellipsis"}
    score_style = {"text-align": "center", "white-space": "nowrap"}

    # KEY CHANGE: use flex so spare width gets used (no big empty right area)
    gb_calls.configure_column(
        "call_time",
        header_name="Call Time",
        minWidth=260,
        flex=2,
        wrapText=False,
        autoHeight=False,
        cellStyle=nowrap_ellipsis_style,
    )
    gb_calls.configure_column(
        "sales_rep",
        header_name="Sales Rep",
        minWidth=200,
        flex=1,
        wrapText=False,
        autoHeight=False,
        cellStyle=nowrap_ellipsis_style,
    )
    gb_calls.configure_column(
        "intent_display",
        header_name="Intent",
        minWidth=320,
        flex=3,
        wrapText=False,
        autoHeight=False,
        cellStyle=nowrap_ellipsis_style,
    )

    # Score columns: give enough min width so headers show fully when space exists
    gb_calls.configure_column("coaching_total_score", header_name="Total", minWidth=110, width=110, wrapText=False, autoHeight=False, cellStyle=score_style)
    gb_calls.configure_column("coaching_opening_score", header_name="Opening", minWidth=130, width=130, wrapText=False, autoHeight=False, cellStyle=score_style)
    gb_calls.configure_column("coaching_discovery_score", header_name="Discovery", minWidth=140, width=140, wrapText=False, autoHeight=False, cellStyle=score_style)
    gb_calls.configure_column("coaching_value_score", header_name="Value", minWidth=110, width=110, wrapText=False, autoHeight=False, cellStyle=score_style)
    gb_calls.configure_column("coaching_closing_score", header_name="Closing", minWidth=130, width=130, wrapText=False, autoHeight=False, cellStyle=score_style)

    gb_calls.configure_grid_options(
        rowHeight=64,
        headerHeight=72,
        suppressSizeToFit=False,
        alwaysShowHorizontalScroll=False,
        domLayout="normal",
    )

    grid_response = AgGrid(
        df_calls,
        gridOptions=gb_calls.build(),
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        height=420,
        fit_columns_on_grid_load=True,
        custom_css=AGGRID_CSS_BIG,
        theme="streamlit",
    )

    selected_rows = grid_response.get("selected_rows", None)
    sel_dict = None
    if isinstance(selected_rows, list) and len(selected_rows) > 0:
        sel_dict = selected_rows[0]
    elif isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
        sel_dict = selected_rows.iloc[0].to_dict()

    if sel_dict is not None:
        chosen_time = sel_dict.get("call_time", "")
        chosen_rep = sel_dict.get("sales_rep", "")
        chosen_intent = sel_dict.get("intent_display", "")

        chosen_dt = pd.to_datetime(chosen_time, errors="coerce")

        mask = pd.Series([True] * len(df_coached))
        if pd.notna(chosen_dt):
            mask &= (df_coached["call_datetime"] == chosen_dt)
        if chosen_rep:
            mask &= (df_coached["sales_rep"] == chosen_rep)
        if chosen_intent:
            mask &= (df_coached["intent"].astype(str) == str(chosen_intent))

        if not mask.any() and pd.notna(chosen_dt):
            mask = df_coached["call_datetime"] == chosen_dt

        if not mask.any():
            st.warning("Could not locate full call details for the selected row.")
            st.stop()

        row_c = df_coached[mask].iloc[0]

        st.markdown("---")
        summary_html = html.escape(str(row_c.get("coaching_summary", "") or ""))
        transcript_html = html.escape(str(row_c.get("transcript", "") or ""))

        st.markdown('<h2 style="font-size: 36px; font-weight: 700;">Coaching Summary</h2>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<h3 style="font-size: 32px; font-weight: 700;">Coaching Summary Text</h3>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px;">{summary_html}</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<h3 style="font-size: 32px; font-weight: 700;">Transcript</h3>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size: 30px; white-space: pre-wrap;">{transcript_html}</div>', unsafe_allow_html=True)
    else:
        st.info("Tick a checkbox in the table above to view the coaching summary and transcript.")
