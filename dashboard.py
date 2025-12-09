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
# PATHS — REWRITTEN FOR STREAMLIT CLOUD
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# This is where your CSV MUST exist in GitHub for Streamlit Cloud:
DATA_CSV = BASE_DIR / "Output" / "all_calls_recordings_enriched_CPD_coached.csv"

# Local mapping CSV
MAPPING_CSV = BASE_DIR / "Config" / "sales_rep_phone_map.csv"

# Validation (Cloud-friendly messaging)
if not DATA_CSV.exists():
    st.error(f"❌ Missing data file:\n`{DATA_CSV}`\n\nUpload it to your GitHub repo exactly in this location.")
    st.stop()

if not MAPPING_CSV.exists():
    st.warning(f"⚠️ Mapping file not found at `{MAPPING_CSV}` — reps may show as Unassigned.")

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
    df = pd.read_csv(DATA_CSV)

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
    if "is_v_
