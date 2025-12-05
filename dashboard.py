# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_CSV)

    # -----------------------------------
    # REPAIR MISSING OR MISNAMED FIELDS
    # -----------------------------------

    # ---- 1) Create unified "intent" field ----
    if "intent" not in df.columns:
        df["intent"] = df["llm_cpd_intent"].fillna(df["rule_intent"])
    df["intent"] = df["intent"].fillna("Unknown")

    # ---- 2) Guarantee sales rep column exists ----
    if "sales_rep" not in df.columns:
        df["sales_rep"] = "Unassigned"
    df["sales_rep"] = df["sales_rep"].fillna("Unassigned")

    # ---- 3) Normalize provider field ----
    if "provider" in df.columns:
        df["provider"] = df["provider"].astype(str).str.title()
    else:
        df["provider"] = "Unknown"

    # ---- 4) Fix missing call_datetime ----
    # Your file contains start_time but NOT call_datetime.
    time_col = None
    for c in ["call_datetime", "start_time", "timestamp", "call_time"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is None:
        raise ValueError("No datetime column found in CSV: expected call_datetime or start_time")

    # Convert to datetime safely
    df["call_datetime"] = pd.to_datetime(df[time_col], errors="coerce", utc=False)

    # Drop impossible or missing datetimes
    df = df[df["call_datetime"].notna()]

    # Ensure datetime is timezone-naive to avoid tz-aware comparison errors
    if hasattr(df["call_datetime"].dt, "tz"):
        if df["call_datetime"].dt.tz is not None:
            df["call_datetime"] = df["call_datetime"].dt.tz_localize(None)

    # ---- 5) Fix call_type classification ----
    df["call_type"] = "Standard"
    df.loc[df.get("is_voicemail", False) == True, "call_type"] = "Voicemail"
    df.loc[df.get("duration_seconds", 999) < 1, "call_type"] = "Too Short"
    df.loc[df.get("status", "").str.lower() == "no-answer", "call_type"] = "No Answer"

    # ---- 6) Guarantee coaching score columns ----
    score_cols = [
        "coaching_opening_score",
        "coaching_discovery_score",
        "coaching_value_score",
        "coaching_closing_score",
        "coaching_total_score",
    ]
    for c in score_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df
