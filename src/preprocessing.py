"""
Data cleaning, encoding, splitting, and horizon table construction.

All functions operate on DataFrames and return DataFrames.
No side effects (no file I/O). Notebook calls these, then saves.
"""

import numpy as np
import pandas as pd

from src.config import (
    FORBIDDEN,
    FOUNDING_SAFE,
    FIRST_FUNDING_SAFE,
    SNAPSHOT_ALL,
    FUNDING_ROUND_COLS,
    CATEGORICAL_FEATURES,
    DATE_FEATURES,
    TEMPORAL_SPLIT,
    TARGET,
    LEAK_REGISTRY,
)


# ── 1. Column whitespace ─────────────────────────────────────────────────

def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from column names and string values."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ── 2. Funding parser ────────────────────────────────────────────────────

def parse_funding(val):
    """Parse a single funding_total_usd value to float.

    Handles:
    - Leading/trailing whitespace
    - Commas (Western and Indian-style grouping)
    - Dash '-' as NaN (undisclosed/missing)
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s in ["-", "", "nan"]:
        return np.nan
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_funding_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply funding parser to funding_total_usd, creating funding_total_clean."""
    df = df.copy()
    df["funding_total_clean"] = df["funding_total_usd"].apply(parse_funding)
    return df


# ── 3. Blank rows and deduplication ──────────────────────────────────────

def remove_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every column is NaN."""
    return df.dropna(how="all").reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows by permalink, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset="permalink", keep="first").reset_index(drop=True)
    n_removed = before - len(df)
    if n_removed > 0:
        print(f"  Deduplication: removed {n_removed} duplicate rows by permalink")
    return df


# ── 4. Date parsing ──────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date strings to datetime, filtering out-of-range values."""
    df = df.copy()
    min_date = pd.Timestamp("1900-01-01")
    max_date = pd.Timestamp("2025-12-31")

    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        if col in df.columns:
            dt_col = col.replace("_at", "_dt")
            df[dt_col] = pd.to_datetime(df[col], errors="coerce")
            # Filter out-of-range dates (malformed entries)
            mask = df[dt_col].notna() & (
                (df[dt_col] < min_date) | (df[dt_col] > max_date)
            )
            n_bad = mask.sum()
            if n_bad > 0:
                print(f"  {col}: {n_bad} out-of-range dates set to NaT")
                df.loc[mask, dt_col] = pd.NaT
    return df


# ── 5. Quarantine impossible dates ───────────────────────────────────────

def flag_impossible_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where first_funding_at < founded_at.

    Adds boolean column `impossible_date_flag`.
    Does NOT remove rows -- the decision on how to handle them
    is made in the notebook.
    """
    df = df.copy()
    both_valid = df["founded_dt"].notna() & df["first_funding_dt"].notna()
    impossible = both_valid & (df["first_funding_dt"] < df["founded_dt"])
    df["impossible_date_flag"] = impossible
    n = impossible.sum()
    print(f"  Impossible dates (first_funding < founded): {n} rows flagged")
    return df


# ── 6. Clean funding round columns ──────────────────────────────────────

def clean_round_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all funding round columns are numeric (float64)."""
    df = df.copy()
    for col in FUNDING_ROUND_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "funding_rounds" in df.columns:
        df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")
    return df


# ── 7. Filter to terminal outcomes ───────────────────────────────────────

def filter_terminal(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only acquired/closed rows. Encode target as binary."""
    df = df[df[TARGET].isin(["acquired", "closed"])].copy()
    df["target"] = (df[TARGET] == "acquired").astype(int)
    df = df.reset_index(drop=True)
    print(f"  Terminal subset: {len(df)} rows "
          f"(acquired={df['target'].sum()}, closed={(1 - df['target']).sum()})")
    return df


# ── 8. Build horizon tables ─────────────────────────────────────────────

def _get_safe_model_features(registry_list: list, engineered: list = None) -> list:
    """Return columns safe for modelling (exclude FORBIDDEN and derive-only)."""
    derive_cols = [c for c, info in LEAK_REGISTRY.items() if info["decision"] == "derive"]
    feats = [c for c in registry_list if c not in FORBIDDEN and c not in derive_cols]
    if engineered:
        feats = feats + engineered
    return feats


def build_h1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build H1 (founding-time) feature matrix.

    Includes: geography, market, category-derived, founding date-derived.
    Excludes: raw date strings, identifiers, all funding columns.
    """
    h1_engineered = [
        "founding_year", "founding_quarter",
        "num_categories", "primary_category",
        "market_clean", "is_usa", "has_state",
    ]
    # Base columns from FOUNDING_SAFE that are actual features
    base = _get_safe_model_features(FOUNDING_SAFE)
    # Remove raw columns superseded by engineered versions:
    # - founded_month/quarter/year → founding_year, founding_quarter
    # - category_list → primary_category + num_categories
    # - market → market_clean
    superseded = ["founded_month", "founded_quarter", "founded_year",
                  "category_list", "market"]
    base = [c for c in base if c not in superseded]
    cols = base + h1_engineered
    # Only keep columns that exist in df
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def build_h2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build H2 (first-funding) feature matrix.

    H1 features + time_to_first_funding_days.
    """
    h1 = build_h1_features(df)
    h2_extra = ["time_to_first_funding_days"]
    for col in h2_extra:
        if col in df.columns:
            h1[col] = df[col].values
    return h1


def build_h3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build H3 (snapshot) feature matrix.

    H2 features + all lifetime funding aggregates + derived snapshot features.
    """
    h2 = build_h2_features(df)
    # Add snapshot-only columns
    snapshot_only = [c for c in SNAPSHOT_ALL if c not in FIRST_FUNDING_SAFE
                     and c not in FORBIDDEN]
    # Remove raw date strings and raw funding_total_usd (replaced by funding_total_clean)
    exclude_raw = set(DATE_FEATURES) | {"funding_total_usd"}
    snapshot_only = [c for c in snapshot_only if c not in exclude_raw]
    # Add engineered H3 features
    h3_engineered = ["num_funding_types", "max_round_reached", "funding_total_clean"]
    all_extra = snapshot_only + h3_engineered
    for col in all_extra:
        if col in df.columns:
            h2[col] = df[col].values
    return h2


# ── 9. Temporal split ────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """Split into train/val/test by first_funding_year.

    Per-row fallback: if a row's first_funding_dt is NaT, uses
    founding_year instead. Rows with neither date go to training.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Primary: first_funding year
    split_year = pd.Series(np.nan, index=df.index)

    if "first_funding_dt" in df.columns:
        split_year = df["first_funding_dt"].dt.year.astype("Float64")

    # Row-level fallback: use founding_year where first_funding is missing
    if "founding_year" in df.columns:
        missing = split_year.isna()
        n_fallback = missing.sum()
        if n_fallback > 0:
            split_year = split_year.fillna(df["founding_year"].astype("Float64"))
            n_still_missing = split_year.isna().sum()
            n_filled = n_fallback - n_still_missing
            if n_filled > 0:
                print(f"  {n_filled} rows used founding_year as split-year fallback")

    if split_year.isna().all():
        raise ValueError("No temporal column available for splitting")

    cfg = TEMPORAL_SPLIT
    train_mask = split_year <= cfg["train_max_year"]
    val_mask = (split_year >= cfg["val_min_year"]) & (split_year <= cfg["val_max_year"])
    test_mask = split_year >= cfg["test_min_year"]

    # Rows with no year at all go to training
    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.sum() > 0:
        print(f"  {unassigned.sum()} rows with no temporal info assigned to train")
        train_mask = train_mask | unassigned

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Temporal split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Class balance: train={y_train.mean():.3f}, val={y_val.mean():.3f}, test={y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ── 10. Full cleaning pipeline ───────────────────────────────────────────

def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute all cleaning steps in order. Returns cleaned DataFrame."""
    print("=== Data Cleaning Pipeline ===")

    print("\n1. Stripping column/value whitespace...")
    df = strip_columns(df)

    print("2. Parsing funding_total_usd...")
    df = parse_funding_column(df)

    print("3. Removing blank rows...")
    n_before = len(df)
    df = remove_blank_rows(df)
    print(f"  Removed {n_before - len(df)} blank rows → {len(df)} remaining")

    print("4. Deduplicating by permalink...")
    df = deduplicate(df)

    print("5. Parsing dates...")
    df = parse_dates(df)

    print("6. Flagging impossible dates...")
    df = flag_impossible_dates(df)

    print("7. Cleaning funding round columns...")
    df = clean_round_columns(df)

    print(f"\n=== Cleaning complete: {len(df)} rows, {len(df.columns)} columns ===")
    return df
