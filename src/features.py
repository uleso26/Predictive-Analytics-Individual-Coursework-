"""
Feature engineering functions.

All functions take a DataFrame and return a DataFrame with new columns added.
Organised by horizon level.
"""

import numpy as np
import pandas as pd

from src.config import FUNDING_ROUND_COLS


# ── H1: Founding-time features ───────────────────────────────────────────

def add_founding_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive year and quarter from founded_dt."""
    df = df.copy()
    dt = df["founded_dt"]
    df["founding_year"] = dt.dt.year.astype("Int64")
    df["founding_quarter"] = dt.dt.quarter.astype("Int64")
    return df


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract primary category and category count from pipe-separated category_list."""
    df = df.copy()

    def _extract_primary(val):
        if pd.isna(val):
            return np.nan
        parts = [p.strip() for p in str(val).split("|") if p.strip()]
        return parts[0] if parts else np.nan

    def _count_categories(val):
        if pd.isna(val):
            return 0
        parts = [p.strip() for p in str(val).split("|") if p.strip()]
        return len(parts)

    df["primary_category"] = df["category_list"].apply(_extract_primary)
    df["num_categories"] = df["category_list"].apply(_count_categories)
    return df


def add_market_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned market column (stripped and lowered)."""
    df = df.copy()
    df["market_clean"] = df["market"].str.strip().str.lower()
    return df


def add_geography_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_usa and has_state binary flags."""
    df = df.copy()
    df["is_usa"] = (df["country_code"] == "USA").astype(int)
    df["has_state"] = df["state_code"].notna().astype(int)
    return df


# ── H2: First-funding features ──────────────────────────────────────────

def add_time_to_first_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days from founding to first funding.

    Only valid for rows with both dates and non-negative lag.
    Rows with impossible dates (negative lag) get NaN.

    NOTE: Extreme outliers exist (max ~37,466 days / ~102 years),
    likely from imprecise founding dates. Consider capping or log-
    transforming this feature during modelling (Phase 4/5).
    """
    df = df.copy()
    if "founded_dt" in df.columns and "first_funding_dt" in df.columns:
        lag = (df["first_funding_dt"] - df["founded_dt"]).dt.days
        # Set negative lags to NaN (impossible dates)
        lag = lag.where(lag >= 0, other=np.nan)
        df["time_to_first_funding_days"] = lag
    else:
        df["time_to_first_funding_days"] = np.nan
    return df


# ── H3: Snapshot features ───────────────────────────────────────────────

def add_num_funding_types(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many distinct funding type columns are non-zero."""
    df = df.copy()
    type_cols = [c for c in FUNDING_ROUND_COLS if c in df.columns]
    df["num_funding_types"] = (df[type_cols] > 0).sum(axis=1)
    return df


def add_max_round_reached(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal encoding of highest lettered round reached (A=1, B=2, ..., H=8).

    0 means no lettered round was reached.
    """
    df = df.copy()
    round_letters = ["round_A", "round_B", "round_C", "round_D",
                     "round_E", "round_F", "round_G", "round_H"]
    present = [c for c in round_letters if c in df.columns]
    if present:
        # For each row, find the highest round with non-zero amount
        max_round = pd.Series(0, index=df.index)
        for i, col in enumerate(present, start=1):
            max_round = max_round.where(~(df[col] > 0), other=i)
        df["max_round_reached"] = max_round
    else:
        df["max_round_reached"] = 0
    return df


# ── Composite pipelines ─────────────────────────────────────────────────

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps. Returns df with new columns added."""
    print("=== Feature Engineering ===")

    print("1. Founding date features (year, quarter)...")
    df = add_founding_date_features(df)

    print("2. Category features (primary_category, num_categories)...")
    df = add_category_features(df)

    print("3. Market cleaning...")
    df = add_market_clean(df)

    print("4. Geography flags (is_usa, has_state)...")
    df = add_geography_flags(df)

    print("5. Time-to-first-funding (H2)...")
    df = add_time_to_first_funding(df)

    print("6. Num funding types (H3)...")
    df = add_num_funding_types(df)

    print("7. Max round reached (H3)...")
    df = add_max_round_reached(df)

    new_cols = [
        "founding_year", "founding_quarter",
        "num_categories", "primary_category", "market_clean",
        "is_usa", "has_state",
        "time_to_first_funding_days",
        "num_funding_types", "max_round_reached",
    ]
    existing = [c for c in new_cols if c in df.columns]
    print(f"\n=== Feature engineering complete: {len(existing)} new columns added ===")
    return df
