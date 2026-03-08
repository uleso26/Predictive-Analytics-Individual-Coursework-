"""
Leakage checks, shape validation, and horizon integrity tests.

These tests verify that the data preparation pipeline produces
valid horizon datasets without temporal leakage.
"""

import os
import sys

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    FORBIDDEN,
    FOUNDING_SAFE,
    FIRST_FUNDING_SAFE,
    SNAPSHOT_ALL,
    LEAK_REGISTRY,
    DATE_FEATURES,
    TEMPORAL_SPLIT,
)
from src.preprocessing import (
    run_cleaning_pipeline,
    filter_terminal,
    build_h1_features,
    build_h2_features,
    build_h3_features,
    temporal_split,
)
from src.features import engineer_all_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "investments_VC 2.csv")


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load raw CSV once for all tests."""
    return pd.read_csv(DATA_PATH, encoding="latin-1")


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    """Run cleaning pipeline once."""
    return run_cleaning_pipeline(raw_df)


@pytest.fixture(scope="module")
def terminal_df(cleaned_df):
    """Filter to terminal outcomes with all features engineered."""
    df = filter_terminal(cleaned_df)
    df = engineer_all_features(df)
    return df


@pytest.fixture(scope="module")
def horizon_tables(terminal_df):
    """Build all three horizon feature matrices."""
    X_h1 = build_h1_features(terminal_df)
    X_h2 = build_h2_features(terminal_df)
    X_h3 = build_h3_features(terminal_df)
    y = terminal_df["target"]
    return X_h1, X_h2, X_h3, y


@pytest.fixture(scope="module")
def splits(terminal_df, horizon_tables):
    """Apply temporal split to all horizons."""
    X_h1, X_h2, X_h3, y = horizon_tables
    result = {}
    for name, X in [("H1", X_h1), ("H2", X_h2), ("H3", X_h3)]:
        X_tr, X_v, X_te, y_tr, y_v, y_te = temporal_split(terminal_df, X, y)
        result[name] = (X_tr, X_v, X_te, y_tr, y_v, y_te)
    return result


# ── 1. FORBIDDEN column tests ───────────────────────────────────────────

class TestForbiddenColumns:
    """No FORBIDDEN column should appear in any horizon dataset."""

    def test_h1_no_forbidden(self, horizon_tables):
        X_h1, _, _, _ = horizon_tables
        overlap = set(X_h1.columns) & set(FORBIDDEN)
        assert overlap == set(), f"H1 contains FORBIDDEN columns: {overlap}"

    def test_h2_no_forbidden(self, horizon_tables):
        _, X_h2, _, _ = horizon_tables
        overlap = set(X_h2.columns) & set(FORBIDDEN)
        assert overlap == set(), f"H2 contains FORBIDDEN columns: {overlap}"

    def test_h3_no_forbidden(self, horizon_tables):
        _, _, X_h3, _ = horizon_tables
        overlap = set(X_h3.columns) & set(FORBIDDEN)
        assert overlap == set(), f"H3 contains FORBIDDEN columns: {overlap}"


# ── 2. Horizon leakage tests ────────────────────────────────────────────

class TestHorizonLeakage:
    """No column from a later horizon should appear in an earlier one."""

    def test_h1_no_snapshot_cols(self, horizon_tables):
        """H1 must not contain any column from SNAPSHOT_ALL - FOUNDING_SAFE."""
        X_h1, _, _, _ = horizon_tables
        snapshot_only = set(SNAPSHOT_ALL) - set(FOUNDING_SAFE)
        # Also check engineered H3/H2 features
        h3_h2_engineered = {
            "time_to_first_funding_days",
            "num_funding_types",
            "max_round_reached",
            "funding_total_clean",
        }
        leaked = set(X_h1.columns) & (snapshot_only | h3_h2_engineered)
        assert leaked == set(), f"H1 contains snapshot-level columns: {leaked}"

    def test_h2_no_snapshot_cols(self, horizon_tables):
        """H2 must not contain snapshot-only columns (except time_to_first_funding_days)."""
        _, X_h2, _, _ = horizon_tables
        snapshot_only = set(SNAPSHOT_ALL) - set(FIRST_FUNDING_SAFE)
        h3_engineered = {"num_funding_types", "max_round_reached", "funding_total_clean"}
        leaked = set(X_h2.columns) & (snapshot_only | h3_engineered)
        assert leaked == set(), f"H2 contains snapshot-level columns: {leaked}"

    def test_no_raw_dates_in_any_horizon(self, horizon_tables):
        """No raw date string column should enter any model."""
        X_h1, X_h2, X_h3, _ = horizon_tables
        raw_dates = {"founded_at", "first_funding_at", "last_funding_at"}
        for name, X in [("H1", X_h1), ("H2", X_h2), ("H3", X_h3)]:
            leaked = set(X.columns) & raw_dates
            assert leaked == set(), f"{name} contains raw date columns: {leaked}"

    def test_no_target_in_features(self, horizon_tables):
        """Target/status must not appear in any feature matrix."""
        X_h1, X_h2, X_h3, _ = horizon_tables
        for name, X in [("H1", X_h1), ("H2", X_h2), ("H3", X_h3)]:
            assert "target" not in X.columns, f"{name} contains target"
            assert "status" not in X.columns, f"{name} contains status"


# ── 3. Shape validation ─────────────────────────────────────────────────

class TestShapeValidation:
    """Row counts must be consistent across horizons and splits."""

    def test_all_horizons_same_rows(self, horizon_tables):
        X_h1, X_h2, X_h3, y = horizon_tables
        assert len(X_h1) == len(y)
        assert len(X_h2) == len(y)
        assert len(X_h3) == len(y)

    def test_h2_has_more_features_than_h1(self, horizon_tables):
        X_h1, X_h2, _, _ = horizon_tables
        assert X_h2.shape[1] > X_h1.shape[1]

    def test_h3_has_more_features_than_h2(self, horizon_tables):
        _, X_h2, X_h3, _ = horizon_tables
        assert X_h3.shape[1] > X_h2.shape[1]

    def test_split_sizes_sum_to_total(self, splits, horizon_tables):
        _, _, _, y = horizon_tables
        for name in ["H1", "H2", "H3"]:
            X_tr, X_v, X_te, y_tr, y_v, y_te = splits[name]
            total = len(X_tr) + len(X_v) + len(X_te)
            assert total == len(y), (
                f"{name}: split total {total} != dataset total {len(y)}"
            )

    def test_y_split_sizes_match_x(self, splits):
        for name in ["H1", "H2", "H3"]:
            X_tr, X_v, X_te, y_tr, y_v, y_te = splits[name]
            assert len(X_tr) == len(y_tr)
            assert len(X_v) == len(y_v)
            assert len(X_te) == len(y_te)


# ── 4. Temporal boundary tests ──────────────────────────────────────────

class TestTemporalBoundary:
    """Temporal split boundaries must be respected."""

    def test_temporal_ordering(self, terminal_df, splits):
        """Train years <= val years <= test years."""
        cfg = TEMPORAL_SPLIT
        split_year = terminal_df["first_funding_dt"].dt.year

        X_tr, _, X_te, _, _, _ = splits["H1"]
        train_idx = X_tr.index
        test_idx = X_te.index

        train_years = split_year.loc[train_idx].dropna()
        test_years = split_year.loc[test_idx].dropna()

        if len(train_years) > 0 and len(test_years) > 0:
            assert train_years.max() <= cfg["train_max_year"], (
                f"Train contains years beyond {cfg['train_max_year']}"
            )
            assert test_years.min() >= cfg["test_min_year"], (
                f"Test contains years before {cfg['test_min_year']}"
            )


# ── 5. Terminal subset tests ────────────────────────────────────────────

class TestTerminalSubset:
    """Terminal subset must only contain acquired and closed."""

    def test_only_terminal_statuses(self, terminal_df):
        statuses = terminal_df["status"].unique()
        assert set(statuses) == {"acquired", "closed"}

    def test_target_is_binary(self, terminal_df):
        assert set(terminal_df["target"].unique()) == {0, 1}

    def test_no_operating(self, terminal_df):
        assert "operating" not in terminal_df["status"].values


# ── 6. Temporal split fallback tests ──────────────────────────────────

class TestTemporalSplitFallback:
    """Rows with missing first_funding_dt should fall back to founding_year."""

    def test_founding_year_fallback_assigns_correctly(self):
        """A row with NaT first_funding_dt but valid founding_year should
        land in the split determined by founding_year, not default to train."""
        import numpy as np
        from src.config import TEMPORAL_SPLIT

        cfg = TEMPORAL_SPLIT
        # Row with founding_year in test range (>= test_min_year)
        test_year = cfg["test_min_year"]
        # Row with founding_year in val range
        val_year = cfg["val_min_year"]

        df = pd.DataFrame({
            "first_funding_dt": pd.NaT,
            "founding_year": [test_year, val_year],
        })
        X = pd.DataFrame({"feat": [1.0, 2.0]}, index=df.index)
        y = pd.Series([1, 0], index=df.index)

        X_tr, X_v, X_te, y_tr, y_v, y_te = temporal_split(df, X, y)

        # The test-year row must be in test, not train
        assert len(X_te) == 1, (
            f"Expected 1 row in test (founding_year={test_year}), got {len(X_te)}"
        )
        # The val-year row must be in val, not train
        assert len(X_v) == 1, (
            f"Expected 1 row in val (founding_year={val_year}), got {len(X_v)}"
        )
        # No rows should default to train
        assert len(X_tr) == 0, (
            f"Expected 0 rows in train (fallback should route both elsewhere), got {len(X_tr)}"
        )
