"""
Column registries, horizon definitions, and leak registry.

Every column in the raw CSV is classified by temporal availability.
This module is the single source of truth for what features are allowed
in each prediction horizon.
"""

# ---------------------------------------------------------------------------
# Raw column names (as they appear after stripping whitespace)
# ---------------------------------------------------------------------------
TARGET = "status"

# ---------------------------------------------------------------------------
# FORBIDDEN — never used as features under any horizon
# ---------------------------------------------------------------------------
FORBIDDEN = [
    "permalink",        # identifier, no predictive value
    "name",             # identifier, no predictive value
    "homepage_url",     # identifier, no predictive value
    "post_ipo_equity",  # post-outcome data — textbook leakage
    "post_ipo_debt",    # post-outcome data — textbook leakage
]

# ---------------------------------------------------------------------------
# H1: FOUNDING-TIME SAFE — available at the moment of founding
# ---------------------------------------------------------------------------
FOUNDING_SAFE = [
    "category_list",    # business category (pipe-separated)
    "market",           # market segment
    "country_code",     # geography
    "state_code",       # US state (if applicable)
    "region",           # geographic region
    "city",             # city
    "founded_at",       # raw date string — derive features (year/quarter/month), then DROP before modelling
    "founded_month",    # derived from founded_at
    "founded_quarter",  # derived from founded_at
    "founded_year",     # derived from founded_at
]

# ---------------------------------------------------------------------------
# H2: FIRST-FUNDING SAFE — H1 features + first funding info
# ---------------------------------------------------------------------------
FIRST_FUNDING_SAFE = FOUNDING_SAFE + [
    "first_funding_at",  # raw date string — derive time_to_first_funding_days, then DROP before modelling
    # Engineered features derived at H2 level:
    # - time_to_first_funding_days (first_funding_at - founded_at)
]

# ---------------------------------------------------------------------------
# H3: SNAPSHOT ALL — all features including lifetime aggregates
# ---------------------------------------------------------------------------
SNAPSHOT_ALL = FIRST_FUNDING_SAFE + [
    "last_funding_at",
    "funding_rounds",
    "funding_total_usd",
    "seed",
    "venture",
    "equity_crowdfunding",
    "undisclosed",
    "convertible_note",
    "debt_financing",
    "angel",
    "grant",
    "private_equity",
    "secondary_market",
    "product_crowdfunding",
    "round_A",
    "round_B",
    "round_C",
    "round_D",
    "round_E",
    "round_F",
    "round_G",
    "round_H",
]

# ---------------------------------------------------------------------------
# Funding round columns (subset of snapshot-only features)
# ---------------------------------------------------------------------------
FUNDING_ROUND_COLS = [
    "seed", "venture", "equity_crowdfunding", "undisclosed",
    "convertible_note", "debt_financing", "angel", "grant",
    "private_equity", "secondary_market", "product_crowdfunding",
    "round_A", "round_B", "round_C", "round_D",
    "round_E", "round_F", "round_G", "round_H",
]

# ---------------------------------------------------------------------------
# Categorical vs numeric classification
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "category_list",
    "market",
    "country_code",
    "state_code",
    "region",
    "city",
]

DATE_FEATURES = [
    "founded_at",
    "first_funding_at",
    "last_funding_at",
]

# ---------------------------------------------------------------------------
# Temporal split boundaries
# ---------------------------------------------------------------------------
TEMPORAL_SPLIT = {
    "train_max_year": 2008,      # train: first_funding_year <= 2008
    "val_min_year": 2009,        # val:   2009-2010
    "val_max_year": 2010,
    "test_min_year": 2011,       # test:  first_funding_year >= 2011
}

# ---------------------------------------------------------------------------
# LEAK REGISTRY — per-column leakage classification
# ---------------------------------------------------------------------------
# safe_at: "founding" | "first_funding" | "snapshot" | "never"
# risk_notes: why this column is/isn't safe at earlier horizons
# decision values:
#   "include"        — used as a feature directly
#   "include_h3_only"— only available at snapshot horizon (H3)
#   "exclude"        — never enters any model (identifier or post-outcome)
#   "derive"         — used to engineer features, then dropped before modelling
#   "target"         — the prediction target (status column)

LEAK_REGISTRY = {
    "permalink":            {"safe_at": "never",          "risk_notes": "Identifier only",                                      "decision": "exclude"},
    "name":                 {"safe_at": "never",          "risk_notes": "Identifier only",                                      "decision": "exclude"},
    "homepage_url":         {"safe_at": "never",          "risk_notes": "Identifier only",                                      "decision": "exclude"},
    "category_list":        {"safe_at": "founding",       "risk_notes": "Business category set at founding",                    "decision": "include"},
    "market":               {"safe_at": "founding",       "risk_notes": "Market segment typically known at founding",            "decision": "include"},
    "funding_total_usd":    {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate — accumulates over entire life",     "decision": "include_h3_only"},
    "status":               {"safe_at": "never",          "risk_notes": "TARGET variable",                                      "decision": "target"},
    "country_code":         {"safe_at": "founding",       "risk_notes": "Geography known at founding",                          "decision": "include"},
    "state_code":           {"safe_at": "founding",       "risk_notes": "Geography known at founding",                          "decision": "include"},
    "region":               {"safe_at": "founding",       "risk_notes": "Geography known at founding",                          "decision": "include"},
    "city":                 {"safe_at": "founding",       "risk_notes": "Geography known at founding",                          "decision": "include"},
    "funding_rounds":       {"safe_at": "snapshot",       "risk_notes": "Count of all rounds — lifetime aggregate",             "decision": "include_h3_only"},
    "founded_at":           {"safe_at": "founding",       "risk_notes": "Founding date",                                        "decision": "derive"},
    "founded_month":        {"safe_at": "founding",       "risk_notes": "Derived from founded_at",                              "decision": "include"},
    "founded_quarter":      {"safe_at": "founding",       "risk_notes": "Derived from founded_at",                              "decision": "include"},
    "founded_year":         {"safe_at": "founding",       "risk_notes": "Derived from founded_at",                              "decision": "include"},
    "first_funding_at":     {"safe_at": "first_funding",  "risk_notes": "Date of first funding event",                          "decision": "derive"},
    "last_funding_at":      {"safe_at": "snapshot",       "risk_notes": "Only known after all funding complete",                "decision": "include_h3_only"},
    "seed":                 {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "venture":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "equity_crowdfunding":  {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "undisclosed":          {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "convertible_note":     {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "debt_financing":       {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "angel":                {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "grant":                {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "private_equity":       {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "post_ipo_equity":      {"safe_at": "never",          "risk_notes": "Post-outcome data — occurs after success",             "decision": "exclude"},
    "post_ipo_debt":        {"safe_at": "never",          "risk_notes": "Post-outcome data — occurs after success",             "decision": "exclude"},
    "secondary_market":     {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "product_crowdfunding": {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_A":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_B":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_C":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_D":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_E":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_F":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_G":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
    "round_H":              {"safe_at": "snapshot",       "risk_notes": "Lifetime aggregate amount",                            "decision": "include_h3_only"},
}

# ---------------------------------------------------------------------------
# Model stack
# ---------------------------------------------------------------------------
MODELS_H1_H2 = [
    "DummyClassifier",
    "LogisticRegression",
    "HistGradientBoostingClassifier",
    "CatBoostClassifier",
    "TabM",
]

MODELS_H3 = [
    "LogisticRegression",
    "CatBoostClassifier",
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "roc_auc"

METRICS = [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "f1",
    "brier_score",
    "ece",
]
