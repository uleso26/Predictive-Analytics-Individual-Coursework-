# Workflow Log
## Horizon-Aware Startup Outcome Prediction
### Individual Coursework

---

**Purpose:** Chronological build diary documenting every decision, action, and agent interaction throughout the project. Each entry maps to a commit and feeds the Decision Register appendix.

**Entry format:**
```
## Step X.Y: [Short Title]

**Action:** What was done
**Agent Role:** What the AI agent contributed
**My Verification:** What you personally checked/changed
**Decision:** Accepted / Accepted-Modified / Rejected / Deferred — one-line rationale
**Risk Type:** leakage | evaluation | coding_bug | data_cleaning | interpretation | citation | scope
**Commit:** `abc1234` — short message
**Report Note:** 1-2 sentence insight for the corresponding report section
```

---

## Step 0.1: Create project directory structure and scaffold

**Action:** Created full directory structure: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `figures/`, `tests/`, `logs/`, `report/figures/`. Created `requirements.txt` with all dependencies (pandas, numpy, scikit-learn, catboost, shap, matplotlib, seaborn, umap-learn, tabm, optuna, jupyter, pytest). Created `.gitignore` covering data files, Python caches, Jupyter checkpoints, IDE files, model artefacts, and secrets. Created `src/__init__.py` and skeleton modules (`preprocessing.py`, `features.py`, `models.py`, `evaluation.py`). Created empty test files (`test_preprocessing.py`, `test_evaluation.py`).
**Agent Role:** Claude Code generated the full scaffold based on the project specification architecture specification. Agent proposed all file contents.
**My Verification:** [TO VERIFY] Review directory structure matches project specification Section 4 exactly. Check .gitignore covers all sensitive patterns. Confirm requirements.txt includes all needed packages.
**Decision:** Accepted — scaffold follows project specification architecture exactly.
**Risk Type:** scope
**Commit:** [PENDING — user will commit]
**Report Note:** Project scaffold established with horizon-aware directory structure separating raw data, processed horizon datasets, and reusable source modules.

---

## Step 0.2: Create src/config.py with column registries and leak registry

**Action:** Created `src/config.py` containing: (1) `FORBIDDEN` list — 5 columns never used as features (permalink, name, homepage_url, post_ipo_equity, post_ipo_debt); (2) `FOUNDING_SAFE` — 10 columns available at founding time; (3) `FIRST_FUNDING_SAFE` — founding columns + first_funding_at; (4) `SNAPSHOT_ALL` — all features including lifetime funding aggregates; (5) `LEAK_REGISTRY` — per-column dictionary with `safe_at`, `risk_notes`, and `decision` fields for all 39 raw columns; (6) `TEMPORAL_SPLIT` config; (7) Model stack and metric definitions.
**Agent Role:** Claude Code constructed all registries from the raw CSV column headers and the project specification horizon definitions. Agent classified each column's temporal availability.
**My Verification:** [TO VERIFY] Check every column classification against the raw CSV. Verify no funding aggregate column appears in FOUNDING_SAFE or FIRST_FUNDING_SAFE. Confirm FORBIDDEN list is complete.
**Decision:** Accepted — column classifications are consistent with the horizon framework. Each column's `safe_at` level is defensible.
**Risk Type:** leakage
**Commit:** [PENDING — user will commit]
**Report Note:** A comprehensive leak registry was built classifying all 39 raw columns by temporal availability, forming the backbone of the three-horizon framework.

---

## Step 0.3: Create skeleton notebooks, logs, and report template

**Action:** Created 6 skeleton Jupyter notebooks (`01_problem_framing.ipynb` through `06_final_solution.ipynb`) with section headers matching the 6 steps in project specification. Created `logs/WORKFLOW_LOG.md` with entry format template. Created `logs/DECISION_REGISTER.md` with table headers and 5 pre-seeded planning entries documenting agent interactions from the planning phase. Created `report/report_draft.md` with 7 report sections, word budgets, and [FILL] prompts.
**Agent Role:** Claude Code generated all skeleton files from the project specification specifications, including the pre-seeded Decision Register entries.
**My Verification:** [TO VERIFY] Check notebook section headers match the step structure. Verify 5 pre-seeded Decision Register entries are accurate. Confirm report sections match the word budget in project specification Section 7.
**Decision:** Accepted — all skeletons faithfully reflect the project specification structure.
**Risk Type:** scope
**Commit:** [PENDING — user will commit]
**Report Note:** Project templates established with pre-seeded agent governance entries documenting 5 planning-phase decisions where agent proposals were rejected or modified.

---

## Step 0.4: Fix LEAK_REGISTRY schema comment to match actual decision values

**Action:** Updated the comment block above `LEAK_REGISTRY` in `src/config.py` to document all five actual `decision` values used: `include`, `include_h3_only`, `exclude`, `derive`, `target`. Previously the comment only listed three values (`include`, `exclude`, `derive`), omitting `include_h3_only` and `target`.
**Agent Role:** Claude Code originally generated the comment with an incomplete set of decision values that did not match the values used in the dictionary entries below it.
**My Verification:** User caught the mismatch between the schema comment and the actual values used in the registry entries.
**Decision:** Rejected original comment — corrected to match actual usage.
**Risk Type:** coding_bug
**Commit:** [PENDING — user will commit]
**Report Note:** Schema documentation corrected to accurately reflect the five-value decision taxonomy used in the leak registry.

---

## Step 0.5: Fix DECISION_REGISTER.md format — use § notation and add phase separator

**Action:** Changed all section references in pre-seeded entries from `S0`/`S1`/`S2`/`S3` to `§0`/`§1`/`§2`/`§3` to match project specification notation. Added section headings ("Pre-seeded Planning Entries" and "Phase 0: Scaffolding") to visually separate the 5 planning entries from the 3 scaffolding entries. Each block now has its own table header for clarity.
**Agent Role:** Claude Code used `S0`/`S1`/etc. shorthand in the Evidence column instead of the `§` section symbol used in the project specification.
**My Verification:** User identified the inconsistency and requested the fix.
**Decision:** Rejected original format — corrected to use § notation with clear phase separators.
**Risk Type:** scope
**Commit:** [PENDING — user will commit]
**Report Note:** Decision register formatting aligned with project specification conventions; phase boundaries now clearly visible.

---

## Step 0.6: Add explicit H1 subsection under 3.5 in notebook 03

**Action:** Split cell-14 in `03_data_preparation.ipynb` so that section 3.5 ("Build Three Horizon Tables") has its own introductory cell, followed by a separate cell for "3.5.1 H1: Founding Dataset" with a note that raw date columns must be dropped before modelling.
**Agent Role:** Claude Code originally combined the section header (3.5) and the first subsection (3.5.1) in a single cell, making the H1 subsection less visually distinct.
**My Verification:** User identified the structural issue during review.
**Decision:** Accepted-Modified — split into two cells for clearer notebook structure.
**Risk Type:** scope
**Commit:** [PENDING — user will commit]
**Report Note:** Notebook structure refined to give each horizon table its own clearly delineated subsection.

---

## Step 0.7: Enhance model subsection descriptions in notebook 04

**Action:** Enhanced existing subsection cells in `04_modelling.ipynb`: (1) Added descriptive text to DummyClassifier under H1 (strategy details, expected AUC ~0.50); (2) Added descriptive text to DummyClassifier under H2 (baseline verification); (3) Added descriptive text to LogisticRegression under H3 (purpose: quantify AUC inflation). All subsections were already present — this fix added master-plan-level detail to each.
**Agent Role:** Claude Code originally created the subsection headers without descriptive context from the project specification (e.g., strategy parameters, purpose statements).
**My Verification:** User requested the additions to ensure each subsection clearly states its role per the project specification.
**Decision:** Accepted-Modified — subsections enriched with project specification context.
**Risk Type:** scope
**Commit:** [PENDING — user will commit]
**Report Note:** Modelling notebook subsections now contain explicit purpose statements matching the project specification's model stack rationale.

---

## Step 0.8: Add date-column drop warnings in src/config.py

**Action:** Updated inline comments for `founded_at` in `FOUNDING_SAFE` and `first_funding_at` in `FIRST_FUNDING_SAFE` to explicitly state that these are raw date strings used to derive features and must be DROPPED before modelling. Previous comments mentioned "used to derive features, then dropped" but the new wording is more emphatic and consistent.
**Agent Role:** Claude Code's original comments noted the derive-then-drop pattern but did not use sufficiently strong language to prevent accidental inclusion of raw date strings in model inputs.
**My Verification:** User flagged that the comments should be unambiguous about the drop requirement.
**Decision:** Accepted-Modified — strengthened wording to prevent raw date leakage into models.
**Risk Type:** leakage
**Commit:** [PENDING — user will commit]
**Report Note:** Raw date columns explicitly marked as derive-only in config, reinforcing the principle that no unparsed temporal strings should enter model feature matrices.

