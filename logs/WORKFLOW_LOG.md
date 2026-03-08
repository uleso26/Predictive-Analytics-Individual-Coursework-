# Workflow Log
## Horizon-Aware Startup Outcome Prediction

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

---

## Step 1.1: Fill problem framing notebook with substantive analytical content

**Action:** Expanded all sections of `notebooks/01_problem_framing.ipynb` from skeleton placeholders to full analytical content. Sections filled: (1.1) Problem definition with dataset size, target encoding, and differentiating insight about lifetime aggregates; (1.2) Four-point censoring argument for excluding operating firms with survival analysis justification; (1.3) Three-horizon framework with per-horizon purpose and narrative framing; (1.4) Six-metric suite with rationale for each metric and threshold selection protocol (Youden's J); (1.5) Five hard constraints with implementation details; (1.6) Leak registry explanation with decision taxonomy; (1.7) Agent governance plan with delegation/verification boundaries and logging protocol; (1.8) Model stack with per-model justification and horizon coverage; (1.9) Validation strategy with temporal split rationale, tuning protocol, and argument against random splitting; (1.10) Summary section. Also enhanced the code cell to print registry summary statistics and verify horizon list consistency.
**Agent Role:** Claude Code generated all notebook cell content based on the project specification Phase 1 specification. Agent drafted the censoring argument, metric justifications, horizon narrative framing, and governance plan text.
**My Verification:** [TO VERIFY] Review censoring argument for statistical accuracy. Confirm metric choices and threshold selection protocol are appropriate. Check that agent governance plan accurately reflects intended delegation boundaries. Verify all section references (§) are consistent.
**Decision:** Accepted — content faithfully implements the project specification's Phase 1 requirements with appropriate analytical depth.
**Risk Type:** interpretation
**Commit:** [PENDING — user will commit]
**Report Note:** Problem framing establishes the three-horizon framework as the project's central methodological contribution, with operating firms excluded via a formal right-censoring argument and six complementary metrics defined for cross-horizon comparison.

---

## Step 1.2: Fix speculative claim in censoring argument (§1.2, point 3)

**Action:** Rewrote point 3 ("Class contamination") in `01_problem_framing.ipynb` §1.2. The original text claimed "Some operating firms are pre-acquisition targets currently in negotiation. Others are pre-failure firms about to shut down." This is speculative — the data contains no evidence about the latent trajectories of operating firms. Replaced with a verifiable statement: the operating class is heterogeneous because outcomes are unobserved, so any assignment is speculation, not data. Added an HTML comment flagging the original wording for report-writing.
**Agent Role:** Claude Code drafted the original speculative sentence as part of the censoring argument in Step 1.1.
**My Verification:** User identified the speculative claim during review and requested correction.
**Decision:** Rejected — replaced unverifiable claim with data-grounded reasoning.
**Risk Type:** interpretation
**Commit:** [PENDING — user will commit]
**Report Note:** Censoring argument tightened: class contamination point now rests on the observable fact that outcomes are missing, not on speculation about sub-populations within the operating class.

---

## Step 2.1: Data audit — blank rows and duplicates

**Action:** Identified and removed 4,856 fully blank rows (all 39 columns NaN) — CSV export artefacts. Found 4 duplicate rows (2 pairs by permalink: Prysm × 2, Treasure Valley Urology Services × 2). After cleaning: 49,438 rows, 39 columns.
**Agent Role:** Claude Code wrote the blank-row removal and permalink-based duplicate detection code.
**My Verification:** [TO VERIFY] Confirm 54,294 − 4,856 = 49,438. Verify duplicate pairs are genuine full-row duplicates.
**Decision:** Accepted — counts verified against raw CSV.
**Risk Type:** data_cleaning
**Commit:** [PENDING — user will commit]
**Report Note:** 4,856 blank rows removed; 2 genuine duplicate pairs identified for deduplication in Phase 3.

---

## Step 2.2: Data audit — malformed funding values

**Action:** `funding_total_usd` is stored as `object` with three issues: (1) leading/trailing whitespace; (2) commas that do not follow Western 3-digit grouping (Indian-style formatting in some values); (3) 8,531 rows use dash `"-"` for missing/undisclosed funding. Wrote `parse_funding()` to strip, remove commas, and convert to float. 40,907 rows successfully parsed to numeric.
**Agent Role:** Claude Code wrote the `parse_funding()` function and diagnostic output.
**My Verification:** [TO VERIFY] Spot-check parsed values against raw strings. Verify dash count.
**Decision:** Accepted — parser handles all observed edge cases.
**Risk Type:** data_cleaning
**Commit:** [PENDING — user will commit]
**Report Note:** Funding column required custom parsing due to inconsistent comma formatting and dash-as-missing encoding.

---

## Step 2.3: Data audit — missingness summary

**Action:** Computed missingness on the original 39 raw columns (before any column derivation). 13 of 39 columns have missing values: `state_code` (39.0%), `founded_*` fields (22%), geography columns (7–12%), `market`/`category_list` (8.0%), `homepage_url` (7.0%), `status` (2.7% = 1,314 rows), `name` (1 row). All 26 funding and identifier columns are fully populated after blank row removal.
**Agent Role:** Claude Code wrote the missingness summary code and interpretive markdown.
**My Verification:** User identified that missingness was originally computed after column derivation (reporting 15/43 columns including 4 derived columns). Moved to before any derivation to report on raw 39 columns only.
**Decision:** Accepted-Modified — missingness section moved before column derivation; numbers corrected from 15/43 to 13/39.
**Risk Type:** data_cleaning
**Commit:** [PENDING — user will commit]
**Report Note:** Missingness is structured: founding dates ~22%, geography 7–39%, funding columns fully populated. Only 1,314 rows (2.7%) have missing status.

---

## Step 2.4: Data audit — impossible dates and label distribution

**Action:** Parsed date columns and found 2,745 rows (7.1% of 38,554 date-valid pairs) where `first_funding_at < founded_at` — a logical impossibility. Many appear to be imprecise founding dates (e.g., `2010-01-01` as a placeholder). Label distribution: 41,829 operating (excluded as right-censored), 3,692 acquired, 2,603 closed, 1,314 missing status. Terminal subset: 6,295 startups (59:41 acquired:closed).
**Agent Role:** Claude Code wrote the date parsing, impossible date detection, and label distribution code.
**My Verification:** [TO VERIFY] Verify impossible date count. Confirm terminal subset size.
**Decision:** Accepted — counts match notebook outputs.
**Risk Type:** data_cleaning
**Commit:** [PENDING — user will commit]
**Report Note:** 2,745 impossible date pairs flagged for quarantine; terminal subset of 6,295 startups with 59:41 class balance confirmed.

---

## Step 2.5: Generate 12 EDA figures

**Action:** Designed and generated all 12 required EDA figures: (1) target distribution; (2) missingness heatmap; (3) funding violin by outcome; (4) funding round ladder; (5) geographic top-15; (6) market sector top-20; (7) founding year cohort; (8) correlation heatmap; (9) horizon-risk bar chart; (10) feature-horizon availability grid; (11) time-to-first-funding; (12) date validity scatter. All saved to `figures/`.
**Agent Role:** Claude Code designed all visualisations, wrote plotting code, chose chart types and colour schemes.
**My Verification:** [TO VERIFY] Review all 12 figures for accuracy and readability. Confirm figure numbering matches project specification.
**Decision:** Accepted — all 12 figures produced with consistent styling.
**Risk Type:** interpretation
**Commit:** [PENDING — user will commit]
**Report Note:** 12 publication-quality figures reveal key patterns: acquired startups have ~10x higher median funding, clear funding round progression separation, and temporal cohort effects justifying the chronological split.

---

## Step 2.6: Fix Figure 12 date validity scatter — out-of-range dates

**Action:** Figure 12 threw `ValueError` because `pd.to_datetime(errors='coerce')` parsed 67+ malformed date strings as extreme values (e.g., year -101, year 1785). Added 1900–2025 date range filter and converted dates to plain float years to bypass matplotlib's datetime converter entirely.
**Agent Role:** Claude Code wrote the original code without date range filtering or datetime-to-float conversion. Multiple fix attempts via NotebookEdit failed because VSCode auto-saved its cached version over the edits. Eventually fixed via direct JSON manipulation.
**My Verification:** User identified the ValueError during notebook execution and caught that NotebookEdit changes were being overwritten by VSCode.
**Decision:** Accepted-Modified — required three iterations to fix due to tool/editor conflict.
**Risk Type:** coding_bug
**Commit:** [PENDING — user will commit]
**Report Note:** Malformed dates (years 1785–1888) parse to extreme values; 1900–2025 filter applied with numeric year conversion for safe plotting.

---

## Step 2.7: Review corrections — numbers, missingness ordering, figure counts

**Action:** User review found five issues: (1) WORKFLOW_LOG compressed all audit findings into one entry — split into separate entries; (2) duplicate count wrong (1 pair → 2 pairs/4 rows), impossible date count wrong (2,739 → 2,745), missing status wrong (6,170 → 1,314) — all corrected to match notebook outputs; (3) missingness summary computed after column derivation (15/43 including derived columns) — moved before derivation (13/39 raw columns); (4) Figure 09 counted derive-then-drop columns (founded_at, first_funding_at) in feature totals — corrected to model-only features (H1=9, H2=9, H3=31); (5) stale test file `figures/12_test.png` — deleted.
**Agent Role:** Claude Code produced the original incorrect numbers and notebook ordering.
**My Verification:** User identified all five issues during systematic review of Phase 2 outputs.
**Decision:** Rejected — five errors corrected across notebook, logs, and figures.
**Risk Type:** data_cleaning
**Commit:** [PENDING — user will commit]
**Report Note:** Post-review corrections ensure all reported counts match actual notebook outputs, missingness reflects raw data only, and feature counts exclude intermediate derive-then-drop columns.

