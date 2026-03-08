# Decision Register
## Agent Contribution Log — Appendix Material
### Horizon-Aware Startup Outcome Prediction

---

**Purpose:** Clean table of all agent contributions throughout the project, with classification of risk type and decision outcome. This file becomes the appendix directly.

---

### Pre-seeded Planning Entries (from planning conversation, before any code)

| ID | Step | Agent Claim/Contribution | Risk Type | Decision | Verification Method | Evidence | Notes |
|----|------|--------------------------|-----------|----------|---------------------|----------|-------|
| 1 | 0.1 | Agent proposed flat-feature approach using all funding columns as standard features | leakage | Rejected | Manual reasoning: funding columns are lifetime aggregates, not point-in-time | project specification §1 Horizons | Led to three-horizon framework design |
| 2 | 0.1 | Agent proposed Random Forest + XGBoost + custom NN as model stack | scope | Accepted-Modified | Compared against CatBoost (native categoricals) and TabM (ICLR 2025 SOTA) | project specification §2 Models | Replaced RF with HGB, XGBoost with CatBoost, custom NN with TabM |
| 3 | 0.1 | Agent proposed random stratified 70/15/15 train/val/test split | evaluation | Rejected | Temporal validation better reflects real-world deployment | project specification §3 Validation | Replaced with chronological split by first_funding_year |
| 4 | 0.1 | Agent did not flag right-censoring issue with 'operating' class | interpretation | Rejected | Operating firms haven't reached terminal outcome — including them contaminates target | project specification §0 Thesis | Fundamental to project framing |
| 5 | 0.1 | Agent generated unified project specification document | scope | Accepted-Modified | Verified structure against project's 6-step workflow and 4 evaluation criteria; added logging trigger rules | project specification | Plan accepted as execution blueprint with logging enhancements |

### Phase 0: Scaffolding

| ID | Step | Agent Claim/Contribution | Risk Type | Decision | Verification Method | Evidence | Notes |
|----|------|--------------------------|-----------|----------|---------------------|----------|-------|
| 6 | 0.1 | Agent generated full project scaffold: directory structure, requirements.txt, .gitignore, skeleton src modules, test files | scope | Accepted | Reviewed against project specification §4 architecture spec | Directory listing, requirements.txt | Scaffold matches project specification architecture |
| 7 | 0.2 | Agent classified all 39 CSV columns into FORBIDDEN/FOUNDING_SAFE/FIRST_FUNDING_SAFE/SNAPSHOT_ALL registries with per-column leak notes | leakage | Accepted | Cross-checked each column's temporal availability against data dictionary and horizon definitions | src/config.py LEAK_REGISTRY | Column registries form backbone of horizon enforcement |
| 8 | 0.3 | Agent generated 6 skeleton notebooks with section headers, WORKFLOW_LOG.md, DECISION_REGISTER.md with pre-seeded entries, and report_draft.md with section headers and [FILL] prompts | scope | Accepted | Verified notebook sections match 6-step structure; confirmed 5 pre-seeded entries accurate; checked report word budgets | Notebook files, log files, report template | All templates faithfully reflect project specification |
| 9 | 0.4 | Agent wrote LEAK_REGISTRY schema comment listing only 3 decision values (include/exclude/derive), omitting include_h3_only and target | coding_bug | Rejected | User compared comment against actual dictionary values | src/config.py L120-122 | Comment corrected to document all 5 actual decision values |
| 10 | 0.5 | Agent used S0/S1/S2/S3 notation in Decision Register instead of § section symbol | scope | Rejected | User compared against project specification notation conventions | DECISION_REGISTER.md | Corrected to §0/§1/§2/§3; added phase separator headings |
| 11 | 0.6 | Agent combined section 3.5 header with 3.5.1 H1 subsection in single notebook cell | scope | Accepted-Modified | User noted H1 subsection was not visually distinct | 03_data_preparation.ipynb cell-14 | Split into separate cells for clearer structure |
| 12 | 0.7 | Agent created model subsection headers without descriptive context from project specification | scope | Accepted-Modified | User requested master-plan-level detail for DummyClassifier (H1/H2) and LR (H3) | 04_modelling.ipynb | Added strategy parameters, expected values, and purpose statements |
| 13 | 0.8 | Agent used insufficiently emphatic language for date-column drop requirement in config comments | leakage | Accepted-Modified | User flagged risk of raw date strings entering model inputs | src/config.py FOUNDING_SAFE, FIRST_FUNDING_SAFE | Comments now explicitly state DROP before modelling |

### Phase 1: Problem Framing

| ID | Step | Agent Claim/Contribution | Risk Type | Decision | Verification Method | Evidence | Notes |
|----|------|--------------------------|-----------|----------|---------------------|----------|-------|
| 14 | 1.1 | Agent generated full problem framing notebook content: problem definition, censoring argument, horizon framework, metric definitions, hard constraints, leak registry explanation, agent governance plan, model stack justification, validation strategy, and summary | interpretation | Accepted | Review analytical content against project specification §0–§3 specifications; verify censoring argument, metric rationale, and governance boundaries | 01_problem_framing.ipynb (12 cells) | Content implements all Phase 1 requirements with analytical depth suitable for report material |
| 15 | 1.2 | Agent wrote speculative claim in censoring argument: "Some operating firms are pre-acquisition targets... Others are pre-failure firms..." — not verifiable from the data | interpretation | Rejected | User identified claim as ungrounded speculation during review | 01_problem_framing.ipynb §1.2 point 3 | Rewrote to data-grounded statement; added HTML comment flagging for report tightening |

### Phase 2: Exploratory Data Analysis

| ID | Step | Agent Claim/Contribution | Risk Type | Decision | Verification Method | Evidence | Notes |
|----|------|--------------------------|-----------|----------|---------------------|----------|-------|
| 16 | 2.1 | Agent wrote blank-row removal (4,856 rows) and duplicate detection code | data_cleaning | Accepted | Verify 54,294 − 4,856 = 49,438; check duplicate pairs | 02_eda.ipynb §2.2.1–2.2.2 | 4,856 blank rows removed; 4 duplicate rows (2 pairs) found |
| 17 | 2.2 | Agent wrote `parse_funding()` to handle Indian-style commas, whitespace, and dash-as-missing in `funding_total_usd` | data_cleaning | Accepted | Spot-check parsed values against raw strings | 02_eda.ipynb §2.2.4 | 8,531 dash-as-missing rows; 40,907 successfully parsed to numeric |
| 18 | 2.3 | Agent wrote missingness summary code and interpretive markdown | data_cleaning | Accepted-Modified | User found missingness was computed after column derivation, reporting 15/43 instead of 13/39 raw columns | 02_eda.ipynb §2.2.3 | Moved before derivation; now reports on original 39 columns only |
| 19 | 2.4 | Agent wrote date parsing and impossible-date detection code; wrote label distribution summary | data_cleaning | Accepted | Verify impossible date count and terminal subset size | 02_eda.ipynb §2.2.5–2.2.6 | 2,745 impossible dates; terminal subset 6,295 (59:41); 1,314 missing status |
| 20 | 2.5 | Agent designed and coded all 12 EDA figures with consistent styling; composed EDA summary section | interpretation | Accepted | Review all 12 figures for accuracy, readability, and consistency with project specification | figures/01–12_*.png, 02_eda.ipynb §2.3–2.4 | All figures saved; summary synthesises key findings across all 12 |
| 21 | 2.6 | Agent wrote Figure 12 date validity scatter without filtering out-of-range dates; 67+ malformed dates (years 1785–1888) caused matplotlib ValueError | coding_bug | Accepted-Modified | User caught ValueError during notebook execution; NotebookEdit changes overwritten by VSCode auto-save — required direct JSON fix | 02_eda.ipynb cell-40 | Added 1900–2025 filter and numeric year conversion; required 3 iterations due to editor conflict |
| 22 | 2.7 | Agent produced incorrect numbers in logs and notebook: 1 duplicate pair (actual: 2 pairs/4 rows), 2,739 impossible dates (actual: 2,745), 6,170 missing status (actual: 1,314). Figure 09 counted derive-then-drop columns in feature totals (10/11/33 instead of 9/9/31). | data_cleaning | Rejected | User cross-checked log numbers against notebook outputs; found 5 discrepancies | WORKFLOW_LOG.md, 02_eda.ipynb | All numbers corrected; missingness reordered; Figure 09 fixed; 12_test.png deleted |
