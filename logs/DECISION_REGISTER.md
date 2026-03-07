# Decision Register
## Agent Contribution Log — Appendix Material
### Individual Coursework

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
| 5 | 0.1 | Agent generated unified project specification document | scope | Accepted-Modified | Verified structure against brief's 6 steps and 4 rubric criteria; added logging trigger rules | project specification | Plan accepted as execution blueprint with logging enhancements |

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
