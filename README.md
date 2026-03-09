# Horizon-Aware Startup Outcome Prediction

Binary classification of startup terminal outcomes (acquired vs closed) using a Crunchbase dataset. The core contribution is a three-horizon framework that measures how much predictive performance is genuine early signal versus temporal contamination from lifetime funding aggregates.

## Requirements

- Python 3.11
- All dependencies listed in `requirements.txt`

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

The raw dataset (`investments_VC 2.csv`) is a Crunchbase export containing 54,294 startup records with 39 columns. Place it in the repository root before running notebooks.

## Notebook Execution Order

Run notebooks sequentially from the `notebooks/` directory:

1. `01_problem_framing.ipynb` — Problem definition, censoring argument, horizon framework, metric definitions
2. `02_eda.ipynb` — Data audit (blank rows, duplicates, impossible dates, missingness) and 12 EDA figures
3. `03_data_preparation.ipynb` — Cleaning pipeline, feature engineering, three horizon tables (H1/H2/H3), temporal split, leakage validation
4. `04_modelling.ipynb` — Five-model stack (Dummy, LR, HGB, CatBoost, TabM), Optuna tuning, cross-horizon comparison
5. `05_evaluation.ipynb` — ROC/PR curves, calibration (Platt/isotonic), SHAP analysis, slice analysis, error analysis
6. `06_final_solution.ipynb` — Model selection summary, model card, limitations, recommendations, agent reflection

## Running Tests

```bash
pytest -p no:debugging -q
```

26 tests covering horizon integrity, leakage prevention, temporal split boundaries, and evaluation output sanity.

## Repository Structure

```
data/raw/               Raw CSV (not tracked by git)
data/processed/         Cleaned horizon splits, modelling results
notebooks/              Jupyter notebooks (01-06)
src/                    Reusable modules
  preprocessing.py      Cleaning pipeline, horizon builders, temporal split
  features.py           Feature engineering (10 engineered features)
  models.py             Per-model encoding strategies
  evaluation.py         6-metric evaluation suite and ResultsCollector
  config.py             Leak registry, column classifications, constants
tests/                  pytest test suite
  test_preprocessing.py Horizon integrity, leakage, split, shape tests
  test_evaluation.py    Modelling results validation, AUC bounds, leakage gap
models/                 Serialised trained models (.joblib, .pt), best_params.json
figures/                All generated figures (01-23)
logs/                   Workflow log and decision register
report/                 Report draft with per-section bullet points
```
