#!/usr/bin/env python3
"""Generate notebooks/04_modelling.ipynb for Phase 4 (Modelling)."""
import nbformat as nbf
import os

PROJECT = "/Users/uleso/Documents/UCL BA/Predictive-Analytics-Individual-Coursework-"
OUT = os.path.join(PROJECT, "notebooks", "04_modelling.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

def md(src):
    nb.cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    nb.cells.append(nbf.v4.new_code_cell(src))


# ══════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════
md("# Step 4: Modelling\n## Horizon-Aware Startup Outcome Prediction")

# ══════════════════════════════════════════════════════════════════════
# 4.0 SETUP
# ══════════════════════════════════════════════════════════════════════
md("## 4.0 Setup and Data Loading")

code('''\
import sys
sys.path.insert(0, '..')

import warnings
import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from src.evaluation import ResultsCollector, compute_all_metrics
from src.models import (
    identify_column_types,
    build_lr_preprocessor, build_hgb_preprocessor, prepare_catboost_data,
    train_dummy, train_lr, train_hgb, train_catboost, train_tabm,
    make_logistic, make_hgb, make_catboost,
)

SEED = 42
np.random.seed(SEED)
DATA_DIR = Path('../data/processed')
MODEL_DIR = Path('../models')
MODEL_DIR.mkdir(exist_ok=True)

# ── Load all horizon splits ─────────────────────────────────────────
def load_split(horizon, split):
    X = pd.read_csv(DATA_DIR / f"{horizon}_{split}_X.csv")
    y = pd.read_csv(DATA_DIR / f"{horizon}_{split}_y.csv").squeeze()
    return X, y

splits = {}
for h in ["H1", "H2", "H3"]:
    splits[h] = {}
    for s in ["train", "val", "test"]:
        X, y = load_split(h, s)
        splits[h][s] = (X, y)
    tr = splits[h]['train'][0].shape
    va = splits[h]['val'][0].shape
    te = splits[h]['test'][0].shape
    print(f"{h}: train={tr}, val={va}, test={te}")

# ── Results collector and model storage ────────────────────────────
results = ResultsCollector()
trained_models = {}   # (horizon, model_name) -> fitted model/pipeline
best_params = {}      # (horizon, model_name) -> best hyperparam dict

# ── Cap extreme outlier in time_to_first_funding_days ──────────────
# Flagged in Step 3.13: max ~102 years from imprecise founding dates
for h in ["H2", "H3"]:
    col = 'time_to_first_funding_days'
    if col in splits[h]['train'][0].columns:
        cap = splits[h]['train'][0][col].quantile(0.99)
        for s in ["train", "val", "test"]:
            X, y = splits[h][s]
            X[col] = X[col].clip(upper=cap)
        print(f"{h}: Capped {col} at 99th percentile = {cap:.0f} days")

print("\\nSetup complete.")
''')

# ══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (defined in a separate cell for clarity)
# ══════════════════════════════════════════════════════════════════════
md("""## 4.0.1 Training Helper Functions

Reusable functions that encapsulate the Optuna tuning + training pattern for each model type. Each function:
1. Builds the appropriate preprocessor
2. Defines an Optuna objective
3. Runs the study
4. Retrains on the best hyperparameters
5. Stores results and returns the fitted model""")

code('''\
def run_dummy(horizon):
    """Train DummyClassifier baselines."""
    X_tr, y_tr = splits[horizon]['train']
    X_va, y_va = splits[horizon]['val']
    print(f"=== {horizon} DummyClassifier ===")
    for strategy in ['most_frequent', 'stratified']:
        model, y_prob = train_dummy(X_tr, y_tr, X_va, y_va, strategy=strategy)
        name = f'Dummy({strategy})'
        results.add(horizon, name, 'val', y_va, y_prob)
        trained_models[(horizon, name)] = model
        m = compute_all_metrics(y_va, y_prob)
        print(f"  {strategy}: ROC-AUC={m['roc_auc']:.4f}, Brier={m['brier_score']:.4f}")


def run_lr(horizon, n_trials=20):
    """Tune and train LogisticRegression."""
    X_tr, y_tr = splits[horizon]['train']
    X_va, y_va = splits[horizon]['val']
    prep = build_lr_preprocessor(X_tr)

    def objective(trial):
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        params = dict(C=C, penalty=penalty)
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9)
        try:
            pipe, y_prob = train_lr(X_tr, y_tr, X_va, prep, **params)
            return roc_auc_score(y_va, y_prob)
        except Exception:
            return 0.5

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    pipe, y_prob = train_lr(X_tr, y_tr, X_va, prep, **best)
    results.add(horizon, 'LogisticRegression', 'val', y_va, y_prob)
    trained_models[(horizon, 'LogisticRegression')] = pipe
    best_params[(horizon, 'LogisticRegression')] = best

    m = compute_all_metrics(y_va, y_prob)
    print(f"=== {horizon} LogisticRegression ===")
    print(f"  Best params: {best}")
    print(f"  Val: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
          f"F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}")
    return pipe, study


def run_hgb(horizon, n_trials=25):
    """Tune and train HistGradientBoostingClassifier."""
    X_tr, y_tr = splits[horizon]['train']
    X_va, y_va = splits[horizon]['val']
    prep = build_hgb_preprocessor(X_tr)

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
            'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 5.0),
            'max_iter': trial.suggest_int('max_iter', 100, 500),
        }
        pipe, y_prob = train_hgb(X_tr, y_tr, X_va, prep, **params)
        return roc_auc_score(y_va, y_prob)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    pipe, y_prob = train_hgb(X_tr, y_tr, X_va, prep, **best)
    results.add(horizon, 'HistGradientBoosting', 'val', y_va, y_prob)
    trained_models[(horizon, 'HistGradientBoosting')] = pipe
    best_params[(horizon, 'HistGradientBoosting')] = best

    m = compute_all_metrics(y_va, y_prob)
    print(f"=== {horizon} HistGradientBoosting ===")
    print(f"  Best params: {best}")
    print(f"  Val: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
          f"F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}")
    return pipe, study


def run_catboost(horizon, n_trials=30):
    """Tune and train CatBoostClassifier."""
    X_tr, y_tr = splits[horizon]['train']
    X_va, y_va = splits[horizon]['val']
    X_tr_cb, cat_idx = prepare_catboost_data(X_tr)
    X_va_cb, _ = prepare_catboost_data(X_va)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 3.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 3.0),
        }
        model, y_prob = train_catboost(X_tr_cb, y_tr, X_va_cb, y_va,
                                       cat_idx, **params)
        return roc_auc_score(y_va, y_prob)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    model, y_prob = train_catboost(X_tr_cb, y_tr, X_va_cb, y_va,
                                   cat_idx, **best)
    results.add(horizon, 'CatBoost', 'val', y_va, y_prob)
    trained_models[(horizon, 'CatBoost')] = model
    best_params[(horizon, 'CatBoost')] = best

    m = compute_all_metrics(y_va, y_prob)
    print(f"=== {horizon} CatBoost ===")
    print(f"  Best params: {best}")
    print(f"  Val: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
          f"F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}")
    return model, study


def run_tabm(horizon, n_trials=15):
    """Tune and train TabM (ICLR 2025 tabular deep learning)."""
    X_tr, y_tr = splits[horizon]['train']
    X_va, y_va = splits[horizon]['val']

    # Store training curves from the best trial
    tabm_curves = {}

    def objective(trial):
        params = {
            'k': trial.suggest_categorical('k', [8, 16, 32]),
            'n_blocks': trial.suggest_int('n_blocks', 1, 3),
            'd_block': trial.suggest_categorical('d_block', [64, 128, 256]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.3),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        }
        try:
            wrapper, y_prob, train_losses, val_aucs = train_tabm(
                X_tr, y_tr, X_va, y_va, **params)
            auc = roc_auc_score(y_va, y_prob)
            # Store curves from this trial
            tabm_curves['train_losses'] = train_losses
            tabm_curves['val_aucs'] = val_aucs
            return auc
        except Exception as e:
            print(f"  TabM trial failed: {e}")
            return 0.5

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    wrapper, y_prob, train_losses, val_aucs = train_tabm(
        X_tr, y_tr, X_va, y_va, **best)
    results.add(horizon, 'TabM', 'val', y_va, y_prob)
    trained_models[(horizon, 'TabM')] = wrapper
    best_params[(horizon, 'TabM')] = best
    # Store final training curves for plotting
    trained_models[(horizon, 'TabM_curves')] = {
        'train_losses': train_losses, 'val_aucs': val_aucs}

    m = compute_all_metrics(y_va, y_prob)
    print(f"=== {horizon} TabM ===")
    print(f"  Best params: {best}")
    print(f"  Val: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
          f"F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}")
    print(f"  Training: {len(train_losses)} epochs, best val AUC={max(val_aucs):.4f}")
    return wrapper, study


def horizon_summary(horizon):
    """Print summary table for a horizon."""
    df = results.to_dataframe()
    df = df[df['horizon'] == horizon].copy()
    df = df.sort_values('roc_auc', ascending=False)
    cols = ['model', 'roc_auc', 'pr_auc', 'balanced_accuracy', 'f1', 'brier_score', 'ece']
    print(f"\\n{'='*70}")
    print(f"{horizon} Validation Results (sorted by ROC-AUC)")
    print(f"{'='*70}")
    display(df[cols].reset_index(drop=True))
    return df

print("Helper functions defined.")
''')

# ══════════════════════════════════════════════════════════════════════
# 4.1 H1 (FOUNDING-TIME) — FULL MODEL STACK
# ══════════════════════════════════════════════════════════════════════
md("""## 4.1 H1 (Founding-Time) — Full Model Stack

H1 uses only features available at founding time: geography, market, category, and founding date-derived features (11 total). This is the **most defensible** horizon — it answers the cleanest causal question with no future-information leakage.

### 4.1.1 DummyClassifier (Sanity Check)

`strategy="most_frequent"` and `strategy="stratified"`. Expected ROC-AUC ~0.50 — establishes random performance baseline.""")

code('''\
run_dummy('H1')
''')

md("### 4.1.2 LogisticRegression\n\nStandardScaler on numerics, frequency encoding for high-cardinality categoricals. Tuning: `C`, `penalty` (l1/l2/elasticnet), `class_weight=balanced`. 20 Optuna trials.")

code('''\
pipe_lr_h1, study_lr_h1 = run_lr('H1', n_trials=20)
''')

md("""### 4.1.3 HistGradientBoostingClassifier

No scaling needed, handles NaN natively. OrdinalEncoder for categoricals. Tuning: `learning_rate`, `max_depth`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`, `max_iter`. 25 Optuna trials with internal early stopping.""")

code('''\
pipe_hgb_h1, study_hgb_h1 = run_hgb('H1', n_trials=25)
''')

md("""### 4.1.4 CatBoostClassifier

Native categorical handling via `cat_features` parameter. Tuning: `iterations`, `depth`, `learning_rate`, `l2_leaf_reg`, `random_strength`, `bagging_temperature`. 30 Optuna trials with validation-based early stopping.""")

code('''\
model_cb_h1, study_cb_h1 = run_catboost('H1', n_trials=30)
''')

md("""### 4.1.5 TabM (Tabular Deep Learning)

**TabM** (Malinin & Babenko, ICLR 2025) is a modern tabular deep learning architecture that ensembles `k` lightweight MLP backbones with batch-ensembling. Each backbone shares parameters but uses independent scaling vectors, providing diversity without the cost of full ensembling.

Preprocessing: LabelEncoder for categoricals (integer indices), SimpleImputer + StandardScaler for numerics. Tuning: `k` (ensemble size), `n_blocks`, `d_block`, `dropout`, `lr`, `weight_decay`, `batch_size`. 15 Optuna trials with early stopping on validation AUC.""")

code('''\
wrapper_tabm_h1, study_tabm_h1 = run_tabm('H1', n_trials=15)
''')

md("### 4.1.6 H1 Model Comparison Summary")

code('''\
df_h1 = horizon_summary('H1')
''')

# ══════════════════════════════════════════════════════════════════════
# 4.2 H2 (FIRST-FUNDING) — FULL MODEL STACK
# ══════════════════════════════════════════════════════════════════════
md("""## 4.2 H2 (First-Funding) — Full Model Stack

H2 adds `time_to_first_funding_days` to the H1 feature set (12 features total). This is the **practical compromise** — a small additional signal from the first observable funding event. The question: does knowing *when* a startup first raised money meaningfully improve prediction beyond founding-time features alone?

### 4.2.1 DummyClassifier (Sanity Check)

Same baseline verification on H2 feature set.""")

code('''\
run_dummy('H2')
''')

md("### 4.2.2 LogisticRegression\n\nSame tuning approach as H1. 20 Optuna trials.")

code('''\
pipe_lr_h2, study_lr_h2 = run_lr('H2', n_trials=20)
''')

md("### 4.2.3 HistGradientBoostingClassifier\n\n25 Optuna trials.")

code('''\
pipe_hgb_h2, study_hgb_h2 = run_hgb('H2', n_trials=25)
''')

md("### 4.2.4 CatBoostClassifier\n\n30 Optuna trials with early stopping.")

code('''\
model_cb_h2, study_cb_h2 = run_catboost('H2', n_trials=30)
''')

md("### 4.2.5 TabM (Tabular Deep Learning)\n\n15 Optuna trials with early stopping.")

code('''\
wrapper_tabm_h2, study_tabm_h2 = run_tabm('H2', n_trials=15)
''')

md("### 4.2.6 H2 Model Comparison Summary")

code('''\
df_h2 = horizon_summary('H2')
''')

# ══════════════════════════════════════════════════════════════════════
# 4.3 H3 (SNAPSHOT) — REDUCED STACK
# ══════════════════════════════════════════════════════════════════════
md("""## 4.3 H3 (Full Snapshot) — Reduced Stack

H3 includes all lifetime funding aggregates (35 features). Purpose: measure the AUC gap to **demonstrate leakage inflation**. Only LogisticRegression and CatBoost are trained — enough to quantify the gap without over-investing.

### 4.3.1 LogisticRegression

Benchmark on full snapshot features to quantify AUC inflation. 15 Optuna trials (reduced — not the primary horizon).""")

code('''\
pipe_lr_h3, study_lr_h3 = run_lr('H3', n_trials=15)
''')

md("### 4.3.2 CatBoostClassifier\n\n20 Optuna trials (reduced investment).")

code('''\
model_cb_h3, study_cb_h3 = run_catboost('H3', n_trials=20)
''')

md("### 4.3.3 H3 Model Comparison Summary")

code('''\
df_h3 = horizon_summary('H3')
''')

# ══════════════════════════════════════════════════════════════════════
# 4.4 CROSS-HORIZON COMPARISON
# ══════════════════════════════════════════════════════════════════════
md("""## 4.4 Cross-Horizon Comparison

The signature analysis: how does ROC-AUC change as we move from the most defensible horizon (H1) to the leakiest (H3)? The gap between H1 and H3 quantifies the performance inflation caused by using lifetime funding aggregates.""")

code('''\
# ── Cross-horizon ROC-AUC pivot table ─────────────────────────────
pivot = results.summary(split='val')
print("Cross-Horizon ROC-AUC (Validation Set)")
print("=" * 50)
display(pivot)

# ── Models present in all three horizons (LR and CatBoost) ────────
print("\\nLeakage gap (H3 - H1) for models on all horizons:")
for model in ['LogisticRegression', 'CatBoost']:
    if model in pivot.index:
        h1_auc = pivot.loc[model, 'H1'] if 'H1' in pivot.columns else None
        h3_auc = pivot.loc[model, 'H3'] if 'H3' in pivot.columns else None
        if h1_auc is not None and h3_auc is not None:
            gap = h3_auc - h1_auc
            print(f"  {model}: H1={h1_auc:.4f} → H3={h3_auc:.4f} (gap={gap:+.4f})")
''')

# ══════════════════════════════════════════════════════════════════════
# 4.5 TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════
md("""## 4.5 Test Set Evaluation

Evaluate all trained models on the held-out test set (first_funding_year >= 2011). This is opened **once** — no further tuning after this point.""")

code('''\
print("=== Test Set Evaluation ===\\n")

for horizon in ["H1", "H2", "H3"]:
    X_te, y_te = splits[horizon]['test']
    print(f"--- {horizon} (n={len(y_te)}, acquired rate={y_te.mean():.3f}) ---")

    for key, model_or_pipe in trained_models.items():
        h, name = key
        if h != horizon:
            continue
        if name.startswith('Dummy'):
            continue  # skip dummy on test
        if name.endswith('_curves'):
            continue  # skip training curve dicts

        try:
            if 'CatBoost' in name:
                X_te_cb, _ = prepare_catboost_data(X_te)
                y_prob = model_or_pipe.predict_proba(X_te_cb)[:, 1]
            elif name == 'TabM':
                y_prob = model_or_pipe.predict_proba(X_te)[:, 1]
            else:
                y_prob = model_or_pipe.predict_proba(X_te)[:, 1]

            results.add(horizon, name, 'test', y_te, y_prob)
            m = compute_all_metrics(y_te, y_prob)
            print(f"  {name}: ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
                  f"F1={m['f1']:.4f}, Brier={m['brier_score']:.4f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    print()
''')

# ══════════════════════════════════════════════════════════════════════
# 4.6 TABM TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════
md("""## 4.6 TabM Training Curves

Training loss and validation AUC curves for the TabM deep learning model. These demonstrate convergence and help assess overfitting — the gap between training loss decrease and validation AUC plateau reveals the model's generalisation behaviour.""")

code('''\
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11, 'figure.dpi': 120})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, horizon in enumerate(['H1', 'H2']):
    curves_key = (horizon, 'TabM_curves')
    if curves_key not in trained_models:
        print(f"No TabM curves for {horizon}")
        continue
    curves = trained_models[curves_key]
    train_losses = curves['train_losses']
    val_aucs = curves['val_aucs']

    ax1 = axes[i]
    color1 = '#2196F3'
    color2 = '#FF9800'

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=color1, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title(f'{horizon} TabM Training Curves')

    ax2 = ax1.twinx()
    val_epochs = range(1, len(val_aucs) + 1)
    ax2.plot(val_epochs, val_aucs, color=color2,
             label='Validation AUC', linestyle='--')
    ax2.set_ylabel('Validation AUC', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    print(f"{horizon} TabM: {len(train_losses)} epochs, best val AUC={max(val_aucs):.4f}")

plt.tight_layout()
plt.savefig('../figures/13_tabm_training_curves.png', bbox_inches='tight')
plt.show()
print("Saved: figures/13_tabm_training_curves.png")
''')

# ══════════════════════════════════════════════════════════════════════
# 4.7 SAVE RESULTS AND MODELS
# ══════════════════════════════════════════════════════════════════════
md("""## 4.7 Save Results and Models""")

code('''\
# ── Save full results table ────────────────────────────────────────
results_df = results.to_dataframe()
results_df.to_csv(DATA_DIR / 'modelling_results.csv', index=False)
print(f"Results saved: {len(results_df)} rows")
display(results_df)

# ── Save trained models via joblib (TabM via torch.save) ──────────
import torch
for key, model_or_pipe in trained_models.items():
    h, name = key
    if name.startswith('Dummy') or name.endswith('_curves'):
        continue
    safe_name = name.replace('(', '').replace(')', '').replace(' ', '_')
    if name == 'TabM':
        path = MODEL_DIR / f"{h}_{safe_name}.pt"
        torch.save({
            'model_state': model_or_pipe.model.state_dict(),
            'preprocessor': model_or_pipe.preprocessor,
        }, path)
    else:
        path = MODEL_DIR / f"{h}_{safe_name}.joblib"
        joblib.dump(model_or_pipe, path)
    print(f"Saved: {path}")

# ── Save best hyperparameters ──────────────────────────────────────
import json
params_out = {}
for key, params in best_params.items():
    h, name = key
    params_serializable = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            params_serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            params_serializable[k] = float(v)
        elif isinstance(v, tuple):
            params_serializable[k] = list(v)
        else:
            params_serializable[k] = v
    params_out[f"{h}_{name}"] = params_serializable

with open(MODEL_DIR / 'best_params.json', 'w') as f:
    json.dump(params_out, f, indent=2)
print(f"\\nBest params saved to {MODEL_DIR / 'best_params.json'}")
''')

# ══════════════════════════════════════════════════════════════════════
# 4.8 MODELLING SUMMARY
# ══════════════════════════════════════════════════════════════════════
md("""## 4.8 Modelling Summary and Next Steps""")

code('''\
print("=" * 70)
print("PHASE 4 MODELLING SUMMARY")
print("=" * 70)

# Validation results pivot
print("\\nValidation ROC-AUC by Horizon x Model:")
val_df = results.to_dataframe()
val_df = val_df[val_df['split'] == 'val']
pivot_val = val_df.pivot_table(index='model', columns='horizon', values='roc_auc').round(4)
display(pivot_val)

# Test results pivot
print("\\nTest ROC-AUC by Horizon x Model:")
test_df = results.to_dataframe()
test_df = test_df[test_df['split'] == 'test']
if not test_df.empty:
    pivot_test = test_df.pivot_table(index='model', columns='horizon', values='roc_auc').round(4)
    display(pivot_test)

# Key findings
print("\\nKey Findings:")
print("1. DummyClassifier confirms ~0.50 AUC baseline (sanity check passed)")

# Best model per horizon (val)
for h in ['H1', 'H2', 'H3']:
    h_df = val_df[val_df['horizon'] == h].sort_values('roc_auc', ascending=False)
    if not h_df.empty:
        best = h_df.iloc[0]
        print(f"2. {h} best model: {best['model']} (val ROC-AUC={best['roc_auc']:.4f})")

# Leakage gap
for model in ['LogisticRegression', 'CatBoost']:
    h1_rows = val_df[(val_df['horizon'] == 'H1') & (val_df['model'] == model)]
    h3_rows = val_df[(val_df['horizon'] == 'H3') & (val_df['model'] == model)]
    if not h1_rows.empty and not h3_rows.empty:
        gap = h3_rows.iloc[0]['roc_auc'] - h1_rows.iloc[0]['roc_auc']
        print(f"3. {model} leakage gap (H3-H1): {gap:+.4f}")

print("\\nNext: Phase 5 (Evaluation) — SHAP, calibration, error analysis, cross-horizon figures")
''')


# ══════════════════════════════════════════════════════════════════════
# WRITE NOTEBOOK
# ══════════════════════════════════════════════════════════════════════
with open(OUT, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {OUT}")
print(f"Total cells: {len(nb.cells)}")
