"""
Model definitions, encoding helpers, and training utilities.

Each model has its own encoding requirements:
- DummyClassifier: no encoding needed
- LogisticRegression: frequency encoding for categoricals, StandardScaler, imputation
- HistGradientBoostingClassifier: OrdinalEncoder for categoricals, handles NaN natively
- CatBoostClassifier: native categorical handling, handles NaN natively
- TabM: LabelEncoder for categoricals (integer indices), SimpleImputer + StandardScaler for numerics
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ── Encoding helpers ──────────────────────────────────────────────────

def identify_column_types(X: pd.DataFrame):
    """Return (cat_cols, num_cols) lists for a feature DataFrame."""
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    return cat_cols, num_cols


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode categoricals by their training-set frequency.

    Unseen categories at transform time get frequency 0.
    NaN values get frequency 0.
    """

    def fit(self, X, y=None):
        self.mappings_ = {}
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            self.mappings_[col] = freq.to_dict()
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.mappings_.get(col, {})
            out[col] = X[col].map(mapping).fillna(0.0).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray(list(self.mappings_.keys()))


def build_lr_preprocessor(X_train: pd.DataFrame):
    """Build a ColumnTransformer for LogisticRegression / MLP.

    - Categoricals: FrequencyEncoder
    - Numerics: SimpleImputer(median) → StandardScaler
    """
    cat_cols, num_cols = identify_column_types(X_train)

    transformers = []
    if cat_cols:
        transformers.append((
            "cat", FrequencyEncoder(), cat_cols
        ))
    if num_cols:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")


def build_hgb_preprocessor(X_train: pd.DataFrame):
    """Build a ColumnTransformer for HistGradientBoosting.

    - Categoricals: OrdinalEncoder (unknown → -1)
    - Numerics: passed through (HGB handles NaN natively)
    """
    cat_cols, num_cols = identify_column_types(X_train)

    transformers = []
    if cat_cols:
        transformers.append((
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                           encoded_missing_value=-2),
            cat_cols,
        ))
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))

    return ColumnTransformer(transformers, remainder="drop")


def prepare_catboost_data(X: pd.DataFrame):
    """Prepare data for CatBoost: fill NaN in categoricals with '__missing__'.

    Returns (X_prepared, cat_feature_indices).
    """
    cat_cols, num_cols = identify_column_types(X)
    X_out = X.copy()
    for col in cat_cols:
        X_out[col] = X_out[col].fillna("__missing__").astype(str)
    cat_indices = [X_out.columns.get_loc(c) for c in cat_cols]
    return X_out, cat_indices


# ── Model factories ────────────────────────────────────────────────────

def make_dummy(strategy="most_frequent"):
    """Create a DummyClassifier."""
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy=strategy, random_state=42)


def make_logistic(C=1.0, penalty="l2", class_weight="balanced",
                  max_iter=2000, solver="saga", l1_ratio=None):
    """Create a LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    kwargs = dict(C=C, penalty=penalty, class_weight=class_weight,
                  max_iter=max_iter, solver=solver, random_state=42, n_jobs=-1)
    if l1_ratio is not None:
        kwargs["l1_ratio"] = l1_ratio
    return LogisticRegression(**kwargs)


def make_hgb(learning_rate=0.1, max_depth=None, max_leaf_nodes=31,
             min_samples_leaf=20, l2_regularization=0.0, max_iter=200):
    """Create a HistGradientBoostingClassifier."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
    )


def make_catboost(iterations=500, depth=6, learning_rate=0.1,
                  l2_leaf_reg=3.0, random_strength=1.0,
                  bagging_temperature=1.0, verbose=0):
    """Create a CatBoostClassifier."""
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        bagging_temperature=bagging_temperature,
        eval_metric="AUC",
        use_best_model=True,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=verbose,
    )


# ── TabM preprocessing ─────────────────────────────────────────────────

class TabMPreprocessor:
    """Preprocess data for TabM: LabelEncode categoricals, impute+scale numerics.

    Stores fitted encoders so transform can be applied to val/test.
    """

    def __init__(self):
        self.cat_cols_ = []
        self.num_cols_ = []
        self.label_encoders_ = {}
        self.cat_cardinalities_ = []
        self.imputer_ = None
        self.scaler_ = None

    def fit(self, X: pd.DataFrame):
        self.cat_cols_, self.num_cols_ = identify_column_types(X)
        # Fit label encoders per categorical column
        self.cat_cardinalities_ = []
        for col in self.cat_cols_:
            vals = X[col].fillna("__missing__").astype(str)
            unique = sorted(vals.unique())
            mapping = {v: i for i, v in enumerate(unique)}
            self.label_encoders_[col] = mapping
            self.cat_cardinalities_.append(len(unique))
        # Fit numeric preprocessing
        if self.num_cols_:
            self.imputer_ = SimpleImputer(strategy="median")
            self.scaler_ = StandardScaler()
            num_data = X[self.num_cols_].values.astype(float)
            num_data = self.imputer_.fit_transform(num_data)
            self.scaler_.fit(num_data)
        return self

    def transform(self, X: pd.DataFrame):
        import torch
        # Categoricals → integer tensor
        cat_arrays = []
        for col in self.cat_cols_:
            vals = X[col].fillna("__missing__").astype(str)
            mapping = self.label_encoders_[col]
            # Unseen categories get index 0
            encoded = vals.map(mapping).fillna(0).astype(int).values
            cat_arrays.append(encoded)
        x_cat = torch.tensor(np.column_stack(cat_arrays), dtype=torch.long) if cat_arrays else None
        # Numerics → float tensor
        if self.num_cols_:
            num_data = X[self.num_cols_].values.astype(float)
            num_data = self.imputer_.transform(num_data)
            num_data = self.scaler_.transform(num_data)
            x_num = torch.tensor(num_data, dtype=torch.float32)
        else:
            x_num = None
        return x_num, x_cat


# ── Training wrappers ──────────────────────────────────────────────────

def train_dummy(X_train, y_train, X_val, y_val, strategy="most_frequent"):
    """Train DummyClassifier, return (model, val_probs)."""
    model = make_dummy(strategy=strategy)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    return model, y_prob


def train_lr(X_train, y_train, X_val, preprocessor, **kwargs):
    """Train LogisticRegression with preprocessor, return (pipeline, val_probs)."""
    model = make_logistic(**kwargs)
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_val)[:, 1]
    return pipe, y_prob


def train_hgb(X_train, y_train, X_val, preprocessor, **kwargs):
    """Train HGB with preprocessor, return (pipeline, val_probs)."""
    model = make_hgb(**kwargs)
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_val)[:, 1]
    return pipe, y_prob


def train_catboost(X_train, y_train, X_val, y_val, cat_indices, **kwargs):
    """Train CatBoost with eval set for early stopping, return (model, val_probs)."""
    from catboost import Pool
    model = make_catboost(**kwargs)
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices)
    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30)
    y_prob = model.predict_proba(X_val)[:, 1]
    return model, y_prob


def train_tabm(X_train, y_train, X_val, y_val,
               k=32, n_blocks=2, d_block=128, dropout=0.1,
               lr=0.001, weight_decay=1e-4, batch_size=256,
               max_epochs=200, patience=15):
    """Train TabM model with early stopping on validation AUC.

    Returns (tabm_wrapper, val_probs, train_losses, val_aucs).
    The wrapper object has .predict_proba() for sklearn-style usage.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from tabm import TabM
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(42)

    # Preprocess
    prep = TabMPreprocessor()
    prep.fit(X_train)
    x_num_tr, x_cat_tr = prep.transform(X_train)
    x_num_va, x_cat_va = prep.transform(X_val)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_va_np = y_val.values

    # Compute class weights for imbalanced data
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()])

    # Build model
    n_num = x_num_tr.shape[1] if x_num_tr is not None else 0
    cat_cards = prep.cat_cardinalities_ if x_cat_tr is not None else None

    model = TabM(
        n_num_features=n_num,
        cat_cardinalities=cat_cards,
        d_out=1,
        k=k,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        arch_type="tabm",
        start_scaling_init="random-signs",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Build dataset
    train_tensors = [t for t in [x_num_tr, x_cat_tr, y_tr] if t is not None]
    # Need to handle the case where cat or num might be None
    if x_num_tr is not None and x_cat_tr is not None:
        train_ds = TensorDataset(x_num_tr, x_cat_tr, y_tr)
    elif x_num_tr is not None:
        train_ds = TensorDataset(x_num_tr, y_tr)
    else:
        train_ds = TensorDataset(x_cat_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Training loop
    train_losses = []
    val_aucs = []
    best_auc = -1
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            if x_num_tr is not None and x_cat_tr is not None:
                bx_num, bx_cat, by = batch
            elif x_num_tr is not None:
                bx_num, by = batch
                bx_cat = None
            else:
                bx_cat, by = batch
                bx_num = None

            optimizer.zero_grad()
            logits = model(bx_num, bx_cat)  # [batch, k, 1]
            logits = logits.mean(dim=1)  # [batch, 1] — ensemble average
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            va_logits = model(x_num_va, x_cat_va).mean(dim=1).squeeze(1)
            va_probs = torch.sigmoid(va_logits).numpy()
        va_auc = roc_auc_score(y_va_np, va_probs)
        val_aucs.append(va_auc)

        # Early stopping
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val predictions
    model.eval()
    with torch.no_grad():
        va_logits = model(x_num_va, x_cat_va).mean(dim=1).squeeze(1)
        va_probs = torch.sigmoid(va_logits).numpy()

    # Wrap for sklearn-style interface
    wrapper = TabMWrapper(model, prep)
    return wrapper, va_probs, train_losses, val_aucs


class TabMWrapper:
    """Sklearn-compatible wrapper around a trained TabM model + preprocessor."""

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict_proba(self, X):
        import torch
        self.model.eval()
        x_num, x_cat = self.preprocessor.transform(X)
        with torch.no_grad():
            logits = self.model(x_num, x_cat).mean(dim=1).squeeze(1)
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])
