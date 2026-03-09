"""
Metric sanity checks for Phase 4-6 outputs.
"""
import os
import pytest
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
RESULTS_PATH = os.path.join(DATA_DIR, 'modelling_results.csv')


@pytest.fixture(scope='module')
def results():
    """Load modelling_results.csv once for all tests."""
    return pd.read_csv(RESULTS_PATH)


class TestModellingResultsShape:
    """Verify modelling_results.csv structure matches expectations."""

    def test_file_exists(self):
        assert os.path.exists(RESULTS_PATH), "modelling_results.csv not found"

    def test_expected_columns(self, results):
        expected = {'horizon', 'model', 'split', 'roc_auc', 'pr_auc',
                    'balanced_accuracy', 'f1', 'brier_score', 'ece'}
        assert set(results.columns) == expected

    def test_row_count(self, results):
        # 14 val rows + 10 test rows = 24 total
        assert len(results) == 24, f"Expected 24 rows, got {len(results)}"

    def test_val_rows(self, results):
        val = results[results['split'] == 'val']
        assert len(val) == 14, f"Expected 14 val rows, got {len(val)}"

    def test_test_rows(self, results):
        test = results[results['split'] == 'test']
        assert len(test) == 10, f"Expected 10 test rows, got {len(test)}"


class TestAUCBounds:
    """All AUC values should be between 0.5 and 1.0 for trained models."""

    def test_roc_auc_range(self, results):
        trained = results[~results['model'].str.startswith('Dummy')]
        assert (trained['roc_auc'] >= 0.5).all(), "ROC-AUC below 0.5 for a trained model"
        assert (trained['roc_auc'] <= 1.0).all(), "ROC-AUC above 1.0"

    def test_pr_auc_range(self, results):
        trained = results[~results['model'].str.startswith('Dummy')]
        assert (trained['pr_auc'] >= 0.0).all(), "PR-AUC below 0.0"
        assert (trained['pr_auc'] <= 1.0).all(), "PR-AUC above 1.0"


class TestLeakageGap:
    """CatBoost H3 test AUC must exceed H1 test AUC (leakage gap)."""

    def test_catboost_h3_exceeds_h1(self, results):
        test = results[results['split'] == 'test']
        cb_h1 = test[(test['model'] == 'CatBoost') & (test['horizon'] == 'H1')]['roc_auc'].values[0]
        cb_h3 = test[(test['model'] == 'CatBoost') & (test['horizon'] == 'H3')]['roc_auc'].values[0]
        gap = cb_h3 - cb_h1
        assert gap > 0.05, f"Leakage gap too small: {gap:.3f}"


class TestCalibration:
    """Verify calibration improves Brier score."""

    def test_catboost_h1_brier_below_dummy(self, results):
        test = results[results['split'] == 'test']
        cb_h1_brier = test[(test['model'] == 'CatBoost') & (test['horizon'] == 'H1')]['brier_score'].values[0]
        # Brier for a constant predictor at base rate ~0.526 would be ~0.249
        assert cb_h1_brier < 0.25, f"CatBoost H1 Brier ({cb_h1_brier:.3f}) not better than baseline"
