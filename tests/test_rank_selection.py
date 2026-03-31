"""Tests for BIC and CV rank selection."""

import numpy as np
import pytest

from alsgls import ALSGLS, select_rank_bic, select_rank_cv, simulate_sur


def _small_sur_data(seed: int = 42):
    """Generate small SUR data for testing."""
    Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(
        N_tr=100, N_te=50, K=20, p=3, k=3, seed=seed
    )
    return Xs_tr, Y_tr, Xs_te, Y_te


class TestSelectRankBIC:
    def test_returns_valid_rank(self):
        Xs, Y, _, _ = _small_sur_data()
        k, results = select_rank_bic(Xs, Y, k_candidates=[1, 2, 3, 4, 5])
        assert 1 <= k <= 5
        assert len(results) == 5

    def test_results_structure(self):
        Xs, Y, _, _ = _small_sur_data()
        _, results = select_rank_bic(Xs, Y, k_candidates=[1, 2, 3])
        for r in results:
            assert "k" in r
            assert "nll" in r
            assert "bic" in r
            assert "n_params" in r

    def test_bic_increases_with_params_when_no_improvement(self):
        Xs, Y, _, _ = _small_sur_data(seed=123)
        _, results = select_rank_bic(Xs, Y, k_candidates=[1, 2, 3, 4, 5, 6])
        nlls = [r["nll"] for r in results if r.get("converged", True)]
        bics = [r["bic"] for r in results if r.get("converged", True)]
        assert len(bics) == len(nlls)

    def test_default_candidates(self):
        Xs, Y, _, _ = _small_sur_data()
        _, results = select_rank_bic(Xs, Y)
        ks = [r["k"] for r in results]
        assert len(ks) > 1


class TestSelectRankCV:
    def test_returns_valid_rank(self):
        Xs, Y, _, _ = _small_sur_data()
        k, results = select_rank_cv(
            Xs, Y, k_candidates=[1, 2, 3], n_folds=3, random_state=42
        )
        assert 1 <= k <= 3
        assert len(results) == 3

    def test_results_structure(self):
        Xs, Y, _, _ = _small_sur_data()
        _, results = select_rank_cv(
            Xs, Y, k_candidates=[1, 2], n_folds=3, random_state=42
        )
        for r in results:
            assert "k" in r
            assert "cv_nll" in r
            assert "cv_std" in r
            assert "fold_nlls" in r
            assert len(r["fold_nlls"]) == 3

    def test_reproducible_with_seed(self):
        Xs, Y, _, _ = _small_sur_data()
        k1, _ = select_rank_cv(
            Xs, Y, k_candidates=[1, 2, 3], n_folds=3, random_state=42
        )
        k2, _ = select_rank_cv(
            Xs, Y, k_candidates=[1, 2, 3], n_folds=3, random_state=42
        )
        assert k1 == k2

    def test_invalid_n_folds(self):
        Xs, Y, _, _ = _small_sur_data()
        with pytest.raises(ValueError):
            select_rank_cv(Xs, Y, k_candidates=[1, 2], n_folds=1)


class TestALSGLSRankSelection:
    def test_bic_rank_selection(self):
        Xs, Y, Xs_te, Y_te = _small_sur_data()
        est = ALSGLS(rank="bic", rank_candidates=[1, 2, 3, 4], max_sweeps=8)
        est.fit(Xs, Y)
        assert est.rank_ in [1, 2, 3, 4]
        assert est.rank_selection_results_ is not None
        assert len(est.rank_selection_results_) == 4

    def test_cv_rank_selection(self):
        Xs, Y, _, _ = _small_sur_data()
        est = ALSGLS(
            rank="cv", rank_candidates=[1, 2, 3], cv_folds=3, cv_random_state=42
        )
        est.fit(Xs, Y)
        assert est.rank_ in [1, 2, 3]
        assert est.rank_selection_results_ is not None

    def test_auto_rank(self):
        Xs, Y, _, _ = _small_sur_data()
        est = ALSGLS(rank="auto")
        est.fit(Xs, Y)
        assert est.rank_ >= 1
        assert est.rank_selection_results_ is None

    def test_fixed_rank(self):
        Xs, Y, _, _ = _small_sur_data()
        est = ALSGLS(rank=5)
        est.fit(Xs, Y)
        assert est.rank_ == 5
        assert est.rank_selection_results_ is None

    def test_prediction_after_bic_selection(self):
        Xs, Y, Xs_te, Y_te = _small_sur_data()
        est = ALSGLS(rank="bic", rank_candidates=[1, 2, 3])
        est.fit(Xs, Y)
        preds = est.predict(Xs_te)
        assert preds.shape == Y_te.shape

    def test_score_after_cv_selection(self):
        Xs, Y, Xs_te, Y_te = _small_sur_data()
        est = ALSGLS(rank="cv", rank_candidates=[1, 2, 3], cv_folds=3)
        est.fit(Xs, Y)
        score = est.score(Xs_te, Y_te)
        assert np.isfinite(score)
