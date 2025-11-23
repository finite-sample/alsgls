import numpy as np

from alsgls import ALSGLS, ALSGLSSystem, als_gls, nll_per_row
from alsgls.ops import XB_from_Blist


def _random_sur(rng, N=60, K=4, p=3):
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B = [rng.standard_normal((p, 1)) for _ in range(K)]
    F = rng.standard_normal((K, 2)) / np.sqrt(K)
    D = 0.4 + 0.2 * rng.random(K)
    Z = rng.standard_normal((N, 2))
    Y = XB_from_Blist(Xs, B) + Z @ F.T + rng.standard_normal((N, K)) * np.sqrt(D)[None, :]
    return Xs, Y


def test_sklearn_api_matches_function():
    rng = np.random.default_rng(123)
    Xs, Y = _random_sur(rng, N=120, K=5, p=4)

    direct = als_gls(Xs, Y, k=3, sweeps=10, rel_tol=1e-8)
    model = ALSGLS(rank=3, max_sweeps=10, rel_tol=1e-8)
    fitted = model.fit(Xs, Y)

    assert fitted is model
    for b_direct, b_model in zip(direct[0], model.B_list_, strict=False):
        assert np.allclose(b_direct, b_model, atol=1e-8)
    assert np.allclose(direct[1], model.F_, atol=1e-8)
    assert np.allclose(direct[2], model.D_, atol=1e-8)

    preds = model.predict(Xs)
    assert np.allclose(preds, XB_from_Blist(Xs, model.B_list_), atol=1e-10)

    score = model.score(Xs, Y)
    nll = -score
    assert np.isclose(nll, nll_per_row(Y - preds, model.F_, model.D_), atol=1e-10)


def test_system_api_mirrors_estimator():
    rng = np.random.default_rng(321)
    Xs, Y = _random_sur(rng, N=80, K=3, p=2)

    system = {f"eq{j}": (Y[:, j], Xs[j]) for j in range(3)}
    sys_model = ALSGLSSystem(system, rank=2, max_sweeps=9, rel_tol=1e-8)
    results = sys_model.fit()

    assert results.model.keqs == 3
    assert results.model.nobs == 80

    preds = results.predict()
    assert np.allclose(preds, XB_from_Blist(Xs, results.B_list), atol=1e-10)
    assert np.allclose(results.fittedvalues, preds)

    # Ensure the summary exposes key scalars
    summary = results.summary_dict()
    assert summary["keqs"] == 3
    assert summary["nobs"] == 80
    assert summary["rank"] == 2

    # Recompute score and ensure consistency with estimator
    estimator_score = sys_model.estimator_.score(Xs, Y)
    assert np.isclose(estimator_score, -results.nll_per_row, atol=1e-10)
