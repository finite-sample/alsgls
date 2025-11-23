import numpy as np

from alsgls.als import als_gls

ABS_FUZZ = 3e-2    # allow tiny non-monotone jiggles per sweep
FINAL_IMPROVE = 1e-3

def make_sur(N=300, K=40, p=3, k=3, seed=2025):
    rng = np.random.default_rng(seed)
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B_true = [rng.standard_normal((p, 1)) for _ in range(K)]
    F_true = rng.standard_normal((K, k)) / np.sqrt(K)
    D_true = 0.4 + rng.random(K)
    Z = rng.standard_normal((N, k))
    E = Z @ F_true.T + rng.standard_normal((N, K)) * np.sqrt(D_true)
    Y = np.column_stack([Xs[j] @ B_true[j] for j in range(K)]) + E
    return Xs, Y

def test_nll_decreases_on_sim():
    Xs, Y = make_sur()
    _, _, _, _, info = als_gls(
        Xs, Y, k=3, lam_F=1e-3, lam_B=1e-3,
        sweeps=20, cg_maxit=4000, cg_tol=1e-8,
        scale_correct=True
    )
    tr = list(map(float, info.get("nll_trace", [])))
    assert len(tr) >= 2

    # Require overall improvement and bound any per-step worsening by small ABS_FUZZ
    assert min(tr) < tr[0] - FINAL_IMPROVE
    for t in range(1, len(tr)):
        assert tr[t] <= tr[t-1] + ABS_FUZZ
