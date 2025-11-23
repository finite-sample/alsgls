import numpy as np


def test_trace_is_nonincreasing_after_sweep_fix():
    from alsgls.als import als_gls

    rng = np.random.default_rng(7)
    N, K, p, k = 300, 40, 3, 3
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B_true = [rng.standard_normal((p, 1)) for _ in range(K)]
    F_true = rng.standard_normal((K, k)) / np.sqrt(K)
    D_true = 0.4 + rng.random(K)
    Z = rng.standard_normal((N, k))
    Y = (
        np.column_stack([Xs[j] @ B_true[j] for j in range(K)])
        + Z @ F_true.T
        + rng.standard_normal((N, K)) * np.sqrt(D_true)
    )

    _, _, _, _, info = als_gls(
        Xs, Y, k=3, sweeps=15, cg_maxit=4000, cg_tol=1e-8, scale_correct=True
    )
    tr = list(map(float, info["nll_trace"]))
    # non-increasing up to 1e-10 jitter
    for t in range(1, len(tr)):
        assert tr[t] <= tr[t - 1] + 1e-10
    # and strictly improved at least once
    assert min(tr) < tr[0] - 1e-3
