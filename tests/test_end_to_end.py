import numpy as np
from alsgls.als import als_gls

def test_nll_decreases_on_sim():
    rng = np.random.default_rng(2025)
    N, K, p, k = 250, 40, 3, 3
    # generate data
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B_true = [rng.standard_normal((p, 1)) for _ in range(K)]
    F_true = rng.standard_normal((K, k)) / np.sqrt(K)
    D_true = 0.4 + rng.random(K)
    Z = rng.standard_normal((N, k))
    E = Z @ F_true.T + rng.standard_normal((N, K)) * np.sqrt(D_true)
    Y = np.column_stack([Xs[j] @ B_true[j] for j in range(K)]) + E

    _, _, _, _, info = als_gls(Xs, Y, k=k, sweeps=8)
    tr = info["nll_trace"]
    assert len(tr) >= 2
    # allow tiny numerical wiggles; just assert end <= start within tolerance
    assert tr[-1] <= tr[0] + 1e-8