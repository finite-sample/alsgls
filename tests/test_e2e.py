import numpy as np

from alsgls.als import als_gls
from alsgls.metrics import nll_per_row

ABS_FUZZ = 3e-2          # allow tiny non-monotone jiggles per sweep (empirically ~0.02)
ABS_FINAL_EPS = 1e-6     # final vs initial must not be worse beyond micro-jitter
TRACE_RATIO_MAX = 1.5    # Σ trace sanity cap on this sim
EIG_MAX_FACTOR = 8.0     # cap for max eigen inflation vs true (generous but protective)

def make_sur(N=400, K=60, p=3, k=4, seed=123):
    rng = np.random.default_rng(seed)
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B_true = [rng.standard_normal((p, 1)) for _ in range(K)]
    F_true = rng.standard_normal((K, k)) / np.sqrt(K)
    D_true = 0.3 + rng.random(K)

    Z = rng.standard_normal((N, k))
    E = Z @ F_true.T + rng.standard_normal((N, K)) * np.sqrt(D_true)

    Y = np.column_stack([Xs[j] @ B_true[j] for j in range(K)]) + E
    return Xs, Y, B_true, F_true, D_true

def test_end_to_end_nll_and_scale_guard():
    Xs, Y, B_true, F_true, D_true = make_sur()

    B_hat, F_hat, D_hat, mem_mb, info = als_gls(
        Xs, Y, k=4,
        lam_F=1e-3, lam_B=1e-3,
        sweeps=25, cg_maxit=4000, cg_tol=1e-8,
        scale_correct=True
    )

    # Residuals and NLL trace
    Yhat = np.column_stack([Xs[j] @ B_hat[j] for j in range(len(Xs))])
    R = Y - Yhat

    nll_trace = info.get("nll_trace", [])
    assert len(nll_trace) >= 1
    nll_trace = list(map(float, nll_trace))

    # 1) Best (min) NLL across sweeps must strictly improve over start
    assert min(nll_trace) < nll_trace[0] - 1e-3

    # 2) Final NLL must not be worse than initial beyond tiny epsilon
    assert nll_trace[-1] <= nll_trace[0] + ABS_FINAL_EPS

    # 3) Per-sweep changes can jiggle, but no step may worsen NLL by > ABS_FUZZ
    for t in range(1, len(nll_trace)):
        assert nll_trace[t] <= nll_trace[t-1] + ABS_FUZZ

    # 4) Σ scale/shape sanity: trace and top eigen cannot be wildly inflated
    Sig_true = F_true @ F_true.T + np.diag(D_true)
    Sig_hat  = F_hat  @ F_hat.T  + np.diag(D_hat)
    tr_ratio = float(np.trace(Sig_hat)) / float(np.trace(Sig_true))
    assert tr_ratio <= TRACE_RATIO_MAX

    # eigen sanity (allow some overshoot but not catastrophic)
    ev_true = np.linalg.eigvalsh(Sig_true)
    ev_hat  = np.linalg.eigvalsh(Sig_hat)
    assert ev_hat[-1] <= EIG_MAX_FACTOR * ev_true[-1] + 1e-9

    # 5) Final NLL is finite and equals direct computation at end
    nll_end_direct = float(nll_per_row(R, F_hat, D_hat))
    assert np.isfinite(nll_end_direct)
    assert abs(nll_end_direct - nll_trace[-1]) <= 1e-9
