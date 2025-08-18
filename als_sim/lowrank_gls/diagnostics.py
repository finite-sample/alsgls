import numpy as np
from numpy.linalg import eigh
from scipy.stats import norm, chi2

from .numerics import predict_Y, cf_matrix, siginv_apply_to_matrix, safe_inv


def _diag_sigma_diag(F, D):
    return (F ** 2).sum(axis=1) + D


def coverage_and_mahalanobis(Y, Xs, B_list, F, D, alphas=(0.10, 0.05, 0.01)):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R = Y - XB

    s = np.sqrt(_diag_sigma_diag(F, D))
    Z = R / s[None, :]
    cov_overall = {}
    for a in alphas:
        thr = norm.ppf(1 - a / 2.0)
        per = (np.abs(Z) <= thr).mean(axis=0)
        cov_overall[f"cov@{int((1 - a) * 100)}%"] = {
            "overall": float(per.mean()),
            "min_eq": float(per.min()),
            "max_eq": float(per.max()),
        }

    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    m2 = np.sum(R * RSig, axis=1)
    q95 = float(chi2.ppf(0.95, df=K))
    q99 = float(chi2.ppf(0.99, df=K))
    W = (R.T @ RSig) / N
    frob = float(np.linalg.norm(W - np.eye(K), "fro") / np.sqrt(K))
    offm = float(np.max(np.abs(W - np.diag(np.diag(W)))))
    return {
        "z_coverage_overall": cov_overall,
        "m2_mean": float(m2.mean()),
        "m2_var": float(m2.var()),
        "m2_frac_gt_95": float((m2 > q95).mean()),
        "m2_frac_gt_99": float((m2 > q99).mean()),
        "m2_q95": q95,
        "m2_q99": q99,
        "whiten_frob_dev": frob,
        "whiten_offdiag_max": offm,
        "K": K,
        "N": N,
    }


def whitening_matrix(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R = Y - XB
    Dinv = 1.0 / D
    Cf_i = safe_inv(cf_matrix(F, Dinv))
    RSig = siginv_apply_to_matrix(R, F, D, Cf_i)
    W = (R.T @ RSig) / N
    return W


def top_eigvecs_sym(M, r=2):
    S = (M + M.T) / 2.0
    vals, vecs = eigh(S)
    return vals[-r:], vecs[:, -r:]


def print_diag(label, d):
    print(f"\n[{label}]  K={d['K']}  N={d['N']}")
    for k in ["cov@90%", "cov@95%", "cov@99%"]:
        if k in d["z_coverage_overall"]:
            v = d["z_coverage_overall"][k]
            print(
                f"  z {k}: overall={v['overall']:.3f}, min_eq={v['min_eq']:.3f}, max_eq={v['max_eq']:.3f}"
            )
    print(
        f"  m2 mean={d['m2_mean']:.2f} (≈K), var={d['m2_var']:.2f} (≈2K)"
    )
    print(
        f"  frac m2>χ2_95={d['m2_frac_gt_95']:.3f} (≈0.05), >χ2_99={d['m2_frac_gt_99']:.3f} (≈0.01)"
    )
    print(
        f"  whiten frob_dev={d['whiten_frob_dev']:.3f}, offdiag_max={d['whiten_offdiag_max']:.3f}"
    )
