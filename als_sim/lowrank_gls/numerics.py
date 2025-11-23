import numpy as np
from numpy.linalg import LinAlgError, solve


def safe_inv(A, lam=1e-6):
    I = np.eye(A.shape[0])
    ridge = max(lam * np.linalg.norm(A, 'fro') / A.shape[0], 1e-8)
    try:
        return solve(A + ridge * I, I)
    except LinAlgError:
        return np.linalg.pinv(A + ridge * I)


def safe_solve(A, b, lam=1e-6):
    I = np.eye(A.shape[0])
    ridge = max(lam * np.linalg.norm(A, 'fro') / A.shape[0], 1e-8)
    try:
        return solve(A + ridge * I, b)
    except LinAlgError:
        return np.linalg.lstsq(A + ridge * I, b, rcond=None)[0]


def wpca_init(R, k, D):
    W = R / np.sqrt(D)[None, :]
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = max(1, min(k, (s > 1e-8).sum()))
    Fw = Vt.T[:, :r] * np.sqrt(s[:r])
    F = Fw * np.sqrt(D)[:, None]
    return np.pad(F, ((0, 0), (0, k - r)))


def cf_matrix(F, Dinv):
    return np.eye(F.shape[1]) + F.T @ (Dinv[:, None] * F)


def siginv_apply_to_matrix(M, F, D, Cf_inv=None):
    Dinv = 1.0 / D
    if Cf_inv is None:
        Cf_inv = safe_inv(cf_matrix(F, Dinv))
    Vd = M * Dinv[None, :]
    FD = Dinv[:, None] * F
    return Vd - ((Vd @ F) @ Cf_inv) @ FD.T


def pack_list_to_vec(B_list):
    return np.concatenate([b.ravel() for b in B_list], axis=0)


def unpack_vec_to_list(v, p_list):
    out, i = [], 0
    for p in p_list:
        out.append(v[i:i + p].reshape(p, 1))
        i += p
    return out


def predict_Y(Xs, B_list):
    return np.column_stack([Xs[j] @ B_list[j] for j in range(len(Xs))])


def mse(Y, Yhat):
    return float(np.mean((Y - Yhat) ** 2))


def _safe_logdet_psd(A):
    sign, logdet = np.linalg.slogdet(A)
    if sign > 0 and np.isfinite(logdet):
        return logdet
    eps = 1e-10
    return np.linalg.slogdet(A + eps * np.eye(A.shape[0]))[1]


def penalized_nll(Y, Xs, B_list, F, D, lam_F=0.0, lam_B=0.0):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R = Y - XB
    Dinv = 1.0 / D
    Cf = cf_matrix(F, Dinv)
    Cf_i = safe_inv(Cf)
    term1 = np.sum(R * (R * Dinv[None, :]))
    M = (R * Dinv[None, :]) @ F
    term2 = np.sum(M * (M @ Cf_i))
    quad = term1 - term2
    logdet = np.sum(np.log(D)) + _safe_logdet_psd(Cf)
    pen = lam_F * np.sum(F ** 2) + lam_B * sum(np.sum(b ** 2) for b in B_list)
    return 0.5 * (N * logdet + quad) + pen


def test_nll(Y, Xs, B_list, F, D):
    N, K = Y.shape
    XB = predict_Y(Xs, B_list)
    R = Y - XB
    Dinv = 1.0 / D
    Cf = cf_matrix(F, Dinv)
    Cf_i = safe_inv(Cf)
    term1 = np.sum(R * (R * Dinv[None, :]))
    M = (R * Dinv[None, :]) @ F
    term2 = np.sum(M * (M @ Cf_i))
    quad = term1 - term2
    logdet = np.sum(np.log(D)) + _safe_logdet_psd(Cf)
    return 0.5 * (N * logdet + quad) / N
