import numpy as np
from .ops import XB_from_Blist

def simulate_sur(N_tr, N_te, K, p, k, seed=0):
    rng = np.random.default_rng(seed)
    N = N_tr + N_te
    base = rng.standard_normal((N, p))
    Xs = [base + 0.5 * rng.standard_normal((N, p)) for _ in range(K)]
    B = [rng.standard_normal((p, 1)) for _ in range(K)]
    F0 = 1.0 * rng.standard_normal((K, k))
    D0 = 0.05 + 0.20 * rng.random(K)
    U = rng.standard_normal((N, k))
    Y = XB_from_Blist(Xs, B) + U @ F0.T + rng.standard_normal((N, K)) * np.sqrt(D0)[None, :]
    return [X[:N_tr] for X in Xs], Y[:N_tr], [X[N_tr:] for X in Xs], Y[N_tr:]

def simulate_gls(N_tr, N_te, p_list, k, seed=0):
    rng = np.random.default_rng(seed)
    K = len(p_list)
    N = N_tr + N_te
    Xs = []
    for p in p_list:
        base = rng.standard_normal((N, p))
        Xs.append(base + 0.5 * rng.standard_normal((N, p)))
    B = [rng.standard_normal((p, 1)) for p in p_list]
    F0 = 1.0 * rng.standard_normal((K, k))
    D0 = 0.05 + 0.20 * rng.random(K)
    U = rng.standard_normal((N, k))
    Y = XB_from_Blist(Xs, B) + U @ F0.T + rng.standard_normal((N, K)) * np.sqrt(D0)[None, :]
    return [X[:N_tr] for X in Xs], Y[:N_tr], [X[N_tr:] for X in Xs], Y[N_tr:]
