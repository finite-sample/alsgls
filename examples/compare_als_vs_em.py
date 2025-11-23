import time

from alsgls import XB_from_Blist, als_gls, em_gls, mse, nll_per_row, simulate_sur


def main():
    N_tr, N_te, K, p, k = 240, 120, 60, 3, 4
    Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr, N_te, K, p, k, seed=123)

    # ALS
    t0 = time.time()
    B_a, F_a, D_a, mem_a, _ = als_gls(Xs_tr, Y_tr, k, lam_F=1e-3, lam_B=1e-3, sweeps=8)
    sec_a = time.time() - t0
    Yhat_te_a = XB_from_Blist(Xs_te, B_a)
    m_a = mse(Y_te, Yhat_te_a)
    nll_a = nll_per_row(Y_te - Yhat_te_a, F_a, D_a)

    # EM
    t0 = time.time()
    B_e, F_e, D_e, mem_e, _ = em_gls(Xs_tr, Y_tr, k, lam_F=1e-3, lam_B=1e-3, iters=20)
    sec_e = time.time() - t0
    Yhat_te_e = XB_from_Blist(Xs_te, B_e)
    m_e = mse(Y_te, Yhat_te_e)
    nll_e = nll_per_row(Y_te - Yhat_te_e, F_e, D_e)

    print("=== SUR (ALS vs EM) ===")
    print(f"K={K}  p={p}  k={k}  N_tr={N_tr}  N_te={N_te}")
    print(
        f"ALS:  sec={sec_a:.3f}  mem_MB≈{mem_a:.3f}  MSE={m_a:.4f}  NLL/N={nll_a:.3f}"
    )
    print(
        f"EM:   sec={sec_e:.3f}  mem_MB≈{mem_e:.3f}  MSE={m_e:.4f}  NLL/N={nll_e:.3f}"
    )
    print("Deltas (ALS - EM):")
    print(
        f"  Δsec={sec_a - sec_e:+.3f}  Δmem={mem_a - mem_e:+.3f}  ΔMSE={m_a - m_e:+.4f}  ΔNLL/N={nll_a - nll_e:+.3f}"
    )


if __name__ == "__main__":
    main()
