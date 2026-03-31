#!/usr/bin/env python3
"""
Real data example: Fama-French 49 Industry Portfolios

Demonstrates ALS-GLS on correlated industry returns where:
- K=49 equations (one per industry)
- Each regressed on Fama-French factors (Mkt-RF, SMB, HML)
- Cross-equation correlation from shared factor exposures

Data downloaded from Kenneth French's Data Library.
"""

import io
import zipfile
from urllib.request import urlopen

import numpy as np
import pandas as pd

from alsgls import ALSGLS
from alsgls.ops import XB_from_Blist


def load_ff_49_industries():
    """Load Fama-French 49 industry portfolios (monthly returns)."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_CSV.zip"
    with urlopen(url) as response:
        zip_data = io.BytesIO(response.read())
    with zipfile.ZipFile(zip_data) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            lines = f.read().decode("utf-8").split("\n")

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("199001"):
            start_idx = i - 1
        if start_idx and line.strip() == "":
            end_idx = i
            break

    header = lines[start_idx].split(",")
    data_lines = lines[start_idx + 1 : end_idx]

    records = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        if len(date_str) != 6:
            continue
        vals = [float(x) if x.strip() and x.strip() != "" else np.nan for x in parts[1:]]
        records.append([date_str] + vals)

    df = pd.DataFrame(records, columns=["date"] + [h.strip() for h in header[1:]])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df = df.set_index("date")
    df = df.dropna()
    return df


def load_ff_factors():
    """Load Fama-French 3 factors (monthly)."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    with urlopen(url) as response:
        zip_data = io.BytesIO(response.read())
    with zipfile.ZipFile(zip_data) as zf:
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        with zf.open(csv_name) as f:
            lines = f.read().decode("utf-8").split("\n")

    start_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("199001"):
            start_idx = i - 1
        if start_idx and line.strip() == "":
            end_idx = i
            break

    data_lines = lines[start_idx + 1 : end_idx]

    records = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        date_str = parts[0].strip()
        if len(date_str) != 6:
            continue
        vals = [float(x.strip()) for x in parts[1:5]]
        records.append([date_str] + vals)

    df = pd.DataFrame(records, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df = df.set_index("date")
    return df


def main():
    print("Loading Fama-French 49 Industry Portfolios...")
    ind = load_ff_49_industries()
    print(f"  Loaded {len(ind)} months, {ind.shape[1]} industries")

    print("Loading Fama-French 3 Factors...")
    ff3 = load_ff_factors()
    print(f"  Loaded {len(ff3)} months")

    common_idx = ind.index.intersection(ff3.index)
    ind = ind.loc[common_idx]
    ff3 = ff3.loc[common_idx]
    print(f"  Aligned to {len(common_idx)} common months")

    N = len(ind)
    K = ind.shape[1]

    Y = (ind.values - ff3["RF"].values[:, None]) / 100
    factors = ff3[["Mkt-RF", "SMB", "HML"]].values / 100
    X_common = np.c_[np.ones(N), factors]
    Xs = [X_common.copy() for _ in range(K)]

    print(f"\nProblem size: N={N} observations, K={K} equations, p=4 features")
    print("Fitting ALS-GLS with BIC rank selection...")

    est = ALSGLS(rank="bic", rank_candidates=list(range(1, 10)), max_sweeps=15)
    est.fit(Xs, Y)

    print("\n--- Results ---")
    print(f"Selected rank: k={est.rank_}")
    print(f"Final NLL per row: {est.info_['nll_trace'][-1]:.4f}")
    print(f"Number of sweeps: {len(est.info_['nll_trace']) - 1}")

    if est.rank_selection_results_ is not None:
        print("\nBIC by rank:")
        for r in est.rank_selection_results_[:6]:
            print(f"  k={r['k']}: BIC={r['bic']:.2f}, NLL={r['nll']:.4f}")

    F = est.F_
    print(f"\nFactor loadings F: {F.shape}")
    print("Top 5 industries loading on factor 1:")
    top_f1 = np.argsort(np.abs(F[:, 0]))[-5:][::-1]
    for i in top_f1:
        print(f"  {ind.columns[i]}: {F[i, 0]:.3f}")

    from sklearn.linear_model import Ridge

    ols_resid = np.zeros((N, K))
    for j in range(K):
        model = Ridge(alpha=0.01, fit_intercept=False)
        model.fit(X_common, Y[:, j])
        ols_resid[:, j] = Y[:, j] - model.predict(X_common)

    corr_ols = np.corrcoef(ols_resid.T)
    off_diag = (corr_ols.sum() - K) / (K * (K - 1))
    print(f"\nOLS residual avg off-diagonal correlation: {off_diag:.3f}")

    gls_resid = Y - XB_from_Blist(Xs, est.B_list_)
    corr_gls = np.corrcoef(gls_resid.T)
    off_diag_gls = (corr_gls.sum() - K) / (K * (K - 1))
    print(f"GLS residual avg off-diagonal correlation: {off_diag_gls:.3f}")

    print("\nALS-GLS captures cross-equation correlation via low-rank factor structure.")


if __name__ == "__main__":
    main()
