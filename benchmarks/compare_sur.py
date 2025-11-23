"""Benchmark ALS-GLS against statsmodels and linearmodels SUR implementations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alsgls import ALSGLS, nll_per_row
from alsgls.ops import XB_from_Blist


def _simulate_sur(N_tr: int, N_te: int, K: int, p: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    N = N_tr + N_te
    Xs = [rng.standard_normal((N, p)) for _ in range(K)]
    B = [rng.standard_normal((p, 1)) for _ in range(K)]
    F = rng.standard_normal((K, k)) / np.sqrt(max(K, 1))
    D = 0.2 + 0.3 * rng.random(K)
    Z = rng.standard_normal((N, k))
    Y = XB_from_Blist(Xs, B) + Z @ F.T + rng.standard_normal((N, K)) * np.sqrt(D)[None, :]
    return (
        [X[:N_tr] for X in Xs],
        Y[:N_tr],
        [X[N_tr:] for X in Xs],
        Y[N_tr:],
        B,
        F,
        D,
    )


def _stack_beta(B_list: Iterable[np.ndarray]) -> np.ndarray:
    blocks = [np.asarray(b).ravel() for b in B_list]
    return np.concatenate(blocks) if blocks else np.empty(0)


def _gaussian_nll(residuals: np.ndarray, sigma: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        raise ValueError("Covariance matrix is not SPD")
    inv = np.linalg.inv(sigma)
    quad = np.sum(residuals @ inv * residuals) / residuals.shape[0]
    return 0.5 * (quad + logdet + residuals.shape[1] * math.log(2.0 * math.pi))


def _maybe_memray_runner(func, *args, **kwargs):
    backend = kwargs.pop("backend", "memray")
    if backend == "none":
        return func(*args, **kwargs), None

    if backend == "memray":
        try:
            import memray
        except ImportError:
            raise RuntimeError("memray is not installed; install memray or use --memory-backend none")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.bin"
            with memray.Tracker(path):
                result = func(*args, **kwargs)
            reader = memray.FileReader(path)
            peak = reader.metadata.peak_memory
            return result, peak

    if backend == "fil":
        raise RuntimeError(
            "Fil profiler integration requires running this script via `fil-profile`; "
            "re-run with --memory-backend none when invoking through fil."
        )

    if backend == "resource":
        import resource

        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = func(*args, **kwargs)
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = max(before, after)
        return result, peak

    raise ValueError(f"Unknown memory backend {backend}")


def _run_statsmodels(system, **fit_kwargs):
    try:
        from statsmodels.sur.sur_model import SUR as SMSUR
    except ImportError:
        return None, "missing"

    model = SMSUR(system)
    results = model.fit(**fit_kwargs)
    sigma = np.asarray(results.sigma)
    params = []
    for name in system.keys():
        params.append(np.asarray(results.params.loc[name]).reshape(-1, 1))
    fitted = np.column_stack([results.predict(eq=name) for name in system.keys()])
    return {"B": params, "sigma": sigma, "fitted": fitted, "resid": np.asarray(results.resid)}, "ok"


def _run_linearmodels(system, **fit_kwargs):
    try:
        from linearmodels.system import SUR as LMSUR
    except ImportError:
        return None, "missing"

    model = LMSUR(system)
    results = model.fit(**fit_kwargs)
    sigma = np.asarray(results.sigma)
    params = [np.asarray(results.params[name]).reshape(-1, 1) for name in system.keys()]
    fitted = np.column_stack([results.predict(eq=name) for name in system.keys()])
    resid = np.column_stack([results.resids[name] for name in system.keys()])
    return {"B": params, "sigma": sigma, "fitted": fitted, "resid": resid}, "ok"


@dataclass
class BenchmarkResult:
    K: int
    N: int
    p: int
    k: int
    method: str
    beta_rmse: float | None
    test_nll: float | None
    peak_memory: float | None
    wall_time: float
    status: str


def run_benchmark(
    grid: Iterable[tuple[int, int, int, int]],
    *,
    seed: int = 0,
    memory_backend: str = "resource",
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for K, N, p, k in grid:
        X_tr, Y_tr, X_te, Y_te, B_true, _, _ = _simulate_sur(
            N_tr=N,
            N_te=N // 2,
            K=K,
            p=p,
            k=k,
            seed=seed,
        )

        system = {f"eq{j}": (Y_tr[:, j], X_tr[j]) for j in range(K)}

        def _als_run():
            model = ALSGLS(rank=k, max_sweeps=12)
            model.fit(X_tr, Y_tr)
            preds = model.predict(X_te)
            return model, preds

        t0 = time.perf_counter()
        (als_model, als_preds), als_peak = _maybe_memray_runner(_als_run, backend=memory_backend)
        wall = time.perf_counter() - t0
        beta_rmse = float(np.sqrt(np.mean((_stack_beta(als_model.B_list_) - _stack_beta(B_true)) ** 2)))
        nll = float(nll_per_row(Y_te - als_preds, als_model.F_, als_model.D_))
        results.append(
            BenchmarkResult(
                K=K,
                N=N,
                p=p,
                k=k,
                method="alsgls",
                beta_rmse=beta_rmse,
                test_nll=nll,
                peak_memory=als_peak,
                wall_time=wall,
                status="ok",
            )
        )

        for name, runner in ("statsmodels", _run_statsmodels), ("linearmodels", _run_linearmodels):
            t0 = time.perf_counter()
            payload, status = runner(system)
            wall = time.perf_counter() - t0
            if status != "ok":
                results.append(
                    BenchmarkResult(
                        K=K,
                        N=N,
                        p=p,
                        k=k,
                        method=name,
                        beta_rmse=None,
                        test_nll=None,
                        peak_memory=None,
                        wall_time=wall,
                        status=status,
                    )
                )
                continue

            beta = _stack_beta(payload["B"])
            beta_rmse = float(np.sqrt(np.mean((beta - _stack_beta(B_true)) ** 2)))
            resid_te = Y_te - XB_from_Blist(X_te, payload["B"])
            nll = float(_gaussian_nll(resid_te, payload["sigma"]))
            results.append(
                BenchmarkResult(
                    K=K,
                    N=N,
                    p=p,
                    k=k,
                    method=name,
                    beta_rmse=beta_rmse,
                    test_nll=nll,
                    peak_memory=None,
                    wall_time=wall,
                    status="ok",
                )
            )

    return results


def parse_grid(K_vals, N_vals, p_vals, k_vals):
    return list(itertools.product(K_vals, N_vals, p_vals, k_vals))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", nargs="*", type=int, default=[20, 40])
    parser.add_argument("--N", nargs="*", type=int, default=[200])
    parser.add_argument("--p", nargs="*", type=int, default=[3])
    parser.add_argument("--k", nargs="*", type=int, default=[2, 4])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--memory-backend",
        choices=["memray", "fil", "resource", "none"],
        default="resource",
    )
    parser.add_argument("--json", type=Path, help="Optional path to dump JSON results")
    args = parser.parse_args(argv)

    grid = parse_grid(args.K, args.N, args.p, args.k)
    results = run_benchmark(grid, seed=args.seed, memory_backend=args.memory_backend)

    for row in results:
        print(
            f"K={row.K:3d} N={row.N:4d} p={row.p:2d} k={row.k:2d} | {row.method:12s} "
            f"beta_RMSE={row.beta_rmse!r} test_NLL={row.test_nll!r} peak_mem={row.peak_memory!r} "
            f"time={row.wall_time:.2f}s status={row.status}"
        )

    if args.json:
        args.json.write_text(json.dumps([row.__dict__ for row in results], indent=2))


if __name__ == "__main__":
    main()
