"""Consistency and accuracy metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .geometry import wrap


def pose_error(x_true, x_est) -> np.ndarray:
    """true - estimate, with the heading component wrapped. Shape (..., 3)."""
    e = np.asarray(x_true, dtype=float) - np.asarray(x_est, dtype=float)
    e[..., 2] = wrap(e[..., 2])
    return e


def nees(e, P) -> float:
    """Normalised estimation error squared e^T P^-1 e for one sample."""
    return float(e @ np.linalg.solve(P, e))


def nees_series(e, P) -> np.ndarray:
    """NEES for stacked errors (T, d) and covariances (T, d, d).

    Samples whose covariance is singular or not finite (a diverged filter)
    get NEES = inf instead of raising.
    """
    e = np.asarray(e, dtype=float)
    P = np.asarray(P, dtype=float)
    try:
        sol = np.linalg.solve(P, e[..., None])[..., 0]
        return np.einsum("ti,ti->t", e, sol)
    except np.linalg.LinAlgError:
        out = np.full(len(e), np.inf)
        for i in range(len(e)):
            try:
                out[i] = e[i] @ np.linalg.solve(P[i], e[i])
            except np.linalg.LinAlgError:
                pass
        return out


def chi2_bounds(dof: int, alpha: float = 0.05):
    """Two-sided (1 - alpha) interval of a chi-squared variable."""
    return chi2.ppf([alpha / 2, 1 - alpha / 2], dof)


def anees_bounds(n_runs: int, dof: int = 3, alpha: float = 0.05):
    """Two-sided (1 - alpha) interval for the average NEES over n_runs runs."""
    return chi2.ppf([alpha / 2, 1 - alpha / 2], dof * n_runs) / n_runs


def frac_inside(values, lo: float, hi: float) -> float:
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return float(np.mean((v >= lo) & (v <= hi))) if v.size else float("nan")


def rmse(e) -> float:
    e = np.asarray(e)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def position_rmse(err) -> float:
    """RMSE of the planar position error for errors (T, >=2)."""
    err = np.asarray(err)
    d2 = err[:, 0] ** 2 + err[:, 1] ** 2
    d2 = d2[np.isfinite(d2)]
    return float(np.sqrt(np.mean(d2))) if d2.size else float("nan")


def truncated_nis_mean(gate: float, dof: int = 2) -> float:
    """Expected NIS of accepted measurements under a chi-squared gate (dof = 2)."""
    if dof != 2:
        raise NotImplementedError
    # chi2 with 2 dof is exponential with mean 2
    p = 1.0 - np.exp(-gate / 2.0)
    return float((2.0 - (gate + 2.0) * np.exp(-gate / 2.0)) / p)
