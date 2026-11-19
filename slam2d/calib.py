"""Noise and sensor calibration against ground truth.

Used on MR.CLAM dataset 1 only. The fitted values are written once to
config/frozen_mrclam.json and then applied unchanged to datasets 2-9.

Corrections applied to the raw data (all fitted here):
- odometry: v' = max(s_v * v - c_v * |w|, 0),  w' = s_w * w + b_w   (per robot);
            the c_v term models the drop in forward speed while turning
- camera:   time shift tau, range scale/offset, bearing offset and a lever
            arm (lx, ly) of the camera in the body frame, applied as a
            pre-correction so the filter keeps the plain range-bearing model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .geometry import wrap

MAD_K = 1.4826


def mad_sigma(x) -> float:
    """Robust sigma: 1.4826 x median absolute deviation about the median."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(MAD_K * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


# ---------------------------------------------------------------- camera
def camera_correct(r, phi, cam):
    """Map raw (range, bearing) to the equivalent measurement from the robot centre.

    cam = dict(a, b, beta, lx, ly): raw_range = a * true_range + b, raw
    bearing = true bearing - beta, camera at (lx, ly) in the body frame.
    """
    rc = (np.asarray(r) - cam["b"]) / cam["a"]
    pc = np.asarray(phi) + cam["beta"]
    x = rc * np.cos(pc) + cam["lx"]
    y = rc * np.sin(pc) + cam["ly"]
    return np.hypot(x, y), np.arctan2(y, x)


IDENTITY_CAM = {"a": 1.0, "b": 0.0, "beta": 0.0, "lx": 0.0, "ly": 0.0}


def measurement_residuals(rb, landmarks, tau: float = 0.0, cam=None):
    """Residuals z - h(ground truth) for every landmark sighting with valid ground truth.

    Returns dict with r (measured, corrected), er, ephi, subject, t.
    """
    m = rb.meas
    a, b = rb.window()
    m = m[(m[:, 0] > a) & (m[:, 0] <= b)]
    pose, valid = rb.gt_at(m[:, 0] + tau)
    lm = np.array([landmarks[int(s)] for s in m[:, 1]])
    d = lm - pose[:, :2]
    r_hat = np.hypot(d[:, 0], d[:, 1])
    phi_hat = wrap(np.arctan2(d[:, 1], d[:, 0]) - pose[:, 2])
    r, phi = camera_correct(m[:, 2], m[:, 3], cam or IDENTITY_CAM)
    return {"t": m[valid, 0], "subject": m[valid, 1], "r": r[valid], "phi": phi[valid],
            "r_hat": r_hat[valid], "phi_hat": phi_hat[valid],
            "er": (r - r_hat)[valid], "ephi": wrap(phi - phi_hat)[valid],
            "raw_r": m[valid, 2], "raw_phi": m[valid, 3],
            "pose": pose[valid], "lm": lm[valid]}


def stationary_mask(rb, odo_tau: float, t) -> np.ndarray:
    """True where the command in force at time t is v = w = 0 (robot told to stand still)."""
    ot = rb.odom[:, 0] + odo_tau
    i = np.clip(np.searchsorted(ot, np.asarray(t), side="right") - 1, 0, None)
    return (rb.odom[i, 1] == 0) & (rb.odom[i, 2] == 0)


def block_kappa(t, subject, z, block: float = 10.0, min_count: int = 3) -> float:
    """Factor by which white noise must be inflated to explain block means.

    z are residuals already divided by their sigma. For white noise the mean
    of n residuals of one landmark inside a time block, times sqrt(n), has
    unit spread. Temporally correlated errors give a larger spread; its
    robust scale is the returned factor (at least 1).
    """
    key = (np.asarray(t) // block).astype(np.int64) * 1000 + np.asarray(subject).astype(np.int64)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    mean = np.bincount(inv, weights=np.clip(z, -4.0, 4.0)) / cnt
    sel = cnt >= min_count
    return float(max(1.0, MAD_K * np.median(np.abs(mean[sel] * np.sqrt(cnt[sel])))))


def fit_camera_time_offset(rb, landmarks, grid=None) -> float:
    """Time shift of the camera stamps that minimises the robust bearing spread."""
    grid = np.arange(-0.30, 0.3001, 0.01) if grid is None else grid
    score = [mad_sigma(measurement_residuals(rb, landmarks, tau)["ephi"]) for tau in grid]
    return float(grid[int(np.argmin(score))])


def fit_camera(res) -> dict:
    """Robust fit of range scale/offset, bearing offset and lever arm for one robot."""
    pose, lm = res["pose"], res["lm"]
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    d = lm - pose[:, :2]
    bx = c * d[:, 0] + s * d[:, 1]          # landmark in the body frame
    by = -s * d[:, 0] + c * d[:, 1]

    def fun(p):
        a, b, beta, lx, ly = p
        cx, cy = bx - lx, by - ly
        return np.concatenate([(a * np.hypot(cx, cy) + b - res["raw_r"]) / 0.15,
                               wrap(np.arctan2(cy, cx) - beta - res["raw_phi"]) / 0.015])

    sol = least_squares(fun, [1.0, 0.0, 0.0, 0.0, 0.0], loss="soft_l1", f_scale=1.5)
    return dict(zip(("a", "b", "beta", "lx", "ly"), map(float, sol.x)))


def range_sigma_model(r, er, n_bins: int = 8, n_boot: int = 300, seed: int = 0) -> dict:
    """Fit sigma_r(r) = c0 + c1 * r on quantile bins of range, with a bootstrap slope interval."""
    r, er = np.asarray(r), np.asarray(er)
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)

    def fit(rr, ee, ii):
        mid = np.array([np.median(rr[ii == k]) for k in range(n_bins)])
        sig = np.array([mad_sigma(ee[ii == k]) for k in range(n_bins)])
        c1, c0 = np.polyfit(mid, sig, 1)
        return c0, c1, mid, sig

    c0, c1, mid, sig = fit(r, er, idx)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_boot):
        j = rng.integers(0, len(r), len(r))
        slopes.append(fit(r[j], er[j], idx[j])[1])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return {"c0": float(c0), "c1": float(c1), "slope_ci": (float(lo), float(hi)),
            "bins_r": mid.tolist(), "bins_sigma": sig.tolist(),
            "range_dependent": bool(lo > 0 or hi < 0)}


# -------------------------------------------------------------- odometry
def odometry_windows(rb, T: float, odo=None, max_windows: int = 100000):
    """Body-frame odometry increments against ground truth over non-overlapping windows.

    Each window starts from the ground-truth pose, integrates the (corrected)
    commands with zero-order hold, and is compared with the ground-truth pose
    at the window end. Returns dict of arrays: d_cmd, dth_cmd (integrated
    command), d_gt (along-track), dth_gt, and the errors e_along, e_lat, e_th
    of the integrated pose in the start frame.
    """
    odo = odo or {"sv": 1.0, "sw": 1.0, "bw": 0.0, "tau": 0.0}
    a, b = rb.window()
    t = rb.odom[:, 0] + odo.get("tau", 0.0)
    v, w = correct_controls(rb.odom[:, 1], rb.odom[:, 2], odo)
    starts = np.arange(a + 1.0, b - T - 1.0, T)[:max_windows]
    p0, ok0 = rb.gt_at(starts)
    p1, ok1 = rb.gt_at(starts + T)
    out = {k: [] for k in ("d_cmd", "dth_cmd", "a_cmd", "d_gt", "dth_gt", "e_along", "e_lat", "e_th", "t")}
    abs_w_raw = np.abs(rb.odom[:, 2])
    for k in np.flatnonzero(ok0 & ok1):
        ts, te = starts[k], starts[k] + T
        i0 = np.searchsorted(t, ts, side="right") - 1
        i1 = np.searchsorted(t, te, side="left")
        if i0 < 0:
            continue
        edges = np.concatenate([[ts], t[i0 + 1:i1], [te]])
        dts = np.diff(edges)
        vv, ww = v[i0:i0 + len(dts)], w[i0:i0 + len(dts)]
        th = np.concatenate([[0.0], np.cumsum(ww * dts)])
        x = np.sum(vv * dts * np.cos(th[:-1]))
        y = np.sum(vv * dts * np.sin(th[:-1]))
        c, s = math.cos(p0[k, 2]), math.sin(p0[k, 2])
        dx, dy = p1[k, 0] - p0[k, 0], p1[k, 1] - p0[k, 1]
        gx, gy = c * dx + s * dy, -s * dx + c * dy
        dth_gt = float(wrap(p1[k, 2] - p0[k, 2]))
        out["d_cmd"].append(np.sum(vv * dts))
        out["dth_cmd"].append(th[-1])
        out["a_cmd"].append(np.sum(abs_w_raw[i0:i0 + len(dts)] * dts))
        out["d_gt"].append(math.hypot(gx, gy))
        out["dth_gt"].append(dth_gt)
        out["e_along"].append(gx - x)
        out["e_lat"].append(gy - y)
        out["e_th"].append(float(wrap(dth_gt - th[-1])))
        out["t"].append(ts)
    return {k: np.asarray(val) for k, val in out.items()}


def correct_controls(v, w, odo: dict):
    """Apply the fitted odometry correction to raw commands (v, w)."""
    v_eff = np.maximum(odo["sv"] * v - odo.get("cv", 0.0) * np.abs(w), 0.0)
    return v_eff, odo["sw"] * w + odo["bw"]


def fit_odometry_scale(rb, T: float = 1.0, tau: float = 0.0) -> dict:
    """Robust odometry correction from windows of length T.

    Distance: d_true = s_v * d_cmd - c_v * integral |w_cmd| dt (forward speed
    drops while turning). Heading: dth_true = s_w * dth_cmd + b_w * T.
    """
    win = odometry_windows(rb, T, {"sv": 1.0, "sw": 1.0, "bw": 0.0, "tau": tau})

    def robust(X, y, scale):
        x0 = np.linalg.lstsq(X, y, rcond=None)[0]
        return least_squares(lambda p: (X @ p - y) / scale, x0, loss="soft_l1").x

    moving = win["d_cmd"] > 0.02 * T
    sv, cv = robust(np.column_stack([win["d_cmd"][moving], -win["a_cmd"][moving]]), win["d_gt"][moving], 0.005)
    sw, bw = robust(np.column_stack([win["dth_cmd"], np.ones_like(win["dth_cmd"])]), win["dth_gt"], 0.01)
    return {"sv": float(sv), "cv": float(cv), "sw": float(sw), "bw": float(bw / T), "tau": float(tau)}


def fit_odometry_time_offset(rb, grid=None, T: float = 0.5) -> float:
    """Shift of the odometry stamps that best aligns commanded and true heading change."""
    grid = np.arange(-0.30, 0.3001, 0.02) if grid is None else grid
    score = []
    for tau in grid:
        win = odometry_windows(rb, T, {"sv": 1.0, "sw": 1.0, "bw": 0.0, "tau": tau}, max_windows=1500)
        score.append(mad_sigma(win["e_th"]))
    return float(grid[int(np.argmin(score))])


def turn_flag(odom_t, odom_w, t, mem: float, thr: float) -> np.ndarray:
    """True where a turn was commanded within the last `mem` seconds before t.

    A time t is "turning" when max |w_cmd| over [t - mem, t] exceeds thr. The
    robots' heading error is several times larger in that state.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    active = np.abs(odom_w) > thr
    csum = np.concatenate([[0], np.cumsum(active)])
    hi = np.searchsorted(odom_t, t, side="right")            # samples with stamp <= t
    lo = np.clip(np.searchsorted(odom_t, t - mem, side="right") - 1, 0, None)  # sample in force at t - mem
    return (csum[hi] - csum[lo]) > 0


def fit_two_state_noise(rb, odo: dict, T: float, mem: float, thr: float, dt_nom: float,
                        vicon=(0.0, 0.0)) -> dict:
    """Per-sample control-noise sigmas for the straight and turning states.

    Each state's sigma comes from the windows that lie entirely in that state:
    the standard deviation of the window error, with Vicon noise removed, is
    converted to the white per-sample sigma that reproduces it. The standard
    deviation (not a robust scale) is the target on purpose: control errors
    cannot be gated, so the model has to cover the whole distribution.
    """
    w = odometry_windows(rb, T, odo)
    ot = rb.odom[:, 0] + odo.get("tau", 0.0)
    grid = np.linspace(0.0, T, 26)[None, :] + w["t"][:, None]
    f_t = turn_flag(ot, rb.odom[:, 2], grid.ravel(), mem, thr).reshape(grid.shape).mean(axis=1)
    out = {"turn_fraction": float(f_t.mean()), "n_straight": int(np.sum(f_t == 0)),
           "n_turning": int(np.sum(f_t == 1))}
    for key, err, vic in (("w", w["e_th"], vicon[1]), ("v", w["e_along"], vicon[0])):
        for state, sel in (("s", f_t == 0), ("t", f_t == 1)):
            var = max(float(np.mean(err[sel] ** 2)) - 2 * vic ** 2, 1e-12)
            out[f"sigma_{key}_{state}"] = per_sample_sigma(math.sqrt(var), T, dt_nom)
    return out


def per_sample_sigma(sigma_window: float, T: float, dt_nom: float) -> float:
    """White per-sample sigma that reproduces a window error sigma.

    White noise of sigma s on each sample of length dt_nom integrates to
    variance s^2 * T * dt_nom over a window of length T.
    """
    return float(sigma_window / math.sqrt(T * dt_nom))
