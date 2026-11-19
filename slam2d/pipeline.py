"""Run the estimators on one MR.CLAM robot log and score them against ground truth."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .ekf import EKFSLAM, ACCEPT, REJECT
from .geometry import wrap
from .metrics import chi2_bounds, frac_inside, nees_series
from .odom import Stream, build_stream

NEES_LO, NEES_HI = chi2_bounds(3)
NIS_LO, NIS_HI = chi2_bounds(2)
SIGMA_R_MIN = 0.03     # [m] lower bound of the range-noise model


# ----------------------------------------------------------------- config
def load_config(path="config/frozen_mrclam.json") -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return cfg


def robot_cfg(cfg: dict, robot: int) -> dict:
    return cfg["robots"][str(robot)]


def _kappa(rc: dict, cfg: dict):
    """Correlation inflation of the measurement sigmas (1 when switched off)."""
    if cfg.get("r_inflation", True):
        return rc["noise"].get("kappa_r", 1.0), rc["noise"].get("kappa_phi", 1.0)
    return 1.0, 1.0


def noise_R(rc: dict, cfg: dict):
    """Measurement covariance as a function of the measured range."""
    kr, kp = _kappa(rc, cfg)
    c0, c1, sp = rc["noise"]["r_c0"], rc["noise"]["r_c1"], kp * rc["noise"]["sigma_phi"]

    def R(r):
        return np.diag([(kr * max(c0 + c1 * r, SIGMA_R_MIN)) ** 2, sp ** 2])
    return R


def noise_R_array(rc: dict, cfg: dict, r: np.ndarray) -> np.ndarray:
    """(n, 2) sigmas [range, bearing] for an array of measured ranges."""
    n = rc["noise"]
    kr, kp = _kappa(rc, cfg)
    return np.column_stack([kr * np.maximum(n["r_c0"] + n["r_c1"] * r, SIGMA_R_MIN),
                            np.full(len(r), kp * n["sigma_phi"])])


def P0_from(cfg: dict) -> np.ndarray:
    pxy, pth = cfg["p0"]
    return np.diag([pxy ** 2, pxy ** 2, pth ** 2])


def stream_for(rb, cfg: dict, t_end=None) -> Stream:
    rc = robot_cfg(cfg, rb.robot)
    return build_stream(rb, rc["odo"], rc["cam"], cfg["camera_tau"], cfg["merge_dt"],
                        cfg["max_step"], t_end, cfg.get("skip_stationary", False))


# -------------------------------------------------------------------- EKF
def run_ekf(stream: Stream, rc: dict, cfg: dict, mode: str = "standard") -> dict:
    """Run the EKF over a stream. Returns estimates at every keyframe."""
    t_start = time.perf_counter()
    R = noise_R(rc, cfg)
    qs = cfg.get("q_inflation", 1.0)
    ekf = EKFSLAM(stream.x0, P0_from(cfg), np.zeros((2, 2)), n_max=15, mode=mode,
                  gate=cfg["gate"], dt_nom=cfg["dt_nom"], floor=cfg["floor"],
                  confirm_dist=cfg.get("confirm_dist"))
    K1 = stream.n_keyframes
    pose = np.empty((K1, 3))
    Prr = np.empty((K1, 3, 3))
    pose[0], Prr[0] = ekf.x[:3], ekf.P[:3, :3]
    nis = np.full(len(stream.meas_id), np.nan)
    status = np.zeros(len(stream.meas_id), dtype=int)
    steps = stream.steps.tolist()
    seg, ptr = stream.seg, stream.kf_ptr
    ids, zs = stream.meas_id.tolist(), stream.meas_z
    for k in range(1, K1):
        for v, w, dt, qv, qw in steps[seg[k - 1]:seg[k]]:
            ekf.predict(v, w, dt, q=(qs * qv, qs * qw))
        a, b = ptr[k], ptr[k + 1]
        res = ekf.update_batch(ids[a:b], zs[a:b], R)
        _assign(res, ids[a:b], nis, status, a)
        pose[k], Prr[k] = ekf.x[:3], ekf.P[:3, :3]
    lm = {j: ekf.landmark(j) for j in ekf.slot}
    lm_cov = {j: ekf.landmark_cov(j) for j in ekf.slot}
    return {"pose": pose, "P": Prr, "nis": nis, "status": status, "lm": lm, "lm_cov": lm_cov,
            "x_full": ekf.x[:ekf.n].copy(), "P_full": ekf.P[:ekf.n, :ekf.n].copy(),
            "slot": dict(ekf.slot), "runtime": time.perf_counter() - t_start}


def _assign(res, batch_ids, nis, status, offset):
    """Map update_batch results back onto measurement rows.

    update_batch returns results grouped (new landmarks, then known ones),
    each group in input order, so rows are matched per landmark id in order.
    """
    pending: dict[int, list] = {}
    for i, j in enumerate(batch_ids):
        pending.setdefault(j, []).append(i)
    for j, n, s in res:
        i = pending[j].pop(0)
        nis[offset + i] = n
        status[offset + i] = s


# ----------------------------------------------------------------- scoring
def score(stream: Stream, pose, P, landmarks_gt: dict, lm: dict, lm_cov: dict | None,
          vicon_sigma=(0.0, 0.0), nis=None, status=None, burn_in: int = 0) -> dict:
    """Accuracy and consistency of one run at the keyframes with valid ground truth."""
    ok = stream.gt_valid.copy()
    ok[:burn_in + 1] = False                 # the start pose is given, never scored
    err = stream.gt - pose
    err[:, 2] = wrap(err[:, 2])
    Pv = P + np.diag([vicon_sigma[0] ** 2, vicon_sigma[0] ** 2, vicon_sigma[1] ** 2])
    nees = np.full(len(pose), np.nan)
    nees[ok] = nees_series(err[ok], Pv[ok])
    e = err[ok]
    out = {
        "n_keyframes": int(len(pose)), "n_scored": int(ok.sum()),
        "pos_rmse": float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2))),
        "heading_rmse_deg": float(np.degrees(np.sqrt(np.mean(e[:, 2] ** 2)))),
        "pos_max": float(np.sqrt(np.max(e[:, 0] ** 2 + e[:, 1] ** 2))),
        # same trajectory after a rigid alignment: map shape error without the anchor error
        "pos_rmse_aligned": float(np.sqrt(np.mean(np.sum(
            (align_se2(pose[ok, :2], stream.gt[ok, :2]) - stream.gt[ok, :2]) ** 2, axis=1)))),
        "nees_mean": float(np.mean(nees[ok])), "nees_median": float(np.median(nees[ok])),
        "nees_inside": frac_inside(nees[ok], NEES_LO, NEES_HI),
    }
    # time-weighted mean NEES: each keyframe weighted by the time to the next one
    w = np.diff(stream.kf_t, append=stream.kf_t[-1])[ok]
    out["nees_time_mean"] = float(np.sum(w * nees[ok]) / np.sum(w)) if w.sum() > 0 else float("nan")
    last = np.flatnonzero(ok)[-1]
    out["nees_final"] = float(nees[last])

    le, ln = [], []
    for j, m in lm.items():
        if j in landmarks_gt:
            d = np.asarray(landmarks_gt[j]) - m
            le.append(d @ d)
            if lm_cov is not None:
                ln.append(float(d @ np.linalg.solve(lm_cov[j], d)))
    out["n_landmarks"] = len(lm)
    out["lm_rmse"] = float(np.sqrt(np.mean(le))) if le else float("nan")
    out["lm_nees_mean"] = float(np.mean(ln)) if ln else float("nan")

    if nis is not None:
        upd = (status == ACCEPT) | (status == REJECT)
        acc = status == ACCEPT
        out.update(n_meas=int(len(nis)), n_updates=int(upd.sum()), n_rejected=int((status == REJECT).sum()),
                   reject_frac=float((status == REJECT).sum() / max(upd.sum(), 1)),
                   nis_mean_all=float(np.mean(nis[upd])) if upd.any() else float("nan"),
                   nis_median_all=float(np.median(nis[upd])) if upd.any() else float("nan"),
                   nis_mean_accepted=float(np.mean(nis[acc])) if acc.any() else float("nan"),
                   nis_inside=frac_inside(nis[upd], NIS_LO, NIS_HI))
    return out, nees


# ------------------------------------------------------------------- graph
def build_problem(stream: Stream, rc: dict, cfg: dict, keep: np.ndarray, odo=None):
    """Factor graph for one run from the measurement rows flagged in `keep`.

    Returns (GraphProblem, lm_ids, rows) where rows are the measurement rows used.
    """
    from .graph import GraphProblem
    from .odom import preintegrate, to_end_frame

    if odo is None:
        z, S = preintegrate(stream, cfg["dt_nom"], cfg["floor"], cfg.get("q_inflation", 1.0))
        odo = (z, to_end_frame(z, S))
    rows = np.flatnonzero(keep)
    lm_ids = sorted(set(stream.meas_id[rows].tolist()))
    index = {j: i for i, j in enumerate(lm_ids)}
    pxy, pth = cfg["p0"]
    problem = GraphProblem(
        prior_mean=stream.x0.copy(), prior_sigma=np.array([pxy, pxy, pth]),
        odo_z=odo[0], odo_cov=odo[1],
        br_pose=stream.meas_kf[rows], br_lm=np.array([index[j] for j in stream.meas_id[rows]], dtype=int),
        br_z=stream.meas_z[rows], br_sigma=noise_R_array(rc, cfg, stream.meas_z[rows, 0]),
        lm_ids=lm_ids, n_poses=stream.n_keyframes)
    return problem, lm_ids, rows, odo


def run_graph(stream: Stream, rc: dict, cfg: dict, init: dict, pose_error: str = "grisetti",
              regate: bool = True, tol: float = 1e-6) -> dict:
    """Own graph optimiser on one run.

    init: an EKF result (run_ekf) providing initial poses, landmarks and the
    set of measurements the filter accepted. With `regate`, after each solve
    every measurement of a mapped landmark is tested at the solution
    (chi-squared with its own noise, threshold cfg["regate"]["threshold"])
    and the factor set is rebuilt, until it stops changing.
    """
    from .ekf import INIT
    from .graph import GraphSLAM, bearing_range_residual

    t_start = time.perf_counter()
    keep = (init["status"] == ACCEPT) | (init["status"] == INIT)
    X = init["pose"].copy()
    lm_est = {j: np.array(m, dtype=float) for j, m in init["lm"].items()}
    thr = cfg["regate"]["threshold"]
    sig_all = noise_R_array(rc, cfg, stream.meas_z[:, 0])
    odo = None
    rounds = 0
    total_iters = 0
    while True:
        problem, lm_ids, rows, odo = build_problem(stream, rc, cfg, keep, odo)
        solver = GraphSLAM(problem, pose_error)
        L0 = np.array([lm_est[j] for j in lm_ids])
        out = solver.optimise(X, L0, method="lm", tol=tol)
        X = out["X"]
        lm_est.update({j: out["L"][i] for i, j in enumerate(lm_ids)})
        total_iters += out["iterations"]
        rounds += 1
        if not regate or rounds >= cfg["regate"]["max_rounds"]:
            break
        known = np.array([j in lm_est for j in stream.meas_id])
        Lall = np.array([lm_est.get(j, (0.0, 0.0)) for j in stream.meas_id])
        e, _, _ = bearing_range_residual(X[stream.meas_kf], Lall, stream.meas_z)
        chi2 = np.sum((e / sig_all) ** 2, axis=1)
        new_keep = known & (chi2 <= thr)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    Pp, Pl = solver.marginals(out["X"], out["L"])
    return {"pose": out["X"], "P": Pp, "lm": {j: out["L"][i] for i, j in enumerate(lm_ids)},
            "lm_cov": {j: Pl[i] for i, j in enumerate(lm_ids)}, "cost": out["cost"],
            "dof": problem.dof, "chi2_dof": out["cost"] / max(problem.dof, 1),
            "iterations": out["iterations"], "total_iterations": total_iters, "rounds": rounds,
            "converged": out["converged"], "keep": keep, "problem": problem, "lm_ids": lm_ids,
            "n_factors_br": int(keep.sum()), "solve_runtime": out["runtime"],
            "runtime": time.perf_counter() - t_start, "L": out["L"]}


def align_se2(est_xy, gt_xy):
    """Rigid (rotation + translation) least-squares alignment of est onto gt. Returns aligned est."""
    mu_e, mu_g = est_xy.mean(axis=0), gt_xy.mean(axis=0)
    H = (est_xy - mu_e).T @ (gt_xy - mu_g)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return (est_xy - mu_e) @ R.T + mu_g
