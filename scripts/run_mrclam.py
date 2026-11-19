"""Run every estimator on MR.CLAM robot logs and score them against ground truth.

Methods per run:
  ekf, fej        standard and First-Estimates-Jacobian EKF-SLAM (frozen noise model)
  ekf_mad, fej_mad the same filters with white MAD-based measurement noise (ablation)
  graph           own optimiser (Grisetti residual), factor set re-gated at the solution
  graph_ekfset    own optimiser on the measurements the FEJ filter accepted (ablation)
  gtsam           GTSAM on the same factor set and initial values as `graph`
An agreement record compares the own optimiser with GTSAM on every run.

Datasets 2-9 are refused until config/frozen_mrclam.json is marked frozen
(python scripts/fit_noise.py --freeze). The configuration hash is written to
every summary row.

Usage: python scripts/run_mrclam.py [--datasets 1-9] [--robots 1-5] [--workers N]
Outputs: results/mrclam/summary.csv, agreement.csv, runs/ds{D}_r{R}.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import argparse
import csv
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from slam2d import graph as gr
from slam2d import pipeline as P
from slam2d.data.mrclam import load_dataset
from slam2d.geometry import wrap

OUT = Path("results/mrclam")


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def run_one(job):
    """Never raises: a failed run is returned as one row with the traceback."""
    try:
        return _run_one(job)
    except Exception:
        _, d, r, cfg, _ = job
        return [{"dataset": d, "robot": r, "config": cfg["sha256"][:12], "method": "ERROR",
                 "error": traceback.format_exc(limit=4), "run_wall_s": 0.0}], None


def _run_one(job):
    root, d, r, cfg, with_gtsam = job
    t_run = time.perf_counter()
    ds = load_dataset(root, d)
    rb = ds.robot(r)
    rc = P.robot_cfg(cfg, r)
    st = P.stream_for(rb, cfg)
    base = {"dataset": d, "robot": r, "config": cfg["sha256"][:12], "duration_s": float(st.kf_t[-1] - st.kf_t[0])}
    rows, arrays = [], {}
    arrays.update(kf_t=st.kf_t, gt=st.gt, gt_valid=st.gt_valid, meas_kf=st.meas_kf, meas_id=st.meas_id,
                  meas_z=st.meas_z, lm_gt_ids=np.array(sorted(ds.landmarks)),
                  lm_gt=np.array([ds.landmarks[j] for j in sorted(ds.landmarks)]))

    def record(name, pose, Pcov, lm, lm_cov, extra, nis=None, status=None, keep_arrays=True):
        s, nees = P.score(st, pose, Pcov, ds.landmarks, lm, lm_cov, cfg["vicon_sigma"], nis, status)
        rows.append({**base, "method": name, **s, **extra})
        if keep_arrays:
            ids = sorted(lm)
            arrays.update({f"{name}_pose": pose, f"{name}_P": Pcov.astype(np.float32),
                           f"{name}_nees": nees.astype(np.float32), f"{name}_lm_ids": np.array(ids),
                           f"{name}_lm": np.array([lm[j] for j in ids]).reshape(-1, 2),
                           f"{name}_lm_cov": np.array([lm_cov[j] for j in ids]).reshape(-1, 2, 2)})
            if nis is not None:
                arrays.update({f"{name}_nis": nis.astype(np.float32), f"{name}_status": status.astype(np.int8)})

    # ---- filters
    ek = {}
    for mode, name in (("standard", "ekf"), ("fej", "fej")):
        ek[name] = res = P.run_ekf(st, rc, cfg, mode)
        record(name, res["pose"], res["P"], res["lm"], res["lm_cov"], {"runtime_s": res["runtime"]},
               res["nis"], res["status"])
    cfg_mad = {**cfg, "r_inflation": False}
    for mode, name in (("standard", "ekf_mad"), ("fej", "fej_mad")):
        res = P.run_ekf(st, rc, cfg_mad, mode)
        record(name, res["pose"], res["P"], res["lm"], res["lm_cov"], {"runtime_s": res["runtime"]},
               res["nis"], res["status"], keep_arrays=False)

    # ---- own graph optimiser
    g = P.run_graph(st, rc, cfg, ek["fej"])
    record("graph", g["pose"], g["P"], g["lm"], g["lm_cov"],
           {"runtime_s": g["runtime"], "chi2_dof": g["chi2_dof"], "iterations": g["iterations"],
            "rounds": g["rounds"], "n_factors": g["n_factors_br"], "converged": g["converged"]})
    arrays["graph_keep"] = g["keep"]
    g0 = P.run_graph(st, rc, cfg, ek["fej"], regate=False)
    record("graph_ekfset", g0["pose"], g0["P"], g0["lm"], g0["lm_cov"],
           {"runtime_s": g0["runtime"], "chi2_dof": g0["chi2_dof"], "iterations": g0["iterations"],
            "rounds": 1, "n_factors": g0["n_factors_br"], "converged": g0["converged"]}, keep_arrays=False)

    # ---- GTSAM on the same factors and initial values
    agree = None
    if with_gtsam:
        try:
            from slam2d import gtsam_backend as gb
            problem, lm_ids = g["problem"], g["lm_ids"]
            X0 = ek["fej"]["pose"]
            L0 = np.array([ek["fej"]["lm"][j] for j in lm_ids])
            conv = gb.pose_convention()
            solver = gr.GraphSLAM(problem, conv)
            own = solver.optimise(X0, L0, tol=1e-9, max_iter=200)
            own_Pp, own_Pl = solver.marginals(own["X"], own["L"])
            ref = gb.solve(problem, X0, L0, method="lm", tol=1e-13, max_iter=200)
            record("gtsam", ref["X"], ref["P_pose"], {j: ref["L"][i] for i, j in enumerate(lm_ids)},
                   {j: ref["P_lm"][i] for i, j in enumerate(lm_ids)},
                   {"runtime_s": ref["runtime"] + ref["marginals_runtime"], "chi2_dof": ref["cost"] / problem.dof,
                    "iterations": ref["iterations"], "n_factors": g["n_factors_br"]})
            scale = np.abs(own_Pp).max(axis=(1, 2))
            agree = {**base, "convention": conv, "n_poses": problem.n_poses, "n_landmarks": problem.n_landmarks,
                     "n_factors": len(problem.br_z) + len(problem.odo_z) + 1,
                     "cost_own": own["cost"], "cost_gtsam": ref["cost"],
                     "cost_rel_diff": abs(own["cost"] - ref["cost"]) / ref["cost"],
                     "max_pos_diff": float(np.abs(own["X"][:, :2] - ref["X"][:, :2]).max()),
                     "max_heading_diff": float(np.abs(wrap(own["X"][:, 2] - ref["X"][:, 2])).max()),
                     "max_lm_diff": float(np.abs(own["L"] - ref["L"]).max()),
                     "max_pose_cov_rel_diff": float((np.abs(own_Pp - ref["P_pose"]).max(axis=(1, 2)) / scale).max()),
                     "max_lm_cov_rel_diff": float((np.abs(own_Pl - ref["P_lm"]).max(axis=(1, 2))
                                                   / np.abs(own_Pl).max(axis=(1, 2))).max()),
                     "iterations_own": own["iterations"], "iterations_gtsam": ref["iterations"],
                     "runtime_own_s": own["runtime"], "runtime_gtsam_s": ref["runtime"],
                     "grad_inf_own": own["grad_inf"],
                     # the Grisetti-residual optimum against GTSAM's (different pose error on GTSAM 4.3)
                     "grisetti_cost": g["cost"], "grisetti_cost_rel_diff": abs(g["cost"] - ref["cost"]) / ref["cost"],
                     "grisetti_max_pos_diff": float(np.abs(g["pose"][:, :2] - ref["X"][:, :2]).max()),
                     "grisetti_max_heading_diff": float(np.abs(wrap(g["pose"][:, 2] - ref["X"][:, 2])).max())}
        except Exception:
            agree = {**base, "error": traceback.format_exc(limit=3)}

    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "runs" / f"ds{d}_r{r}.npz", **arrays)
    for row in rows:
        row["run_wall_s"] = time.perf_counter() - t_run
    return rows, agree


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        keys.extend(k for k in row if k not in keys)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="1-9")
    ap.add_argument("--robots", default="1-5")
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--config", default="config/frozen_mrclam.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-gtsam", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the summary files (e.g. a development run)")
    args = ap.parse_args()

    cfg = P.load_config(args.config)
    datasets = parse_range(args.datasets)
    if any(d != 1 for d in datasets) and not cfg.get("frozen"):
        sys.exit("refusing datasets 2-9: the configuration is not frozen "
                 "(develop on dataset 1, then run scripts/fit_noise.py --freeze)")
    jobs = [(args.root, d, r, cfg, not args.no_gtsam) for d in datasets for r in parse_range(args.robots)]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows, agree = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for run_rows, run_agree in ex.map(run_one, jobs):
            rows.extend(run_rows)
            if run_agree is not None:
                agree.append(run_agree)
            r0 = run_rows[0]
            print(f"ds{r0['dataset']} r{r0['robot']} done ({r0['run_wall_s']:.0f} s)", flush=True)
    suffix = f"_{args.tag}" if args.tag else ""
    write_csv(OUT / f"summary{suffix}.csv", rows)
    if agree:
        write_csv(OUT / f"agreement{suffix}.csv", agree)
    print(f"{len(jobs)} runs in {time.time() - t0:.0f} s -> {OUT}/summary{suffix}.csv "
          f"(config {cfg['sha256'][:12]}, frozen={cfg.get('frozen', False)})")


if __name__ == "__main__":
    main()
