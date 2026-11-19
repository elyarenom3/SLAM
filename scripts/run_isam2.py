"""Optional extra: GTSAM's incremental solver iSAM2 on the MR.CLAM factor graphs.

For each selected run the frozen factor set of the batch graph is fed to iSAM2
one keyframe at a time (only the new factors and new variables per update).
Recorded per run: the causal ("online") pose error, the difference between the
final incremental estimate and the batch GTSAM solution, and timing.

Usage: python scripts/run_isam2.py [--datasets 2-8] [--robot 3]
Output: results/isam2.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import argparse
import csv
import time

import numpy as np

from run_mrclam import parse_range
from slam2d import pipeline as P
from slam2d.data.mrclam import load_dataset
from slam2d.geometry import se2_compose, wrap
from slam2d.models import inverse_measure


def run(d: int, r: int, cfg: dict) -> dict:
    import gtsam
    from gtsam.symbol_shorthand import L, X
    from slam2d import gtsam_backend as gb

    ds = load_dataset("data/raw", d)
    rb = ds.robot(r)
    rc = P.robot_cfg(cfg, r)
    st = P.stream_for(rb, cfg)
    saved = np.load(f"results/mrclam/runs/ds{d}_r{r}.npz")
    keep = saved["graph_keep"]
    problem, lm_ids, rows, _ = P.build_problem(st, rc, cfg, keep)

    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(0.01)
    params.relinearizeSkip = 1
    isam = gtsam.ISAM2(params)

    by_pose: dict[int, list] = {}
    for m, k in enumerate(problem.br_pose):
        by_pose.setdefault(int(k), []).append(m)
    K1 = problem.n_poses
    online = np.empty((K1, 3))
    seen: set[int] = set()
    t_updates = np.empty(K1)

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(*problem.prior_mean),
                                     gtsam.noiseModel.Diagonal.Sigmas(problem.prior_sigma)))
    values.insert(X(0), gtsam.Pose2(*problem.prior_mean))
    cur = problem.prior_mean.copy()
    t_total = time.perf_counter()
    for k in range(K1):
        if k > 0:
            graph = gtsam.NonlinearFactorGraph()
            values = gtsam.Values()
            graph.add(gtsam.BetweenFactorPose2(X(k - 1), X(k), gtsam.Pose2(*problem.odo_z[k - 1]),
                                               gtsam.noiseModel.Gaussian.Covariance(problem.odo_cov[k - 1])))
            cur = se2_compose(cur, problem.odo_z[k - 1])
            values.insert(X(k), gtsam.Pose2(*cur))
        for m in by_pose.get(k, ()):
            j = int(problem.br_lm[m])
            rng, phi = problem.br_z[m]
            sr, sp = problem.br_sigma[m]
            graph.add(gtsam.BearingRangeFactor2D(X(k), L(j), gtsam.Rot2(phi), rng,
                                                 gtsam.noiseModel.Diagonal.Sigmas(np.array([sp, sr]))))
            if j not in seen:
                seen.add(j)
                values.insert(L(j), gtsam.Point2(*inverse_measure(cur, (rng, phi))))
        t0 = time.perf_counter()
        isam.update(graph, values)
        pose = isam.calculateEstimatePose2(X(k))
        t_updates[k] = time.perf_counter() - t0
        cur = np.array([pose.x(), pose.y(), pose.theta()])
        online[k] = cur
    t_total = time.perf_counter() - t_total

    final = isam.calculateEstimate()
    Xf = np.array([[final.atPose2(X(k)).x(), final.atPose2(X(k)).y(), final.atPose2(X(k)).theta()]
                   for k in range(K1)])
    batch = gb.solve(problem, saved["fej_pose"], np.array([saved["fej_lm"][list(saved["fej_lm_ids"]).index(j)]
                                                           for j in lm_ids]),
                     method="lm", tol=1e-13, with_marginals=False)
    ok = st.gt_valid.copy()
    ok[0] = False

    def rmse(pose):
        e = st.gt[ok, :2] - pose[ok, :2]
        return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))

    return {"dataset": d, "robot": r, "n_poses": K1, "n_factors": len(problem.br_z) + len(problem.odo_z) + 1,
            "online_pos_rmse": rmse(online), "final_pos_rmse": rmse(Xf), "batch_pos_rmse": rmse(batch["X"]),
            "fej_pos_rmse": rmse(saved["fej_pose"]), "ekf_pos_rmse": rmse(saved["ekf_pose"]),
            "final_vs_batch_max_pos": float(np.abs(Xf[:, :2] - batch["X"][:, :2]).max()),
            "final_vs_batch_max_heading": float(np.abs(wrap(Xf[:, 2] - batch["X"][:, 2])).max()),
            "update_ms_median": float(1e3 * np.median(t_updates)), "update_ms_p99": float(1e3 * np.percentile(t_updates, 99)),
            "isam2_total_s": t_total, "batch_solve_s": batch["runtime"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="2-8")
    ap.add_argument("--robot", type=int, default=3)
    args = ap.parse_args()
    cfg = P.load_config()
    rows = []
    for d in parse_range(args.datasets):
        row = run(d, args.robot, cfg)
        rows.append(row)
        print(f"ds{d} r{args.robot}: {row['n_poses']} poses | position RMSE online {row['online_pos_rmse']:.3f} m, "
              f"FEJ filter {row['fej_pos_rmse']:.3f} m, batch {row['batch_pos_rmse']:.3f} m | final vs batch "
              f"{row['final_vs_batch_max_pos']:.1e} m | update {row['update_ms_median']:.2f} ms median, "
              f"{row['update_ms_p99']:.2f} ms p99 | total {row['isam2_total_s']:.1f} s, batch {row['batch_solve_s']:.2f} s",
              flush=True)
    with open("results/isam2.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"median over {len(rows)} runs: online {med('online_pos_rmse'):.3f} m, FEJ {med('fej_pos_rmse'):.3f} m, "
          f"standard EKF {med('ekf_pos_rmse'):.3f} m, batch {med('batch_pos_rmse'):.3f} m; "
          f"final vs batch {med('final_vs_batch_max_pos'):.1e} m; update {med('update_ms_median'):.2f} ms")


if __name__ == "__main__":
    main()
