"""Optional scale test: standard 2D pose graphs (INTEL, MITb, M3500) in g2o format.

The files come from Luca Carlone's dataset page (https://lucacarlone.mit.edu/datasets/).
Each graph is solved from the initial guess stored in the file by the own
optimiser and by GTSAM (which loads the same file with gtsam.readG2o), with a
tight prior on pose 0 in both.

Usage: python scripts/run_g2o.py
Output: results/g2o.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import csv
import shutil
import time
import urllib.request

import numpy as np

from slam2d import graph as gr
from slam2d import gtsam_backend as gb

FILES = {
    "INTEL": "https://www.dropbox.com/s/vcz8cag7bo0zlaj/input_INTEL_g2o.g2o?dl=1",
    "MITb": "https://www.dropbox.com/s/d8fcn1jg1mebx8f/input_MITb_g2o.g2o?dl=1",
    "M3500": "https://www.dropbox.com/s/gmdzo74b3tzvbrw/input_M3500_g2o.g2o?dl=1",
}
DATA = Path("data/g2o")
PRIOR_SIGMA = np.array([1e-4, 1e-4, 1e-4])


def fetch(name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.g2o"
    if not path.exists():
        req = urllib.request.Request(FILES[name], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    return path


def parse(path: Path):
    """VERTEX_SE2 id x y th; EDGE_SE2 i j dx dy dth I11 I12 I13 I22 I23 I33 (upper triangle)."""
    verts, edges = {}, []
    for line in path.read_text().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "VERTEX_SE2":
            verts[int(tok[1])] = [float(v) for v in tok[2:5]]
        elif tok[0] == "EDGE_SE2":
            i, j = int(tok[1]), int(tok[2])
            z = [float(v) for v in tok[3:6]]
            a, b, c, d, e, f = (float(v) for v in tok[6:12])
            edges.append((i, j, z, np.array([[a, b, c], [b, d, e], [c, e, f]])))
    ids = sorted(verts)
    assert ids == list(range(len(ids)))
    X0 = np.array([verts[i] for i in ids])
    X0[:, 2] = gr.wrap(X0[:, 2])
    problem = gr.GraphProblem(
        prior_mean=X0[0].copy(), prior_sigma=PRIOR_SIGMA,
        odo_z=np.array([e[2] for e in edges]), odo_cov=np.array([np.linalg.inv(e[3]) for e in edges]),
        br_pose=np.zeros(0, dtype=int), br_lm=np.zeros(0, dtype=int), br_z=np.zeros((0, 2)), br_sigma=np.zeros((0, 2)),
        odo_i=np.array([e[0] for e in edges]), odo_j=np.array([e[1] for e in edges]), n_poses=len(ids))
    return problem, X0


def solve_gtsam(path: Path, problem, max_iter=500):
    """GTSAM Levenberg-Marquardt with its default damping. Returns None fields on failure."""
    import gtsam
    graph, values = gtsam.readG2o(str(path), False)
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(*problem.prior_mean),
                                     gtsam.noiseModel.Diagonal.Sigmas(PRIOR_SIGMA)))
    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-13)
    params.setAbsoluteErrorTol(1e-13)
    params.setMaxIterations(max_iter)
    t0 = time.perf_counter()
    try:
        opt = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
        result = opt.optimize()
    except RuntimeError as exc:
        return None, float("nan"), -1, time.perf_counter() - t0, str(exc).strip().splitlines()[0]
    dt = time.perf_counter() - t0
    X = np.array([[result.atPose2(k).x(), result.atPose2(k).y(), result.atPose2(k).theta()]
                  for k in range(problem.n_poses)])
    return X, 2 * graph.error(result), int(opt.iterations()), dt, ""


def solve_own(solver, X0, L0):
    """Gauss-Newton first; Levenberg-Marquardt when Gauss-Newton does not converge."""
    F0 = solver.cost(X0, L0)
    try:
        out = solver.optimise(X0, L0, method="gn", tol=1e-9, max_iter=50)
        if out["converged"] and out["cost"] < F0:
            return out, "gn"
    except Exception:
        pass
    return solver.optimise(X0, L0, method="lm", tol=1e-9, max_iter=3000), "lm"


def main() -> None:
    conv = gb.pose_convention()
    rows = []
    for name in FILES:
        path = fetch(name)
        problem, X0 = parse(path)
        L0 = np.zeros((0, 2))
        Xg, Fg, itg, tg, err = solve_gtsam(path, problem)
        row = {"graph": name, "poses": problem.n_poses, "edges": len(problem.odo_z), "gtsam_convention": conv}
        for mode in (conv, gr.GRISETTI if conv == gr.SE2LOG else gr.SE2LOG):
            solver = gr.GraphSLAM(problem, mode)
            out, method = solve_own(solver, X0, L0)
            tag = "same" if mode == conv else "other"
            row.update({f"{tag}_mode": mode, f"{tag}_method": method, f"{tag}_initial_cost": solver.cost(X0, L0),
                        f"{tag}_cost": out["cost"], f"{tag}_iterations": out["iterations"],
                        f"{tag}_time_s": out["runtime"], f"{tag}_converged": out["converged"],
                        f"{tag}_max_pos_diff_vs_gtsam": (float(np.abs(out["X"][:, :2] - Xg[:, :2]).max())
                                                         if Xg is not None else float("nan"))})
        row.update(gtsam_cost=Fg, gtsam_iterations=itg, gtsam_time_s=tg, gtsam_error=err,
                   cost_rel_diff=abs(row["same_cost"] - Fg) / Fg if Xg is not None else float("nan"))
        rows.append(row)
        print(f"{name}: {row['poses']} poses, {row['edges']} edges | own {row['same_method']} ({row['same_mode']}) "
              f"F {row['same_initial_cost']:.4g} -> {row['same_cost']:.6f} in {row['same_iterations']} it, "
              f"{row['same_time_s']:.2f} s | GTSAM LM F {Fg:.6f} in {itg} it, {tg:.2f} s {err} | rel cost diff "
              f"{row['cost_rel_diff']:.1e}, max pose diff {row['same_max_pos_diff_vs_gtsam']:.1e} m | own "
              f"{row['other_method']} ({row['other_mode']}) F {row['other_cost']:.6f} in {row['other_iterations']} it",
              flush=True)
    with open("results/g2o.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
