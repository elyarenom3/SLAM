"""Own optimiser against the installed GTSAM on small synthetic graphs.

Run it once per environment (GTSAM 4.3 in .venv, GTSAM 4.2 in .venv42). It
records, for both pose-error forms of the own optimiser, how closely the two
solvers agree, and writes results/gtsam_small_<convention>.json.

Usage: python scripts/compare_small_graphs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import json

import numpy as np

from slam2d import graph as gr
from slam2d import gtsam_backend as gb
from slam2d.sim import small_graph


def main() -> None:
    from importlib.metadata import version as pkg_version
    conv = gb.pose_convention()
    version = pkg_version("gtsam")
    rows = []
    for noise in (1.0, 0.1):
        for seed in range(10):
            problem, _, _, X0, L0 = small_graph(seed=seed, noise=noise)
            ref = gb.solve(problem, X0, L0, method="lm+gn", tol=1e-14)
            for mode in (gr.GRISETTI, gr.SE2LOG):
                solver = gr.GraphSLAM(problem, mode)
                own = solver.optimise(X0, L0, tol=1e-11, max_iter=200)
                Pp, _ = solver.marginals(own["X"], own["L"])
                graph, values, _, _ = gb.build(problem, own["X"], own["L"])
                rows.append({
                    "noise": noise, "seed": seed, "own_mode": mode, "same_convention": mode == conv,
                    "cost_own": own["cost"], "cost_gtsam": ref["cost"],
                    "gtsam_cost_at_own_optimum": 2 * graph.error(values),
                    "cost_rel_diff": abs(own["cost"] - ref["cost"]) / ref["cost"],
                    "max_pos_diff": float(np.abs(own["X"][:, :2] - ref["X"][:, :2]).max()),
                    "max_heading_diff": float(np.abs(gr.wrap(own["X"][:, 2] - ref["X"][:, 2])).max()),
                    "max_cov_rel_diff": float(np.abs(Pp - ref["P_pose"]).max() / np.abs(Pp).max()),
                    "iterations_own": own["iterations"], "iterations_gtsam": ref["iterations"]})
    print(f"GTSAM {version}, pose error convention: {conv}")
    print("noise own_mode   | median rel cost diff | median max pos diff | median max cov rel diff | same cost function")
    for noise in (1.0, 0.1):
        for mode in (gr.GRISETTI, gr.SE2LOG):
            sel = [r for r in rows if r["noise"] == noise and r["own_mode"] == mode]
            same = np.median([abs(r["gtsam_cost_at_own_optimum"] - r["cost_own"]) / r["cost_own"] for r in sel])
            print(f"{noise:4.1f}  {mode:9s} | {np.median([r['cost_rel_diff'] for r in sel]):.1e}"
                  f"              | {np.median([r['max_pos_diff'] for r in sel]):.1e}"
                  f"             | {np.median([r['max_cov_rel_diff'] for r in sel]):.1e}"
                  f"                 | {'yes' if same < 1e-10 else 'no'} ({same:.0e})")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/gtsam_small_{conv}.json").write_text(json.dumps(
        {"gtsam_version": version, "convention": conv, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
