"""Sweep the declared simulation grid and record every point.

The headline configuration (config/sim.json) is the first grid point, in the
order swept here, at which the ideal filter's heading sigma reaches 3 degrees
before the first loop closure. This script reproduces the whole table.

Usage: python scripts/run_sim_grid.py [--trials 20]
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import argparse
import json
import math

from run_sim_mc import MODES, load_config, summarise
from slam2d.sim import build_world, run_monte_carlo

GRID = [(r, w, p) for r in (3.0, 2.0) for w in (0.1, 0.2) for p in (2.0, 5.0, 10.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--config", default="config/sim.json")
    args = ap.parse_args()
    rows = []
    for r_max, sigma_w, phi_deg in GRID:
        cfg = load_config(args.config, [f"r_max={r_max}", f"sigma_w={sigma_w}",
                                        f"sigma_phi={math.radians(phi_deg)}"])
        world = build_world(cfg)
        s = summarise(world, run_monte_carlo(world, args.trials, MODES), args.trials)
        row = {"r_max": r_max, "sigma_w": sigma_w, "sigma_phi_deg": phi_deg,
               "heading_sigma_deg": s["ideal"]["heading_sigma_first_lap_deg"]}
        for m in MODES:
            row[f"anees_{m}"] = s[m]["anees_mean"]
            row[f"pos_rmse_{m}"] = s[m]["pos_rmse"]
        rows.append(row)
        print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
              flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/sim_grid.json").write_text(json.dumps({"trials": args.trials, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
