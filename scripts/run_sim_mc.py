"""Monte Carlo consistency study in simulation: ideal, standard and FEJ EKF-SLAM.

Writes results/sim_mc[_tag].npz with ANEES over time, RMSE over time and the
configuration used, and prints a summary.

Usage: python scripts/run_sim_mc.py [--trials 50] [--config config/sim.json]
                                    [--gate] [--tag NAME] [--set key=value ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import argparse
import json
import time

import numpy as np

from slam2d.metrics import anees_bounds
from slam2d.sim import SimConfig, build_world, config_dict, run_monte_carlo

MODES = ("ideal", "standard", "fej")


def load_config(path: str | None, overrides: list[str]) -> SimConfig:
    kw = {}
    if path and Path(path).exists():
        kw.update(json.loads(Path(path).read_text()))
    for item in overrides:
        key, val = item.split("=")
        kw[key] = json.loads(val)
    for key in ("centre", "size"):
        if key in kw:
            kw[key] = tuple(kw[key])
    return SimConfig(**kw)


def summarise(world, mc, n_trials: int) -> dict:
    lo, hi = anees_bounds(n_trials)
    t = world.t
    lap = world.lap_ticks * world.cfg.dt
    out = {"bounds": (lo, hi), "lap_s": lap}
    for m, r in mc.items():
        anees = r["nees"].mean(axis=0)
        pos = np.sqrt((r["err"][:, :, :2] ** 2).sum(axis=2).mean(axis=0))
        th = np.sqrt((r["err"][:, :, 2] ** 2).mean(axis=0))
        out[m] = {
            "anees_mean": float(anees.mean()),
            "anees_last_lap": float(anees[t > t[-1] - lap].mean()),
            "frac_inside": float(np.mean((anees >= lo) & (anees <= hi))),
            "frac_above": float(np.mean(anees > hi)),
            "pos_rmse": float(np.sqrt((pos ** 2).mean())),
            "heading_rmse_deg": float(np.degrees(np.sqrt((th ** 2).mean()))),
            "lm_nees": float(np.nanmean(r["lm_nees"])),
            "lm_rmse": float(np.sqrt(np.nanmean((r["lm_err"] ** 2).sum(axis=2)))),
            "nis": float(np.nanmean(r["nis_mean"])),
            "heading_sigma_first_lap_deg": float(np.degrees(r["sig"][:, t < lap, 2].mean(axis=0).max())),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--config", default="config/sim.json")
    ap.add_argument("--gate", action="store_true", help="apply the 99%% NIS gate")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="key=value")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    world = build_world(cfg)
    t0 = time.time()
    mc = run_monte_carlo(world, args.trials, MODES, gate=9.21 if args.gate else None,
                         seed0=args.seed, workers=args.workers)
    s = summarise(world, mc, args.trials)
    lo, hi = s["bounds"]
    print(f"{args.trials} trials, {cfg.laps} laps of {s['lap_s']:.0f} s, gate {'on' if args.gate else 'off'}, "
          f"{time.time() - t0:.0f} s; ANEES 95% band [{lo:.3f}, {hi:.3f}]")
    print(f"criterion: ideal heading sigma before first loop closure = "
          f"{s['ideal']['heading_sigma_first_lap_deg']:.2f} deg (target >= 3)")
    print(f"{'mode':9s} {'ANEES':>7s} {'last lap':>9s} {'inside':>7s} {'above':>6s} "
          f"{'pos RMSE':>9s} {'head RMSE':>10s} {'lm NEES':>8s} {'lm RMSE':>8s} {'NIS':>5s}")
    for m in MODES:
        r = s[m]
        print(f"{m:9s} {r['anees_mean']:7.2f} {r['anees_last_lap']:9.2f} {r['frac_inside']:7.2f} "
              f"{r['frac_above']:6.2f} {r['pos_rmse']:9.3f} {r['heading_rmse_deg']:10.2f} "
              f"{r['lm_nees']:8.2f} {r['lm_rmse']:8.3f} {r['nis']:5.2f}")

    if not args.no_save:
        out = Path("results")
        out.mkdir(exist_ok=True)
        name = "sim_mc" + (f"_{args.tag}" if args.tag else "")
        arrays = {"t": world.t, "bounds": np.array(s["bounds"]), "lap_s": s["lap_s"],
                  "n_trials": args.trials, "gate": args.gate,
                  "truth_xy": world.X[::50, :2], "landmarks": world.landmarks}
        for m in MODES:
            arrays[f"{m}_anees"] = mc[m]["nees"].mean(axis=0)
            arrays[f"{m}_nees_all"] = mc[m]["nees"].astype(np.float32)
            arrays[f"{m}_pos_rmse"] = np.sqrt((mc[m]["err"][:, :, :2] ** 2).sum(axis=2).mean(axis=0))
            arrays[f"{m}_th_rmse"] = np.sqrt((mc[m]["err"][:, :, 2] ** 2).mean(axis=0))
            arrays[f"{m}_sig"] = mc[m]["sig"].mean(axis=0)
            arrays[f"{m}_lm_nees"] = mc[m]["lm_nees"]
        np.savez_compressed(out / f"{name}.npz", **arrays)
        (out / f"{name}.json").write_text(json.dumps(
            {"config": config_dict(cfg), "summary": {k: v for k, v in s.items()}}, indent=1, default=float))
        print(f"saved results/{name}.npz")


if __name__ == "__main__":
    main()
