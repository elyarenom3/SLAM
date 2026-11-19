"""Calibrate sensors and noise on MR.CLAM dataset 1 and write the run configuration.

Everything the estimators need on real data is fitted here from dataset 1 only
and written to config/frozen_mrclam.json. `--freeze` marks the file as frozen;
scripts/run_mrclam.py refuses datasets 2-9 until that mark is set, and this
script refuses to overwrite a frozen file.

Usage: python scripts/fit_noise.py [--dataset 1] [--freeze] [--q-inflation X]
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import argparse
import json
import math

import numpy as np

from slam2d import calib as C
from slam2d.data.mrclam import load_dataset

DT_NOM = 0.015
WINDOWS = (0.5, 1.0, 2.0, 5.0, 10.0)
TURN_MEM = 0.5     # [s] a turn commanded within this time marks the "turning" state
TURN_THR = 0.02    # [rad/s] commanded turn rate above which the robot counts as turning
BLOCK = 10.0       # [s] block length for the correlation inflation of the measurement noise
SIGMA_R_MIN = 0.03
CONFIG = Path("config/frozen_mrclam.json")


def split_half(res: dict, first: bool) -> dict:
    t_mid = np.median(res["t"])
    sel = res["t"] <= t_mid if first else res["t"] > t_mid
    return {k: v[sel] for k, v in res.items()}


def offset_camera(res: dict) -> dict:
    """Simplest correction: constant range and bearing offsets."""
    return {"a": 1.0, "b": float(np.median(res["raw_r"] - res["r_hat"])),
            "beta": float(-np.median(C.wrap(res["raw_phi"] - res["phi_hat"]))), "lx": 0.0, "ly": 0.0}


def corrected(res: dict, cam: dict):
    r, phi = C.camera_correct(res["raw_r"], res["raw_phi"], cam)
    return r - res["r_hat"], C.wrap(phi - res["phi_hat"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=1)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--freeze", action="store_true", help="mark the existing config as frozen")
    ap.add_argument("--q-inflation", type=float, default=None,
                    help="set the process-noise inflation factor in the existing config")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if CONFIG.exists():
        old = json.loads(CONFIG.read_text())
        if args.freeze or args.q_inflation is not None:
            if old.get("frozen") and not args.force:
                sys.exit("config is frozen; nothing changed")
            if args.q_inflation is not None:
                old["q_inflation"] = args.q_inflation
            if args.freeze:
                old["frozen"] = True
            CONFIG.write_text(json.dumps(old, indent=1))
            print(f"updated {CONFIG}: frozen={old.get('frozen', False)} q_inflation={old['q_inflation']}")
            return
        if old.get("frozen") and not args.force:
            sys.exit("config is frozen; refusing to refit")

    if args.dataset != 1:
        sys.exit("calibration is defined on dataset 1 only")
    ds = load_dataset(args.root, args.dataset)
    report = {"dataset": args.dataset, "landmark_fix": {str(k): v for k, v in ds.landmark_fix.items()},
              "robots": {}}

    # horizon for the odometry sigma: 90th percentile gap between landmark updates
    gaps = []
    for rb in ds.robots:
        a, b = rb.window()
        gaps.append(np.diff(np.unique(rb.meas[(rb.meas[:, 0] > a) & (rb.meas[:, 0] <= b), 0])))
    gap90 = float(np.percentile(np.concatenate(gaps), 90))
    horizon = min(WINDOWS, key=lambda T: abs(T - gap90))
    report["gap_p90_s"] = gap90
    report["horizon_s"] = horizon

    # Vicon noise from stationary stretches (commanded v = w = 0 for more than 3 s)
    vic = []
    for rb in ds.robots:
        t, still = rb.odom[:, 0], (rb.odom[:, 1] == 0) & (rb.odom[:, 2] == 0)
        idx = np.flatnonzero(still)
        for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run) > 2 and t[run[-1]] - t[run[0]] > 3:
                g = rb.gt[(rb.gt[:, 0] > t[run[0]] + 1.5) & (rb.gt[:, 0] < t[run[-1]])]
                if len(g) > 50:
                    vic.append((math.hypot(g[:, 1].std(), g[:, 2].std()) / math.sqrt(2),
                                np.unwrap(g[:, 3]).std()))
    vicon = [float(np.median([v[0] for v in vic])), float(np.median([v[1] for v in vic]))] if vic else [1e-3, 1e-3]
    report["vicon_sigma"] = vicon

    # camera time offset: one pooled value
    taus = [C.fit_camera_time_offset(rb, ds.landmarks) for rb in ds.robots]
    camera_tau = float(np.median(taus))
    report["camera_tau_per_robot"] = taus

    # camera model choice by a time split of dataset 1: fit on the first half, score on the second
    split = {"offset": [], "full": []}
    res_all = {}
    for rb in ds.robots:
        res = C.measurement_residuals(rb, ds.landmarks, camera_tau)
        res_all[rb.robot] = res
        first, second = split_half(res, True), split_half(res, False)
        base_r, base_p = C.mad_sigma(second["er"]), C.mad_sigma(second["ephi"])
        for name, cam in (("offset", offset_camera(first)), ("full", C.fit_camera(first))):
            er, ep = corrected(second, cam)
            # robust total error (spread and bias), relative to the raw spread
            split[name].append(math.hypot(C.mad_sigma(er), np.median(er)) / base_r
                               + math.hypot(C.mad_sigma(ep), np.median(ep)) / base_p)
    model = min(split, key=lambda k: float(np.mean(split[k])))
    report["camera_model_split_score"] = {k: float(np.mean(v)) for k, v in split.items()}
    report["camera_model"] = model

    robots = {}
    for rb in ds.robots:
        res = res_all[rb.robot]
        cam = C.fit_camera(res) if model == "full" else offset_camera(res)
        tau_o = C.fit_odometry_time_offset(rb)
        odo = C.fit_odometry_scale(rb, 1.0, tau_o)

        # noise statistics use sightings taken while moving: a robot standing
        # still repeats one reading, which would shrink any spread estimate
        moving = ~C.stationary_mask(rb, tau_o, res["t"])
        er, ep = corrected(res, cam)
        r_corr, _ = C.camera_correct(res["raw_r"], res["raw_phi"], cam)
        rs = C.range_sigma_model(r_corr[moving], er[moving])
        if not rs["range_dependent"]:
            rs["c0"], rs["c1"] = C.mad_sigma(er[moving]), 0.0
        sigma_phi = C.mad_sigma(ep[moving])
        sig_r = np.maximum(rs["c0"] + rs["c1"] * r_corr, SIGMA_R_MIN)
        kappa_r = C.block_kappa(res["t"][moving], res["subject"][moving], (er / sig_r)[moving], BLOCK)
        kappa_phi = C.block_kappa(res["t"][moving], res["subject"][moving], (ep / sigma_phi)[moving], BLOCK)
        table = []
        for T in WINDOWS:
            w = C.odometry_windows(rb, T, odo)
            s_th = math.sqrt(max(C.mad_sigma(w["e_th"]) ** 2 - 2 * vicon[1] ** 2, 1e-12))
            s_al = math.sqrt(max(C.mad_sigma(w["e_along"]) ** 2 - 2 * vicon[0] ** 2, 1e-12))
            table.append({"T": T, "n": int(len(w["t"])), "sigma_th": s_th, "sigma_along": s_al,
                          "sigma_lat": C.mad_sigma(w["e_lat"]),
                          "sigma_w_eq": C.per_sample_sigma(s_th, T, DT_NOM),
                          "sigma_v_eq": C.per_sample_sigma(s_al, T, DT_NOM)})
        two = C.fit_two_state_noise(rb, odo, horizon, TURN_MEM, TURN_THR, DT_NOM, vicon)
        odo.update(turn_mem=TURN_MEM, turn_thr=TURN_THR,
                   **{k: two[k] for k in ("sigma_v_s", "sigma_v_t", "sigma_w_s", "sigma_w_t")})

        robots[str(rb.robot)] = {"odo": odo, "cam": cam,
                                 "noise": {"r_c0": rs["c0"], "r_c1": rs["c1"], "sigma_phi": sigma_phi,
                                           "kappa_r": kappa_r, "kappa_phi": kappa_phi}}
        report["robots"][str(rb.robot)] = {
            "n_sightings": int(len(res["er"])), "stationary_fraction": float(1 - moving.mean()),
            "kappa_by_block": {str(B): [C.block_kappa(res["t"][moving], res["subject"][moving], (er / sig_r)[moving], B),
                                        C.block_kappa(res["t"][moving], res["subject"][moving], (ep / sigma_phi)[moving], B)]
                               for B in (1, 2, 5, 10, 20)},
            "raw": {"range_median": float(np.median(res["er"])), "range_sigma": C.mad_sigma(res["er"]),
                    "range_std": float(res["er"].std()),
                    "bearing_median": float(np.median(res["ephi"])), "bearing_sigma": C.mad_sigma(res["ephi"]),
                    "bearing_std": float(res["ephi"].std())},
            "corrected": {"range_median": float(np.median(er)), "range_sigma": C.mad_sigma(er),
                          "bearing_median": float(np.median(ep)), "bearing_sigma": sigma_phi},
            "range_model": rs, "odometry_windows": table, "two_state": two}
        print(f"R{rb.robot}: odo tau {odo['tau']:+.2f} s, sv {odo['sv']:.3f}, cv {odo['cv']:.3f}, sw {odo['sw']:.3f}, "
              f"bw {odo['bw']:+.4f}; sigma_v {odo['sigma_v_s']:.3f}/{odo['sigma_v_t']:.3f}, "
              f"sigma_w {odo['sigma_w_s']:.3f}/{odo['sigma_w_t']:.3f} (straight/turning, "
              f"turning {100 * two['turn_fraction']:.0f}% of the time) | "
              f"sigma_r = {rs['c0']:.3f} + {rs['c1']:.3f} r, sigma_phi {sigma_phi:.4f}, "
              f"kappa {kappa_r:.2f}/{kappa_phi:.2f}, stationary {100 * (1 - moving.mean()):.0f}%")

    cfg = {
        "fitted_on": "MR.CLAM dataset 1",
        "frozen": False,
        "dt_nom": DT_NOM,
        "gate": 9.21,
        "floor": [1e-6, math.radians(0.1) ** 2],     # (1 mm)^2 / s and (0.1 deg)^2 / s
        "merge_dt": DT_NOM,
        "max_step": 0.02,
        "p0": [1e-3, 1e-3],
        "vicon_sigma": vicon,
        "camera_tau": camera_tau,
        "camera_model": model,
        "odometry_horizon_s": horizon,
        "confirm_dist": None,
        "skip_stationary": True,
        "r_inflation": True,
        "r_inflation_block_s": BLOCK,
        "q_inflation": 1.0,
        "regate": {"threshold": 9.21, "max_rounds": 5},
        "robots": robots,
    }
    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=1))
    Path("results").mkdir(exist_ok=True)
    Path("results/calibration_ds1.json").write_text(json.dumps(report, indent=1))
    print(f"gap p90 {gap90:.2f} s -> horizon {horizon} s; camera tau {camera_tau:+.3f} s; "
          f"camera model '{model}' (split scores {report['camera_model_split_score']}); "
          f"vicon sigma {vicon[0] * 1e3:.2f} mm, {vicon[1] * 1e3:.2f} mrad")
    print(f"wrote {CONFIG} and results/calibration_ds1.json")


if __name__ == "__main__":
    main()
