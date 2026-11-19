"""A 60-second animation of the standard and FEJ EKF running on one MR.CLAM log.

Left: the room from above with ground truth, both filter estimates, 3-sigma
ellipses of the robot and of every mapped landmark, and the current sightings.
Right: running NEES of the robot pose with the 95 % region, and position error.

The run shown is the same one as in the paper's map figure: the held-out run
whose standard-EKF position RMSE is the median of the 40 held-out runs.

Usage: python scripts/make_animation.py [--dataset D --robot R] [--seconds 60] [--fps 30]
Output: results/animation.mp4
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection

from make_figures import load_rows, pick_run
from slam2d import pipeline as P
from slam2d.data.mrclam import load_dataset
from slam2d.ekf import EKFSLAM
from slam2d.geometry import cov_ellipse, wrap
from slam2d.metrics import chi2_bounds, nees_series
from slam2d.viz import BAND, COLORS, GRID, INK, INK2, LABELS, use_paper_style

TRAIL_S = 150.0     # seconds of trajectory kept on screen


def run_recorded(stream, rc, cfg, mode, frame_kf):
    """Run the filter and snapshot the full state at the keyframes in frame_kf."""
    R = P.noise_R(rc, cfg)
    qs = cfg.get("q_inflation", 1.0)
    ekf = EKFSLAM(stream.x0, P.P0_from(cfg), np.zeros((2, 2)), n_max=15, mode=mode, gate=cfg["gate"],
                  dt_nom=cfg["dt_nom"], floor=cfg["floor"], confirm_dist=cfg.get("confirm_dist"))
    K1 = stream.n_keyframes
    pose = np.empty((K1, 3))
    Prr = np.empty((K1, 3, 3))
    pose[0], Prr[0] = ekf.x[:3], ekf.P[:3, :3]
    snaps = {}
    wanted = set(int(k) for k in frame_kf)
    steps = stream.steps.tolist()
    ids, zs = stream.meas_id.tolist(), stream.meas_z
    for k in range(1, K1):
        for v, w, dt, qv, qw in steps[stream.seg[k - 1]:stream.seg[k]]:
            ekf.predict(v, w, dt, q=(qs * qv, qs * qw))
        a, b = stream.kf_ptr[k], stream.kf_ptr[k + 1]
        ekf.update_batch(ids[a:b], zs[a:b], R)
        pose[k], Prr[k] = ekf.x[:3], ekf.P[:3, :3]
        if k in wanted:
            snaps[k] = {j: (ekf.landmark(j), ekf.landmark_cov(j)) for j in ekf.slot}
    return pose, Prr, snaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=None)
    ap.add_argument("--robot", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default="results/animation.mp4")
    args = ap.parse_args()

    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    use_paper_style()
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})

    cfg = P.load_config()
    if args.dataset is None:
        d, r = pick_run(load_rows("results/mrclam/summary.csv"))
    else:
        d, r = args.dataset, args.robot
    ds = load_dataset("data/raw", d)
    rb = ds.robot(r)
    rc = P.robot_cfg(cfg, r)
    st = P.stream_for(rb, cfg)
    t = st.kf_t - st.kf_t[0]
    n_frames = int(round(args.seconds * args.fps))
    frame_t = np.linspace(0.0, t[-1], n_frames)
    frame_kf = np.clip(np.searchsorted(t, frame_t, side="right") - 1, 0, len(t) - 1)
    speed = t[-1] / args.seconds

    res = {}
    for mode, name in (("standard", "ekf"), ("fej", "fej")):
        pose, Prr, snaps = run_recorded(st, rc, cfg, mode, frame_kf)
        err = st.gt - pose
        err[:, 2] = wrap(err[:, 2])
        Pv = Prr + np.diag([cfg["vicon_sigma"][0] ** 2] * 2 + [cfg["vicon_sigma"][1] ** 2])
        nees = np.where(st.gt_valid, nees_series(err, Pv), np.nan)
        perr = np.where(st.gt_valid, np.hypot(err[:, 0], err[:, 1]), np.nan)
        res[name] = {"pose": pose, "P": Prr, "snaps": snaps, "nees": nees, "perr": perr}
    lm_gt = np.array([ds.landmarks[j] for j in sorted(ds.landmarks)])
    lm_ids = sorted(ds.landmarks)
    lo, hi = chi2_bounds(3)

    # ---------------------------------------------------------------- layout
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1.0], left=0.055, right=0.985, top=0.90, bottom=0.09,
                          wspace=0.16, hspace=0.30)
    ax = fig.add_subplot(gs[:, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_e = fig.add_subplot(gs[1, 1])
    title = fig.text(0.055, 0.945, "", fontsize=14, color=INK, ha="left", va="center")
    fig.text(0.985, 0.945, "Ellipses show 3 sigma", fontsize=10, color=INK2, ha="right", va="center")

    # map, drawn with world y on the horizontal axis (inverted) and world x up
    pad = 0.9
    ax.set_xlim(lm_gt[:, 1].max() + pad, lm_gt[:, 1].min() - pad)
    ax.set_ylim(min(lm_gt[:, 0].min(), st.gt[st.gt_valid, 0].min()) - 2.0 * pad,
                max(lm_gt[:, 0].max(), st.gt[st.gt_valid, 0].max()) + pad)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("x (m)")
    ax.plot(lm_gt[:, 1], lm_gt[:, 0], ls="none", marker="+", ms=9, mew=1.4, color=INK, zorder=6,
            label="Landmark (truth)")
    rays = LineCollection([], colors=INK2, linewidths=0.8, alpha=0.6, zorder=2)
    ax.add_collection(rays)
    art = {"truth_trail": ax.plot([], [], color=INK, lw=1.6, zorder=5, label=LABELS["truth"])[0],
           "truth_dot": ax.plot([], [], ls="none", marker="o", ms=6, color=INK, mec="white", mew=1.0, zorder=7)[0]}
    for name in ("ekf", "fej"):
        c = COLORS[name]
        art[name] = {
            "trail": ax.plot([], [], color=c, lw=1.8, zorder=4, label=LABELS[name])[0],
            "dot": ax.plot([], [], ls="none", marker="o", ms=6, color=c, mec="white", mew=1.0, zorder=7)[0],
            "ell": ax.plot([], [], color=c, lw=1.4, zorder=4)[0],
            "lm": ax.plot([], [], ls="none", marker="o", ms=4, color=c, zorder=3)[0],
            "lm_ell": [ax.plot([], [], color=c, lw=1.1, zorder=3)[0] for _ in lm_ids],
        }
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=4, columnspacing=1.6, handletextpad=0.5)

    # NEES panel
    ax_n.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax_n.axhline(3.0, color=INK2, lw=0.8, zorder=1)
    ax_n.set_yscale("log")
    ax_n.set_xlim(0, t[-1])
    top = max(np.nanpercentile(res["ekf"]["nees"], 99.5), np.nanpercentile(res["fej"]["nees"], 99.5), 20.0)
    ax_n.set_ylim(0.05, top * 1.5)
    ax_n.set_ylabel("NEES of the robot pose")
    ax_n.grid(axis="x", visible=False)
    ax_n.minorticks_off()
    ax_n.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_n.text(0.99, 0.04, "95% region and expected value 3", transform=ax_n.transAxes, ha="right", va="bottom",
              fontsize=9, color=INK2)
    # position error panel
    ax_e.set_xlim(0, t[-1])
    ax_e.set_ylim(0, 1.1 * max(np.nanmax(res["ekf"]["perr"]), np.nanmax(res["fej"]["perr"])))
    ax_e.set_ylabel("Position error (m)")
    ax_e.set_xlabel("Time (s)")
    ax_e.grid(axis="x", visible=False)
    lines = {}
    for name in ("ekf", "fej"):
        lines[name] = (ax_n.plot([], [], color=COLORS[name], lw=1.3, label=LABELS[name])[0],
                       ax_e.plot([], [], color=COLORS[name], lw=1.6, label=LABELS[name])[0])
    ax_n.legend(loc="upper left", ncol=2, columnspacing=1.2, handletextpad=0.4)
    cursor = [ax_n.axvline(0, color=GRID, lw=1.0, zorder=1), ax_e.axvline(0, color=GRID, lw=1.0, zorder=1)]

    kf_of_meas = st.meas_kf
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=3500,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, args.out, dpi=100):
        prev_k = 0
        for f in range(n_frames):
            k = int(frame_kf[f])
            k0 = int(np.searchsorted(t, t[k] - TRAIL_S))
            sl = slice(k0, k + 1)
            g = np.where(st.gt_valid[sl, None], st.gt[sl, :2], np.nan)
            art["truth_trail"].set_data(g[:, 1], g[:, 0])
            if st.gt_valid[k]:
                art["truth_dot"].set_data([st.gt[k, 1]], [st.gt[k, 0]])
            for name in ("ekf", "fej"):
                rr, a = res[name], art[name]
                a["trail"].set_data(rr["pose"][sl, 1], rr["pose"][sl, 0])
                a["dot"].set_data([rr["pose"][k, 1]], [rr["pose"][k, 0]])
                e = cov_ellipse(rr["pose"][k], rr["P"][k, :2, :2], 3.0, 40)
                a["ell"].set_data(e[:, 1], e[:, 0])
                snap = rr["snaps"].get(k, {})
                pts = np.array([snap[j][0] for j in snap]).reshape(-1, 2)
                a["lm"].set_data(pts[:, 1], pts[:, 0])
                for line, j in zip(a["lm_ell"], lm_ids):
                    if j in snap:
                        e = cov_ellipse(snap[j][0], snap[j][1], 3.0, 40)
                        line.set_data(e[:, 1], e[:, 0])
                    else:
                        line.set_data([], [])
                lines[name][0].set_data(t[:k + 1], rr["nees"][:k + 1])
                lines[name][1].set_data(t[:k + 1], rr["perr"][:k + 1])
            # sightings received since the previous frame, drawn from the FEJ estimate
            rows = np.flatnonzero((kf_of_meas > prev_k) & (kf_of_meas <= k))
            snap = res["fej"]["snaps"].get(k, {})
            p = res["fej"]["pose"][k]
            segs = [[(p[1], p[0]), (snap[j][0][1], snap[j][0][0])] for j in set(st.meas_id[rows].tolist()) if j in snap]
            rays.set_segments(segs)
            prev_k = k
            for c in cursor:
                c.set_xdata([t[k], t[k]])
            title.set_text(f"EKF-SLAM on MR.CLAM dataset {d}, robot {r}    t = {t[k]:6.0f} s    "
                           f"({speed:.0f} times real time)")
            writer.grab_frame()
    print(f"wrote {args.out}: {n_frames} frames at {args.fps} fps = {n_frames / args.fps:.1f} s "
          f"(dataset {d}, robot {r}, {t[-1]:.0f} s of data)")


if __name__ == "__main__":
    main()
