"""Figures, LaTeX tables and the numbers quoted in the paper, all from saved results.

Inputs:  results/sim_mc.npz, results/sim_mc_gated.json, results/sim_grid.json,
         results/mrclam/summary.csv, agreement.csv, runs/*.npz,
         results/gtsam_small_*.json, results/calibration_ds1.json
Outputs: paper/figures/fig_anees.pdf, fig_map.pdf, fig_consistency.pdf
         paper/tables/tab_accuracy.tex, tab_agreement.tex
         results/numbers.json

Usage: python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "scripts")]   # run from a checkout, also with python -I

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slam2d.geometry import cov_ellipse
from slam2d.metrics import chi2_bounds, truncated_nis_mean
from slam2d.viz import BAND, COL_W, COLORS, GRID, INK, INK2, LABELS, PAGE_W, use_paper_style

RES = Path("results")
FIG = Path("paper/figures")
TAB = Path("paper/tables")
HOLDOUT = range(2, 10)


def load_rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k in ("method", "config", "convention", "error", "converged"):
                continue
            try:
                r[k] = float(v) if v != "" else np.nan
            except ValueError:
                pass
        r["dataset"], r["robot"] = int(r["dataset"]), int(r["robot"])
    return rows


def q(values, p):
    v = np.asarray(values, dtype=float)
    return float(np.nanpercentile(v, p))


def med_iqr(values, fmt="{:.2f}"):
    return f"{fmt.format(q(values, 50))} [{fmt.format(q(values, 25))}, {fmt.format(q(values, 75))}]"


# ------------------------------------------------------------------ figure 1
def fig_anees(numbers: dict) -> None:
    d = np.load(RES / "sim_mc.npz")
    t, (lo, hi), lap = d["t"], d["bounds"], float(d["lap_s"])
    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    ax.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    win = 51      # 10 s moving average of the 5 Hz series, for legibility only
    kern = np.ones(win) / win
    ends = {}
    for mode in ("standard", "fej", "ideal"):
        y = d[f"{mode}_anees"]
        ys = np.convolve(y, kern, mode="valid")
        ts = t[win // 2: win // 2 + len(ys)]
        ax.plot(ts, ys, color=COLORS[mode], lw=1.2, label=LABELS[mode], zorder=3 if mode != "ideal" else 2)
        ends[mode] = ys[-1]
    for k in range(1, int(t[-1] // lap) + 1):
        ax.axvline(k * lap, color=GRID, lw=0.5, zorder=1)
    ax.set_yscale("log")
    ax.set_ylim(1.5, 60)
    ax.set_yticks([2, 3, 5, 10, 20, 50])
    ax.set_yticklabels(["2", "3", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlim(0, t[-1] * 1.17)
    ax.set_xticks(np.arange(0, t[-1], 300))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ANEES of the robot pose (50 runs)")
    ax.grid(axis="x", visible=False)
    for mode, y in ends.items():
        ax.annotate(LABELS[mode], (t[-1], y), xytext=(4, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, color=INK)
    ax.annotate("95% band", (t[-1], np.sqrt(lo * hi)), xytext=(4, -9), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=INK2)
    ax.legend(loc="upper left", ncol=3, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(FIG / "fig_anees.pdf")
    plt.close(fig)

    s = json.loads((RES / "sim_mc.json").read_text())
    sg = json.loads((RES / "sim_mc_gated.json").read_text())
    numbers["sim"] = {
        "band": [float(lo), float(hi)], "lap_s": lap, "config": s["config"],
        **{m: s["summary"][m] for m in ("ideal", "standard", "fej")},
        "gated": {m: sg["summary"][m] for m in ("ideal", "standard", "fej")},
        "grid": json.loads((RES / "sim_grid.json").read_text()),
    }


# ------------------------------------------------------------------ figure 2
def pick_run(rows) -> tuple[int, int]:
    """Holdout run whose standard-EKF position RMSE is the median of the 40 (rule fixed in advance)."""
    cand = sorted((r["pos_rmse"], r["dataset"], r["robot"]) for r in rows
                  if r["method"] == "ekf" and r["dataset"] in HOLDOUT)
    _, d, r = cand[(len(cand) - 1) // 2]
    return d, r


def fig_map(rows, numbers: dict) -> None:
    d, r = pick_run(rows)
    z = np.load(RES / "mrclam" / "runs" / f"ds{d}_r{r}.npz")
    kf_t, gt, ok = z["kf_t"], z["gt"], z["gt_valid"]
    t = kf_t - kf_t[0]
    span = (t >= 0.45 * t[-1]) & (t <= 0.45 * t[-1] + 240.0)     # a 4 minute stretch, for legibility
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.45), sharex=True, sharey=True)
    panels = [("ekf", "fej"), ("graph",)]
    for ax, methods in zip(axes, panels):
        g = np.where((span & ok)[:, None], gt[:, :2], np.nan)
        ax.plot(g[:, 1], g[:, 0], color=COLORS["truth"], lw=0.9, label=LABELS["truth"], zorder=4)
        ax.plot(z["lm_gt"][:, 1], z["lm_gt"][:, 0], ls="none", marker="+", ms=5, mew=0.9,
                color=INK, zorder=5, label="Landmark (truth)")
        for m in methods:
            pose, P = z[f"{m}_pose"], z[f"{m}_P"].astype(float)
            p = np.where(span[:, None], pose[:, :2], np.nan)
            ax.plot(p[:, 1], p[:, 0], color=COLORS[m], lw=1.0, label=LABELS[m], zorder=3)
            idx = np.flatnonzero(span)
            for k in idx[:: max(len(idx) // 9, 1)]:
                e = cov_ellipse(pose[k], P[k, :2, :2], 3.0)
                ax.plot(e[:, 1], e[:, 0], color=COLORS[m], lw=0.6, zorder=2)
            for j, (lm, C) in enumerate(zip(z[f"{m}_lm"], z[f"{m}_lm_cov"])):
                e = cov_ellipse(lm, C, 3.0)
                ax.plot(e[:, 1], e[:, 0], color=COLORS[m], lw=0.7, zorder=2)
                ax.plot(lm[1], lm[0], ls="none", marker="o", ms=2.2, color=COLORS[m], zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("y (m)")
        ax.grid(False)
    axes[0].set_ylabel("x (m)")
    axes[0].invert_xaxis()        # looking down on the room with x up and y to the left
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(0.015, 0.03, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK)
    handles, labels = {}, []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=5,
               columnspacing=1.4, handletextpad=0.5)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(FIG / "fig_map.pdf")
    plt.close(fig)
    sel = {m: next(x for x in rows if x["method"] == m and x["dataset"] == d and x["robot"] == r)
           for m in ("ekf", "fej", "graph")}
    numbers["map_run"] = {"dataset": d, "robot": r, "window_s": [float(0.45 * t[-1]), float(0.45 * t[-1] + 240)],
                          **{m: {k: sel[m][k] for k in ("pos_rmse", "pos_rmse_aligned", "heading_rmse_deg",
                                                        "nees_time_mean")} for m in sel}}


# ------------------------------------------------------------------ figure 3
def fig_consistency(rows, numbers: dict) -> None:
    rng = np.random.default_rng(0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 3.5), gridspec_kw={"height_ratios": [3, 4]})

    def strip(ax, y, vals, color, hollow):
        jit = rng.uniform(-0.22, 0.22, len(vals))
        vals = np.asarray(vals)
        ax.plot(vals[~hollow], y + jit[~hollow], ls="none", marker="o", ms=3.0, mfc=color, mec="white",
                mew=0.4, alpha=0.9, zorder=3)
        ax.plot(vals[hollow], y + jit[hollow], ls="none", marker="o", ms=3.0, mfc="none", mec=color,
                mew=0.7, zorder=3)
        med = np.median(vals)
        ax.plot([med, med], [y - 0.34, y + 0.34], color=INK, lw=1.2, zorder=4)

    # (a) time-averaged NEES per holdout run
    names = [("ekf", "Standard EKF"), ("fej", "FEJ-EKF"), ("graph", "Graph (own)")]
    for i, (m, lab) in enumerate(names):
        rr = [r for r in rows if r["method"] == m and r["dataset"] in HOLDOUT]
        strip(ax1, len(names) - 1 - i, [r["nees_time_mean"] for r in rr], COLORS[m],
              np.array([r["dataset"] == 9 for r in rr]))
    ax1.axvline(3.0, color=INK2, lw=0.7, zorder=1)
    ax1.set_xscale("log")
    ax1.set_xlim(1.5, 4000)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels([lab for _, lab in names][::-1])
    ax1.set_ylim(-0.6, len(names) - 0.4)
    ax1.set_xlabel("Time-averaged NEES per run (40 holdout runs; 3 if consistent)")
    ax1.grid(axis="y", visible=False)
    ax1.tick_params(axis="y", length=0)

    # (b) mean NIS per holdout run, two noise models
    groups = [("ekf_mad", "Standard, white noise", "ekf"), ("fej_mad", "FEJ, white noise", "fej"),
              ("ekf", "Standard, corr.-aware", "ekf"), ("fej", "FEJ, corr.-aware", "fej")]
    for i, (m, lab, c) in enumerate(groups):
        rr = [r for r in rows if r["method"] == m and r["dataset"] in HOLDOUT]
        strip(ax2, len(groups) - 1 - i, [r["nis_mean_all"] for r in rr], COLORS[c],
              np.array([r["dataset"] == 9 for r in rr]))
    ax2.axvline(2.0, color=INK2, lw=0.7, zorder=1)
    ax2.set_xscale("log")
    ax2.set_xlim(0.15, 400)
    ax2.set_yticks(range(len(groups)))
    ax2.set_yticklabels([lab for _, lab, _ in groups][::-1])
    ax2.set_ylim(-0.6, len(groups) - 0.4)
    ax2.set_xlabel("Mean NIS per run, before gating (2 if consistent)")
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        ax.xaxis.set_major_formatter(plain)
        ax.minorticks_off()
        ax.text(-0.42, 1.0, tag, transform=ax.transAxes, ha="left", va="center", fontsize=8, color=INK)
    proxies = [Line2D([], [], ls="none", marker="o", ms=3.0, mfc=INK2, mec="white", mew=0.4),
               Line2D([], [], ls="none", marker="o", ms=3.0, mfc="none", mec=INK2, mew=0.7),
               Line2D([], [], color=INK, lw=1.2)]
    ax1.legend(proxies, ["Datasets 2-8", "Dataset 9", "Median"], loc="lower center", bbox_to_anchor=(0.4, 1.0),
               ncol=3, columnspacing=1.2, handletextpad=0.3)
    fig.subplots_adjust(hspace=0.62)
    fig.savefig(FIG / "fig_consistency.pdf")
    plt.close(fig)


# -------------------------------------------------------------------- tables
METHODS = [("ekf", "Standard EKF"), ("fej", "FEJ-EKF"), ("graph", "Graph (own)"), ("gtsam", "Graph (GTSAM)")]


def tab_accuracy(rows, numbers: dict) -> None:
    def block(sel):
        out = {}
        for m, _ in METHODS + [("ekf_mad", ""), ("fej_mad", ""), ("graph_ekfset", "")]:
            rr = [r for r in rows if r["method"] == m and sel(r)]
            out[m] = {k: [q([r[k] for r in rr], p) for p in (25, 50, 75)]
                      for k in ("pos_rmse", "pos_rmse_aligned", "heading_rmse_deg", "lm_rmse", "nees_time_mean",
                                "nees_median", "nees_inside", "nees_final", "lm_nees_mean", "nis_mean_all",
                                "nis_mean_accepted", "reject_frac", "chi2_dof", "runtime_s")}
            out[m]["n"] = len(rr)
        return out

    blocks = {"holdout": block(lambda r: r["dataset"] in HOLDOUT),
              "ds2to8": block(lambda r: 2 <= r["dataset"] <= 8),
              "ds9": block(lambda r: r["dataset"] == 9),
              "ds1": block(lambda r: r["dataset"] == 1),
              "all45": block(lambda r: True)}
    numbers["real"] = blocks

    def cell(b, m, k, fmt, scale=1.0):
        lo, md, hi = (scale * v for v in b[m][k])
        return f"{fmt.format(md)} [{fmt.format(lo)}, {fmt.format(hi)}]"

    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Datasets & Method & Position RMSE (m) & After alignment (m) & Heading RMSE (deg) & Landmark RMSE (m) "
             r"& NEES \\", r"\midrule"]
    spec = [("holdout", "2--9 (40 runs)"), ("ds2to8", "2--8 (35 runs)"),
            ("ds9", "9 (5 runs)"), ("ds1", "1 (5 runs)")]
    for bi, (key, title) in enumerate(spec):
        b = blocks[key]
        for mi, (m, lab) in enumerate(METHODS):
            first = title if mi == 0 else ""
            lines.append(" & ".join([first, lab, cell(b, m, "pos_rmse", "{:.2f}"),
                                     cell(b, m, "pos_rmse_aligned", "{:.2f}"),
                                     cell(b, m, "heading_rmse_deg", "{:.1f}"), cell(b, m, "lm_rmse", "{:.2f}"),
                                     cell(b, m, "nees_time_mean", "{:.1f}")]) + r" \\")
        if bi < len(spec) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_accuracy.tex").write_text("\n".join(lines) + "\n")

    # paired comparisons on the holdout
    def paired(m1, m2, k):
        a = {(r["dataset"], r["robot"]): r[k] for r in rows if r["method"] == m1 and r["dataset"] in HOLDOUT}
        b = {(r["dataset"], r["robot"]): r[k] for r in rows if r["method"] == m2 and r["dataset"] in HOLDOUT}
        d = np.array([a[key] - b[key] for key in a])
        return {"wins": int((d < 0).sum()), "n": int(len(d)), "median_diff": float(np.median(d))}
    numbers["paired"] = {f"{a}_vs_{b}": {k: paired(a, b, k) for k in
                                          ("pos_rmse", "pos_rmse_aligned", "heading_rmse_deg", "lm_rmse",
                                           "nees_time_mean")}
                         for a, b in (("fej", "ekf"), ("graph", "ekf"), ("graph", "fej"))}
    numbers["per_dataset"] = {str(d): {m: {k: q([r[k] for r in rows if r["method"] == m and r["dataset"] == d], 50)
                                           for k in ("pos_rmse", "heading_rmse_deg", "nees_time_mean")}
                                       for m in ("ekf", "fej", "graph")} for d in range(1, 10)}
    numbers["n_keyframes"] = {"min": q([r["n_keyframes"] for r in rows if r["method"] == "ekf"], 0),
                              "median": q([r["n_keyframes"] for r in rows if r["method"] == "ekf"], 50),
                              "max": q([r["n_keyframes"] for r in rows if r["method"] == "ekf"], 100)}


def sci(x: float) -> str:
    if x == 0 or not np.isfinite(x):
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    return rf"${x / 10 ** e:.1f}{{\cdot}}10^{{{e}}}$"


def tab_agreement(numbers: dict) -> None:
    ag = load_rows(RES / "mrclam" / "agreement.csv")
    same = [a for a in ag if a["max_pos_diff"] <= 1e-5]
    other = [a for a in ag if a["max_pos_diff"] > 1e-5]
    small = {c: json.loads((RES / f"gtsam_small_{c}.json").read_text()) for c in ("se2log", "grisetti")}

    def stats(rows_, key):
        v = np.array([r[key] for r in rows_], dtype=float)
        return float(np.median(v)), float(v.max())

    def small_rows(conv, mode, noise=1.0):
        return [r for r in small[conv]["rows"] if r["own_mode"] == mode and r["noise"] == noise]

    entries = [
        (f"Synthetic, 10 graphs & log-map & {small['se2log']['gtsam_version']}", small_rows("se2log", "se2log"),
         ("cost_rel_diff", "max_pos_diff", "max_cov_rel_diff", "iterations_own", "iterations_gtsam")),
        (f"Synthetic, 10 graphs & Grisetti & {small['grisetti']['gtsam_version']}", small_rows("grisetti", "grisetti"),
         ("cost_rel_diff", "max_pos_diff", "max_cov_rel_diff", "iterations_own", "iterations_gtsam")),
        (f"MR.CLAM, {len(same)} runs & log-map & {small['se2log']['gtsam_version']}", same,
         ("cost_rel_diff", "max_pos_diff", "max_pose_cov_rel_diff", "iterations_own", "iterations_gtsam")),
    ]
    lines = [r"\begin{tabular}{@{}lllccccc@{}}", r"\toprule",
             r"Problem & Residual & GTSAM & Cost, rel. diff. & Pose diff. (m) & Covariance, rel. diff. "
             r"& Iterations & Time (s) \\", r"\midrule"]
    out = {}
    for label, rr, keys in entries:
        c, p, v = (stats(rr, k) for k in keys[:3])
        it_o, it_g = np.median([r[keys[3]] for r in rr]), np.median([r[keys[4]] for r in rr])
        if "runtime_own_s" in rr[0]:
            tm = f"{np.median([r['runtime_own_s'] for r in rr]):.2f} / {np.median([r['runtime_gtsam_s'] for r in rr]):.2f}"
        else:
            tm = "--"
        lines.append(f"{label} & {sci(c[0])} ({sci(c[1])}) & {sci(p[0])} ({sci(p[1])}) & {sci(v[0])} ({sci(v[1])}) "
                     f"& {it_o:.0f} / {it_g:.0f} & {tm}" + r" \\")
        out[label.split(" &")[0] + " | " + label.split("& ")[1].strip()] = {
            "cost_rel_diff": c, "max_pos_diff": p, "cov_rel_diff": v, "iterations": [float(it_o), float(it_g)]}
    gc, gp = stats(same, "grisetti_cost_rel_diff"), stats(same, "grisetti_max_pos_diff")
    lines.append(f"MR.CLAM, {len(same)} runs & Grisetti & {small['se2log']['gtsam_version']} & {sci(gc[0])} ({sci(gc[1])}) "
                 f"& {sci(gp[0])} ({sci(gp[1])}) & -- & -- & --" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_agreement.tex").write_text("\n".join(lines) + "\n")
    numbers["agreement"] = {
        "table": out, "n_same": len(same), "n_other": len(other),
        "other_runs": [{"dataset": a["dataset"], "robot": a["robot"], "cost_own": a["cost_own"],
                        "cost_gtsam": a["cost_gtsam"], "max_pos_diff": a["max_pos_diff"]} for a in other],
        "grisetti_vs_logmap": {"cost_rel_diff": gc, "max_pos_diff": gp},
        "runtime_own_median": float(np.median([a["runtime_own_s"] for a in same])),
        "runtime_gtsam_median": float(np.median([a["runtime_gtsam_s"] for a in same])),
        "n_poses": [float(min(a["n_poses"] for a in ag)), float(np.median([a["n_poses"] for a in ag])),
                    float(max(a["n_poses"] for a in ag))],
        "gtsam42_same_cost": float(np.median([abs(r["gtsam_cost_at_own_optimum"] - r["cost_own"]) / r["cost_own"]
                                              for r in small_rows("grisetti", "grisetti")])),
        "gtsam42_low_noise": {"cost_rel_diff": stats(small_rows("grisetti", "grisetti", 0.1), "cost_rel_diff"),
                              "max_pos_diff": stats(small_rows("grisetti", "grisetti", 0.1), "max_pos_diff")},
    }


def main() -> None:
    use_paper_style()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rows = load_rows(RES / "mrclam" / "summary.csv")
    numbers: dict = {"nees_bounds_1": list(map(float, chi2_bounds(3))), "nis_bounds_1": list(map(float, chi2_bounds(2))),
                     "nis_truncated_mean": truncated_nis_mean(9.21),
                     "config_sha256": rows[0]["config"]}
    fig_anees(numbers)
    fig_map(rows, numbers)
    fig_consistency(rows, numbers)
    tab_accuracy(rows, numbers)
    tab_agreement(numbers)
    numbers["calibration"] = json.loads((RES / "calibration_ds1.json").read_text())
    (RES / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")), sorted(p.name for p in TAB.glob("*.tex")))
    print("map run:", numbers["map_run"])


if __name__ == "__main__":
    main()
