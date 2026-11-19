"""Loader for the UTIAS MR.CLAM dataset, one single-robot SLAM problem per robot log.

Layout expected: <root>/ds{N}/{Barcodes,Landmark_Groundtruth,Robot{1-5}_{Groundtruth,
Measurement,Odometry}}.dat (see scripts/download_mrclam.py).

Facts the loader relies on (checked against the files):
- every file has 4 header lines starting with '#', whitespace-separated columns;
- the measurement file's second column is a *barcode* number;
- Barcodes.dat differs between datasets, so it is read per dataset;
- subjects 1-5 are robots, 6-20 landmarks; dataset 8 has 14 mapped landmarks;
- odometry is the commanded (v, w); timestamps can repeat or step backwards;
- Vicon ground truth has gaps (up to several minutes in some logs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..geometry import wrap

N_ROBOTS = 5
ROBOT_SUBJECTS = range(1, 6)
MAX_GT_GAP = 0.2      # [s] largest Vicon bracket still treated as valid ground truth


@dataclass
class RobotLog:
    dataset: int
    robot: int
    gt: np.ndarray            # (n, 4) t, x, y, th      (time relative to Dataset.t_ref)
    odom: np.ndarray          # (n, 3) t, v, w          cleaned, strictly increasing time
    meas: np.ndarray          # (n, 4) t, subject, range, bearing   landmarks only
    dropped: dict = field(default_factory=dict)   # counts of rows removed, by reason

    @property
    def name(self) -> str:
        return f"ds{self.dataset}_r{self.robot}"

    # ------------------------------------------------------- ground truth
    def gt_at(self, t, max_gap: float = MAX_GT_GAP):
        """Interpolated ground-truth pose(s) at time(s) t and a validity mask.

        Position is interpolated linearly and heading through its unwrapped
        series. A sample is valid when t lies inside the log and its two
        bracketing Vicon stamps are at most max_gap apart.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        g = self.gt
        i = np.clip(np.searchsorted(g[:, 0], t, side="right"), 1, len(g) - 1)
        gap = g[i, 0] - g[i - 1, 0]
        valid = (t >= g[0, 0]) & (t <= g[-1, 0]) & (gap <= max_gap)
        pose = np.column_stack([np.interp(t, g[:, 0], g[:, 1]),
                                np.interp(t, g[:, 0], g[:, 2]),
                                wrap(np.interp(t, g[:, 0], self._th_unwrapped))])
        return pose, valid

    @property
    def _th_unwrapped(self) -> np.ndarray:
        cached = self.__dict__.get("_thu")
        if cached is None:
            cached = self.__dict__["_thu"] = np.unwrap(self.gt[:, 3])
        return cached

    def window(self) -> tuple[float, float]:
        """Run window: where odometry and valid ground truth are both available.

        Starts at the first time at or after the first odometry stamp with a
        valid ground-truth pose, and ends at the earlier of the last odometry
        and last ground-truth stamps.
        """
        t0 = max(self.odom[0, 0], self.gt[0, 0])
        t1 = min(self.odom[-1, 0], self.gt[-1, 0])
        if not self.gt_at(t0)[1][0]:
            g = self.gt[:, 0]
            i = int(np.searchsorted(g, t0, side="left"))
            good = np.flatnonzero(np.diff(g[i:]) <= MAX_GT_GAP)
            t0 = g[i + good[0]]
        return float(t0), float(t1)


@dataclass
class Dataset:
    index: int
    t_ref: float                              # epoch seconds subtracted from all times
    barcodes: dict                            # barcode -> subject
    landmarks: dict                           # subject -> (x, y) ground truth
    robots: list
    landmark_fix: dict = field(default_factory=dict)   # subject -> subject whose position it took

    def robot(self, r: int) -> RobotLog:
        return self.robots[r - 1]


# ------------------------------------------------------------------ parsing
def _read(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", ndmin=2)


def clean_odometry(odom: np.ndarray):
    """Stable sort by time and keep the last row of any repeated timestamp."""
    order = np.argsort(odom[:, 0], kind="stable")
    n_unsorted = int(np.sum(np.diff(odom[:, 0]) < 0))
    odom = odom[order]
    keep = np.append(np.diff(odom[:, 0]) > 0, True)
    return odom[keep], {"odom_backward_steps": n_unsorted, "odom_repeated_stamps": int(np.sum(~keep))}


def load_dataset(root, index: int, cache: bool = True, fix_landmarks: bool = True) -> Dataset:
    """Load one MR.CLAM dataset. Parsed arrays are cached in <root>/../cache/."""
    root = Path(root)
    cache_file = root.parent / "cache" / f"ds{index}.npz"
    if cache and cache_file.exists():
        raw = dict(np.load(cache_file))
    else:
        d = root / f"ds{index}"
        raw = {"barcodes": _read(d / "Barcodes.dat"), "landmarks": _read(d / "Landmark_Groundtruth.dat")}
        for r in range(1, N_ROBOTS + 1):
            raw[f"gt{r}"] = _read(d / f"Robot{r}_Groundtruth.dat")
            raw[f"odom{r}"] = _read(d / f"Robot{r}_Odometry.dat")
            raw[f"meas{r}"] = _read(d / f"Robot{r}_Measurement.dat")
        if cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_file, **raw)

    barcodes = {int(b): int(s) for s, b in raw["barcodes"]}
    landmarks = {int(row[0]): (float(row[1]), float(row[2])) for row in raw["landmarks"]}
    t_ref = float(np.floor(min(raw[f"gt{r}"][0, 0] for r in range(1, N_ROBOTS + 1))))

    robots = []
    for r in range(1, N_ROBOTS + 1):
        gt = raw[f"gt{r}"].copy()
        gt[:, 0] -= t_ref
        gt[:, 3] = wrap(gt[:, 3])
        odom, dropped = clean_odometry(raw[f"odom{r}"])
        odom[:, 0] -= t_ref

        m = raw[f"meas{r}"]
        subj = np.array([barcodes.get(int(b), 0) for b in m[:, 1]])
        is_robot = np.isin(subj, list(ROBOT_SUBJECTS))
        is_lm = np.array([s in landmarks for s in subj])
        dropped.update(meas_robot_sightings=int(is_robot.sum()),
                       meas_unmapped_barcodes=int(np.sum(subj == 0)),
                       meas_landmark_rows=int(is_lm.sum()))
        meas = np.column_stack([m[is_lm, 0] - t_ref, subj[is_lm], m[is_lm, 2], m[is_lm, 3]])
        meas = meas[np.argsort(meas[:, 0], kind="stable")]
        robots.append(RobotLog(index, r, gt, odom, meas, dropped))

    ds = Dataset(index, t_ref, barcodes, landmarks, robots)
    if fix_landmarks:
        apply_landmark_fix(ds)
    return ds


# ------------------------------------------------ landmark identity check
def landmark_identity_check(ds: Dataset, near: float = 0.5, far: float = 1.0) -> dict:
    """Find landmarks whose sightings sit on another landmark's ground-truth position.

    For each subject, every sighting is projected into the world with the
    ground-truth robot pose and the median projected point is compared with
    all ground-truth landmark positions. Returns {subject: matching subject}
    for subjects whose median lies within `near` of another landmark and
    farther than `far` from their own. Used only for scoring and calibration.
    """
    proj: dict[int, list] = {s: [] for s in ds.landmarks}
    for rb in ds.robots:
        if not len(rb.meas):
            continue
        pose, valid = rb.gt_at(rb.meas[:, 0])
        a = pose[:, 2] + rb.meas[:, 3]
        p = pose[:, :2] + rb.meas[:, 2:3] * np.column_stack([np.cos(a), np.sin(a)])
        for s in proj:
            sel = valid & (rb.meas[:, 1] == s)
            if sel.any():
                proj[s].append(p[sel])
    subjects = sorted(ds.landmarks)
    pos = np.array([ds.landmarks[s] for s in subjects])
    out = {}
    for s in subjects:
        if not proj[s]:
            continue
        med = np.median(np.vstack(proj[s]), axis=0)
        d = np.hypot(*(pos - med).T)
        best = subjects[int(np.argmin(d))]
        if best != s and d.min() < near and d[subjects.index(s)] > far:
            out[s] = best
    return out


def apply_landmark_fix(ds: Dataset) -> None:
    """Reassign ground-truth positions where the identity check finds a clean permutation."""
    mapping = landmark_identity_check(ds)
    if not mapping:
        return
    if sorted(mapping) != sorted(mapping.values()):
        raise ValueError(f"dataset {ds.index}: landmark mismatch is not a permutation: {mapping}")
    old = dict(ds.landmarks)
    for s, other in mapping.items():
        ds.landmarks[s] = old[other]
    ds.landmark_fix = mapping


def all_runs(root, datasets=range(1, 10)):
    """Yield (Dataset, RobotLog) for every robot of every requested dataset."""
    for n in datasets:
        ds = load_dataset(root, n)
        for rb in ds.robots:
            yield ds, rb
