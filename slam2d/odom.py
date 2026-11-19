"""Odometry handling shared by the filter and the graph.

`build_stream` merges odometry and landmark measurements of one robot log into
one time-sorted stream: keyframes (one per measurement timestamp) and the
zero-order-hold odometry steps between them. `preintegrate` turns the steps
between consecutive keyframes into relative-pose factors with covariance,
using the same motion model and noise recursion as the EKF.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .calib import IDENTITY_CAM, camera_correct, correct_controls, turn_flag
from .ekf import EKFSLAM
from .geometry import wrap


@dataclass
class Stream:
    name: str
    x0: np.ndarray        # (3,) ground-truth start pose
    kf_t: np.ndarray      # (K + 1,) keyframe times; kf_t[0] is the start time
    steps: np.ndarray     # (S, 5) v, w, dt, sigma_v^2, sigma_w^2   (corrected controls)
    seg: np.ndarray       # (K + 1,) steps[seg[k-1]:seg[k]] lead from keyframe k-1 to k
    meas_id: np.ndarray   # (M,) landmark subject number
    meas_z: np.ndarray    # (M, 2) corrected range, bearing
    kf_ptr: np.ndarray    # (K + 2,) measurements of keyframe k are rows kf_ptr[k]:kf_ptr[k+1]
    gt: np.ndarray        # (K + 1, 3) ground truth at the keyframes
    gt_valid: np.ndarray  # (K + 1,) bool

    @property
    def n_keyframes(self) -> int:
        return len(self.kf_t)

    @property
    def meas_kf(self) -> np.ndarray:
        """Keyframe index of every measurement row."""
        return np.repeat(np.arange(len(self.kf_t)), np.diff(self.kf_ptr))


def build_stream(rb, odo=None, cam=None, camera_tau: float = 0.0, merge_dt: float = 0.015,
                 max_step: float = 0.02, t_end: float | None = None,
                 skip_stationary: bool = False) -> Stream:
    """Merge one robot log into keyframes and odometry steps.

    odo = dict(sv, cv, sw, bw, tau): corrected controls (calib.correct_controls),
    stamps shifted by tau. cam: camera pre-correction (see calib.camera_correct).
    Measurement timestamps closer than merge_dt to the current keyframe share it.
    Steps longer than max_step are subdivided for integration accuracy.
    With skip_stationary, sightings taken while the command in force is
    v = w = 0 are dropped: a robot standing still repeats the same image, and
    a white-noise estimator would count every repeat as new information.
    """
    odo = odo or {"sv": 1.0, "sw": 1.0, "bw": 0.0, "tau": 0.0}
    ot = rb.odom[:, 0] + odo.get("tau", 0.0)
    ov, ow = correct_controls(rb.odom[:, 1], rb.odom[:, 2], odo)

    t0, t1 = rb.window()
    t0 = max(t0, ot[0])
    if not rb.gt_at(t0)[1][0]:                      # start needs a valid ground-truth pose
        g = rb.gt[:, 0]
        i = int(np.searchsorted(g, t0, side="left"))
        t0 = float(g[i + np.flatnonzero(np.diff(g[i:]) <= 0.2)[0]])
    if t_end is not None:
        t1 = min(t1, t_end)

    m = rb.meas.copy()
    m[:, 0] += camera_tau
    m = m[(m[:, 0] > t0 + 1e-9) & (m[:, 0] <= t1)]
    m = m[np.argsort(m[:, 0], kind="stable")]
    if skip_stationary:
        i = np.clip(np.searchsorted(ot, m[:, 0], side="right") - 1, 0, None)
        m = m[~((rb.odom[i, 1] == 0) & (rb.odom[i, 2] == 0))]
    r, phi = camera_correct(m[:, 2], m[:, 3], cam or IDENTITY_CAM)

    # keyframes: unique measurement times, merged when closer than merge_dt
    kf_t = [t0]
    kf_of = np.empty(len(m), dtype=int)
    for i, t in enumerate(m[:, 0]):
        if len(kf_t) == 1 or t - kf_t[-1] >= merge_dt:
            kf_t.append(t)
        kf_of[i] = len(kf_t) - 1
    kf_t = np.array(kf_t)
    K1 = len(kf_t)
    kf_ptr = np.searchsorted(kf_of, np.arange(K1 + 1), side="left")

    # zero-order-hold steps between consecutive breakpoints
    inside = ot[(ot > t0) & (ot < kf_t[-1])]
    T = np.unique(np.concatenate([kf_t, inside]))
    L = np.diff(T)
    ctrl = np.searchsorted(ot, T[:-1], side="right") - 1
    n_sub = np.maximum(np.ceil(L / max_step - 1e-9).astype(int), 1)
    if "sigma_w_t" in odo:      # two-state control noise keyed on the commanded turn rate
        turning = turn_flag(ot, rb.odom[:, 2], T[:-1], odo["turn_mem"], odo["turn_thr"])
        qv = np.where(turning, odo["sigma_v_t"], odo["sigma_v_s"]) ** 2
        qw = np.where(turning, odo["sigma_w_t"], odo["sigma_w_s"]) ** 2
    else:
        qv = np.full(len(L), odo.get("sigma_v", 0.0) ** 2)
        qw = np.full(len(L), odo.get("sigma_w", 0.0) ** 2)
    steps = np.column_stack([np.repeat(ov[ctrl], n_sub), np.repeat(ow[ctrl], n_sub),
                             np.repeat(L / n_sub, n_sub), np.repeat(qv, n_sub), np.repeat(qw, n_sub)])
    cum = np.concatenate([[0], np.cumsum(n_sub)])
    seg = cum[np.searchsorted(T, kf_t)]

    gt, valid = rb.gt_at(kf_t)
    return Stream(rb.name, gt[0].copy(), kf_t, steps, seg, m[:, 1].astype(int),
                  np.column_stack([r, wrap(phi)]), kf_ptr, gt, valid)


def preintegrate(stream: Stream, dt_nom: float = 0.015, floor=(0.0, 0.0), q_scale: float = 1.0):
    """Relative pose and covariance between consecutive keyframes.

    For each keyframe pair the steps are integrated in the frame of the first
    keyframe, starting from the zero pose with zero covariance, with the EKF's
    own predict step (Sigma <- G Sigma G^T + Q). Returns z (K, 3) and
    Sigma (K, 3, 3), where row k-1 links keyframe k-1 to keyframe k and Sigma
    is expressed in the frame of keyframe k-1.
    """
    K = stream.n_keyframes - 1
    z = np.zeros((K, 3))
    S = np.zeros((K, 3, 3))
    f = EKFSLAM(np.zeros(3), np.zeros((3, 3)), np.zeros((2, 2)), n_max=0, dt_nom=dt_nom, floor=floor)
    steps = stream.steps.tolist()
    seg = stream.seg
    for k in range(1, K + 1):
        f.x[:] = 0.0
        f.P[:] = 0.0
        for v, w, dt, qv, qw in steps[seg[k - 1]:seg[k]]:
            f.predict(v, w, dt, q=(q_scale * qv, q_scale * qw))
        z[k - 1] = f.x
        S[k - 1] = f.P
    return z, S


def to_end_frame(z, S):
    """Rotate odometry covariances from the start frame to the end frame.

    The pose-pose residual e = [R_z^T (R_i^T (t_j - t_i) - t_z), .] lives in
    the end frame of the measurement, so its covariance is T S T^T with
    T = blkdiag(R(theta_z)^T, 1).
    """
    c, s = np.cos(z[:, 2]), np.sin(z[:, 2])
    T = np.zeros((len(z), 3, 3))
    T[:, 0, 0], T[:, 0, 1] = c, s
    T[:, 1, 0], T[:, 1, 1] = -s, c
    T[:, 2, 2] = 1.0
    return T @ S @ np.transpose(T, (0, 2, 1))


def dead_reckon(stream: Stream) -> np.ndarray:
    """Open-loop poses at the keyframes from the start pose and the odometry steps."""
    out = np.empty((stream.n_keyframes, 3))
    x, y, th = stream.x0
    out[0] = stream.x0
    steps = stream.steps.tolist()
    for k in range(1, stream.n_keyframes):
        for v, w, dt, _, _ in steps[stream.seg[k - 1]:stream.seg[k]]:
            x += v * dt * math.cos(th)
            y += v * dt * math.sin(th)
            th += w * dt
        out[k] = (x, y, th)
    out[:, 2] = wrap(out[:, 2])
    return out
