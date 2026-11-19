"""Simulator: a unicycle robot looping through the MR.CLAM landmark layout.

Truth follows the discrete motion model with clean controls. The filter
receives controls with N(0, M) noise and range-bearing measurements with
N(0, R) noise, so the only modelling error left is linearisation.
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict

import numpy as np

from .data.landmarks_mrclam import LANDMARKS
from .ekf import EKFSLAM, ACCEPT, REJECT
from .geometry import wrap
from .metrics import nees_series
from .models import motion


@dataclass
class SimConfig:
    v: float = 0.2                      # forward speed [m/s]
    laps: int = 10
    centre: tuple = (2.9, 0.0)          # centre of the rounded-rectangle track [m]
    size: tuple = (4.4, 10.4)           # track extent in x and y [m]
    corner_radius: float = 1.0
    dt: float = 0.015                   # odometry period [s] (67 Hz)
    meas_every: int = 13                # one measurement epoch every 13 ticks (about 5 Hz)
    r_max: float = 3.0                  # sensing range [m]
    r_min: float = 0.8                  # closest detectable range [m] (MR.CLAM camera minimum)
    fov: float = 2.0 * math.pi          # full field of view [rad]
    sigma_v: float = 0.05               # per-sample control noise [m/s]
    sigma_w: float = 0.1                # per-sample control noise [rad/s]
    sigma_r: float = 0.1                # range noise [m]
    sigma_phi: float = math.radians(2)  # bearing noise [rad]
    p0_xy: float = 1e-3                 # initial position sigma [m]
    p0_th: float = 1e-3                 # initial heading sigma [rad]

    def M(self) -> np.ndarray:
        return np.diag([self.sigma_v ** 2, self.sigma_w ** 2])

    def R(self) -> np.ndarray:
        return np.diag([self.sigma_r ** 2, self.sigma_phi ** 2])

    def P0(self) -> np.ndarray:
        return np.diag([self.p0_xy ** 2, self.p0_xy ** 2, self.p0_th ** 2])


@dataclass
class World:
    cfg: SimConfig
    landmarks: np.ndarray      # (L, 2)
    u: np.ndarray              # (K, 2) clean controls
    X: np.ndarray              # (K + 1, 3) true poses
    epochs: np.ndarray         # (T,) state indices with a measurement epoch
    vis: list                  # per epoch: array of visible landmark ids
    z: list                    # per epoch: (n, 2) clean range-bearing
    lap_ticks: int

    @property
    def t(self) -> np.ndarray:
        return self.epochs * self.cfg.dt


def build_world(cfg: SimConfig, landmarks: np.ndarray = LANDMARKS) -> World:
    """Controls and true trajectory for a counter-clockwise rounded rectangle.

    Corner turn rates are set so each corner turns exactly 90 degrees, which
    makes the discrete loop close exactly.
    """
    step = cfg.v * cfg.dt
    rho = cfg.corner_radius
    n_x = round((cfg.size[0] - 2 * rho) / step)
    n_y = round((cfg.size[1] - 2 * rho) / step)
    n_c = round(0.5 * math.pi * rho / step)
    w_c = 0.5 * math.pi / (n_c * cfg.dt)
    lap = ([(cfg.v, 0.0)] * n_x + [(cfg.v, w_c)] * n_c
           + [(cfg.v, 0.0)] * n_y + [(cfg.v, w_c)] * n_c) * 2
    u = np.array(lap * cfg.laps)
    K = len(u)

    X = np.empty((K + 1, 3))
    X[0] = (cfg.centre[0] - cfg.size[0] / 2 + rho, cfg.centre[1] - cfg.size[1] / 2, 0.0)
    for k in range(K):
        X[k + 1] = motion(X[k], u[k, 0], u[k, 1], cfg.dt)

    epochs = np.arange(cfg.meas_every, K + 1, cfg.meas_every)
    vis, zs = [], []
    for k in epochs:
        d = landmarks - X[k, :2]
        r = np.hypot(d[:, 0], d[:, 1])
        phi = wrap(np.arctan2(d[:, 1], d[:, 0]) - X[k, 2])
        ok = (r <= cfg.r_max) & (r >= cfg.r_min) & (np.abs(phi) <= cfg.fov / 2)
        ids = np.flatnonzero(ok)
        vis.append(ids)
        zs.append(np.column_stack([r[ids], phi[ids]]))
    return World(cfg, np.asarray(landmarks, dtype=float), u, X, epochs, vis, zs, len(lap))


def draw_noise(world: World, seed: int, zero: bool = False):
    """Control, measurement and initial-state noise for one trial."""
    cfg = world.cfg
    rng = np.random.default_rng(seed)
    du = rng.standard_normal(world.u.shape) * (cfg.sigma_v, cfg.sigma_w)
    dz = [rng.standard_normal(z.shape) * (cfg.sigma_r, cfg.sigma_phi) for z in world.z]
    dx0 = rng.standard_normal(3) * (cfg.p0_xy, cfg.p0_xy, cfg.p0_th)
    if zero:
        du[:] = 0.0
        dx0[:] = 0.0
        dz = [np.zeros_like(d) for d in dz]
    return du, dz, dx0


def run_filter(world: World, noise, mode: str, gate=None, hook=None) -> dict:
    """Run one filter over one noise realisation and score it against truth."""
    cfg = world.cfg
    du, dz, dx0 = noise
    X, lm = world.X, world.landmarks
    ekf = EKFSLAM(X[0] + dx0, cfg.P0(), cfg.M(), n_max=len(lm), mode=mode,
                  gate=gate, dt_nom=cfg.dt)
    ekf.hook = hook
    R = cfg.R()
    um = world.u + du
    T = len(world.epochs)
    err = np.empty((T, 3))
    Prr = np.empty((T, 3, 3))
    nis = []
    i = 0
    next_epoch = world.epochs[0]
    for k in range(len(um)):
        ekf.predict(um[k, 0], um[k, 1], cfg.dt, truth=(X[k], X[k + 1]))
        if k + 1 == next_epoch:
            ids = world.vis[i]
            if len(ids):
                zs = world.z[i] + dz[i]
                zs[:, 1] = wrap(zs[:, 1])
                res = ekf.update_batch(ids, zs, R, truth=(X[k + 1], lm))
                nis.extend(n for _, n, s in res if s in (ACCEPT, REJECT))
            e = X[k + 1] - ekf.x[:3]
            e[2] = wrap(e[2])
            err[i] = e
            Prr[i] = ekf.P[:3, :3]
            i += 1
            next_epoch = world.epochs[i] if i < T else -1

    nees = nees_series(err, Prr)
    lm_nees = np.full(len(lm), np.nan)
    lm_err = np.full((len(lm), 2), np.nan)
    for j in ekf.slot:
        e = lm[j] - ekf.landmark(j)
        lm_err[j] = e
        lm_nees[j] = nees_series(e[None], ekf.landmark_cov(j)[None])[0]
    return {"err": err, "nees": nees, "sig": np.sqrt(np.einsum("tii->ti", Prr)),
            "nis": np.asarray(nis), "lm_nees": lm_nees, "lm_err": lm_err}


def _trial(args):
    world, seed, modes, gate = args
    noise = draw_noise(world, seed)
    return {m: run_filter(world, noise, m, gate) for m in modes}


def run_monte_carlo(world: World, n_trials: int = 50, modes=("ideal", "standard", "fej"),
                    gate=None, seed0: int = 1000, workers: int | None = None) -> dict:
    """Run n_trials realisations; every mode sees identical noise per trial.

    Returns {mode: {"err": (N, T, 3), "nees": (N, T), "sig": (N, T, 3),
                    "nis_mean": (N,), "lm_nees": (N, L), "lm_err": (N, L, 2)}}.
    """
    jobs = [(world, seed0 + i, tuple(modes), gate) for i in range(n_trials)]
    if workers == 1:
        results = [_trial(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_trial, jobs))
    out = {}
    for m in modes:
        out[m] = {
            "err": np.stack([r[m]["err"] for r in results]),
            "nees": np.stack([r[m]["nees"] for r in results]),
            "sig": np.stack([r[m]["sig"] for r in results]),
            "nis_mean": np.array([r[m]["nis"].mean() if r[m]["nis"].size else np.nan
                                  for r in results]),
            "lm_nees": np.stack([r[m]["lm_nees"] for r in results]),
            "lm_err": np.stack([r[m]["lm_err"] for r in results]),
        }
    return out


def config_dict(cfg: SimConfig) -> dict:
    return asdict(cfg)


# ----------------------------------------------------- small synthetic graphs
def small_graph(seed: int = 0, n_poses: int = 30, n_landmarks: int = 6, noise: float = 1.0,
                sigma_odo=(0.04, 0.02, 0.02), sigma_r: float = 0.08, sigma_phi: float = 0.03,
                r_max: float = 6.0):
    """A random planar SLAM graph with known truth, for tests and solver comparisons.

    Returns (problem, X_true, L_true, X_init, L_init). Odometry noise is drawn
    in the frame of the pose-pose residual, so the residual at the true poses
    is N(0, odo_cov) exactly; `noise` scales all noise draws (0 gives exact data).
    Initial values are dead reckoning and first-sighting landmark positions.
    """
    from .geometry import se2_between, se2_compose, se2_inverse
    from .graph import GraphProblem
    from .models import inverse_measure, measure

    rng = np.random.default_rng(seed)
    X = np.zeros((n_poses, 3))
    for k in range(1, n_poses):
        rel = np.array([rng.uniform(0.4, 0.9), rng.uniform(-0.05, 0.05), rng.uniform(-0.6, 0.6)])
        X[k] = se2_compose(X[k - 1], rel)
    centre = X[:, :2].mean(axis=0)
    spread = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]), 2.0)
    L = centre + rng.uniform(-0.6, 0.6, (n_landmarks, 2)) * spread

    A = rng.normal(size=(n_poses - 1, 3, 3)) * 0.3 + np.eye(3)
    S = np.diag(sigma_odo)
    cov = A @ (S @ S)[None] @ np.transpose(A, (0, 2, 1))        # full, well-conditioned covariances
    odo_z = np.empty((n_poses - 1, 3))
    for k in range(1, n_poses):
        delta = noise * rng.multivariate_normal(np.zeros(3), cov[k - 1])
        odo_z[k - 1] = se2_compose(se2_between(X[k - 1], X[k]), se2_inverse(delta))

    br_pose, br_lm, br_z = [], [], []
    for k in range(n_poses):
        d = np.hypot(*(L - X[k, :2]).T)
        seen = np.flatnonzero(d < r_max)
        if len(seen) == 0:
            seen = np.array([int(np.argmin(d))])
        for j in seen:
            z = measure(X[k], L[j]) + noise * rng.normal(size=2) * (sigma_r, sigma_phi)
            br_pose.append(k)
            br_lm.append(j)
            br_z.append([z[0], wrap(z[1])])
    br_pose, br_lm, br_z = np.array(br_pose), np.array(br_lm), np.array(br_z)
    used = np.unique(br_lm)                         # relabel so every landmark has a factor
    br_lm = np.searchsorted(used, br_lm)
    L = L[used]

    problem = GraphProblem(prior_mean=X[0].copy(), prior_sigma=np.array([1e-3, 1e-3, 1e-3]),
                           odo_z=odo_z, odo_cov=cov, br_pose=br_pose, br_lm=br_lm, br_z=br_z,
                           br_sigma=np.tile([sigma_r, sigma_phi], (len(br_z), 1)),
                           lm_ids=list(range(len(L))))
    X_init = np.zeros_like(X)
    X_init[0] = X[0]
    for k in range(1, n_poses):
        X_init[k] = se2_compose(X_init[k - 1], odo_z[k - 1])
    L_init = np.zeros_like(L)
    for j in range(len(L)):
        m = int(np.flatnonzero(br_lm == j)[0])
        L_init[j] = inverse_measure(X_init[br_pose[m]], br_z[m])
    return problem, X, L, X_init, L_init
