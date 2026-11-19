"""EKF-SLAM behaviour in simulation, including the observability property of FEJ."""
import math

import numpy as np
import pytest

from slam2d import models
from slam2d.ekf import EKFSLAM, STANDARD, FEJ, IDEAL
from slam2d.geometry import J2, wrap
from slam2d.metrics import anees_bounds
from slam2d.sim import SimConfig, build_world, draw_noise, run_filter, run_monte_carlo

MODES = (STANDARD, FEJ, IDEAL)


@pytest.fixture(scope="module")
def world():
    return build_world(SimConfig(laps=1))


@pytest.mark.parametrize("mode", MODES)
def test_zero_noise_recovers_truth(world, mode):
    out = run_filter(world, draw_noise(world, 0, zero=True), mode)
    assert np.abs(out["err"]).max() < 1e-9
    assert np.nanmax(np.abs(out["lm_err"])) < 1e-9


def test_zero_noise_with_measurements_off_the_odometry_ticks():
    """Split an odometry step at an arbitrary measurement time; still exact."""
    rng = np.random.default_rng(3)
    lm = np.array([[2.0, 1.0], [0.5, 3.0], [-1.0, 0.5]])
    x_true = np.array([0.0, 0.0, 0.3])
    ekf = EKFSLAM(x_true, np.diag([1e-6] * 3), np.diag([0.05 ** 2, 0.1 ** 2]), n_max=3, mode=FEJ)
    R = np.diag([0.1 ** 2, 0.03 ** 2])
    dt = 0.015
    for k in range(400):
        v, w = 0.2, 0.25 * math.sin(k / 40)
        frac = rng.uniform(0.1, 0.9)         # measurement falls inside the step
        for part in (frac * dt, (1 - frac) * dt):
            x_true = models.motion(x_true, v, w, part)
            ekf.predict(v, w, part)
            if part == frac * dt and k % 10 == 0:
                zs = np.array([models.measure(x_true, l) for l in lm])
                ekf.update_batch(range(3), zs, R)
    assert np.abs(ekf.pose - x_true).max() < 1e-9
    for j in range(3):
        assert np.abs(ekf.landmark(j) - lm[j]).max() < 1e-9


def _filter_with_map(mode=STANDARD, seed=0):
    rng = np.random.default_rng(seed)
    ekf = EKFSLAM([1.0, 2.0, 0.4], np.diag([0.04, 0.03, 0.01]), np.diag([0.05 ** 2, 0.1 ** 2]),
                  n_max=4, mode=mode, gate=None)
    R = np.diag([0.1 ** 2, 0.03 ** 2])
    zs = np.column_stack([rng.uniform(1, 4, 4), rng.uniform(-1, 1, 4)])
    ekf.update_batch(range(4), zs, R)
    ekf.update_batch(range(4), zs + rng.normal(0, 0.02, zs.shape), R)
    return ekf, R


def test_predict_matches_dense_form_and_brief_formula():
    ekf, _ = _filter_with_map()
    n = ekf.n
    P0, x0 = ekf.P[:n, :n].copy(), ekf.x.copy()
    v, w, dt = 0.3, -0.4, 0.015
    ekf.predict(v, w, dt)
    F = np.eye(n)
    F[:3, :3] = models.motion_G(v, x0[2], dt)
    V = models.motion_V(x0[2], dt)
    Q = np.zeros((n, n))
    Q[:3, :3] = V @ np.diag([0.05 ** 2, 0.1 ** 2]) @ V.T      # V M V^T at the nominal period
    assert np.allclose(ekf.P[:n, :n], F @ P0 @ F.T + Q, atol=1e-15, rtol=1e-12)
    assert np.allclose(ekf.P[3:n, 3:n], P0[3:, 3:])           # P_mm untouched


def test_split_step_noise_is_additive():
    M = np.diag([0.05 ** 2, 0.1 ** 2])
    P0 = np.diag([0.01, 0.02, 0.005])
    for v, tol in ((0.0, 1e-14), (0.3, 2e-3)):
        a = EKFSLAM([0, 0, 0.3], P0, M)
        b = EKFSLAM([0, 0, 0.3], P0, M)
        a.predict(v, 0.0, 0.015)
        b.predict(v, 0.0, 0.004)
        b.predict(v, 0.0, 0.011)
        assert np.allclose(a.x, b.x, atol=1e-14)
        added_a, added_b = a.P[:3, :3] - P0, b.P[:3, :3] - P0
        assert np.allclose(added_a, added_b, atol=tol * np.abs(added_a).max())


@pytest.mark.parametrize("mode", MODES)
def test_covariance_stays_symmetric_psd(world, mode):
    cfg = world.cfg
    noise = draw_noise(world, 5)
    X, lm = world.X, world.landmarks
    ekf = EKFSLAM(X[0], cfg.P0(), cfg.M(), n_max=len(lm), mode=mode, gate=9.21, dt_nom=cfg.dt)
    um = world.u + noise[0]
    i = 0
    for k in range(3000):
        ekf.predict(um[k, 0], um[k, 1], cfg.dt, truth=(X[k], X[k + 1]))
        if k + 1 == world.epochs[i]:
            ekf.update_batch(world.vis[i], world.z[i] + noise[1][i], cfg.R(), truth=(X[k + 1], lm))
            i += 1
    P = ekf.P[:ekf.n, :ekf.n]
    assert np.allclose(P, P.T, atol=0)
    assert np.linalg.eigvalsh(P).min() > -1e-12


def test_landmark_initial_covariance_matches_monte_carlo():
    rng = np.random.default_rng(11)
    x = np.array([1.0, -0.5, 0.8])
    Prr = np.array([[4e-4, 1e-4, 2e-5], [1e-4, 3e-4, -1e-5], [2e-5, -1e-5, 1e-4]])
    R = np.diag([0.02 ** 2, 0.01 ** 2])
    z = np.array([2.5, 0.3])
    ekf = EKFSLAM(x, Prr, np.eye(2), n_max=1)
    ekf.update_batch([0], [z], R)
    xs = rng.multivariate_normal(x, Prr, 200000)
    zs = rng.multivariate_normal(z, R, 200000)
    a = xs[:, 2] + zs[:, 1]
    ms = np.column_stack([xs[:, 0] + zs[:, 0] * np.cos(a), xs[:, 1] + zs[:, 0] * np.sin(a)])
    C = np.cov(np.column_stack([ms, xs]).T)
    assert np.allclose(ekf.P[3:5, 3:5], C[:2, :2], rtol=0.03, atol=2e-6)
    assert np.allclose(ekf.P[3:5, :3], C[:2, 2:], rtol=0.05, atol=2e-6)


# ------------------------------------------------------------- observability
def _replay(mode, steps=4000, seed=2):
    """Run a filter, propagating the 3-column null-space basis N alongside.

    N has rows for the robot (3) and each landmark (2). Returns the largest
    |H_k N_k| per column over all updates, the largest deviation of the
    landmark rows from [I, J m_first], and the change of N^T P^-1 N across
    all measurement batches.
    """
    cfg = SimConfig(laps=1, p0_xy=0.05, p0_th=0.03)
    world = build_world(cfg)
    noise = draw_noise(world, seed)
    X, lm = world.X, world.landmarks
    x0 = X[0] + noise[2]
    ekf = EKFSLAM(x0, cfg.P0(), cfg.M(), n_max=len(lm), mode=mode, gate=None, dt_nom=cfg.dt)

    p0 = X[0] if mode == IDEAL else x0
    state = {"Nr": np.array([[1, 0, -p0[1]], [0, 1, p0[0]], [0, 0, 1.0]]), "Nm": {},
             "hn": np.zeros(3)}

    def hook(ev):
        if ev[0] == "predict":
            G = np.eye(3)
            G[0, 2], G[1, 2] = ev[1], ev[2]
            state["Nr"] = G @ state["Nr"]
        elif ev[0] == "init":
            state["Nm"][ev[1]] = ev[2] @ state["Nr"]
        else:
            _, s, H5, _ = ev
            hn = H5[:, :3] @ state["Nr"] + H5[:, 3:] @ state["Nm"][s]
            state["hn"] = np.maximum(state["hn"], np.abs(hn).max(axis=0))

    ekf.hook = hook

    def info():
        n = ekf.n
        N = np.vstack([state["Nr"]] + [state["Nm"][s] for s in range((n - 3) // 2)])
        return N.T @ np.linalg.solve(ekf.P[:n, :n], N)

    um = world.u + noise[0]
    gain = np.zeros((3, 3))
    i = 0
    for k in range(steps):
        ekf.predict(um[k, 0], um[k, 1], cfg.dt, truth=(X[k], X[k + 1]))
        if k + 1 == world.epochs[i]:
            before = info() if ekf.n > 3 else None
            n_before = ekf.n
            ekf.update_batch(world.vis[i], world.z[i] + noise[1][i], cfg.R(), truth=(X[k + 1], lm))
            if before is not None and ekf.n == n_before:
                gain += info() - before
            i += 1

    rows_dev = 0.0
    for s, Nm in state["Nm"].items():
        m_first = lm[[j for j, t in ekf.slot.items() if t == s][0]] if mode == IDEAL else ekf.m_first[s]
        expect = np.column_stack([np.eye(2), J2 @ m_first])
        rows_dev = max(rows_dev, np.abs(Nm - expect).max())
    return state["hn"], rows_dev, gain, info()


@pytest.mark.parametrize("mode", (FEJ, IDEAL))
def test_fej_and_ideal_keep_three_unobservable_directions(mode):
    hn, rows_dev, gain, info = _replay(mode)
    assert hn.max() < 1e-9                      # H_k N_k = 0 for all three columns
    assert rows_dev < 1e-9                      # landmark rows are [I, J m_first]
    assert np.abs(gain).max() < 1e-6 * np.abs(info).max()   # no information gained along N


def test_standard_ekf_gains_spurious_heading_information():
    hn, _, gain, info = _replay(STANDARD)
    assert hn[:2].max() < 1e-9                  # the two translations stay unobservable
    assert hn[2] > 1e-3                         # the global rotation does not
    assert gain[2, 2] > 1e-3 * info[2, 2]       # and the filter gains information along it


# ------------------------------------------------------------------- Monte Carlo
def test_short_monte_carlo_ideal_filter_is_consistent():
    world = build_world(SimConfig(laps=1))
    mc = run_monte_carlo(world, n_trials=20, modes=(IDEAL, FEJ), workers=1)
    lo, hi = anees_bounds(20)
    for mode in (IDEAL, FEJ):
        anees = mc[mode]["nees"].mean(axis=0)
        inside = np.mean((anees >= lo) & (anees <= hi))
        assert 2.5 < anees.mean() < 3.6, (mode, anees.mean())
        assert inside > 0.7, (mode, inside)
        assert abs(mc[mode]["nis_mean"].mean() - 2.0) < 0.1
