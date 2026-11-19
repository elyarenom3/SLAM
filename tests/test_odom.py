"""Odometry pre-integration: mean, covariance and the frame of the pose-pose factor."""
import math

import numpy as np
import pytest

from slam2d import graph as gr
from slam2d.ekf import EKFSLAM
from slam2d.geometry import se2_compose, wrap
from slam2d.models import motion
from slam2d.odom import Stream, dead_reckon, preintegrate, to_end_frame

DT = 0.015
SV, SW = 0.05, 0.1


def make_stream(controls, seg_len, x0=(0.0, 0.0, 0.0), sv=SV, sw=SW):
    """A stream with a keyframe every seg_len steps and no measurements."""
    controls = np.asarray(controls, dtype=float)
    S = len(controls)
    steps = np.column_stack([controls, np.full(S, DT), np.full(S, sv ** 2), np.full(S, sw ** 2)])
    seg = np.arange(0, S + 1, seg_len)
    K1 = len(seg)
    return Stream("synthetic", np.array(x0, dtype=float), seg * DT, steps, seg,
                  np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros(K1 + 1, dtype=int),
                  np.zeros((K1, 3)), np.ones(K1, dtype=bool))


def controls(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([0.2 + 0.05 * np.sin(np.arange(n) / 40), 0.6 * np.sin(np.arange(n) / 25 + rng.uniform(0, 6))])


def test_preintegration_composes_and_matches_dead_reckoning():
    u = controls(600)
    fine, _ = preintegrate(make_stream(u, 50))
    coarse, _ = preintegrate(make_stream(u, 150))
    for k in range(len(coarse)):
        rel = np.zeros(3)
        for z in fine[3 * k:3 * k + 3]:
            rel = se2_compose(rel, z)
        assert np.allclose(rel[:2], coarse[k, :2], atol=1e-12)
        assert abs(wrap(rel[2] - coarse[k, 2])) < 1e-12
    st = make_stream(u, 50, x0=(1.0, -2.0, 0.7))
    dr = dead_reckon(st)
    x = st.x0.copy()
    for z in fine:
        x = se2_compose(x, z)
    assert np.allclose(x[:2], dr[-1, :2], atol=1e-10) and abs(wrap(x[2] - dr[-1, 2])) < 1e-10


def test_preintegrated_covariance_is_singular_without_a_floor_when_stationary():
    """V M V^T has rank 2: a robot that does not move has no lateral noise."""
    st = make_stream(np.zeros((40, 2)), 40)
    _, S = preintegrate(st)
    assert np.linalg.matrix_rank(S[0], tol=1e-14) == 2
    _, Sf = preintegrate(st, floor=(1e-6, 3e-6))
    assert np.linalg.eigvalsh(Sf[0]).min() > 0


def test_whitened_odometry_residual_is_chi2_with_three_dof():
    """Residual at the true poses, whitened with the end-frame covariance, has mean 3.

    The same residual whitened with the start-frame covariance does not: the
    recursion gives the covariance in the frame of X_k, the residual lives in
    the end frame, and they differ by the rotation R(theta_z).
    """
    rng = np.random.default_rng(4)
    n = 80
    u_true = np.column_stack([np.full(n, 0.25), np.full(n, 1.2)])      # turns about 83 degrees
    x_true = np.zeros(3)
    for v, w in u_true:
        x_true = motion(x_true, v, w, DT)
    good, bad = [], []
    for _ in range(4000):
        u_meas = u_true + rng.standard_normal(u_true.shape) * (SV, SW) * 3
        z, S = preintegrate(make_stream(u_meas, n, sv=3 * SV, sw=3 * SW), floor=(1e-9, 1e-9))
        e, _, _ = gr.between_residual(np.zeros(3), x_true, z[0])
        good.append(e[0] @ np.linalg.solve(to_end_frame(z, S)[0], e[0]))
        bad.append(e[0] @ np.linalg.solve(S[0], e[0]))
    assert abs(np.mean(good) - 3.0) < 0.15
    assert np.mean(bad) > 4.0


def test_odometry_only_graph_matches_ekf_dead_reckoning_covariance():
    u = controls(900, seed=2)
    st = make_stream(u, 60, x0=(0.5, 0.3, -0.4), sv=0.02, sw=0.04)
    floor = (1e-10, 1e-10)
    z, S = preintegrate(st, floor=floor)
    P0 = np.diag([1e-8, 1e-8, 1e-8])
    problem = gr.GraphProblem(st.x0, np.sqrt(np.diag(P0)), z, to_end_frame(z, S),
                              np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros((0, 2)), np.zeros((0, 2)))
    solver = gr.GraphSLAM(problem)
    X = dead_reckon(st)
    out = solver.optimise(X, np.zeros((0, 2)))
    assert out["cost"] < 1e-12 and np.abs(out["X"] - X).max() < 1e-7      # chain is already optimal
    Pp, _ = solver.marginals(out["X"], out["L"])

    ekf = EKFSLAM(st.x0, P0, np.zeros((2, 2)), n_max=0, floor=floor)
    for v, w, dt, qv, qw in st.steps:
        ekf.predict(v, w, dt, q=(qv, qw))
    assert np.allclose(ekf.pose, out["X"][-1], atol=1e-7)
    assert np.allclose(Pp[-1], ekf.P_rr, rtol=2e-2, atol=1e-12)
    assert np.allclose(Pp, solver.marginals_dense(out["X"], out["L"])[0], rtol=1e-6, atol=1e-16)
