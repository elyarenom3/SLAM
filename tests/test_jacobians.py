"""Every analytic Jacobian against central differences (step 1e-6)."""
import math

import numpy as np
import pytest

from slam2d import geometry as g
from slam2d import models as m

H = 1e-6
TOL = 1e-6


def numjac(f, x, angle_rows=()):
    """Central-difference Jacobian of f at x; listed output rows are angles."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x))
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        d = np.zeros_like(x)
        d[i] = H
        diff = np.asarray(f(x + d)) - np.asarray(f(x - d))
        for r in angle_rows:
            diff[r] = g.wrap_scalar(diff[r])
        J[:, i] = diff / (2 * H)
    return J


def poses(n=40, seed=0):
    rng = np.random.default_rng(seed)
    out = [np.array([*rng.uniform(-5, 5, 2), rng.uniform(-math.pi, math.pi)]) for _ in range(n)]
    # states at and next to the angle wrap
    out += [np.array([1.0, -2.0, math.pi - 1e-9]), np.array([0.3, 0.2, -math.pi + 1e-9]),
            np.array([0.0, 0.0, 0.0])]
    return out


@pytest.mark.parametrize("xr", poses())
def test_motion_G_and_V(xr):
    v, w, dt = 0.37, -0.52, 0.015
    G_num = numjac(lambda x: m.motion(x, v, w, dt), xr, angle_rows=(2,))
    assert np.allclose(m.motion_G(v, xr[2], dt), G_num, atol=TOL)
    V_num = numjac(lambda u: m.motion(xr, u[0], u[1], dt), np.array([v, w]), angle_rows=(2,))
    assert np.allclose(m.motion_V(xr[2], dt), V_num, atol=TOL)


@pytest.mark.parametrize("xr", poses(10, seed=1))
def test_motion_G_displacement_form_equals_standard(xr):
    v, w, dt = 0.2, 0.3, 0.015
    nxt = m.motion(xr, v, w, dt)
    assert np.allclose(m.motion_G_disp(nxt, xr), m.motion_G(v, xr[2], dt), atol=1e-14)


@pytest.mark.parametrize("xr", poses(40, seed=2))
def test_measurement_jacobians(xr):
    rng = np.random.default_rng(int(abs(xr[0]) * 1e6) % 2**32)
    lm = xr[:2] + rng.uniform(0.8, 6.0) * np.array([math.cos(a := rng.uniform(-3.1, 3.1)), math.sin(a)])
    Hr, Hm = m.measure_jac(xr[:2], lm)
    Hr_num = numjac(lambda x: m.measure(x, lm), xr, angle_rows=(1,))
    Hm_num = numjac(lambda l: m.measure(xr, l), lm, angle_rows=(1,))
    assert np.allclose(Hr, Hr_num, atol=TOL)
    assert np.allclose(Hm, Hm_num, atol=TOL)
    # structure used by the observability argument: H_r = H_m [-I, -J delta]
    delta = lm - xr[:2]
    assert np.allclose(Hr, Hm @ np.column_stack([-np.eye(2), -g.J2 @ delta]), atol=1e-12)


@pytest.mark.parametrize("xr", poses(40, seed=3))
def test_inverse_measurement_jacobians(xr):
    z = np.array([2.7, 0.4])
    Jx, Jz = m.inverse_measure_jac(xr, z)
    assert np.allclose(Jx, numjac(lambda x: m.inverse_measure(x, z), xr), atol=TOL)
    assert np.allclose(Jz, numjac(lambda zz: m.inverse_measure(xr, zz), z), atol=TOL)
    # inverse and forward models are consistent
    assert np.allclose(m.measure(xr, m.inverse_measure(xr, z)), z, atol=1e-12)


@pytest.mark.parametrize("p", [np.array([0.4, -0.3, 0.7]), np.array([-1.2, 0.9, -2.5]),
                               np.array([0.05, 0.02, 1e-3]), np.array([0.3, 0.1, 0.0]),
                               np.array([2.0, -1.0, 3.0])])
def test_se2_dlog(p):
    assert np.allclose(g.se2_dlog(p), numjac(g.se2_log, p), atol=TOL)


# ------------------------------------------------------------------ graph factors
from slam2d import graph as gr  # noqa: E402


@pytest.mark.parametrize("mode", (gr.GRISETTI, gr.SE2LOG))
@pytest.mark.parametrize("seed", range(12))
def test_pose_pose_jacobians(mode, seed):
    rng = np.random.default_rng(seed)
    xi = np.array([*rng.uniform(-3, 3, 2), rng.uniform(-math.pi, math.pi)])
    z = np.array([*rng.uniform(-1, 1, 2), rng.uniform(-1.5, 1.5)])
    # second pose near the measured relative pose so the log-map residual is well inside (-pi, pi)
    xj = g.se2_compose(xi, g.se2_compose(z, np.array([*rng.normal(0, 0.2, 2), rng.normal(0, 0.3)])))
    e, A, B = gr.between_residual(xi, xj, z, mode)
    f_i = lambda x: gr.between_residual(x, xj, z, mode)[0][0]
    f_j = lambda x: gr.between_residual(xi, x, z, mode)[0][0]
    assert np.allclose(A[0], numjac(f_i, xi, angle_rows=(2,)), atol=TOL)
    assert np.allclose(B[0], numjac(f_j, xj, angle_rows=(2,)), atol=TOL)


@pytest.mark.parametrize("mode", (gr.GRISETTI, gr.SE2LOG))
def test_prior_jacobian(mode):
    """The prior is the pose-pose residual with a fixed first pose and zero measurement."""
    mu = np.array([1.0, -2.0, 0.7])
    x = mu + np.array([0.03, -0.02, 0.05])
    e, _, J = gr.between_residual(mu, x, np.zeros(3), mode)
    f = lambda v: gr.between_residual(mu, v, np.zeros(3), mode)[0][0]
    assert np.allclose(J[0], numjac(f, x, angle_rows=(2,)), atol=TOL)
    assert np.allclose(gr.between_residual(mu, mu, np.zeros(3), mode)[0], 0.0, atol=1e-15)


@pytest.mark.parametrize("xr", poses(20, seed=5))
def test_bearing_range_factor_jacobians_reuse_ekf_blocks(xr):
    lm = xr[:2] + np.array([2.0, -1.5])
    z = np.array([2.2, 0.3])
    e, Jp, Jl = gr.bearing_range_residual(xr, lm, z)
    Hr, Hm = m.measure_jac(xr[:2], lm)
    assert np.allclose(Jp[0], Hr, atol=1e-14) and np.allclose(Jl[0], Hm, atol=1e-14)
    f_p = lambda x: gr.bearing_range_residual(x, lm, z)[0][0]
    f_l = lambda l: gr.bearing_range_residual(xr, l, z)[0][0]
    assert np.allclose(Jp[0], numjac(f_p, xr, angle_rows=(1,)), atol=TOL)
    assert np.allclose(Jl[0], numjac(f_l, lm, angle_rows=(1,)), atol=TOL)
