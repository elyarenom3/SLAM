import math

import numpy as np
import pytest

from slam2d import geometry as g


def test_wrap_range_and_identity():
    a = np.linspace(-20, 20, 2001)
    w = g.wrap(a)
    assert np.all(w >= -np.pi) and np.all(w < np.pi)
    assert np.allclose(np.exp(1j * w), np.exp(1j * a))
    assert g.wrap_scalar(3 * math.pi / 2) == pytest.approx(-math.pi / 2)


def test_interp_angle_crosses_wrap():
    tp = np.array([0.0, 1.0])
    thp = np.array([math.pi - 0.1, -math.pi + 0.1])   # 0.2 rad apart through pi
    mid = g.interp_angle(0.5, tp, thp)
    assert abs(abs(mid) - math.pi) < 1e-12


def test_se2_group_identities():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-3, 3, 3)
        b = rng.uniform(-3, 3, 3)
        assert np.allclose(g.se2_compose(a, g.se2_inverse(a)), 0, atol=1e-12)
        d = g.se2_between(a, b)
        back = g.se2_compose(a, d)
        assert np.allclose(back[:2], b[:2], atol=1e-12)
        assert abs(g.wrap_scalar(back[2] - b[2])) < 1e-12


def test_se2_log_matches_exponential():
    """exp(log(p)) recovers p, using the closed-form SE(2) exponential."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        p = np.array([*rng.uniform(-2, 2, 2), rng.uniform(-3, 3)])
        xi = g.se2_log(p)
        w = xi[2]
        V = np.array([[math.sin(w), -(1 - math.cos(w))], [1 - math.cos(w), math.sin(w)]]) / w
        assert np.allclose(V @ xi[:2], p[:2], atol=1e-10)
        assert xi[2] == pytest.approx(p[2])


def test_se2_log_small_angle_branch_is_continuous():
    p_small = np.array([0.7, -0.4, 5e-5])
    p_large = np.array([0.7, -0.4, 2e-4])
    assert np.allclose(g.se2_log(p_small)[:2], p_small[:2], atol=1e-4)
    assert np.allclose(g.se2_log(p_large)[:2], p_large[:2], atol=1e-3)
    assert np.allclose(g.se2_dlog(p_small), g.se2_dlog(p_large), atol=1e-3)


def test_cov_ellipse_axes():
    pts = g.cov_ellipse([1.0, 2.0], np.diag([4.0, 1.0]), nsig=3.0, n=400)
    assert np.max(pts[:, 0]) == pytest.approx(1.0 + 6.0, abs=1e-3)
    assert np.max(pts[:, 1]) == pytest.approx(2.0 + 3.0, abs=1e-3)
