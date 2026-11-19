"""Motion and measurement models with their Jacobians.

State: x = [x_r, y_r, th | m1x, m1y, ...]. Controls u = (v, w).
Every Jacobian takes its linearisation point explicitly, so the same code
serves the standard EKF (latest estimates), the FEJ EKF (first estimates)
and the ideal EKF (true state).
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import wrap_scalar


def motion(xr, v: float, w: float, dt: float) -> np.ndarray:
    """One odometry step of the unicycle model."""
    th = xr[2]
    return np.array([xr[0] + v * dt * math.cos(th),
                     xr[1] + v * dt * math.sin(th),
                     wrap_scalar(th + w * dt)])


def motion_G(v: float, th: float, dt: float) -> np.ndarray:
    """G = d f / d(x_r, y_r, th), evaluated at heading th."""
    return np.array([[1.0, 0.0, -v * dt * math.sin(th)],
                     [0.0, 1.0, v * dt * math.cos(th)],
                     [0.0, 0.0, 1.0]])


def motion_G_disp(p_next, p_lin) -> np.ndarray:
    """G written from a displacement: off-diagonals -(y_next - y_lin), (x_next - x_lin).

    With p_lin the pose the step started from this equals `motion_G`. The FEJ
    filter passes the previous *predicted* position as p_lin instead.
    """
    return np.array([[1.0, 0.0, -(p_next[1] - p_lin[1])],
                     [0.0, 1.0, p_next[0] - p_lin[0]],
                     [0.0, 0.0, 1.0]])


def motion_V(th: float, dt: float) -> np.ndarray:
    """V = d f / d(v, w), evaluated at heading th."""
    return np.array([[dt * math.cos(th), 0.0],
                     [dt * math.sin(th), 0.0],
                     [0.0, dt]])


def measure(xr, m) -> np.ndarray:
    """Expected range and bearing from robot pose xr to landmark m."""
    dx, dy = m[0] - xr[0], m[1] - xr[1]
    return np.array([math.hypot(dx, dy), wrap_scalar(math.atan2(dy, dx) - xr[2])])


def measure_jac(p, m):
    """H_r (2x3) and H_m (2x2) for robot position p and landmark m.

    The Jacobians depend only on delta = m - p; the heading enters as the
    constant -1 in the bearing row.
    """
    dx, dy = m[0] - p[0], m[1] - p[1]
    q = dx * dx + dy * dy
    sq = math.sqrt(q)
    Hr = np.array([[-dx / sq, -dy / sq, 0.0],
                   [dy / q, -dx / q, -1.0]])
    Hm = np.array([[dx / sq, dy / sq],
                   [-dy / q, dx / q]])
    return Hr, Hm


def inverse_measure(xr, z) -> np.ndarray:
    """Landmark position implied by a range-bearing measurement z = (r, phi)."""
    a = xr[2] + z[1]
    return np.array([xr[0] + z[0] * math.cos(a), xr[1] + z[0] * math.sin(a)])


def inverse_measure_jac(xr, z):
    """J_x (2x3) and J_z (2x2) of `inverse_measure`."""
    r = z[0]
    a = xr[2] + z[1]
    c, s = math.cos(a), math.sin(a)
    Jx = np.array([[1.0, 0.0, -r * s],
                   [0.0, 1.0, r * c]])
    Jz = np.array([[c, -r * s],
                   [s, r * c]])
    return Jx, Jz
