"""Planar geometry helpers: angle wrapping, rotations, SE(2) operations."""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi
J2 = np.array([[0.0, -1.0], [1.0, 0.0]])  # rotation by +90 degrees


def wrap(a):
    """Wrap angle(s) to [-pi, pi). Works on scalars and arrays."""
    return (np.asarray(a, dtype=float) + np.pi) % TWO_PI - np.pi


def wrap_scalar(a: float) -> float:
    """Wrap one Python float to [-pi, pi). Faster than `wrap` in tight loops."""
    return (a + math.pi) % TWO_PI - math.pi


def rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def drot(th: float) -> np.ndarray:
    """Derivative of rot(th) with respect to th."""
    c, s = math.cos(th), math.sin(th)
    return np.array([[-s, -c], [c, -s]])


def interp_angle(t, tp, thp):
    """Interpolate a wrapped angle series at times t (unwrap, interpolate, wrap)."""
    return wrap(np.interp(t, tp, np.unwrap(thp)))


# ----------------------------------------------------------------- SE(2)
def se2_compose(a, b) -> np.ndarray:
    """Pose a followed by pose b (b expressed in the frame of a)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1],
                     a[1] + s * b[0] + c * b[1],
                     wrap_scalar(a[2] + b[2])])


def se2_inverse(a) -> np.ndarray:
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), wrap_scalar(-a[2])])


def se2_between(a, b) -> np.ndarray:
    """Relative pose a^-1 * b: pose b expressed in the frame of a."""
    c, s = math.cos(a[2]), math.sin(a[2])
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap_scalar(b[2] - a[2])])


def _vinv_terms(w):
    """a(w) = (w/2) cot(w/2) and its derivative, with series near zero."""
    w = np.asarray(w, dtype=float)
    small = np.abs(w) < 1e-4
    ws = np.where(small, 1.0, w)  # placeholder avoids division by zero
    half = 0.5 * ws
    a = np.where(small, 1.0 - w * w / 12.0, half / np.tan(half))
    da = np.where(small, -w / 6.0, 0.5 / np.tan(half) - 0.25 * ws / np.sin(half) ** 2)
    return a, da


def se2_log(p):
    """SE(2) logarithm of pose(s) p = (x, y, th). Shape (..., 3) in and out.

    Log(t, th) = (V(th)^-1 t, th) with V^-1 = [[a, th/2], [-th/2, a]],
    a = (th/2) cot(th/2). This matches gtsam.Pose2.Logmap.
    """
    p = np.asarray(p, dtype=float)
    w = p[..., 2]
    a, _ = _vinv_terms(w)
    out = np.empty_like(p)
    out[..., 0] = a * p[..., 0] + 0.5 * w * p[..., 1]
    out[..., 1] = -0.5 * w * p[..., 0] + a * p[..., 1]
    out[..., 2] = w
    return out


def se2_dlog(p):
    """Derivative of se2_log with respect to the components (x, y, th).

    Shape (..., 3) in, (..., 3, 3) out.
    """
    p = np.asarray(p, dtype=float)
    w = p[..., 2]
    a, da = _vinv_terms(w)
    D = np.zeros(p.shape[:-1] + (3, 3))
    D[..., 0, 0] = a
    D[..., 0, 1] = 0.5 * w
    D[..., 1, 0] = -0.5 * w
    D[..., 1, 1] = a
    D[..., 0, 2] = da * p[..., 0] + 0.5 * p[..., 1]
    D[..., 1, 2] = -0.5 * p[..., 0] + da * p[..., 1]
    D[..., 2, 2] = 1.0
    return D


def cov_ellipse(mean, P, nsig: float = 3.0, n: int = 60) -> np.ndarray:
    """Points (n, 2) on the nsig-sigma ellipse of a 2x2 covariance."""
    vals, vecs = np.linalg.eigh(0.5 * (P + P.T))
    vals = np.clip(vals, 0.0, None)
    t = np.linspace(0.0, TWO_PI, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    return (np.asarray(mean)[:2, None] + nsig * vecs @ (np.sqrt(vals)[:, None] * circ)).T
