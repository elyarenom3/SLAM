"""EKF-SLAM with three Jacobian modes on one filter.

mode = "standard": Jacobians at the latest estimates.
mode = "fej":      First-Estimates Jacobians (Huang, Mourikis, Roumeliotis).
                   Robot Jacobians use the predicted pose, landmark Jacobians
                   use the landmark's initial estimate.
mode = "ideal":    Jacobians at the true state (simulation reference only).

In all modes the state is propagated and corrected the same way; only the
points at which the Jacobians are evaluated differ.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import wrap_scalar

STANDARD, FEJ, IDEAL = "standard", "fej", "ideal"
GATE_99 = 9.210340371976184  # chi2.ppf(0.99, 2)

# status codes returned per measurement by update_batch
INIT, ACCEPT, REJECT, PENDING = 0, 1, 2, 3


class EKFSLAM:
    """Range-bearing EKF-SLAM for a unicycle robot and point landmarks.

    Parameters
    ----------
    x0, P0 : initial robot pose (3,) and covariance (3, 3).
    M : (2, 2) diagonal control-noise covariance diag(sigma_v^2, sigma_w^2),
        defined for one odometry sample of length dt_nom.
    mode : "standard", "fej" or "ideal".
    gate : NIS threshold for rejecting a measurement; None disables gating.
    dt_nom : nominal odometry period. A step of length dt gets process noise
        dt * dt_nom * Vhat M Vhat^T, which equals V M V^T when dt == dt_nom and
        stays additive when steps are split or irregular.
    floor : (rate_xy, rate_th) extra process-noise rates [m^2/s, rad^2/s].
    confirm_dist : if set, a landmark is added only after two consecutive
        sightings agree within this distance (guards against a misread first
        sighting on real data).
    """

    def __init__(self, x0, P0, M, n_max: int = 15, mode: str = STANDARD,
                 gate: float | None = GATE_99, dt_nom: float = 0.015,
                 floor=(0.0, 0.0), confirm_dist: float | None = None):
        if mode not in (STANDARD, FEJ, IDEAL):
            raise ValueError(f"unknown mode {mode!r}")
        n = 3 + 2 * n_max
        self.mode = mode
        self.gate = gate
        self.dt_nom = float(dt_nom)
        self.qv = float(M[0][0])
        self.qw = float(M[1][1])
        self.floor_xy, self.floor_th = float(floor[0]), float(floor[1])
        self.confirm_dist = confirm_dist
        self.x = np.zeros(n)
        self.x[:3] = x0
        self.P = np.zeros((n, n))
        self.P[:3, :3] = P0
        self.n = 3                      # active state size
        self.slot: dict[int, int] = {}  # landmark id -> slot
        self.x_lin = np.array(x0, dtype=float)   # FEJ robot linearisation point
        self.m_first = np.zeros((n_max, 2))      # FEJ landmark linearisation points
        self.pending: dict[int, np.ndarray] = {}
        self.hook = None                # optional callable(event tuple) for tests
        self._G = np.eye(3)

    # ------------------------------------------------------------ accessors
    @property
    def pose(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def P_rr(self) -> np.ndarray:
        return self.P[:3, :3].copy()

    def landmark(self, j: int) -> np.ndarray:
        i = 3 + 2 * self.slot[j]
        return self.x[i:i + 2].copy()

    def landmark_cov(self, j: int) -> np.ndarray:
        i = 3 + 2 * self.slot[j]
        return self.P[i:i + 2, i:i + 2].copy()

    # -------------------------------------------------------------- predict
    def predict(self, v: float, w: float, dt: float, truth=None, q=None) -> None:
        """One odometry step. O(N): only P_rr and P_rm change.

        truth = (x_true_k, x_true_k+1) is required in "ideal" mode.
        q = (sigma_v^2, sigma_w^2) overrides M for this step (used when the
        control noise depends on the command, as on the real robots).
        """
        x = self.x
        th = x[2]
        c, s = math.cos(th), math.sin(th)
        xn = x[0] + v * dt * c
        yn = x[1] + v * dt * s
        thn = wrap_scalar(th + w * dt)

        if self.mode == STANDARD:
            gx, gy = -(yn - x[1]), xn - x[0]
        elif self.mode == FEJ:
            # subtract the previous *predicted* position; the new mean is still
            # propagated from the updated pose
            gx, gy = -(yn - self.x_lin[1]), xn - self.x_lin[0]
        else:
            xt, xt1 = truth
            gx, gy = -(xt1[1] - xt[1]), xt1[0] - xt[0]
            c, s = math.cos(xt[2]), math.sin(xt[2])

        P = self.P
        n = self.n
        if n > 3:  # P_rm <- G P_rm ; G = I + e0*gx*e2^T + e1*gy*e2^T
            P[0, 3:n] += gx * P[2, 3:n]
            P[1, 3:n] += gy * P[2, 3:n]
            P[3:n, 0] = P[0, 3:n]
            P[3:n, 1] = P[1, 3:n]
        G = self._G
        G[0, 2] = gx
        G[1, 2] = gy
        Prr = G @ P[:3, :3] @ G.T
        scale = dt * self.dt_nom
        qv, qw = (self.qv, self.qw) if q is None else q
        qv *= scale
        Prr[0, 0] += qv * c * c + self.floor_xy * dt
        Prr[0, 1] += qv * c * s
        Prr[1, 0] += qv * c * s
        Prr[1, 1] += qv * s * s + self.floor_xy * dt
        Prr[2, 2] += scale * qw + self.floor_th * dt
        P[:3, :3] = Prr

        x[0], x[1], x[2] = xn, yn, thn
        self.x_lin[0], self.x_lin[1], self.x_lin[2] = xn, yn, thn
        if self.hook is not None:
            self.hook(("predict", gx, gy))

    # --------------------------------------------------------------- update
    def update_batch(self, ids, zs, R, truth=None):
        """Process all measurements sharing one timestamp.

        ids : landmark ids; zs : (k, 2) rows of (range, bearing);
        R : (2, 2) covariance or callable(range) -> (2, 2).
        truth = (x_true, {id: m_true}) is required in "ideal" mode.

        New landmarks are initialised first, from the predicted pose, then
        known landmarks are updated one at a time. Returns a list of
        (id, nis, status) with nis = nan for initialisations.
        """
        out = []
        known = []
        for j, z in zip(ids, zs):
            if j in self.slot:
                known.append((j, z))
                continue
            Rj = R(z[0]) if callable(R) else R
            status = self._try_init(j, z, Rj, truth)
            out.append((j, math.nan, status))
        for j, z in known:
            Rj = R(z[0]) if callable(R) else R
            nis, ok = self._update(j, z, Rj, truth)
            out.append((j, nis, ACCEPT if ok else REJECT))
        return out

    def _try_init(self, j, z, R, truth):
        if self.confirm_dist is not None:
            a = self.x[2] + z[1]
            cand = np.array([self.x[0] + z[0] * math.cos(a), self.x[1] + z[0] * math.sin(a)])
            prev = self.pending.get(j)
            if prev is None or np.hypot(*(cand - prev)) > self.confirm_dist:
                self.pending[j] = cand
                return PENDING
            del self.pending[j]
        self._init(j, z, R, truth)
        return INIT

    def _init(self, j, z, R, truth):
        x, P, n = self.x, self.P, self.n
        s = len(self.slot)
        i = 3 + 2 * s
        r, phi = z[0], z[1]
        a = x[2] + phi
        ca, sa = math.cos(a), math.sin(a)
        mx, my = x[0] + r * ca, x[1] + r * sa

        if self.mode == STANDARD:
            dx, dy = mx - x[0], my - x[1]
        elif self.mode == FEJ:
            dx, dy = mx - self.x_lin[0], my - self.x_lin[1]
        else:
            xt, lm = truth
            dx, dy = lm[j][0] - xt[0], lm[j][1] - xt[1]
            r = math.hypot(dx, dy)
            ca, sa = dx / r, dy / r
        Jx = np.array([[1.0, 0.0, -dy], [0.0, 1.0, dx]])
        Jz = np.array([[ca, -r * sa], [sa, r * ca]])

        P[i:i + 2, :n] = Jx @ P[:3, :n]
        P[:n, i:i + 2] = P[i:i + 2, :n].T
        P[i:i + 2, i:i + 2] = Jx @ P[:3, :3] @ Jx.T + Jz @ R @ Jz.T
        x[i], x[i + 1] = mx, my
        self.m_first[s] = (mx, my)
        self.slot[j] = s
        self.n = n + 2
        if self.hook is not None:
            self.hook(("init", s, Jx.copy()))

    def _update(self, j, z, R, truth):
        x, P, n = self.x, self.P, self.n
        s = self.slot[j]
        i = 3 + 2 * s

        # innovation, always from the latest estimates
        ex, ey = x[i] - x[0], x[i + 1] - x[1]
        nu = np.array([z[0] - math.hypot(ex, ey),
                       wrap_scalar(z[1] - (math.atan2(ey, ex) - x[2]))])

        # Jacobian linearisation point
        if self.mode == STANDARD:
            dx, dy = ex, ey
        elif self.mode == FEJ:
            dx, dy = self.m_first[s, 0] - self.x_lin[0], self.m_first[s, 1] - self.x_lin[1]
        else:
            xt, lm = truth
            dx, dy = lm[j][0] - xt[0], lm[j][1] - xt[1]
        q = dx * dx + dy * dy
        sq = math.sqrt(q)
        H5 = np.array([[-dx / sq, -dy / sq, 0.0, dx / sq, dy / sq],
                       [dy / q, -dx / q, -1.0, -dy / q, dx / q]])
        idx = [0, 1, 2, i, i + 1]

        PHt = P[:n, idx] @ H5.T                 # (n, 2)
        S = H5 @ PHt[idx] + R
        S = 0.5 * (S + S.T)
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:      # covariance lost positive definiteness (diverged filter)
            return math.inf, False
        y = np.linalg.solve(L, nu)
        nis = float(y @ y)
        accepted = self.gate is None or nis <= self.gate
        if self.hook is not None:
            self.hook(("update", s, H5.copy(), accepted))
        if not accepted:
            return nis, False

        # K = P H^T S^-1 through the Cholesky factor
        K = np.linalg.solve(L.T, np.linalg.solve(L, PHt.T)).T   # (n, 2)
        x[:n] += K @ nu
        x[2] = wrap_scalar(x[2])

        # Joseph form: (I - KH) P (I - KH)^T + K R K^T
        A = P[:n, :n] - K @ PHt.T
        AHt = A[:, idx] @ H5.T
        Pn = A - AHt @ K.T + K @ R @ K.T
        P[:n, :n] = 0.5 * (Pn + Pn.T)
        return nis, True
