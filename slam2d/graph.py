"""Graph-SLAM: a sparse Gauss-Newton / Levenberg-Marquardt optimiser for planar landmark SLAM.

Variables: keyframe poses X_k = (x, y, th) and landmark positions L_j = (x, y).
Factors:   a prior on X_0, pose-pose odometry factors, bearing-range factors.

Cost F(x) = sum e^T Omega e. Normal equations H = sum J^T Omega J, b = sum J^T Omega e.
LM step (H + lambda diag(H)) dx = -b, x <- x + dx with angles wrapped.

Pose-pose residual, `pose_error`:
  "grisetti": e = [R_z^T (R_i^T (t_j - t_i) - t_z), wrap(th_j - th_i - th_z)]
              (Grisetti et al. 2010; identical to GTSAM 4.2's BetweenFactorPose2)
  "se2log":   e = Log(grisetti residual read as a pose)
              (the full SE(2) logarithm, GTSAM 4.3's BetweenFactorPose2)
The log-map Jacobians are the Grisetti A, B premultiplied by dLog.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from .geometry import se2_dlog, se2_log, wrap

GRISETTI, SE2LOG = "grisetti", "se2log"


@dataclass
class GraphProblem:
    prior_mean: np.ndarray        # (3,) pose of X_0
    prior_sigma: np.ndarray       # (3,) sigma x, y, th (x and y equal)
    odo_z: np.ndarray             # (K, 3) relative pose X_{k-1} -> X_k
    odo_cov: np.ndarray           # (K, 3, 3) covariance of the residual (end frame)
    br_pose: np.ndarray           # (M,) pose index of each bearing-range factor
    br_lm: np.ndarray             # (M,) landmark index (0..L-1)
    br_z: np.ndarray              # (M, 2) range, bearing
    br_sigma: np.ndarray          # (M, 2) sigma range, sigma bearing
    lm_ids: list = field(default_factory=list)   # landmark index -> external id
    odo_i: np.ndarray | None = None              # (K,) first pose index (default k-1)
    odo_j: np.ndarray | None = None              # (K,) second pose index (default k)
    n_poses: int | None = None

    def __post_init__(self):
        K = len(self.odo_z)
        if self.odo_i is None:
            self.odo_i, self.odo_j = np.arange(K), np.arange(1, K + 1)
        if self.n_poses is None:
            self.n_poses = int(max(self.odo_j.max(initial=0), self.br_pose.max(initial=0))) + 1

    @property
    def n_landmarks(self) -> int:
        return len(self.lm_ids) if self.lm_ids else (int(self.br_lm.max()) + 1 if len(self.br_lm) else 0)

    @property
    def dof(self) -> int:
        """Residual dimensions minus variable dimensions."""
        return 3 + 3 * len(self.odo_z) + 2 * len(self.br_z) - 3 * self.n_poses - 2 * self.n_landmarks


# ------------------------------------------------------------------ residuals
def between_residual(xi, xj, z, mode: str = GRISETTI):
    """Pose-pose residual and its Jacobians A = de/dx_i, B = de/dx_j. Arrays (n, 3)."""
    xi, xj, z = np.atleast_2d(xi), np.atleast_2d(xj), np.atleast_2d(z)
    n = len(xj)
    ci, si = np.cos(xi[:, 2]), np.sin(xi[:, 2])
    cz, sz = np.cos(z[:, 2]), np.sin(z[:, 2])
    dx, dy = xj[:, 0] - xi[:, 0], xj[:, 1] - xi[:, 1]
    # u = R_i^T d - t_z ;  e_t = R_z^T u
    ux = ci * dx + si * dy - z[:, 0]
    uy = -si * dx + ci * dy - z[:, 1]
    e = np.empty((n, 3))
    e[:, 0] = cz * ux + sz * uy
    e[:, 1] = -sz * ux + cz * uy
    e[:, 2] = wrap(xj[:, 2] - xi[:, 2] - z[:, 2])

    # M = R_z^T R_i^T = R(th_i + th_z)^T
    c, s = np.cos(xi[:, 2] + z[:, 2]), np.sin(xi[:, 2] + z[:, 2])
    B = np.zeros((n, 3, 3))
    B[:, 0, 0], B[:, 0, 1] = c, s
    B[:, 1, 0], B[:, 1, 1] = -s, c
    B[:, 2, 2] = 1.0
    A = -B.copy()
    # R_z^T (dR_i^T / dth_i) d, with dR^T/dth = [[-s, c], [-c, -s]]
    wx = -si * dx + ci * dy
    wy = -ci * dx - si * dy
    A[:, 0, 2] = cz * wx + sz * wy
    A[:, 1, 2] = -sz * wx + cz * wy

    if mode == SE2LOG:
        D = se2_dlog(e)
        e = se2_log(e)
        A, B = D @ A, D @ B
    elif mode != GRISETTI:
        raise ValueError(f"unknown pose_error {mode!r}")
    return e, A, B


def bearing_range_residual(xp, lm, z):
    """e = h(x, l) - z with rows (range, bearing), and Jacobians wrt pose and landmark."""
    xp, lm, z = np.atleast_2d(xp), np.atleast_2d(lm), np.atleast_2d(z)
    n = len(xp)
    dx, dy = lm[:, 0] - xp[:, 0], lm[:, 1] - xp[:, 1]
    q = dx * dx + dy * dy
    sq = np.sqrt(q)
    e = np.column_stack([sq - z[:, 0], wrap(np.arctan2(dy, dx) - xp[:, 2] - z[:, 1])])
    Jp = np.zeros((n, 2, 3))
    Jp[:, 0, 0], Jp[:, 0, 1] = -dx / sq, -dy / sq
    Jp[:, 1, 0], Jp[:, 1, 1], Jp[:, 1, 2] = dy / q, -dx / q, -1.0
    Jl = np.empty((n, 2, 2))
    Jl[:, 0, 0], Jl[:, 0, 1] = dx / sq, dy / sq
    Jl[:, 1, 0], Jl[:, 1, 1] = -dy / q, dx / q
    return e, Jp, Jl


# ------------------------------------------------------------------ optimiser
class GraphSLAM:
    def __init__(self, problem: GraphProblem, pose_error: str = GRISETTI):
        self.p = p = problem
        self.mode = pose_error
        self.nP, self.nL = p.n_poses, p.n_landmarks
        self.n = 3 * self.nP + 2 * self.nL
        self.prior_info = np.diag(1.0 / np.asarray(p.prior_sigma, dtype=float) ** 2)
        self.odo_info = np.linalg.inv(p.odo_cov) if len(p.odo_cov) else np.zeros((0, 3, 3))
        self.br_info = 1.0 / np.asarray(p.br_sigma, dtype=float) ** 2          # (M, 2) diagonal
        self._pattern()

    # sparsity pattern (constant across iterations)
    def _pattern(self):
        p = self.p
        pi = 3 * p.odo_i[:, None] + np.arange(3)          # (K, 3) variable indices
        pj = 3 * p.odo_j[:, None] + np.arange(3)
        bp = 3 * p.br_pose[:, None] + np.arange(3)        # (M, 3)
        bl = 3 * self.nP + 2 * p.br_lm[:, None] + np.arange(2)   # (M, 2)
        self._idx = (pi, pj, bp, bl)

        def block(r, c):
            return np.repeat(r, c.shape[1], axis=1).ravel(), np.tile(c, (1, r.shape[1])).ravel()

        parts = [block(np.arange(3)[None, :], np.arange(3)[None, :]),
                 block(pi, pi), block(pi, pj), block(pj, pi), block(pj, pj),
                 block(bp, bp), block(bp, bl), block(bl, bp), block(bl, bl)]
        self._rows = np.concatenate([a for a, _ in parts])
        self._cols = np.concatenate([b for _, b in parts])

    # ---------------------------------------------------------------- pieces
    def _residuals(self, X, L):
        p = self.p
        e0, _, J0 = between_residual(p.prior_mean[None], X[:1], np.zeros((1, 3)), self.mode)
        eo, A, B = between_residual(X[p.odo_i], X[p.odo_j], p.odo_z, self.mode)
        eb, Jp, Jl = bearing_range_residual(X[p.br_pose], L[p.br_lm], p.br_z)
        return (e0[0], J0[0]), (eo, A, B), (eb, Jp, Jl)

    def cost(self, X, L) -> float:
        (e0, _), (eo, _, _), (eb, _, _) = self._residuals(X, L)
        return float(e0 @ self.prior_info @ e0 + np.einsum("ni,nij,nj->", eo, self.odo_info, eo)
                     + np.sum(self.br_info * eb * eb))

    def linearise(self, X, L):
        """Return H (CSC), b and the cost at (X, L)."""
        (e0, J0), (eo, A, B), (eb, Jp, Jl) = self._residuals(X, L)
        W0, Wo, Wb = self.prior_info, self.odo_info, self.br_info
        pi, pj, bp, bl = self._idx

        AtW = np.einsum("nji,njk->nik", A, Wo)
        BtW = np.einsum("nji,njk->nik", B, Wo)
        JptW = np.transpose(Jp, (0, 2, 1)) * Wb[:, None, :]
        JltW = np.transpose(Jl, (0, 2, 1)) * Wb[:, None, :]
        vals = np.concatenate([
            (J0.T @ W0 @ J0).ravel(),
            (AtW @ A).ravel(), (AtW @ B).ravel(), (BtW @ A).ravel(), (BtW @ B).ravel(),
            (JptW @ Jp).ravel(), (JptW @ Jl).ravel(), (JltW @ Jp).ravel(), (JltW @ Jl).ravel()])
        H = sparse.coo_matrix((vals, (self._rows, self._cols)), shape=(self.n, self.n)).tocsc()

        b = np.zeros(self.n)
        b[:3] += J0.T @ W0 @ e0
        np.add.at(b, pi, np.einsum("nij,nj->ni", AtW, eo))
        np.add.at(b, pj, np.einsum("nij,nj->ni", BtW, eo))
        np.add.at(b, bp, np.einsum("nij,nj->ni", JptW, eb))
        np.add.at(b, bl, np.einsum("nij,nj->ni", JltW, eb))
        F = float(e0 @ W0 @ e0 + np.einsum("ni,nij,nj->", eo, Wo, eo) + np.sum(Wb * eb * eb))
        return H, b, F

    def _apply(self, X, L, dx):
        Xn = X + dx[:3 * self.nP].reshape(-1, 3)
        Xn[:, 2] = wrap(Xn[:, 2])
        return Xn, L + dx[3 * self.nP:].reshape(-1, 2)

    # --------------------------------------------------------------- optimise
    def optimise(self, X0, L0, method: str = "lm", tol: float = 1e-6, max_iter: int = 100,
                 lam0: float = 1e-4, lam_factor: float = 10.0, verbose: bool = False) -> dict:
        """Gauss-Newton ("gn") or Levenberg-Marquardt ("lm").

        LM: lambda shrinks by lam_factor after a step that lowers the cost and
        grows by it otherwise. Stops when the largest component of an accepted
        step is below tol. `iterations` counts accepted steps.
        """
        t0 = time.perf_counter()
        X, L = np.array(X0, dtype=float), np.array(L0, dtype=float).reshape(-1, 2)
        lam = lam0 if method == "lm" else 0.0
        iters = solves = 0
        converged = False
        history = []
        H, b, F = self.linearise(X, L)
        history.append(F)
        while iters < max_iter:
            if method == "gn":
                dx = spsolve(H, -b)
                solves += 1
                X, L = self._apply(X, L, dx)
            else:
                d = H.diagonal()
                accepted = False
                while lam <= 1e15:
                    dx = spsolve(H + sparse.diags(lam * d, format="csc"), -b)
                    solves += 1
                    Xn, Ln = self._apply(X, L, dx)
                    Fn = self.cost(Xn, Ln)
                    if Fn < F:
                        X, L, accepted = Xn, Ln, True
                        lam = max(lam / lam_factor, 1e-15)
                        break
                    if np.max(np.abs(dx)) < 0.1 * tol:      # no descent left at this damping
                        break
                    lam *= lam_factor
                if not accepted:
                    # No damped step lowers the cost any further: the iterate sits at a
                    # minimum to within rounding. Take the undamped (Gauss-Newton) step
                    # if it does not raise the cost, and call the point converged when
                    # that step is below the standard tolerance.
                    dx = spsolve(H, -b)
                    solves += 1
                    Xn, Ln = self._apply(X, L, dx)
                    if self.cost(Xn, Ln) <= F * (1.0 + 1e-13):
                        X, L = Xn, Ln
                        H, b, F = self.linearise(X, L)
                        history.append(F)
                    converged = bool(np.max(np.abs(dx)) < max(tol, 1e-6))
                    break
            iters += 1
            step = float(np.max(np.abs(dx)))
            H, b, F = self.linearise(X, L)
            history.append(F)
            if verbose:
                print(f"  it {iters:3d}  F = {F:.6f}  step = {step:.2e}  lambda = {lam:.1e}")
            if step < tol:
                converged = True
                break
        return {"X": X, "L": L, "cost": F, "iterations": iters, "solves": solves,
                "converged": converged, "history": history, "grad_inf": float(np.max(np.abs(b))),
                "runtime": time.perf_counter() - t0}

    # --------------------------------------------------------------- marginals
    def marginals_dense(self, X, L):
        """All marginal blocks from a dense inverse of H (small problems, used in tests)."""
        H, _, _ = self.linearise(X, L)
        S = np.linalg.inv(H.toarray())
        Pp = np.array([S[3 * k:3 * k + 3, 3 * k:3 * k + 3] for k in range(self.nP)])
        o = 3 * self.nP
        Pl = np.array([S[o + 2 * j:o + 2 * j + 2, o + 2 * j:o + 2 * j + 2] for j in range(self.nL)])
        return Pp, Pl

    def marginals(self, X, L):
        """Marginal covariance of every pose (nP, 3, 3) and landmark (nL, 2, 2).

        These are the diagonal blocks of H^-1 at the solution, computed exactly
        in O(nP) for an odometry chain: with H = [[A, B], [B^T, C]] (poses,
        landmarks), A is block-tridiagonal and C is small, so
            Sigma_ll = (C - B^T A^-1 B)^-1,
            Sigma_pp = A^-1 + (A^-1 B) Sigma_ll (A^-1 B)^T,
        and the diagonal blocks of A^-1 follow from a two-sweep recursion.
        """
        p = self.p
        if not (np.array_equal(p.odo_i, np.arange(self.nP - 1)) and np.array_equal(p.odo_j, np.arange(1, self.nP))):
            return self._marginals_solve(X, L)
        H, _, _ = self.linearise(X, L)
        nP, o = self.nP, 3 * self.nP
        A = H[:o, :o].tocsc()
        Bm = H[:o, o:].toarray()
        C = H[o:, o:].toarray()

        Ad = A.toarray() if nP <= 400 else None
        D = np.empty((nP, 3, 3))
        U = np.empty((nP - 1, 3, 3))
        if Ad is not None:
            for k in range(nP):
                D[k] = Ad[3 * k:3 * k + 3, 3 * k:3 * k + 3]
            for k in range(nP - 1):
                U[k] = Ad[3 * k:3 * k + 3, 3 * k + 3:3 * k + 6]
        else:
            idx = np.arange(nP)[:, None, None] * 3
            r = (idx + np.arange(3)[None, :, None]).repeat(3, axis=2)
            c = (idx + np.arange(3)[None, None, :]).repeat(3, axis=1)
            D[:] = np.asarray(A[r.ravel(), c.ravel()]).reshape(nP, 3, 3)
            U[:] = np.asarray(A[r[:-1].ravel(), (c[:-1] + 3).ravel()]).reshape(nP - 1, 3, 3)

        fwd = np.empty_like(D)
        bwd = np.empty_like(D)
        fwd[0] = D[0]
        for k in range(1, nP):
            fwd[k] = D[k] - U[k - 1].T @ np.linalg.solve(fwd[k - 1], U[k - 1])
        bwd[-1] = D[-1]
        for k in range(nP - 2, -1, -1):
            bwd[k] = D[k] - U[k] @ np.linalg.solve(bwd[k + 1], U[k].T)
        Ainv_diag = np.linalg.inv(fwd + bwd - D)

        if self.nL == 0:
            return Ainv_diag, np.zeros((0, 2, 2))
        W = splu(A).solve(Bm)                         # A^-1 B, (3 nP, 2 nL)
        Sll = np.linalg.inv(C - Bm.T @ W)
        Sll = 0.5 * (Sll + Sll.T)
        Wk = W.reshape(nP, 3, -1)
        Pp = Ainv_diag + Wk @ Sll @ np.transpose(Wk, (0, 2, 1))
        Pl = np.array([Sll[2 * j:2 * j + 2, 2 * j:2 * j + 2] for j in range(self.nL)])
        return Pp, Pl

    def _marginals_solve(self, X, L):
        """General fallback (e.g. pose graphs with loop closures): sparse solves per block."""
        H, _, _ = self.linearise(X, L)
        lu = splu(H.tocsc())
        Pp = np.empty((self.nP, 3, 3))
        for k in range(self.nP):
            rhs = np.zeros((self.n, 3))
            rhs[3 * k:3 * k + 3] = np.eye(3)
            Pp[k] = lu.solve(rhs)[3 * k:3 * k + 3]
        Pl = np.empty((self.nL, 2, 2))
        o = 3 * self.nP
        for j in range(self.nL):
            rhs = np.zeros((self.n, 2))
            rhs[o + 2 * j:o + 2 * j + 2] = np.eye(2)
            Pl[j] = lu.solve(rhs)[o + 2 * j:o + 2 * j + 2]
        return Pp, Pl

    def chi2_per_dof(self, X, L) -> float:
        return self.cost(X, L) / max(self.p.dof, 1)

    def br_chi2(self, X, L) -> np.ndarray:
        """Whitened squared residual of every bearing-range factor."""
        eb, _, _ = bearing_range_residual(X[self.p.br_pose], L[self.p.br_lm], self.p.br_z)
        return np.sum(self.br_info * eb * eb, axis=1)
