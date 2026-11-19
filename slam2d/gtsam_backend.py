"""The same factor graph solved with GTSAM, for cross-checking the own optimiser.

Conventions that matter when comparing:
- BearingRangeFactor2D takes the bearing before the range, in the measurement
  and in the noise sigmas. The own code stores (range, bearing).
- NonlinearFactorGraph.error is 0.5 * sum of squared whitened errors, so the
  cost comparable with F = sum e^T Omega e is 2 * error.
- Marginals returns a pose covariance in the pose's local (body) frame. It is
  rotated here with blkdiag(R(theta), 1) into the world frame, where the
  blocks of H^-1 of the own optimiser live. Landmark covariances are world-frame.
- BetweenFactorPose2 / PriorFactorPose2 use the plain components of
  z^-1 (x_i^-1 x_j) in GTSAM 4.2 and the SE(2) logarithm in GTSAM 4.3
  (see `pose_convention`).
"""
from __future__ import annotations

import time

import numpy as np

from .graph import GraphProblem


def _gtsam():
    import gtsam
    return gtsam


def pose_convention() -> str:
    """Which pose error the installed GTSAM uses: "grisetti" (plain) or "se2log"."""
    gtsam = _gtsam()
    from .geometry import se2_between, se2_log
    a, b, z = np.array([0.3, -0.2, 0.4]), np.array([1.1, 0.5, 1.3]), np.array([0.5, 0.3, 0.2])
    f = gtsam.BetweenFactorPose2(0, 1, gtsam.Pose2(*z), gtsam.noiseModel.Unit.Create(3))
    v = gtsam.Values()
    v.insert(0, gtsam.Pose2(*a))
    v.insert(1, gtsam.Pose2(*b))
    err = np.asarray(f.unwhitenedError(v))
    plain = se2_between(z, se2_between(a, b))
    if np.allclose(err, plain, atol=1e-9):
        return "grisetti"
    if np.allclose(err, se2_log(plain), atol=1e-9):
        return "se2log"
    raise RuntimeError("unrecognised GTSAM Pose2 error convention")


def build(problem: GraphProblem, X0, L0):
    """Return (graph, values, pose_keys, landmark_keys)."""
    gtsam = _gtsam()
    from gtsam.symbol_shorthand import L, X
    p = problem
    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.PriorFactorPose2(X(0), gtsam.Pose2(*p.prior_mean),
                                     gtsam.noiseModel.Diagonal.Sigmas(np.asarray(p.prior_sigma, dtype=float))))
    for k in range(len(p.odo_z)):
        graph.add(gtsam.BetweenFactorPose2(X(int(p.odo_i[k])), X(int(p.odo_j[k])), gtsam.Pose2(*p.odo_z[k]),
                                           gtsam.noiseModel.Gaussian.Covariance(p.odo_cov[k])))
    for m in range(len(p.br_z)):
        r, phi = p.br_z[m]
        sr, sp = p.br_sigma[m]
        graph.add(gtsam.BearingRangeFactor2D(X(int(p.br_pose[m])), L(int(p.br_lm[m])), gtsam.Rot2(phi), r,
                                             gtsam.noiseModel.Diagonal.Sigmas(np.array([sp, sr]))))
    values = gtsam.Values()
    for k, x in enumerate(X0):
        values.insert(X(k), gtsam.Pose2(*x))
    for j, l in enumerate(np.asarray(L0).reshape(-1, 2)):
        values.insert(L(j), gtsam.Point2(*l))
    return graph, values, [X(k) for k in range(len(X0))], [L(j) for j in range(len(np.asarray(L0).reshape(-1, 2)))]


def solve(problem: GraphProblem, X0, L0, method: str = "lm", tol: float = 1e-12, max_iter: int = 200,
          lam0: float = 1e-4, marginal_poses=None, with_marginals: bool = True) -> dict:
    """Optimise with GTSAM and return estimates, cost, iterations, runtime and covariances.

    method: "lm", "gn", or "lm+gn" (LM followed by a Gauss-Newton polish from
    the LM result, which removes the last digits LM's stopping rule leaves).
    marginal_poses: pose indices whose covariance to compute (default all).
    """
    gtsam = _gtsam()
    graph, values, xk, lk = build(problem, X0, L0)
    t0 = time.perf_counter()
    if method == "gn":
        params = gtsam.GaussNewtonParams()
        params.setRelativeErrorTol(tol)
        params.setAbsoluteErrorTol(tol)
        params.setMaxIterations(max_iter)
        opt = gtsam.GaussNewtonOptimizer(graph, values, params)
    else:
        params = gtsam.LevenbergMarquardtParams()
        params.setRelativeErrorTol(tol)
        params.setAbsoluteErrorTol(tol)
        params.setMaxIterations(max_iter)
        params.setlambdaInitial(lam0)
        params.setlambdaFactor(10.0)
        params.setDiagonalDamping(True)
        opt = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = opt.optimize()
    runtime = time.perf_counter() - t0
    iterations = int(opt.iterations())
    polish_iterations = 0
    if method == "lm+gn":
        gn_params = gtsam.GaussNewtonParams()
        gn_params.setRelativeErrorTol(0.0)
        gn_params.setAbsoluteErrorTol(0.0)
        gn_params.setMaxIterations(5)
        polish = gtsam.GaussNewtonOptimizer(graph, result, gn_params)
        polished = polish.optimize()
        if graph.error(polished) <= graph.error(result):
            result, polish_iterations = polished, int(polish.iterations())

    X = np.array([[result.atPose2(k).x(), result.atPose2(k).y(), result.atPose2(k).theta()] for k in xk])
    L = np.array([result.atPoint2(k) for k in lk]).reshape(-1, 2)
    out = {"X": X, "L": L, "cost": 2.0 * graph.error(result), "iterations": iterations,
           "polish_iterations": polish_iterations, "runtime": runtime,
           "initial_cost": 2.0 * graph.error(values)}
    if with_marginals:
        t1 = time.perf_counter()
        marg = gtsam.Marginals(graph, result)
        sel = range(len(xk)) if marginal_poses is None else marginal_poses
        Pp = np.full((len(xk), 3, 3), np.nan)
        for k in sel:
            c, s = np.cos(X[k, 2]), np.sin(X[k, 2])
            A = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            Pp[k] = A @ marg.marginalCovariance(xk[k]) @ A.T     # body frame -> world frame
        out["P_pose"] = Pp
        out["P_lm"] = np.array([marg.marginalCovariance(k) for k in lk]).reshape(-1, 2, 2)
        out["marginals_runtime"] = time.perf_counter() - t1
    return out


def factor_errors(problem: GraphProblem, X, L):
    """Unwhitened error of every factor at (X, L), in GTSAM's own conventions.

    Returns (prior (3,), odometry (K, 3), bearing_range (M, 2) as [bearing, range]).
    """
    graph, values, _, _ = build(problem, X, L)
    K, M = len(problem.odo_z), len(problem.br_z)
    errs = [np.asarray(graph.at(i).unwhitenedError(values)) for i in range(graph.size())]
    return errs[0], np.array(errs[1:1 + K]).reshape(K, 3), np.array(errs[1 + K:1 + K + M]).reshape(M, 2)
