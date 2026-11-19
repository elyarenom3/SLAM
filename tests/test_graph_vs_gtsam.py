"""The own optimiser against GTSAM, and its internal consistency checks."""
import numpy as np
import pytest

from slam2d import graph as gr
from slam2d.sim import small_graph

gtsam = pytest.importorskip("gtsam")
from slam2d import gtsam_backend as gb  # noqa: E402

CONVENTION = gb.pose_convention()          # "se2log" for GTSAM 4.3, "grisetti" for 4.2
OTHER = gr.GRISETTI if CONVENTION == gr.SE2LOG else gr.SE2LOG


@pytest.fixture(scope="module")
def case():
    return small_graph(seed=3, n_poses=30, n_landmarks=6)


def test_installed_gtsam_convention_is_known():
    assert CONVENTION in (gr.GRISETTI, gr.SE2LOG)
    from importlib.metadata import version
    major_minor = tuple(int(v) for v in version("gtsam").split(".")[:2])
    assert CONVENTION == (gr.SE2LOG if major_minor >= (4, 3) else gr.GRISETTI)


def test_factor_residuals_match_gtsam_at_random_values(case):
    problem, X, L, _, _ = case
    rng = np.random.default_rng(0)
    Xr = X + rng.normal(0, 0.15, X.shape)          # far from any optimum
    Lr = L + rng.normal(0, 0.3, L.shape)
    solver = gr.GraphSLAM(problem, CONVENTION)
    (e0, _), (eo, _, _), (eb, _, _) = solver._residuals(Xr, Lr)
    g0, go, gbr = gb.factor_errors(problem, Xr, Lr)
    assert np.allclose(eo, go, atol=1e-9)
    assert np.allclose(eb[:, ::-1], gbr, atol=1e-9)             # GTSAM orders bearing, range
    # the prior is isotropic in xy, so only the norm of its translation part is convention-free
    assert np.hypot(*e0[:2]) == pytest.approx(np.hypot(*g0[:2]), abs=1e-9)
    assert e0[2] == pytest.approx(g0[2], abs=1e-9)
    graph, values, _, _ = gb.build(problem, Xr, Lr)
    assert solver.cost(Xr, Lr) == pytest.approx(2 * graph.error(values), rel=1e-10)


EXACT_JACOBIANS = CONVENTION == gr.SE2LOG   # GTSAM 4.3 linearises its pose factors exactly, 4.2 does not


@pytest.mark.parametrize("seed", (3, 7, 11))
def test_optimum_cost_estimates_and_covariances_match_gtsam(seed):
    problem, _, _, X0, L0 = small_graph(seed=seed)
    solver = gr.GraphSLAM(problem, CONVENTION)
    own = solver.optimise(X0, L0, method="lm", tol=1e-11, max_iter=200)
    ref = gb.solve(problem, X0, L0, method="lm+gn", tol=1e-14, max_iter=200)
    assert own["converged"]
    Pp, Pl = solver.marginals(own["X"], own["L"])
    # the two cost functions are the same function: evaluate GTSAM's at the own optimum
    graph, values, _, _ = gb.build(problem, own["X"], own["L"])
    assert own["cost"] == pytest.approx(2 * graph.error(values), rel=1e-11)
    if EXACT_JACOBIANS:
        assert own["cost"] == pytest.approx(ref["cost"], rel=1e-8)
        assert np.abs(own["X"][:, :2] - ref["X"][:, :2]).max() < 1e-7
        assert np.abs(gr.wrap(own["X"][:, 2] - ref["X"][:, 2])).max() < 1e-7
        assert np.abs(own["L"] - ref["L"]).max() < 1e-7
        assert np.allclose(Pp, ref["P_pose"], rtol=1e-5, atol=1e-12)   # GTSAM's rotated into the world frame
        assert np.allclose(Pl, ref["P_lm"], rtol=1e-5, atol=1e-12)
    else:
        # GTSAM 4.2 drops the chart Jacobian of its between and prior factors, so its
        # iteration stops at a point of slightly higher cost; the gap is second order
        # in the residual
        assert own["cost"] <= ref["cost"] * (1 + 1e-12)
        assert (ref["cost"] - own["cost"]) / own["cost"] < 1e-3
        assert np.abs(own["X"][:, :2] - ref["X"][:, :2]).max() < 1e-2
        assert np.abs(Pp - ref["P_pose"]).max() / np.abs(Pp).max() < 5e-2


def test_agreement_tightens_as_residuals_shrink():
    """With ten times less noise the remaining gap to GTSAM shrinks (to zero for 4.3)."""
    gaps = []
    for noise in (1.0, 0.1):
        problem, _, _, X0, L0 = small_graph(seed=7, noise=noise)
        own = gr.GraphSLAM(problem, CONVENTION).optimise(X0, L0, tol=1e-11)
        ref = gb.solve(problem, X0, L0, method="lm+gn", tol=1e-14, with_marginals=False)
        gaps.append(np.abs(own["X"][:, :2] - ref["X"][:, :2]).max())
    assert gaps[1] < 1e-7 if EXACT_JACOBIANS else gaps[1] < 0.05 * gaps[0]


def test_gtsam_levenberg_marquardt_reaches_the_same_optimum(case):
    problem, _, _, X0, L0 = case
    solver = gr.GraphSLAM(problem, CONVENTION)
    own = solver.optimise(X0, L0, tol=1e-11)
    ref = gb.solve(problem, X0, L0, method="lm", tol=1e-14, with_marginals=False)
    assert own["cost"] == pytest.approx(ref["cost"], rel=1e-6 if EXACT_JACOBIANS else 1e-3)
    assert np.abs(own["X"][:, :2] - ref["X"][:, :2]).max() < (1e-4 if EXACT_JACOBIANS else 1e-2)


def test_other_convention_differs_only_at_second_order(case):
    """Grisetti and log-map residuals give nearby but distinct optima."""
    problem, _, _, X0, L0 = case
    a = gr.GraphSLAM(problem, CONVENTION).optimise(X0, L0, tol=1e-11)
    b = gr.GraphSLAM(problem, OTHER).optimise(X0, L0, tol=1e-11)
    diff = np.abs(a["X"][:, :2] - b["X"][:, :2]).max()
    assert 1e-8 < diff < 5e-3
    assert abs(a["cost"] - b["cost"]) / a["cost"] < 5e-2


@pytest.mark.parametrize("mode", (gr.GRISETTI, gr.SE2LOG))
def test_gauss_newton_and_lm_agree(case, mode):
    problem, _, _, X0, L0 = case
    solver = gr.GraphSLAM(problem, mode)
    gn = solver.optimise(X0, L0, method="gn", tol=1e-11)
    lm = solver.optimise(X0, L0, method="lm", tol=1e-11)
    assert gn["converged"] and lm["converged"]
    assert np.abs(gn["X"] - lm["X"]).max() < 1e-8
    assert gn["cost"] == pytest.approx(lm["cost"], rel=1e-10)
    assert gn["grad_inf"] < 1e-5 and lm["grad_inf"] < 1e-5


@pytest.mark.parametrize("mode", (gr.GRISETTI, gr.SE2LOG))
def test_zero_noise_recovers_truth(mode):
    problem, X, L, _, _ = small_graph(seed=5, noise=0.0)
    rng = np.random.default_rng(1)
    solver = gr.GraphSLAM(problem, mode)
    out = solver.optimise(X + rng.normal(0, 0.05, X.shape), L + rng.normal(0, 0.1, L.shape), tol=1e-12)
    assert out["cost"] < 1e-15
    assert np.abs(out["X"] - X).max() < 1e-8 and np.abs(out["L"] - L).max() < 1e-8


@pytest.mark.parametrize("mode", (gr.GRISETTI, gr.SE2LOG))
def test_chain_marginals_equal_dense_inverse(case, mode):
    problem, _, _, X0, L0 = case
    solver = gr.GraphSLAM(problem, mode)
    out = solver.optimise(X0, L0, tol=1e-10)
    Pp, Pl = solver.marginals(out["X"], out["L"])
    Dp, Dl = solver.marginals_dense(out["X"], out["L"])
    assert np.allclose(Pp, Dp, rtol=1e-7, atol=1e-14)
    assert np.allclose(Pl, Dl, rtol=1e-7, atol=1e-14)
    Sp, Sl = solver._marginals_solve(out["X"], out["L"])
    assert np.allclose(Sp, Dp, rtol=1e-7, atol=1e-14) and np.allclose(Sl, Dl, rtol=1e-7, atol=1e-14)


def test_chi2_per_dof_is_one_on_average():
    vals = []
    for seed in range(40):
        problem, _, _, X0, L0 = small_graph(seed=100 + seed, n_poses=25)
        solver = gr.GraphSLAM(problem, gr.GRISETTI)
        out = solver.optimise(X0, L0)
        assert out["converged"]
        assert problem.dof == 2 * len(problem.br_z) - 2 * problem.n_landmarks
        vals.append(out["cost"] / problem.dof)
    assert abs(np.mean(vals) - 1.0) < 0.06


def test_graph_nees_is_consistent_in_simulation():
    """Smoothed pose marginals match the actual errors (NEES near 3)."""
    nees = []
    for seed in range(30):
        problem, X, L, X0, L0 = small_graph(seed=200 + seed, n_poses=25)
        solver = gr.GraphSLAM(problem, gr.GRISETTI)
        out = solver.optimise(X0, L0)
        Pp, _ = solver.marginals(out["X"], out["L"])
        e = X - out["X"]
        e[:, 2] = gr.wrap(e[:, 2])
        nees.append(np.einsum("ki,ki->k", e[1:], np.linalg.solve(Pp[1:], e[1:, :, None])[..., 0]).mean())
    assert 2.6 < np.mean(nees) < 3.4
