# slam2d: EKF-SLAM, FEJ-EKF and graph-SLAM on UTIAS MR.CLAM

Two planar SLAM estimators built from scratch and run on the same real dataset with ground truth:

- an EKF-SLAM with standard, First-Estimates-Jacobian (FEJ) and ideal Jacobians on one code path;
- a sparse Gauss-Newton / Levenberg-Marquardt graph optimiser, with the same graph solved in GTSAM.

The project tests whether the reported uncertainty matches the actual error (NEES, NIS), reproduces the overconfidence of the standard EKF and the FEJ correction in simulation, and checks the optimiser against GTSAM.

Deliverables: [paper/main.pdf](paper/main.pdf) (6 pages, IEEE two-column), `results/animation.mp4` (60 s), this code and its tests.

## Results in brief

Simulation, 50 runs on the MR.CLAM landmark layout (95% band for the average NEES: 2.36 to 3.72):

| Filter | Average NEES | Time inside band | Position RMSE |
|---|---|---|---|
| Ideal EKF (Jacobians at the truth) | 3.2 | 90% | 0.18 m |
| Standard EKF | 14.9 | 1% | 0.39 m |
| FEJ-EKF | 4.2 | 31% | 0.20 m |

MR.CLAM, 40 held-out runs (datasets 2 to 9), all parameters fitted on dataset 1. Median over runs:

| Method | Position RMSE | After rigid alignment | Heading RMSE | Time-averaged NEES | NEES samples in 95% region |
|---|---|---|---|---|---|
| Standard EKF | 0.37 m | 0.11 m | 6.4 deg | 19.7 | 30% |
| FEJ-EKF | 0.29 m | 0.12 m | 5.7 deg | 7.9 | 77% |
| Graph, own optimiser | 0.19 m | 0.05 m | 3.0 deg | 6.9 | 82% |
| Graph, GTSAM 4.3 | 0.22 m | 0.05 m | 3.2 deg | 7.0 | 80% |

- FEJ has the lower NEES in 35 of 40 runs; the graph has the lowest aligned error in all 40.
- Dataset 9 fails for every method (6.7 to 10.3 m). Its odometry behaves differently (turn-rate gain 0.62 to 0.65 compared with 0.91 to 0.96 fitted on dataset 1). Nothing was retuned; it is reported on its own row in the paper.
- With white measurement noise from the MAD alone the median NEES is 75.8 (standard) and 18.8 (FEJ). Measurement errors are strongly correlated in time, and the frozen noise model accounts for that.
- Own optimiser compared with GTSAM 4.3 (log-map pose error): relative cost difference 1e-11 median, poses within 1e-7 m median, on 43 of 45 runs. On the other two runs the solvers stop in different local minima of the same cost.
- Own optimiser compared with GTSAM 4.2 (Grisetti pose error): identical cost function to 1e-15, and GTSAM stops 2e-5 higher in relative cost because that version omits a chart Jacobian.

## Reproduce

```
/opt/anaconda3/bin/python3 -m venv .venv            # any arm64 Python 3.11 to 3.14
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                                  # 245 tests
.venv/bin/python scripts/run_sim_mc.py --trials 50   # headline simulation -> results/sim_mc.npz
.venv/bin/python scripts/run_sim_mc.py --trials 50 --gate --tag gated
.venv/bin/python scripts/run_sim_grid.py             # the 12-point sweep behind the chosen configuration
.venv/bin/python scripts/download_mrclam.py          # anonymous FTP, about 75 MB
.venv/bin/python -I scripts/fit_noise.py             # calibration on dataset 1 (refuses to overwrite a frozen file)
.venv/bin/python -I scripts/run_mrclam.py --datasets 1-9
.venv/bin/python -I scripts/compare_small_graphs.py
.venv/bin/python -I scripts/make_figures.py
.venv/bin/python -I scripts/make_animation.py
cd paper && latexmk -pdf main.tex
```

Scripts add the repository root to `sys.path` themselves, so they also run with `python -I`. `pip install -e .` is optional.

GTSAM 4.2 cross-check (second environment, Python 3.10 with `numpy<2`):

```
python3.10 -m venv .venv42 && .venv42/bin/pip install "gtsam==4.2" "numpy<2" scipy pytest
.venv42/bin/python -m pytest -q tests/test_graph_vs_gtsam.py
.venv42/bin/python scripts/compare_small_graphs.py
```

Optional extras:

```
.venv/bin/python -I scripts/run_isam2.py             # iSAM2 on the same factors
.venv/bin/python -I scripts/run_g2o.py               # INTEL, MITb, M3500 pose graphs
```

## Layout

```
slam2d/   geometry.py   angles, SE(2), SE(2) log and its derivative
          models.py     motion and measurement models with Jacobians
          ekf.py        EKF-SLAM: standard, FEJ, ideal
          sim.py        simulator and Monte Carlo driver, small synthetic graphs
          metrics.py    NEES, NIS, chi-squared bounds
          odom.py       merged odometry/measurement stream, pre-integration
          graph.py      sparse Gauss-Newton / LM optimiser, marginals
          gtsam_backend.py   the same graph in GTSAM
          calib.py      sensor and noise calibration against ground truth
          pipeline.py   run and score the estimators on one robot log
          data/mrclam.py     dataset loader
tests/    Jacobians, geometry, EKF in simulation, odometry, graph and GTSAM, loader
scripts/  download, calibration, runs, figures, animation, extras
config/   sim.json, frozen_mrclam.json (fitted on dataset 1, then frozen)
paper/    main.tex, refs.bib, generated figures and tables
```

## Decisions that go beyond the brief

- **FEJ with many predictions per update.** The filter keeps the predicted pose as the robot linearisation point; updates never overwrite it. Only the first prediction after an update differs from the standard filter, and all landmarks of one timestamp share the linearisation point. `tests/test_ekf_sim.py` checks the three unobservable directions directly: `H N = 0` to 1e-14 for FEJ, 0.26 in the rotation column for the standard filter.
- **Process noise under irregular steps.** `Q = dt * dt_nom * Vhat M Vhat^T`, equal to `V M V^T` at the nominal 15 ms and additive when a step is split at a measurement time.
- **Odometry covariance frame.** Pre-integration gives the covariance in the frame of the first pose; the Grisetti residual lives in the end frame. The factor uses `T Sigma T^T` with `T = blkdiag(R(theta_z)^T, 1)`. A Monte Carlo test confirms it.
- **Singular odometry covariance.** `V M V^T` has rank 2. A small noise rate, (1 mm)^2 and (0.1 deg)^2 per second, is added in the filter and in the pre-integration. Results on dataset 1 change little between 0.1x and 10x that value (`results/floor_sensitivity_ds1.json`).
- **Odometry model on real data.** The logged odometry is the commanded velocity. Fitted on dataset 1: a delay of 0.14 to 0.24 s, `v' = s_v v - c_v |w|` (forward speed drops while turning), a turn-rate gain, and two noise levels (straight, and within 0.5 s of a commanded turn). A filter with a MAD-based control sigma diverged on dataset 1.
- **Measurement model on real data.** MAD-based sigmas per robot, range sigma linear in range, a 40 ms camera latency and a small lever arm. Residuals are correlated in time, so sigmas are inflated by a factor computed from 10 s block means of the residuals (2.3 to 4.2 for range, 1.9 to 2.9 for bearing), and sightings taken while the robot is commanded to stand still are dropped.
- **Marginals in O(K).** Schur complement on the landmarks plus a two-sweep block-tridiagonal recursion give every pose covariance exactly; tested against a dense inverse.
- **Two pose-error forms.** `pose_error="grisetti"` (the brief, equal to GTSAM 4.2) and `"se2log"` (GTSAM 4.3). The log-map Jacobians are the Grisetti `A`, `B` premultiplied by `dLog`.
- **LM termination.** When no damped step lowers the cost, the undamped step decides convergence, so a large lambda cannot end the iteration early.

## Dataset notes

- Download is FTP only; dataset 4 unpacks to `MRSLAM_Dataset4/`.
- `Barcodes.dat` differs between datasets and is read per dataset. Dataset 8 maps 14 landmarks; its barcode 61 is unmapped and dropped.
- In dataset 1 the ground-truth positions of subjects 11 and 17 are exchanged. The loader detects and corrects this (scoring and calibration only).
- Odometry stamps repeat and, in dataset 9, step backwards once per file. Ground truth has gaps of up to 662 s, which are masked in scoring.

## Holdout protocol

Everything the estimators need on real data is fitted on dataset 1 by `scripts/fit_noise.py` and stored in `config/frozen_mrclam.json`. The file was marked frozen and committed (commit `5271612`) before any estimator ran on datasets 2 to 9, `run_mrclam.py` refuses those datasets until the mark is set, and the file hash is written to every result row. Datasets 2 to 9 were processed once. The dataset 9 diagnosis in the paper was made afterwards and changed nothing.

## Optional extras

- **iSAM2** on seven held-out runs: the causal estimate has a median position RMSE of 0.11 m, compared with 0.20 m for the FEJ filter and 0.09 m for the batch solution; the final incremental estimate is within 4 mm of the batch solution; 2.1 ms per update.
- **Pose graphs**: M3500 (3500 poses) converges in 9 Gauss-Newton iterations and 0.27 s to the same cost as GTSAM (relative difference 1e-12); MITb in 44 LM iterations (5e-14). On INTEL own Gauss-Newton reaches the known optimum (cost 215.8) in 12 to 25 iterations; GTSAM 4.3.0 LM returned the initial values for that file.

## Limits

- No method is consistent on the real data in the strict sense (NEES of 3). Correlated sensor error is the main reason.
- The five robots are the same in every dataset, so the holdout is over sessions.
- Filter covariances are filtered and graph covariances are smoothed; the final pose gives the like-for-like comparison.
- In simulation with accurate bearings the standard EKF is close to consistent, and under strong nonlinearity (2 m sensing range with high turn-rate noise) FEJ diverged at two of twelve swept configurations.
