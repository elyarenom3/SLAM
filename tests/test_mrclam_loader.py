"""MR.CLAM loader checks. Skipped when the dataset has not been downloaded."""
import math
from pathlib import Path

import numpy as np
import pytest

from slam2d.data import mrclam
from slam2d.odom import build_stream

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(not (ROOT / "ds1" / "Barcodes.dat").exists(), reason="MR.CLAM not downloaded")


@pytest.fixture(scope="module")
def ds1():
    return mrclam.load_dataset(ROOT, 1)


def test_barcodes_are_read_per_dataset(ds1):
    assert len(ds1.barcodes) == 20
    assert [ds1.barcodes[b] for b in (5, 14, 41, 32, 23)] == [1, 2, 3, 4, 5]
    assert ds1.barcodes[72] == 6 and ds1.barcodes[63] == 20
    ds2 = mrclam.load_dataset(ROOT, 2)
    assert ds2.barcodes[18] == 6          # landmark barcodes differ between datasets
    assert ds2.barcodes != ds1.barcodes


def test_landmark_counts_and_dataset_8():
    assert len(mrclam.load_dataset(ROOT, 1).landmarks) == 15
    ds8 = mrclam.load_dataset(ROOT, 8)
    assert len(ds8.landmarks) == 14 and 61 not in ds8.barcodes
    assert all(rb.dropped["meas_unmapped_barcodes"] > 500 for rb in ds8.robots)   # barcode 61 rows


def test_dataset_4_folder_was_flattened():
    assert (ROOT / "ds4" / "Robot1_Odometry.dat").exists()
    assert len(mrclam.load_dataset(ROOT, 4).robots) == 5


def test_measurements_are_landmark_only_and_sorted(ds1):
    for rb in ds1.robots:
        assert set(np.unique(rb.meas[:, 1]).astype(int)) <= set(range(6, 21))
        assert np.all(np.diff(rb.meas[:, 0]) >= 0)
        assert rb.dropped["meas_robot_sightings"] > 0
        assert np.all(rb.meas[:, 2] > 0) and np.all(np.abs(rb.meas[:, 3]) < math.pi / 2)


def test_cleaned_odometry_time_is_strictly_increasing():
    for n in (1, 3, 9):
        for rb in mrclam.load_dataset(ROOT, n).robots:
            assert np.all(np.diff(rb.odom[:, 0]) > 0)
    assert any(rb.dropped["odom_backward_steps"] for rb in mrclam.load_dataset(ROOT, 9).robots)


def test_clean_odometry_keeps_last_of_repeated_stamps():
    raw = np.array([[2.0, 0.1, 0.0], [1.0, 0.2, 0.0], [2.0, 0.3, 0.0], [3.0, 0.4, 0.0]])
    out, info = mrclam.clean_odometry(raw)
    assert out[:, 0].tolist() == [1.0, 2.0, 3.0] and out[1, 1] == 0.3
    assert info == {"odom_backward_steps": 1, "odom_repeated_stamps": 1}


def test_dataset_1_landmark_identity_fix(ds1):
    assert ds1.landmark_fix == {11: 17, 17: 11}
    raw = mrclam.load_dataset(ROOT, 1, fix_landmarks=False)
    assert raw.landmarks[11] == ds1.landmarks[17] and raw.landmarks[17] == ds1.landmarks[11]
    assert mrclam.landmark_identity_check(ds1) == {}          # consistent after the fix
    assert mrclam.load_dataset(ROOT, 2).landmark_fix == {}


def test_ground_truth_interpolation_handles_the_wrap_and_gaps(ds1):
    rb = ds1.robot(1)
    g = rb.gt
    jumps = np.flatnonzero(np.abs(np.diff(g[:, 3])) > 6.0)      # heading crossing +-pi
    assert len(jumps) > 0
    i = jumps[0]
    pose, valid = rb.gt_at(0.5 * (g[i, 0] + g[i + 1, 0]))
    assert valid[0] and abs(abs(pose[0, 2]) - math.pi) < 0.1     # stays near pi, not near 0
    # a sample inside a long Vicon gap is flagged invalid
    rb3 = mrclam.load_dataset(ROOT, 3).robot(3)
    gaps = np.diff(rb3.gt[:, 0])
    k = int(np.argmax(gaps))
    assert gaps[k] > 100
    assert not rb3.gt_at(rb3.gt[k, 0] + 0.5 * gaps[k])[1][0]
    assert not rb3.gt_at(rb3.gt[0, 0] - 1.0)[1][0]


def test_run_window_and_stream(ds1):
    rb = ds1.robot(1)
    t0, t1 = rb.window()
    assert t0 >= rb.odom[0, 0] and t0 >= rb.gt[0, 0] and t1 <= min(rb.odom[-1, 0], rb.gt[-1, 0])
    st = build_stream(rb)
    assert st.gt_valid[0] and np.allclose(st.x0, st.gt[0])
    assert np.all(np.diff(st.kf_t) >= 0.015 - 1e-9)             # merged keyframes
    assert st.seg[0] == 0 and st.seg[-1] == len(st.steps) and np.all(np.diff(st.seg) >= 1)
    assert st.kf_ptr[-1] == len(st.meas_id) and st.kf_ptr[1] == 0
    assert np.all(st.steps[:, 2] > 0) and st.steps[:, 2].max() <= 0.02 + 1e-9
    # the steps of each segment span exactly the time between its keyframes
    span = np.add.reduceat(st.steps[:, 2], st.seg[:-1])
    assert np.allclose(span, np.diff(st.kf_t), atol=1e-9)
