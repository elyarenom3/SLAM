"""The 15 landmark positions of UTIAS MR.CLAM dataset 1 (Landmark_Groundtruth.dat).

Used by the simulator so the simulated room has the real landmark layout.
Rows are subjects 6..20, columns x [m], y [m].
"""
import numpy as np

LANDMARKS = np.array([
    [5.70928255, 4.96404466],
    [5.25292609, 5.53656921],
    [3.69987701, 4.46642332],
    [2.31446307, 3.37498700],
    [0.48179893, 4.39597180],
    [3.15071999, 2.38294871],
    [4.06328771, 0.94429372],
    [2.68034388, 0.26835185],
    [0.94828519, 0.75601306],
    [3.76290153, -2.03091517],
    [1.69836770, -2.08966931],
    [0.03596156, -2.84396626],
    [0.94291375, -5.43445742],
    [2.59017016, -5.52265035],
    [2.09939861, -3.91527208],
])
