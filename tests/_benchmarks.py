"""A benchmark for tests: a Franka in end-effector deltas on a robosuite tabletop.

It publishes a scene and a wrist camera at 256x256 with the metric depth of each,
all bottom-up as robosuite renders them and each calibrated, plus an instruction, so
a test can exercise every camera axis the check compares.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.core.conventions import CameraAxes, Frame
from manifold.core.sensor import CameraCalibration, CameraIntrinsics, CameraOrientation
from manifold.embodiments.franka_ee_delta import FRANKA_EE_DELTA
from manifold.sensors.cameras import agentview, agentview_depth, wrist, wrist_depth

# The square resolution every camera renders at, which the intrinsics are derived for.
CAMERA_RES = 256

# The scene and wrist cameras have different fields of view, so each has its own
# intrinsics.
_SCENE_CALIBRATION = CameraCalibration(
    intrinsics=CameraIntrinsics.from_fov(fovy_degrees=45.0, height=CAMERA_RES, width=CAMERA_RES),
    axes=CameraAxes.OPENCV,
    frame=Frame.WORLD,
)
_WRIST_CALIBRATION = CameraCalibration(
    intrinsics=CameraIntrinsics.from_fov(fovy_degrees=75.0, height=CAMERA_RES, width=CAMERA_RES),
    axes=CameraAxes.OPENCV,
    frame=Frame.WORLD,
)

# robosuite hands back MuJoCo's render buffer bottom-up.
_ORIENTATION = CameraOrientation.FLIPPED_VERTICAL

TABLETOP = Benchmark(
    name="tabletop",
    embodiment=FRANKA_EE_DELTA,
    sensors=[
        agentview((CAMERA_RES, CAMERA_RES, 3), _SCENE_CALIBRATION, _ORIENTATION),
        wrist((CAMERA_RES, CAMERA_RES, 3), _WRIST_CALIBRATION, _ORIENTATION),
        agentview_depth((CAMERA_RES, CAMERA_RES, 1), _SCENE_CALIBRATION, _ORIENTATION),
        wrist_depth((CAMERA_RES, CAMERA_RES, 1), _WRIST_CALIBRATION, _ORIENTATION),
    ],
    instruction=True,
)
