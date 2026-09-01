"""LIBERO: a Franka manipulation suite on robosuite.

One benchmark stands for the LIBERO family (spatial, object, goal, 10, 90): they
share this contract and differ only in tasks. The arm is driven in end-effector
deltas, with a third-person and a wrist camera at 256x256, the metric depth of
each, and a language instruction.

Depth is published alongside the colour views rather than under a second
benchmark identity (ADR 0007): matching reads the cameras a *policy* consumes, so
a colour-only policy pairs with this contract untouched, and one identity keeps
the family's scores comparable.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.core.conventions import CameraAxes, Frame
from manifold.core.sensor import CameraCalibration, CameraIntrinsics
from manifold.embodiments.franka_ee_delta import FRANKA_EE_DELTA
from manifold.sensors.cameras import agentview, agentview_depth, wrist, wrist_depth

# The square resolution every LIBERO camera renders at, and what the intrinsics below
# are derived for: a focal length in pixels is only meaningful against a pixel count.
CAMERA_RES = 256

# The vertical field of view robosuite gives each LIBERO camera, in degrees, read off
# `sim.model.cam_fovy` and identical across every task of every suite. They differ from
# each other: the wrist camera is much wider than the scene camera, so a single set of
# intrinsics for the pair would be wrong for one of them.
_AGENTVIEW_FOVY = 45.0
_WRIST_FOVY = 75.0

# The pose robosuite publishes is camera-to-world in OpenCV axes: its
# `get_camera_extrinsic_matrix` corrects MuJoCo's own camera body axes into that
# convention before returning. Declared rather than assumed, because the two
# conventions differ by a half turn and mistaking one for the other points a camera
# backwards with nothing to raise (ADR 0008).
_AGENTVIEW_CALIBRATION = CameraCalibration(
    intrinsics=CameraIntrinsics.from_fov(
        fovy_degrees=_AGENTVIEW_FOVY, height=CAMERA_RES, width=CAMERA_RES
    ),
    axes=CameraAxes.OPENCV,
    frame=Frame.WORLD,
)
_WRIST_CALIBRATION = CameraCalibration(
    intrinsics=CameraIntrinsics.from_fov(
        fovy_degrees=_WRIST_FOVY, height=CAMERA_RES, width=CAMERA_RES
    ),
    axes=CameraAxes.OPENCV,
    frame=Frame.WORLD,
)

LIBERO = Benchmark(
    name="libero",
    embodiment=FRANKA_EE_DELTA,
    sensors=[
        agentview((CAMERA_RES, CAMERA_RES, 3), _AGENTVIEW_CALIBRATION),
        wrist((CAMERA_RES, CAMERA_RES, 3), _WRIST_CALIBRATION),
        agentview_depth((CAMERA_RES, CAMERA_RES, 1), _AGENTVIEW_CALIBRATION),
        wrist_depth((CAMERA_RES, CAMERA_RES, 1), _WRIST_CALIBRATION),
    ],
    instruction=True,
)
