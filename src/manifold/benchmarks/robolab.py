"""ROBOLAB: NVIDIA's 120-task Isaac Lab manipulation suite (RoboLab-120).

The 120 tasks in `robolab/tasks/benchmark/` are pick-place and tidying problems
tagged by the competency they probe (semantics, spatial, color, counting, sorting,
stacking) and graded simple / moderate / complex. The tasks themselves are
robot-agnostic — an embodiment and an action space are bound at
environment-registration time — so this constant takes the pairing the standard
joint-position registration binds (`registrations/droid/`, jointpos): the DROID
arm, which `robolab/robots/README.md` tags the default benchmark embodiment, in
absolute joint targets.

Sensors are every camera RoboLab's registration presets can render: the
over-the-shoulder left and right scene views, the head view facing the arm, and the
gripper-mounted wrist view, the latter the 720p camera the robots README notes is
calibrated to match pi05 / DreamZero training data. A policy names the ones it
reads. All four render natively at 720x1280, and a benchmark driver restates each
shape as the size it publishes. The mirrored egocentric camera the
registration also attaches is viewport-only — it feeds recorded video, not the
policy — so it is not published here. Every task carries a language instruction.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.droid_joint_absolute import DROID_JOINT_ABSOLUTE
from manifold.sensors.cameras import head, over_shoulder_left, over_shoulder_right, wrist

ROBOLAB = Benchmark(
    name="robolab",
    embodiment=DROID_JOINT_ABSOLUTE,
    sensors=[
        over_shoulder_left((360, 640, 3)),
        over_shoulder_right((360, 640, 3)),
        head((360, 640, 3)),
        wrist((360, 640, 3)),
    ],
    instruction=True,
)
