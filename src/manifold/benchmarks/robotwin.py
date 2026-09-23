"""ROBOTWIN: the 50-task dual-arm SAPIEN manipulation suite (RoboTwin 2.0).

The tasks in `envs/` are bimanual tabletop problems, such as pick-place, handover,
stacking, and opening articulated objects. Each task is a Python scene program that
spawns the objects, scripts an expert, and defines the success condition.

The suite runs the same tasks under two configs, `demo_clean` and
`demo_randomized`. The second randomises the scene, but under both configs
RoboTwin publishes the same sensors and accepts the same actions.

The cameras follow `_camera_config.yml`:

    head_camera       head
    observer_camera   agentview, rendered only when the config sets `third_view: true`
    front_camera      front
    left_camera       wrist_left
    right_camera      wrist_right

RoboTwin can also publish depth, but both configs leave it off.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.aloha_agilex_joint_absolute import ALOHA_AGILEX_JOINT_ABSOLUTE
from manifold.sensors.cameras import agentview, front, head, wrist_left, wrist_right

_CAMERA_SHAPE = (240, 320, 3)

ROBOTWIN = Benchmark(
    name="robotwin",
    embodiment=ALOHA_AGILEX_JOINT_ABSOLUTE,
    sensors=[
        head(_CAMERA_SHAPE),
        agentview(_CAMERA_SHAPE),
        front(_CAMERA_SHAPE),
        wrist_left(_CAMERA_SHAPE),
        wrist_right(_CAMERA_SHAPE),
    ],
    instruction=True,
)
