"""ROBOCASA: a RoboCasa kitchen suite on robosuite/MuJoCo.

A mobile Franka (the PandaOmron) through 24 atomic kitchen tasks. The whole body is
driven by a 12-D action (arm EE-delta + gripper + mobile base + control mode),
declared on the PANDA_OMRON_WHOLE_BODY embodiment. The benchmark publishes three cameras — a
left + right workspace pair and a wrist camera, each 256x256 — plus a language
instruction.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.core.sensor import CameraOrientation
from manifold.embodiments.panda_omron_whole_body import PANDA_OMRON_WHOLE_BODY
from manifold.sensors.cameras import agentview_left, agentview_right, wrist

# Every RoboCasa camera publishes MuJoCo's render buffer as robosuite hands it back,
# and that buffer is bottom-up: row 0 is the bottom of the scene. Declared rather than
# left at the `UPRIGHT` default for the reason `benchmarks.libero` declares it -- an
# upside-down frame has the right shape and dtype, so nothing but this field can catch
# it. RoboCasa had no runner correcting it at serve time, so the entry and every
# pairing built on it agreed on a value that no published frame carried.
_ORIENTATION = CameraOrientation.FLIPPED_VERTICAL

ROBOCASA = Benchmark(
    name="robocasa",
    embodiment=PANDA_OMRON_WHOLE_BODY,
    sensors=[
        agentview_left((256, 256, 3), _ORIENTATION),
        agentview_right((256, 256, 3), _ORIENTATION),
        wrist((256, 256, 3), orientation=_ORIENTATION),
    ],
    instruction=True,
)
