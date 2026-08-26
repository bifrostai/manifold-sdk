"""ROBOCASA: a RoboCasa kitchen suite on robosuite/MuJoCo.

A mobile Franka (the PandaOmron) through 24 atomic kitchen tasks. The whole body is
driven by a 12-D action (arm EE-delta + gripper + mobile base + control mode),
declared on the PANDA_OMRON_WHOLE_BODY embodiment. The benchmark publishes three cameras — a
left + right workspace pair and a wrist camera, each 256x256 — plus a language
instruction.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.panda_omron_whole_body import PANDA_OMRON_WHOLE_BODY
from manifold.sensors.cameras import agentview_left, agentview_right, wrist

ROBOCASA = Benchmark(
    name="robocasa",
    embodiment=PANDA_OMRON_WHOLE_BODY,
    sensors=[
        agentview_left((256, 256, 3)),
        agentview_right((256, 256, 3)),
        wrist((256, 256, 3)),
    ],
    instruction=True,
)
