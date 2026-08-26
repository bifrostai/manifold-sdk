"""SIMPLER: a WidowX / Bridge manipulation suite (SimplerEnv).

The arm is driven in end-effector deltas with euler-XYZ rotation. A single
third-person camera at the env's native 480x640 render (no wrist) and a language
instruction; the policy resizes the frame to 256x256 on its own side.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.widowx_ee_delta import WIDOWX_EE_DELTA
from manifold.sensors.cameras import agentview

SIMPLER = Benchmark(
    name="simpler",
    embodiment=WIDOWX_EE_DELTA,
    sensors=[agentview((480, 640, 3))],
    instruction=True,
)
