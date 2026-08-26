"""LIBERO: a Franka manipulation suite on robosuite.

One benchmark stands for the LIBERO family (spatial, object, goal, 10, 90): they
share this contract and differ only in tasks. The arm is driven in end-effector
deltas, with a third-person and a wrist camera at 256x256, and a language
instruction.
"""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.franka_ee_delta import FRANKA_EE_DELTA
from manifold.sensors.cameras import agentview, wrist

LIBERO = Benchmark(
    name="libero",
    embodiment=FRANKA_EE_DELTA,
    sensors=[agentview((256, 256, 3)), wrist((256, 256, 3))],
    instruction=True,
)
