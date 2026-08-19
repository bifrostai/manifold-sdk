"""IsaacLab Arena pick-and-place tasks using the DROID assembly."""

from __future__ import annotations

from manifold.core.benchmark import Benchmark
from manifold.embodiments.droid_joint_absolute import DROID_JOINT_ABSOLUTE
from manifold.sensors.cameras import agentview, wrist

ISAACLAB_ARENA_DROID = Benchmark(
    name="isaaclab-arena-droid",
    embodiment=DROID_JOINT_ABSOLUTE,
    sensors=[agentview((360, 640, 3)), wrist((360, 640, 3))],
    instruction=True,
)

__all__ = ["ISAACLAB_ARENA_DROID"]
