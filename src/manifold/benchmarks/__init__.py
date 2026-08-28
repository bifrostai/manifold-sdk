"""Catalog of benchmarks, one task suite per file.

Each benchmark imports its embodiment and declares the sensors and task signals
it publishes. `ALL` lists every shipped benchmark, for a consumer enumerating
them.
"""

from __future__ import annotations

from manifold.benchmarks.libero import LIBERO
from manifold.benchmarks.robocasa import ROBOCASA
from manifold.benchmarks.robolab import ROBOLAB
from manifold.benchmarks.simpler import SIMPLER

ALL = (LIBERO, ROBOCASA, ROBOLAB, SIMPLER)

__all__ = ["ALL", "LIBERO", "ROBOCASA", "ROBOLAB", "SIMPLER"]
