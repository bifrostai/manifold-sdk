"""Catalog of embodiments, one robot-and-control-mode per file.

An embodiment pairs a robot with the one canonical action space it is driven in,
so the same physical arm under two control modes is two embodiments: the Franka
appears here once driven in end-effector deltas (`franka_ee`) and once in joint
targets (`franka_joint`).

`ALL` lists every shipped embodiment, for a consumer that wants to enumerate them.
"""

from __future__ import annotations

from manifold.embodiments.franka_ee import FRANKA_EE
from manifold.embodiments.franka_joint import FRANKA_JOINT
from manifold.embodiments.panda_omron import PANDA_OMRON
from manifold.embodiments.widowx import WIDOWX

ALL = (FRANKA_EE, FRANKA_JOINT, PANDA_OMRON, WIDOWX)

__all__ = ["ALL", "FRANKA_EE", "FRANKA_JOINT", "PANDA_OMRON", "WIDOWX"]
