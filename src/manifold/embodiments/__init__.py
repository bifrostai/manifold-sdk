"""Catalog of embodiments, one robot-and-control-mode per file.

An embodiment pairs a robot with the one canonical action space it is driven in,
so the same physical arm under two control modes is two embodiments: the Franka
appears here once driven in end-effector deltas (`franka_ee_delta`) and once in
absolute joint targets (`franka_joint_absolute`).

Every name is `<robot-or-assembly>_<ControlMode>` and a member of `EmbodimentName`,
and the filename, the exported constant and `Embodiment.name` say the same thing.
Nothing else belongs in a name: rotation encoding, gripper polarity, frame and
proprioception stay fields of the declaration.

`ALL` lists every shipped embodiment, for a consumer enumerating them.
"""

from __future__ import annotations

from manifold.embodiments.droid_joint_absolute import DROID_JOINT_ABSOLUTE
from manifold.embodiments.franka_ee_delta import FRANKA_EE_DELTA
from manifold.embodiments.franka_joint_absolute import FRANKA_JOINT_ABSOLUTE
from manifold.embodiments.panda_omron_whole_body import PANDA_OMRON_WHOLE_BODY
from manifold.embodiments.widowx_ee_delta import WIDOWX_EE_DELTA

ALL = (
    DROID_JOINT_ABSOLUTE,
    FRANKA_EE_DELTA,
    FRANKA_JOINT_ABSOLUTE,
    PANDA_OMRON_WHOLE_BODY,
    WIDOWX_EE_DELTA,
)

__all__ = [
    "ALL",
    "DROID_JOINT_ABSOLUTE",
    "FRANKA_EE_DELTA",
    "FRANKA_JOINT_ABSOLUTE",
    "PANDA_OMRON_WHOLE_BODY",
    "WIDOWX_EE_DELTA",
]
