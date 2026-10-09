"""The names a policy and a benchmark share for cameras and embodiments.

Each name is a member of an enum, and a spec can only use a name this SDK lists.
A benchmark received from a peer built against a newer SDK may publish a camera
this one does not list; `Benchmark.from_received` drops it before parsing.
"""

from __future__ import annotations

from manifold.lib.compat import StrEnum


class _Name(StrEnum):
    # A name reads as its string in a repr, so an error message shows 'wrist' rather
    # than the enum's own repr.
    def __repr__(self) -> str:
        return repr(self.value)


class CameraName(_Name):
    """The name of each camera viewpoint."""

    AGENTVIEW = "agentview"
    AGENTVIEW_DEPTH = "agentview_depth"
    AGENTVIEW_LEFT = "agentview_left"
    AGENTVIEW_RIGHT = "agentview_right"
    OVER_SHOULDER_LEFT = "over_shoulder_left"
    WRIST = "wrist"
    WRIST_DEPTH = "wrist_depth"


class ControlMode(_Name):
    """The canonical control mode that ends every embodiment name.

    The action space plus its absolute-or-delta sense. `WHOLE_BODY` is a padded
    buffer whose extra slots are real action. `EE_ABSOLUTE` and `JOINT_DELTA` are
    reserved for an embodiment that needs them.
    """

    EE_ABSOLUTE = "ee_absolute"
    EE_DELTA = "ee_delta"
    JOINT_ABSOLUTE = "joint_absolute"
    JOINT_DELTA = "joint_delta"
    WHOLE_BODY = "whole_body"


class EmbodimentName(_Name):
    """The name of each embodiment: a robot or assembly, then its `ControlMode`."""

    DROID_JOINT_ABSOLUTE = "droid_joint_absolute"
    FRANKA_EE_DELTA = "franka_ee_delta"
    FRANKA_JOINT_ABSOLUTE = "franka_joint_absolute"
    PANDA_OMRON_WHOLE_BODY = "panda_omron_whole_body"
    WIDOWX_EE_DELTA = "widowx_ee_delta"


__all__ = [
    "CameraName",
    "ControlMode",
    "EmbodimentName",
]
