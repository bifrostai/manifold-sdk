"""Robotics conventions attached to every action and observation spec.

A spec declares the shape of a value and its meaning. Two specs can agree on
length and still be incompatible: one opens the gripper at +1, the other at -1.
The enums here name these conventions, so a mismatch is a parameter difference a
check can detect before a rollout.

This is the lowest layer of `core/`. The action and observation specs
(`embodiment.py`) and the camera spec (`sensor.py`) import the enums and the two
length helpers from here.

Schema-wide conventions, fixed and not per-spec: end-effector position is in
meters; numerical values are float32 unless a spec says otherwise.
"""

from __future__ import annotations

from typing import Any

from manifold.lib.compat import StrEnum, assert_never


class RotationFormat(StrEnum):
    """Numerical encoding of a 3D rotation."""

    # 3 floats: axis direction scaled by rotation angle (a.k.a. rotation vector).
    AXIS_ANGLE = "axis_angle"
    # 4 floats, ordered (qx, qy, qz, qw).
    QUATERNION = "quaternion"
    # 3 floats, XYZ extrinsic Euler angles in radians. CALVIN-style convention.
    EULER_XYZ = "euler_xyz"
    # 6 floats, two non-collinear column vectors of the rotation matrix
    # (Zhou et al., 2018). RDT-1B and several diffusion-based policies use this.
    ROTATION_6D = "rotation_6d"


class GripperFormat(StrEnum):
    """Numerical encoding of a gripper command or observation value.

    A gripper convention has two parts: the numeric range, and which end of the
    range means open. Both must match. If the polarity is reversed the gripper
    moves opposite to the commanded value, and the task fails without raising.
    Robots disagree on polarity (some treat the high value as open, some as
    closed), so it is named here rather than assumed.

    SIGNED, UNSIGNED, and BINARY treat the high value as open. The _OPEN_LOW
    variants invert that, for benchmarks where the low value is open.
    """

    # Continuous in [-1, +1]; +1 = open, -1 = closed.
    SIGNED = "signed"
    # Continuous in [-1, +1]; -1 = open, +1 = closed.
    SIGNED_OPEN_LOW = "signed_open_low"
    # Continuous in [0, 1]; 1 = open, 0 = closed.
    UNSIGNED = "unsigned"
    # Continuous in [0, 1]; 0 = open, 1 = closed.
    UNSIGNED_OPEN_LOW = "unsigned_open_low"
    # Discrete {0, 1}; 1 = open, 0 = closed.
    BINARY = "binary"
    # Discrete {0, 1}; 0 = open, 1 = closed.
    BINARY_OPEN_LOW = "binary_open_low"


class Frame(StrEnum):
    """Reference frame an end-effector pose or delta is expressed in.

    The same values mean different positions in different frames, with no error
    if mismatched. Naming the frame lets a check detect the mismatch and an
    adapter convert between frames.
    """

    # Fixed world / table frame. The default for the benchmarks shipped so far.
    WORLD = "world"
    # Robot base frame.
    BASE = "base"
    # End-effector / tool frame (deltas relative to the current pose).
    TOOL = "tool"
    # The WidowX/Bridge env's fixed EE reference orientation — a static
    # reorientation of BASE, not a kinematic transform. GR00T/RLDX's
    # WidowXBridgeEnv reads its proprio euler in this frame.
    WIDOWX_BRIDGE_EE = "widowx_bridge_ee"


def rotation_dims(rotation: RotationFormat) -> int:
    """Number of floats the chosen rotation encoding takes per step."""
    if rotation is RotationFormat.AXIS_ANGLE or rotation is RotationFormat.EULER_XYZ:
        return 3
    if rotation is RotationFormat.QUATERNION:
        return 4
    if rotation is RotationFormat.ROTATION_6D:
        return 6
    return assert_never(rotation)


def ee_step_layout(rotation: RotationFormat, gripper: GripperFormat | None) -> tuple[int, int, int]:
    """Per-step float counts of an end-effector value: (position, rotation, gripper).

    Position is always 3 (xyz). The single source of truth for how an EE pose or
    action is laid out, used by the specs and by the adapters that slice it.
    """
    return 3, rotation_dims(rotation), (1 if gripper is not None else 0)


def check_length(value: Any, expected: int, label: str) -> None:
    """Raise if `value` is not a sequence of `expected` length.

    Anything with `__len__` is accepted: a numpy array, a torch tensor, or a
    plain list.
    """
    if not hasattr(value, "__len__"):
        raise TypeError(f"{label}: expected a sequence with len(), got {type(value).__name__}")
    actual = len(value)
    if actual != expected:
        raise ValueError(f"{label}: length mismatch: expected {expected}, got {actual}")


__all__ = [
    "Frame",
    "GripperFormat",
    "RotationFormat",
    "check_length",
    "ee_step_layout",
    "rotation_dims",
]
