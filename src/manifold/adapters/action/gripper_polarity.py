"""Convert the gripper encoding of an end-effector action to a target encoding.

A policy trained with one gripper convention (say a binary gripper that opens on
the low value) can drive a benchmark that expects another (a signed gripper that
opens on the high value). Everything else about the action stays the same; only
the gripper element is remapped. This is a lossless representation
conversion.

The adapter is parameterized by the target encoding: `GripperPolarityAdapter(
target=GripperFormat.SIGNED)` converts any end-effector action whose gripper
differs into one with a signed gripper.

`JointGripperPolarityAdapter` does the same for a joint action, whose arms end
in a gripper value too. A policy trained on a real robot often outputs the
gripper as a continuous value and leaves the open-or-close decision to its robot
client. A benchmark that takes the robot's own command expects a binary
gripper. `remap` rounds to the nearer end of a binary encoding, so the same
remap makes that decision at half open.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import EEActionSpace, JointActionSpace
from manifold.lib.gripper import remap


class GripperPolarityAdapter(ActionAdapter):
    """Remap the gripper element of an end-effector action to `target`."""

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = EEActionSpace
    lossless: ClassVar[bool] = True

    def __init__(self, target: GripperFormat) -> None:
        self.target = target

    def applies(self, source: BaseModel) -> bool:
        """True when `source` carries a gripper that differs from the target."""
        return (
            isinstance(source, EEActionSpace)
            and source.gripper is not None
            and source.gripper != self.target
        )

    def produce(self, source: BaseModel) -> EEActionSpace:
        """The same action space with its gripper set to the target encoding."""
        if not isinstance(source, EEActionSpace):
            raise TypeError("GripperPolarityAdapter transforms EEActionSpace only")
        return source.model_copy(update={"gripper": self.target})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Remap the gripper element of each arm to the target encoding."""
        if not isinstance(source, EEActionSpace) or source.gripper is None:
            raise TypeError("GripperPolarityAdapter needs an EEActionSpace with a gripper")
        out = [float(v) for v in values]
        source.validate_value(out)  # the remap below assumes the declared length
        # The gripper is the last element of each arm's values, and an action holds
        # `arm_count` of those groups. If the loop stepped over the whole action at
        # once, it would remap the last arm's gripper only, and the other arms would
        # keep the source encoding.
        per_arm = source.per_arm_length()
        for gripper_index in range(per_arm - 1, len(out), per_arm):
            out[gripper_index] = remap(out[gripper_index], source.gripper, self.target)
        return out


class JointGripperPolarityAdapter(ActionAdapter):
    """Remap the gripper element of each arm of a joint action to `target`.

    It is declared lossy because a binary `target` discards how far open the
    gripper was, so a policy opts in by placing it in its `Pipeline`.
    """

    from_spec: ClassVar[type[BaseModel]] = JointActionSpace
    to_spec: ClassVar[type[BaseModel]] = JointActionSpace
    lossless: ClassVar[bool] = False

    def __init__(self, target: GripperFormat) -> None:
        self.target = target

    def applies(self, source: BaseModel) -> bool:
        """True when `source` carries a gripper that differs from the target."""
        return (
            isinstance(source, JointActionSpace)
            and source.gripper is not None
            and source.gripper != self.target
        )

    def produce(self, source: BaseModel) -> JointActionSpace:
        """The same action space with its gripper set to the target encoding."""
        if not isinstance(source, JointActionSpace):
            raise TypeError("JointGripperPolarityAdapter transforms JointActionSpace only")
        return source.model_copy(update={"gripper": self.target})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Remap the gripper element of each arm to the target encoding."""
        if not isinstance(source, JointActionSpace) or source.gripper is None:
            raise TypeError("JointGripperPolarityAdapter needs a JointActionSpace with a gripper")
        out = [float(v) for v in values]
        source.validate_value(out)  # the remap below assumes the declared length
        # Each arm's joints come first, then that arm's gripper, as on the
        # end-effector adapter above.
        per_arm = source.per_arm_length()
        for gripper_index in range(per_arm - 1, len(out), per_arm):
            out[gripper_index] = remap(out[gripper_index], source.gripper, self.target)
        return out


__all__ = ["GripperPolarityAdapter", "JointGripperPolarityAdapter"]
