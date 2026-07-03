"""Convert the gripper encoding of an end-effector action to a target encoding.

A policy trained with one gripper convention (say a binary gripper that opens on
the low value) can drive a benchmark that expects another (a signed gripper that
opens on the high value). Everything else about the action stays the same; only
the gripper element is remapped, per step. This is a lossless representation
conversion.

The adapter is parameterized by the target encoding: `GripperPolarityAdapter(
target=GripperFormat.SIGNED)` converts any end-effector action whose gripper
differs into one with a signed gripper.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.conventions import GripperFormat
from manifold.core.embodiment import EEActionSpace
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
        """Remap the gripper element of each per-step block to the target encoding."""
        if not isinstance(source, EEActionSpace) or source.gripper is None:
            raise TypeError("GripperPolarityAdapter needs an EEActionSpace with a gripper")
        out = [float(v) for v in values]
        source.validate_value(out)  # the remap below assumes the declared length
        # The gripper is the last element of each per-step block.
        per_step = source.expected_length() // source.chunk_size
        for gripper_index in range(per_step - 1, len(out), per_step):
            out[gripper_index] = remap(out[gripper_index], source.gripper, self.target)
        return out


__all__ = ["GripperPolarityAdapter"]
