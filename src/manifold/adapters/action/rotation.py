"""Convert the rotation encoding of an end-effector action to a target format.

A policy trained to emit euler-XYZ rotations can drive a benchmark that expects
axis-angle (or quaternion, or 6D). Only the rotation portion of each step is
re-encoded; the position and gripper pass through. Re-encoding the same rotation
in another format is lossless.

Parameterized by the target format: `RotationFormatAdapter(target=AXIS_ANGLE)`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from manifold.core.adapter import ActionAdapter
from manifold.core.conventions import RotationFormat, ee_step_layout
from manifold.core.embodiment import EEActionSpace
from manifold.lib.rotation import convert


class RotationFormatAdapter(ActionAdapter):
    """Re-encode the rotation of an end-effector action to `target`."""

    from_spec: ClassVar[type[BaseModel]] = EEActionSpace
    to_spec: ClassVar[type[BaseModel]] = EEActionSpace
    lossless: ClassVar[bool] = True

    def __init__(self, target: RotationFormat) -> None:
        self.target = target

    def applies(self, source: BaseModel) -> bool:
        """True when `source` is an end-effector action in a different rotation format."""
        return isinstance(source, EEActionSpace) and source.rotation != self.target

    def produce(self, source: BaseModel) -> EEActionSpace:
        """The same action space with its rotation set to the target format."""
        if not isinstance(source, EEActionSpace):
            raise TypeError("RotationFormatAdapter transforms EEActionSpace only")
        return source.model_copy(update={"rotation": self.target})

    def adapt(self, values: Any, *, source: BaseModel) -> list[float]:
        """Re-encode the rotation of each per-step block; pass position and gripper through."""
        if not isinstance(source, EEActionSpace):
            raise TypeError("RotationFormatAdapter transforms EEActionSpace only")
        buffer = [float(v) for v in values]
        source.validate_value(buffer)
        pos_len, rot_len, _ = ee_step_layout(source.rotation, source.gripper)
        per_step = pos_len + rot_len + (1 if source.gripper is not None else 0)
        out: list[float] = []
        for step in range(source.chunk_size):
            block = buffer[step * per_step : (step + 1) * per_step]
            out.extend(block[:pos_len])  # position
            out.extend(convert(block[pos_len : pos_len + rot_len], source.rotation, self.target))
            out.extend(block[pos_len + rot_len :])  # gripper, if any
        return out


__all__ = ["RotationFormatAdapter"]
