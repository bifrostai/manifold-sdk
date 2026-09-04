"""Re-encode the rotation of a reported end-effector pose to a target format.

A benchmark may report its end-effector pose as a quaternion while the policy was
trained on axis-angle. This adapter re-encodes the rotation portion of the
proprioception channel's `ee_pose` to the target format; position and gripper pass
through, as do the other channels (cameras, instruction). It is lossless.

Parameterized by the target format: `ProprioRotationAdapter(target=AXIS_ANGLE)`.

By default it converts through scipy (`lib.rotation.convert`), which canonicalizes
a quaternion to an axis-angle in [0, pi]. For the QUATERNION -> AXIS_ANGLE case a
policy may have been trained on a non-wrapping convention (angle = 2*acos(w),
keeping the sign so w < 0 gives an angle in (pi, 2*pi)); pass `wrap=False` to route
that one case through `quat_to_axisangle_nonwrapping`. That path re-narrows the
block to float32 before the conversion (a float32 quaternion keeps its axis at
float32 before scaling), so a policy trained on float32 axis-angles sees values
identical to its training data — not merely close. `wrap=True` (the default)
preserves the original scipy behavior for every format, so existing pairings are
unchanged. `wrap` only affects QUATERNION -> AXIS_ANGLE; every other conversion
takes the scipy path regardless.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.conventions import RotationFormat, ee_step_layout
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation
from manifold.lib.rotation import convert, quat_to_axisangle_nonwrapping


class ProprioRotationAdapter(ObservationAdapter):
    """Re-encode the `ee_pose` rotation of the reported proprioception to `target`."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(self, target: RotationFormat, *, wrap: bool = True) -> None:
        self.target = target
        self.wrap = wrap

    def _encode_rotation(self, rot_block: list[float], source: RotationFormat) -> list[float]:
        """Re-encode one rotation block from `source` to `self.target`.

        Routes QUATERNION -> AXIS_ANGLE through the non-wrapping conversion when
        `wrap=False`; every other case (and `wrap=True`) takes the scipy path.
        """
        if (
            not self.wrap
            and source == RotationFormat.QUATERNION
            and self.target == RotationFormat.AXIS_ANGLE
        ):
            # Re-narrow to float32 before the non-wrapping conversion. `adapt` decodes
            # the block to Python floats (float64), but the runner this path mirrors
            # bakes from a float32 quaternion, and numpy keeps `quat[:3] / den` at
            # float32 — rounding the axis to float32 before scaling by the float64
            # angle. Converting from the float64-promoted block instead keeps the axis
            # at float64 and diverges by a ULP for some quaternions (e.g. w<0). The
            # input is a lossless float32->float64 promotion, so narrowing back is
            # exact and reproduces the runner's float32 arithmetic byte-for-byte.
            quat_f32 = np.asarray(rot_block, dtype=np.float32)
            return quat_to_axisangle_nonwrapping(quat_f32)
        return convert(rot_block, source, self.target)

    def applies(self, source: BaseModel) -> bool:
        """True when the reported pose carries a rotation in a different format."""
        return (
            isinstance(source, ObservationSpace)
            and source.proprioception.ee_pose is not None
            and source.proprioception.ee_pose.rotation != self.target
        )

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with its `ee_pose` rotation set to the target format."""
        spec = self._spec(source)
        ee_pose = spec.proprioception.ee_pose
        if ee_pose is None:
            raise TypeError("ProprioRotationAdapter needs a proprioception with an ee_pose")
        new_proprio = spec.proprioception.model_copy(
            update={"ee_pose": ee_pose.model_copy(update={"rotation": self.target})}
        )
        return spec.with_proprioception(new_proprio)

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Re-encode the rotation portion of the `ee_pose` state array."""
        ee_pose = self._spec(source).proprioception.ee_pose
        if ee_pose is None:
            raise TypeError("ProprioRotationAdapter needs a proprioception with an ee_pose")
        block = [float(v) for v in observation.state["ee_pose"]]
        # Only the position and rotation lengths are needed to slice out the
        # rotation block; the gripper slot (the discarded third element) does not
        # affect those, so pass None rather than the GripperObservationSpec.
        pos_len, rot_len, _ = ee_step_layout(ee_pose.rotation, None)
        reencoded = (
            block[:pos_len]  # position
            + self._encode_rotation(block[pos_len : pos_len + rot_len], ee_pose.rotation)
            + block[pos_len + rot_len :]  # gripper, if any
        )
        state = {**observation.state, "ee_pose": np.asarray(reencoded, dtype=np.float32)}
        return replace(observation, state=state)

    @staticmethod
    def _spec(source: BaseModel) -> ObservationSpace:
        if not isinstance(source, ObservationSpace):
            raise TypeError("ProprioRotationAdapter transforms an ObservationSpace source only")
        return source


__all__ = ["ProprioRotationAdapter"]
