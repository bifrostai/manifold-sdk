"""Re-encode the rotation of a reported end-effector pose to a target format.

A benchmark may report its end-effector pose as a quaternion while the policy was
trained on axis-angle. This adapter re-encodes the rotation portion of the
proprioception channel's `ee_pose` to the target format; position and gripper pass
through, as do the other channels (cameras, instruction). It is lossless.

Parameterized by the target format: `ProprioRotationAdapter(target=AXIS_ANGLE)`.

By default, QUATERNION -> AXIS_ANGLE uses the non-wrapping conversion
(`quat_to_axisangle_nonwrapping`): angle = 2*acos(w), keeping the sign, so w < 0
gives an angle in (pi, 2*pi). LIBERO's dataset scripts and the LIBERO training sets
on Hugging Face store axis-angle this way. That path re-narrows the block to float32
before the conversion (a float32 quaternion keeps its axis at float32 before
scaling), so a policy trained on float32 axis-angles sees values identical to its
training data. Pass `wrap=True` for a policy trained on scipy's `as_rotvec`, which
wraps the angle into [0, pi] and flips the axis when w < 0. `wrap` only affects
QUATERNION -> AXIS_ANGLE; every other conversion goes through scipy
(`lib.rotation.convert`).

`channel="base_pose"` re-encodes the rotation of a mobile robot's `base_pose`
instead, with the same `target` and `wrap`. A base pose is one pose with no
gripper.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from manifold.core.adapter import ObservationAdapter
from manifold.core.conventions import RotationFormat, ee_step_layout
from manifold.core.embodiment import EEObservationSpec, Proprioception
from manifold.core.observation_space import ObservationSpace
from manifold.core.values import Observation
from manifold.lib.rotation import convert, quat_to_axisangle_nonwrapping


class ProprioRotationAdapter(ObservationAdapter):
    """Re-encode the rotation of one proprioception pose, `ee_pose` by default, to `target`."""

    from_spec: ClassVar[type[BaseModel]] = ObservationSpace
    to_spec: ClassVar[type[BaseModel]] = ObservationSpace
    lossless: ClassVar[bool] = True

    def __init__(
        self,
        target: RotationFormat,
        *,
        wrap: bool = False,
        channel: str = "ee_pose",
    ) -> None:
        poses = Proprioception.pose_fields()
        if channel not in poses:
            raise ValueError(
                f"ProprioRotationAdapter re-encodes a pose ({', '.join(poses)}), not {channel!r}"
            )
        self.target = target
        self.wrap = wrap
        self.channel = channel

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
        """True when the channel's pose carries a rotation in a different format."""
        if not isinstance(source, ObservationSpace):
            return False
        spec = getattr(source.proprioception, self.channel)
        return spec is not None and spec.rotation != self.target

    def produce(self, source: BaseModel) -> ObservationSpace:
        """The same spec with the channel's rotation set to the target format."""
        spec = self._spec(source)
        pose = self._pose(spec)
        new_proprio = spec.proprioception.model_copy(
            update={self.channel: pose.model_copy(update={"rotation": self.target})}
        )
        return spec.with_proprioception(new_proprio)

    def adapt(self, observation: Any, *, source: BaseModel) -> Observation:
        """Re-encode the rotation portion of the channel's state array."""
        pose = self._pose(self._spec(source))
        values = [float(v) for v in observation.state[self.channel]]
        # The loop below reads exactly `arm_count` arms. If the value is longer than
        # the spec declares, the extra arms would be dropped without any error, so
        # check the length first and raise.
        pose.validate_value(values)
        # An ee_pose repeats a position, a rotation and the gripper's fingers once
        # per arm, and a base pose is one such arm with no gripper. If the whole
        # value were treated as a single arm, only arm 0 would be transformed.
        pos_len, rot_len, _ = ee_step_layout(pose.rotation, None)
        arms, per_arm = pose.arm_count, pose.per_arm_length()
        reencoded: list[float] = []
        for arm in range(arms):
            arm_values = values[arm * per_arm : (arm + 1) * per_arm]
            reencoded += (
                arm_values[:pos_len]  # position
                + self._encode_rotation(arm_values[pos_len : pos_len + rot_len], pose.rotation)
                + arm_values[pos_len + rot_len :]  # gripper, if any
            )
        state = {**observation.state, self.channel: np.asarray(reencoded, dtype=np.float32)}
        return replace(observation, state=state)

    def _pose(self, spec: ObservationSpace) -> EEObservationSpec:
        pose = getattr(spec.proprioception, self.channel)
        if pose is None:
            raise TypeError(f"ProprioRotationAdapter needs a proprioception with {self.channel}")
        return pose

    @staticmethod
    def _spec(source: BaseModel) -> ObservationSpace:
        if not isinstance(source, ObservationSpace):
            raise TypeError("ProprioRotationAdapter transforms an ObservationSpace source only")
        return source


__all__ = ["ProprioRotationAdapter"]
